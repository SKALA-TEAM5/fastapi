"""
DB에서 직접 usage_statement 데이터를 가져와 validator를 테스트하는 스크립트.

사용법:
  uv run python -m tests.agents.validator_agent.test_validator_from_db
  uv run python -m tests.agents.validator_agent.test_validator_from_db --id 1
  uv run python -m tests.agents.validator_agent.test_validator_from_db --id 1 --verbose
"""

import argparse
import json
import os

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.agents.validator_agent.agent import summarize_audit_response, validate_usage_statement
from src.core import llm_config
from src.schemas.validator import AuditResponse
from src.services.orchestrator_service import _legal_item_results_from_audit

load_dotenv()


# ── DB 연결 ──────────────────────────────────────────────────────────────────

def _get_conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "safety"),
        user=os.getenv("SERVICE_APP_USER", "safety_service_app"),
        password=os.getenv("SERVICE_APP_PASSWORD", "safety_service_app_password"),
        options="-c search_path=service",
    )


# ── DB → v1_input 조립 ───────────────────────────────────────────────────────

def fetch_v1_input(usage_statement_id: int) -> dict:
    """DB 3개 테이블을 조회해 validator가 받는 v1_input 구조로 조립."""
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:

            # ① projects + usage_statements JOIN
            cur.execute(
                """
                SELECT p.id            AS project_id,
                       p.appropriated_amount,
                       us.id           AS usage_statement_id,
                       us.project_id,
                       us.source_file_id,
                       us.report_month,
                       us.revision_no,
                       us.document_written_date,
                       us.cumulative_progress_rate
                FROM usage_statements us
                JOIN projects p ON p.id = us.project_id
                WHERE us.id = %s
                """,
                (usage_statement_id,),
            )
            row = cur.fetchone()
            if not row:
                raise ValueError(f"usage_statement id={usage_statement_id} 없음")

            # ② usage_statement_summaries
            cur.execute(
                """
                SELECT category_code,
                       previous_amount,
                       current_amount,
                       cumulative_amount
                FROM usage_statement_summaries
                WHERE usage_statement_id = %s
                """,
                (usage_statement_id,),
            )
            summaries = [dict(r) for r in cur.fetchall()]

            # ③ usage_statement_items
            cur.execute(
                """
                SELECT id, category_code, used_on, item_name,
                       unit, quantity, unit_price, total_amount, remark
                FROM usage_statement_items
                WHERE usage_statement_id = %s
                ORDER BY category_code, id
                """,
                (usage_statement_id,),
            )
            items = [dict(r) for r in cur.fetchall()]

    # source_row_id = id (스키마에 source_row_id 컬럼 없음)
    for i, item in enumerate(items, 1):
        item["source_row_id"] = item["id"]
        # used_on은 date 객체일 수 있으므로 문자열 변환
        if item.get("used_on") and not isinstance(item["used_on"], str):
            item["used_on"] = item["used_on"].isoformat()

    return {
        "project": {
            "id": row["project_id"],
        },
        "usage_statement": {
            "id": usage_statement_id,
            "project_id": row["project_id"],
            "source_file_id": row["source_file_id"],
            "report_month": str(row["report_month"]) if row["report_month"] else None,
            "revision_no": row["revision_no"],
            "document_written_date": str(row["document_written_date"]) if row["document_written_date"] else None,
            "cumulative_progress_rate": float(row["cumulative_progress_rate"]) if row["cumulative_progress_rate"] is not None else None,
        },
        "usage_statement_summaries": summaries,
        "usage_statement_items": items,
        "validator_context": {
            "safety_budget_total": int(row["appropriated_amount"]) if row["appropriated_amount"] else None,
            "cumulative_progress_rate": float(row["cumulative_progress_rate"]) if row["cumulative_progress_rate"] is not None else None,
        },
    }


# ── v1_input → validator 입력 구조 ───────────────────────────────────────────

def _validator_input_from_v1(v1_input: dict) -> dict:
    """test_validator_agent.py의 동일 함수와 동일한 로직."""
    usage_statement = v1_input["usage_statement"]
    context = v1_input.get("validator_context") or {}
    summaries_by_code = {
        row["category_code"]: row
        for row in v1_input.get("usage_statement_summaries", [])
    }
    items_by_code: dict[str, list[dict]] = {}
    for item in v1_input.get("usage_statement_items", []):
        items_by_code.setdefault(item["category_code"], []).append(item)

    categories = []
    for category_code, summary in summaries_by_code.items():
        rows = sorted(items_by_code.get(category_code, []), key=lambda r: r["id"])
        categories.append(
            {
                "카테고리코드": category_code,
                "집계정보": {
                    "전회사용금액": summary.get("previous_amount") or 0,
                    "금회사용금액": summary.get("current_amount") or 0,
                    "누적사용금액": summary.get("cumulative_amount") or 0,
                },
                "항목목록": [
                    {
                        "행ID": row.get("source_row_id") or row["id"],
                        "사용일자": row.get("used_on"),
                        "항목명": row.get("item_name"),
                        "단위": row.get("unit"),
                        "수량": row.get("quantity"),
                        "단가": row.get("unit_price"),
                        "금액": row.get("total_amount"),
                        "비고": row.get("remark") or "",
                    }
                    for row in rows
                ],
            }
        )

    return {
        "사용내역서ID": usage_statement["id"],
        "기본정보": {
            "산안비총액": context.get("safety_budget_total"),
            "누계공정률": context.get("cumulative_progress_rate")
            if context.get("cumulative_progress_rate") is not None
            else usage_statement.get("cumulative_progress_rate"),
        },
        "카테고리별데이터": categories,
    }


def _agent_logs_from_response(
    *,
    v1_input: dict,
    summary_response,
    item_results: list[dict],
    result_code: str,
    reason: str,
    todos: list[dict],
    model_name: str,
) -> dict:
    """DB에 쓰지 않고 orchestrator _run_legal_agent()의 upsert row payload를 만든다."""
    project_id = int(v1_input["project"]["id"])
    usage_statement_id = int(v1_input["usage_statement"]["id"])
    category_results = [
        result.model_dump(mode="json", by_alias=False)
        for result in summary_response.results
    ]

    return {
        "project_id": project_id,
        "usage_statement_id": usage_statement_id,
        "usage_statement_item_id": None,
        "agent_type_code": "legal",
        "status_code": "success",
        "result_code": result_code,
        "reason": reason,
        "details": {
            "event": "legal_completed",
            "summary": reason,
            "payload": {
                "she_user_id": None,
                "usage_statement_id": usage_statement_id,
                "category_results": category_results,
                "results": item_results,
                "todos": todos,
            },
        },
        "model_name": model_name,
        "token": None,
    }


# ── 실행 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="DB 데이터로 validator 테스트")
    parser.add_argument("--id", type=int, default=1, help="usage_statement id (기본값: 1)")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        llm_config.configure(ChatOpenAI(model=args.model, temperature=0))

    print(f"\n{'=' * 70}")
    print(f"  usage_statement id={args.id} DB 조회 → validator 실행")
    print(f"{'=' * 70}\n")

    # 1. DB 조회
    v1_input = fetch_v1_input(args.id)
    validator_input = _validator_input_from_v1(v1_input)

    print("[v1_input 요약]")
    print(f"  project_id       : {v1_input['project']['id']}")
    print(f"  산안비총액        : {v1_input['validator_context']['safety_budget_total']:,}")
    print(f"  누계공정률        : {v1_input['validator_context']['cumulative_progress_rate']}%")
    print(f"  summaries 수      : {len(v1_input['usage_statement_summaries'])}")
    print(f"  items 수          : {len(v1_input['usage_statement_items'])}")

    if args.verbose:
        print("\n[validator 입력 전체]")
        print(json.dumps(validator_input, ensure_ascii=False, indent=2, default=str))

    # 2. Validator 실행
    print("\nValidator 실행 중...\n")
    response: AuditResponse = validate_usage_statement(document=validator_input)
    summary_response = summarize_audit_response(response=response, usage_statement_id=args.id)

    # 3. orchestrator와 동일한 방식으로 결과 조립
    # category_rows: orchestrator의 _build_validator_document와 동일한 구조
    items_by_code: dict[str, list[dict]] = {}
    for row in v1_input.get("usage_statement_items", []):
        items_by_code.setdefault(row["category_code"], []).append(row)

    category_rows = {
        code: [{"행ID": r["id"], **r} for r in rows]
        for code, rows in items_by_code.items()
    }

    item_results = _legal_item_results_from_audit(
        audit_response=response,
        summary_response=summary_response,
        category_rows=category_rows,
    )

    # orchestrator와 동일한 result_code 계산
    review_count = sum(1 for row in item_results if row["status"] != "적절")
    category_issue_count = sum(
        1 for s in summary_response.results
        if s.status in {"부적절", "검토필요"}
    )
    result_code = "hil" if (review_count or category_issue_count) else "success"
    reason = "법령 검토 결과 특이사항 없음" if review_count == 0 else f"법령 검토 결과 보고서 반영 대상 {review_count}건"
    todos = [
        {
            "usage_statement_item_id": row.get("item_id"),
            "reason": f"법령 검토 필요: {row.get('reason') or row.get('status')}",
        }
        for row in item_results
        if row["status"] != "적절"
    ]
    agent_log = _agent_logs_from_response(
        v1_input=v1_input,
        summary_response=summary_response,
        item_results=item_results,
        result_code=result_code,
        reason=reason,
        todos=todos,
        model_name="validator_agent",
    )

    print(f"[결과] result_code={result_code} | {reason}")
    print(f"  항목 이슈: {review_count}건 / 카테고리 이슈: {category_issue_count}건")
    print("\n[item_results]")
    print(json.dumps(item_results, ensure_ascii=False, indent=2, default=str))
    if todos:
        print("\n[todos]")
        print(json.dumps(todos, ensure_ascii=False, indent=2, default=str))

    print("\n[agent_logs payload]")
    print(json.dumps(agent_log, ensure_ascii=False, indent=2, default=str))

    if args.verbose:
        print("\n[AuditResponse 전체]")
        print(json.dumps(response.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
