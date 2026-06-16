"""
project_id=3, usage_statement_id=2 validator 테스트 스크립트.

토큰 추적 수정(CategoryRuleBundle.token_usage → CategoryAuditResult.token_usage
→ AuditResponse.total_token_usage) 검증 포함.

사용법:
  cd fastapi
  python tests/agents/validator_agent/test_validator_project3.py
  python tests/agents/validator_agent/test_validator_project3.py --model gpt-4o
  python tests/agents/validator_agent/test_validator_project3.py --verbose
  python tests/agents/validator_agent/test_validator_project3.py --full-payload
"""

import argparse
import json
import logging
import os
import sys
from time import perf_counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.agents.validator_agent.agent import (
    summarize_audit_response,
    validate_usage_statement,
)
from src.core import llm_config
from src.schemas.classifier import CATEGORIES
from src.schemas.validator import AuditResponse
from src.services.orchestrator_service import (
    _legal_apply_generated_item_reasons,
    _legal_basis_by_citation,
    _legal_citations_from_results,
    _legal_frontend_categories,
    _legal_item_results_from_audit,
    _legal_payload_item_results,
    _linked_files_by_item_id,
)

load_dotenv()

TARGET_PROJECT_ID = 3
TARGET_USAGE_STATEMENT_ID = 2


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
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
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

            if int(row["project_id"]) != TARGET_PROJECT_ID:
                raise ValueError(
                    f"usage_statement id={usage_statement_id}의 project_id={row['project_id']}이나 "
                    f"기대값은 {TARGET_PROJECT_ID}입니다."
                )

            cur.execute(
                """
                SELECT category_code, previous_amount, current_amount, cumulative_amount
                FROM usage_statement_summaries
                WHERE usage_statement_id = %s
                """,
                (usage_statement_id,),
            )
            summaries = [dict(r) for r in cur.fetchall()]

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

    for item in items:
        item["source_row_id"] = item["id"]
        if item.get("used_on") and not isinstance(item["used_on"], str):
            item["used_on"] = item["used_on"].isoformat()

    return {
        "project": {"id": int(row["project_id"])},
        "usage_statement": {
            "id": usage_statement_id,
            "project_id": int(row["project_id"]),
            "source_file_id": row["source_file_id"],
            "report_month": str(row["report_month"]) if row["report_month"] else None,
            "revision_no": row["revision_no"],
            "document_written_date": str(row["document_written_date"])
            if row["document_written_date"]
            else None,
            "cumulative_progress_rate": float(row["cumulative_progress_rate"])
            if row["cumulative_progress_rate"] is not None
            else None,
        },
        "usage_statement_summaries": summaries,
        "usage_statement_items": items,
        "validator_context": {
            "safety_budget_total": int(row["appropriated_amount"])
            if row["appropriated_amount"]
            else None,
            "cumulative_progress_rate": float(row["cumulative_progress_rate"])
            if row["cumulative_progress_rate"] is not None
            else None,
        },
    }


def _validator_input_from_v1(v1_input: dict) -> dict:
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


# ── 실행 ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"project_id={TARGET_PROJECT_ID}, usage_statement_id={TARGET_USAGE_STATEMENT_ID} validator 테스트 (토큰 추적 포함)"
    )
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--full-payload", action="store_true")
    parser.add_argument(
        "--debug-items",
        action="store_true",
        help="항목별 judgment_source, allowed, reasoning, reason_text 출력",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        )
        for noisy in ("httpx", "httpcore", "huggingface_hub"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY 없음 — .env 파일 확인")
        sys.exit(1)
    llm_config.configure(ChatOpenAI(model=args.model, temperature=0))

    W = 70
    print(f"\n{'=' * W}")
    print(
        f"  project_id={TARGET_PROJECT_ID} / usage_statement_id={TARGET_USAGE_STATEMENT_ID}  |  model: {args.model}"
    )
    print(f"{'=' * W}\n")

    # 1. DB 조회
    t0 = perf_counter()
    v1_input = fetch_v1_input(TARGET_USAGE_STATEMENT_ID)
    validator_input = _validator_input_from_v1(v1_input)
    print("[DB 조회 완료]")
    print(f"  project_id       : {v1_input['project']['id']}")
    budget = v1_input["validator_context"]["safety_budget_total"]
    print(f"  산안비총액        : {budget:,}" if budget else "  산안비총액        : -")
    rate = v1_input["validator_context"]["cumulative_progress_rate"]
    print(
        f"  누계공정률        : {rate}%"
        if rate is not None
        else "  누계공정률        : -"
    )
    print(f"  summaries 수      : {len(v1_input['usage_statement_summaries'])}")
    print(f"  items 수          : {len(v1_input['usage_statement_items'])}")

    if args.verbose:
        print("\n[validator 입력 전체]")
        print(json.dumps(validator_input, ensure_ascii=False, indent=2, default=str))

    # 2. Validator 실행
    print("\nValidator 실행 중...\n")
    t1 = perf_counter()
    response: AuditResponse = validate_usage_statement(document=validator_input)
    elapsed_validate = perf_counter() - t1
    summary_response = summarize_audit_response(
        response=response, usage_statement_id=TARGET_USAGE_STATEMENT_ID
    )
    elapsed_total = perf_counter() - t0

    # 3. 토큰 집계 출력 (핵심 검증 포인트)
    print("[토큰 집계]")
    total_tokens = response.total_token_usage
    print(
        f"  전체 토큰 합계    : {total_tokens:,}"
        if total_tokens
        else "  전체 토큰 합계    : 0 (LLM 미사용 또는 집계 오류)"
    )
    print("  카테고리별:")
    for cat_code, cat_result in sorted(response.categories.items()):
        cat_name = CATEGORIES.get(cat_code, "")
        label = (
            f"{cat_code} ({cat_name})"
            if cat_name and cat_name != cat_code
            else cat_code
        )
        item_count = len(cat_result.items)
        print(f"    {label}: {cat_result.token_usage:,} tokens  ({item_count}항목)")

    # 3-b. 항목별 내부 판정 디버그
    if args.debug_items:
        print("\n[항목별 내부 판정]")
        for cat_code, cat_result in sorted(response.categories.items()):
            cat_name = CATEGORIES.get(cat_code, cat_code)
            for item in cat_result.items:
                verdict = "✓ 허용" if item.allowed else "✗ 불허"
                print(
                    f"  [{cat_code}] {item.item:30s}  {verdict}  source={item.judgment_source}  conf={item.confidence:.2f}"
                )
                if item.reasoning:
                    print(f"    evidence : {item.reasoning[:150]}")
                if item.reason_text:
                    print(f"    reason   : {item.reason_text[:150]}")
        print()

    # 4. 결과 조립
    items_by_code: dict[str, list[dict]] = {}
    for row in v1_input.get("usage_statement_items", []):
        items_by_code.setdefault(row["category_code"], []).append(row)

    category_rows = {}
    for code, rows in items_by_code.items():
        category_rows[code] = [
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
        ]

    item_results = _legal_item_results_from_audit(
        audit_response=response,
        summary_response=summary_response,
        category_rows=category_rows,
    )
    linked_files_by_item_id = _linked_files_by_item_id(TARGET_USAGE_STATEMENT_ID)
    legal_basis_by_source_id = _legal_basis_by_citation(
        _legal_citations_from_results(item_results)
    )
    _legal_apply_generated_item_reasons(
        item_results=item_results,
        legal_basis_by_source_id=legal_basis_by_source_id,
    )
    categories = _legal_frontend_categories(
        item_results=item_results,
        category_rows=category_rows,
        linked_files_by_item_id=linked_files_by_item_id,
        legal_basis_by_source_id=legal_basis_by_source_id,
    )
    payload_item_results = _legal_payload_item_results(item_results)

    review_count = sum(1 for row in item_results if row["status"] != "적절")
    result_code = "hil" if review_count else "success"
    reason = (
        "법령 검토 결과 특이사항 없음"
        if review_count == 0
        else f"법령 검토 결과 보고서 반영 대상 {review_count}건"
    )

    # 5. 최종 결과 출력
    print("\n[검증 결과]")
    print(f"  result_code      : {result_code}")
    print(f"  reason           : {reason}")
    print(f"  이슈 항목         : {review_count}건")
    print(
        f"  소요 시간         : {elapsed_validate:.1f}s (validate) / {elapsed_total:.1f}s (total)"
    )

    print("\n[item_results]")
    print(json.dumps(payload_item_results, ensure_ascii=False, indent=2, default=str))

    todos = [
        {
            "usage_statement_item_id": row.get("item_id"),
            "category_code": row.get("category_code"),
            "category_name": CATEGORIES.get(str(row.get("category_code") or "")),
            "reason": f"법령 검토 필요: {row.get('reason') or row.get('status')}",
        }
        for row in item_results
        if row["status"] != "적절"
    ]
    if todos:
        print("\n[todos]")
        print(json.dumps(todos, ensure_ascii=False, indent=2, default=str))

    if args.full_payload:
        print("\n[categories payload 전체]")
        print(json.dumps(categories, ensure_ascii=False, indent=2, default=str))

    if args.verbose:
        print("\n[AuditResponse 전체]")
        print(json.dumps(response.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
