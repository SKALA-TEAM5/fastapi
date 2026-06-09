"""
DB에서 usage_statement_items를 가져와 classi Agent를 실행하는 확인 스크립트.

사용법:
  uv run python -m tests.agents.classifier_agent.test_classifier_from_db --id 1
  uv run python -m tests.agents.classifier_agent.test_classifier_from_db --id 1 --verbose
  uv run python -m tests.agents.classifier_agent.test_classifier_from_db --id 1 --item-id 10

기본 모드는 실제 카테고리를 수정하거나 agent_logs에 INSERT하지 않고,
orchestrator의 classi 완료 로그와 같은 aggregate row payload를 출력한다.
--item-id를 주면 orchestrator의 classify_existing_usage_statement() 단건 요청/응답/agent_logs
payload 형식을 INSERT 없이 재현한다.
"""

from __future__ import annotations

import argparse
import json
import os
from decimal import Decimal
from typing import Any

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.agents.classifier_agent.agent import review_usage_statement
from src.core import llm_config
from src.schemas.orchestrator import UsageStatementClassifyRequest

load_dotenv()


def _get_conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "safety"),
        user=os.getenv("SERVICE_APP_USER", "safety_service_app"),
        password=os.getenv("SERVICE_APP_PASSWORD", "safety_service_app_password"),
        options="-c search_path=service",
    )


def fetch_usage_statement_rows(usage_statement_id: int) -> dict[str, Any]:
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    us.id AS usage_statement_id,
                    us.project_id,
                    us.report_month,
                    us.revision_no,
                    us.document_written_date,
                    us.cumulative_progress_rate,
                    p.appropriated_amount
                FROM usage_statements us
                JOIN projects p ON p.id = us.project_id
                WHERE us.id = %s
                """,
                (usage_statement_id,),
            )
            header = cur.fetchone()
            if not header:
                raise ValueError(f"usage_statement id={usage_statement_id} 없음")

            cur.execute(
                """
                SELECT
                    id,
                    category_code,
                    used_on,
                    item_name,
                    unit,
                    quantity,
                    unit_price,
                    total_amount,
                    remark
                FROM usage_statement_items
                WHERE usage_statement_id = %s
                ORDER BY id
                """,
                (usage_statement_id,),
            )
            items = [dict(row) for row in cur.fetchall()]

    if not items:
        raise ValueError(f"usage_statement id={usage_statement_id} 항목 없음")

    rows = []
    for item in items:
        used_on = item.get("used_on")
        rows.append(
            {
                "row_id": item["id"],
                "given_category_code": item.get("category_code") or "",
                "used_on": used_on.isoformat() if hasattr(used_on, "isoformat") else used_on,
                "item_name": item.get("item_name") or "",
                "unit": item.get("unit"),
                "quantity": float(item["quantity"]) if item.get("quantity") is not None else None,
                "unit_price": float(item["unit_price"]) if item.get("unit_price") is not None else None,
                "total_amount": float(item["total_amount"]) if item.get("total_amount") is not None else 0,
                "remark": item.get("remark") or "",
            }
        )

    basic_info = {
        "report_month": str(header["report_month"]) if header.get("report_month") else None,
        "revision_no": header.get("revision_no"),
        "document_written_date": (
            str(header["document_written_date"])
            if header.get("document_written_date")
            else None
        ),
        "cumulative_progress_rate": (
            float(header["cumulative_progress_rate"])
            if header.get("cumulative_progress_rate") is not None
            else None
        ),
        "safety_budget_total": (
            int(header["appropriated_amount"])
            if header.get("appropriated_amount") is not None
            else None
        ),
    }

    return {
        "project_id": int(header["project_id"]),
        "usage_statement_id": int(header["usage_statement_id"]),
        "basic_info": {key: value for key, value in basic_info.items() if value is not None},
        "rows": rows,
    }


def _result_details(response_results: list[Any]) -> dict[str, Any]:
    results = []
    changed_count = 0
    for result in response_results:
        status = (
            "appropriate"
            if result.decision_status == "유지"
            else "inappropriate"
        )
        if result.decision_status == "카테고리변경":
            changed_count += 1
        results.append(
            {
                "row_id": result.row_id,
                "item_id": result.row_id,
                "item_name": result.item_name,
                "original_category_code": result.given_category_code,
                "final_category_code": result.final_category_code,
                "status": status,
                "decision_status": result.decision_status,
                "needs_human_review": result.needs_human_review,
                "reason": result.reason,
            }
        )

    summary = (
        f"세부내역 {changed_count}건을 올바른 항목으로 이동했습니다."
        if changed_count
        else "세부내역 분류 이동 없음"
    )
    return {
        "event": "classification_updated" if changed_count else "classification_checked",
        "summary": summary,
        "payload": {
            "changed_count": changed_count,
            "kept_count": len(results) - changed_count,
            "changes": [
                {
                    "row_id": row["row_id"],
                    "item_id": row["item_id"],
                    "item_name": row["item_name"],
                    "before": {"category_code": row["original_category_code"]},
                    "after": {"category_code": row["final_category_code"]},
                    "reason": row["reason"],
                }
                for row in results
                if row["decision_status"] == "카테고리변경"
            ],
            "results": results,
        },
    }


def _aggregate_agent_log_payload(
    *,
    project_id: int,
    usage_statement_id: int,
    details: dict[str, Any],
) -> dict[str, Any]:
    return {
        "project_id": project_id,
        "usage_statement_id": usage_statement_id,
        "usage_statement_item_id": None,
        "agent_type_code": "classi",
        "status_code": "success",
        "result_code": "success",
        "reason": details["summary"],
        "details": details,
        "model_name": "classifier_agent",
        "token": None,
    }


def _to_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    return Decimal(str(value))


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(Decimal(str(value)).to_integral_value())


def _build_orchestrator_request(
    *,
    project_id: int,
    usage_statement_id: int,
    row: dict[str, Any],
) -> UsageStatementClassifyRequest:
    return UsageStatementClassifyRequest(
        project_id=project_id,
        usage_statement_id=usage_statement_id,
        item_id=row["row_id"],
        category_code=row.get("given_category_code") or "",
        item_name=row.get("item_name") or "",
        used_on=row.get("used_on"),
        unit=row.get("unit"),
        quantity=_to_decimal(row.get("quantity")),
        unit_price=_to_decimal(row.get("unit_price")),
        total_amount=_to_int(row.get("total_amount")),
        remark=row.get("remark") or None,
    )


def _orchestrator_single_item_payload(
    *,
    request: UsageStatementClassifyRequest,
) -> tuple[dict[str, Any], dict[str, Any]]:
    submitted_category = request.category_code or ""
    review_response = review_usage_statement(
        usage_statement_id=request.usage_statement_id,
        rows=[
            {
                "row_id": 1,
                "given_category_code": submitted_category,
                "item_name": request.item_name,
            }
        ],
        basic_info={},
    )
    review_map = {result.row_id: result for result in review_response.results}
    review = review_map.get(1)

    if review is None:
        updated_category = submitted_category
        status = "appropriate"
        reason = "classifier result was missing, so the submitted category was kept."
        item_name = request.item_name
    else:
        updated_category = review.final_category_code or submitted_category
        status = "appropriate" if review.decision_status == "유지" else "inappropriate"
        reason = review.reason
        item_name = review.item_name

    changes: list[dict[str, Any]] = []
    if updated_category != submitted_category:
        changes.append(
            {
                "row_id": 1,
                "item_id": request.item_id,
                "item_name": item_name,
                "before": {"category_code": submitted_category},
                "after": {"category_code": updated_category},
                "reason": reason,
            }
        )

    changed_count = len(changes)
    summary = (
        f"세부내역 {changed_count}건을 올바른 항목으로 이동했습니다."
        if changed_count
        else "세부내역 분류 이동 없음"
    )
    details = {
        "event": "classification_updated" if changed_count else "classification_checked",
        "summary": summary,
        "payload": {
            "changed_count": changed_count,
            "kept_count": 0 if changed_count else 1,
            "changes": changes,
            "results": [
                {
                    "row_id": 1,
                    "item_id": request.item_id,
                    "item_name": item_name,
                    "original_category_code": submitted_category,
                    "final_category_code": updated_category,
                    "status": status,
                    "reason": reason,
                }
            ],
        },
    }
    response_payload = {
        "status": "success",
        "message": summary,
        "usage_statement_id": request.usage_statement_id,
        "target_agents": ["classi"],
        "hil_agents": [],
        "result": details,
    }
    agent_log_payload = {
        "project_id": request.project_id,
        "usage_statement_id": request.usage_statement_id,
        "usage_statement_item_id": None,
        "agent_type_code": "classi",
        "status_code": "success",
        "result_code": "success",
        "reason": summary,
        "details": details,
        "model_name": "classifier_agent",
        "token": None,
    }
    return response_payload, agent_log_payload


def run_aggregate_classi(
    *,
    usage_statement_id: int,
    model: str,
    verbose: bool,
) -> None:
    data = fetch_usage_statement_rows(usage_statement_id)
    project_id = data["project_id"]
    rows = data["rows"]

    print(f"\n{'=' * 76}")
    print(f"  usage_statement id={usage_statement_id} DB 조회 -> classi 실행")
    print(f"{'=' * 76}\n")
    print(f"[DB 요약] project_id={project_id} | items={len(rows)}")

    if verbose:
        print("\n[classi 입력 rows]")
        print(json.dumps(rows, ensure_ascii=False, indent=2, default=str))

    response = review_usage_statement(
        usage_statement_id=usage_statement_id,
        rows=rows,
        basic_info=data["basic_info"],
        model_name=model,
    )
    details = _result_details(response.results)
    result_code = (
        "hil"
        if details["payload"]["changed_count"]
        else "success"
    )
    agent_log = _aggregate_agent_log_payload(
        project_id=project_id,
        usage_statement_id=usage_statement_id,
        details=details,
    )

    print(f"\n[결과] result_code={result_code} | {details['summary']}")
    print(f"  changed_count={details['payload']['changed_count']}")
    print(f"  kept_count={details['payload']['kept_count']}")
    print("  agent_logs 기록: no")

    print("\n[results]")
    print(json.dumps(details["payload"]["results"], ensure_ascii=False, indent=2, default=str))

    print("\n[agent_logs payload]")
    print(json.dumps(agent_log, ensure_ascii=False, indent=2, default=str))


def run_orchestrator_single_item(
    *,
    usage_statement_id: int,
    item_id: int,
    verbose: bool,
) -> None:
    data = fetch_usage_statement_rows(usage_statement_id)
    project_id = data["project_id"]
    row = next((row for row in data["rows"] if row["row_id"] == item_id), None)
    if row is None:
        raise ValueError(f"usage_statement id={usage_statement_id} item_id={item_id} 없음")

    request = _build_orchestrator_request(
        project_id=project_id,
        usage_statement_id=usage_statement_id,
        row=row,
    )

    print(f"\n{'=' * 76}")
    print("  orchestrator classify_existing_usage_statement 단건 경로")
    print(f"{'=' * 76}\n")
    print(f"[요청] project_id={project_id} | usage_statement_id={usage_statement_id} | item_id={item_id}")
    if verbose:
        print("\n[orchestrator request]")
        print(json.dumps(request.model_dump(), ensure_ascii=False, indent=2, default=str))

    response_payload, agent_log = _orchestrator_single_item_payload(request=request)
    print("\n[모드] dry-run: 실제 orchestrator 호출/agent_logs INSERT 없음")
    print("\n[orchestrator response payload]")
    print(json.dumps(response_payload, ensure_ascii=False, indent=2, default=str))
    print("\n[agent_logs payload]")
    print(json.dumps(agent_log, ensure_ascii=False, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="DB 데이터로 classi Agent 실행 및 agent_logs 확인")
    parser.add_argument("--id", type=int, default=1, help="usage_statement id (기본값: 1)")
    parser.add_argument("--item-id", type=int, default=None, help="단건 orchestrator 재분류 경로로 실행할 item id")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if os.getenv("OPENAI_API_KEY"):
        llm_config.configure(ChatOpenAI(model=args.model, temperature=0))

    if args.item_id is not None:
        run_orchestrator_single_item(
            usage_statement_id=args.id,
            item_id=args.item_id,
            verbose=args.verbose,
        )
        return

    run_aggregate_classi(
        usage_statement_id=args.id,
        model=args.model,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
