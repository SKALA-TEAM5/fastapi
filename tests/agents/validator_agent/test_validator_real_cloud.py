import os

import pytest
from dotenv import load_dotenv

from src.agents.validator_agent.agent import summarize_audit_response, validate_usage_statement
from src.schemas.validator import AuditResponse
from src.services.orchestrator_service import (
    _build_validator_document,
    _legal_apply_generated_item_reasons,
    _legal_basis_by_citation,
    _legal_citations_from_results,
    _legal_frontend_categories,
    _legal_item_results_from_audit,
    _legal_payload_item_results,
    _linked_files_by_item_id,
    run_legal_review,
)

load_dotenv()


def _enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "y"}


pytestmark = pytest.mark.skipif(
    not _enabled("RUN_LEGAL_REAL_DB_TEST"),
    reason="set RUN_LEGAL_REAL_DB_TEST=1 to run DB/Qdrant-backed legal smoke",
)


def _target_ids() -> tuple[int, int]:
    return (
        int(os.getenv("LEGAL_REAL_PROJECT_ID", "5")),
        int(os.getenv("LEGAL_REAL_USAGE_STATEMENT_ID", "3")),
    )


def _build_legal_payload(
    *,
    audit_response: AuditResponse,
    summary_response,
    category_rows: dict[str, list[dict]],
    usage_statement_id: int,
) -> dict:
    item_results = _legal_item_results_from_audit(
        audit_response=audit_response,
        summary_response=summary_response,
        category_rows=category_rows,
    )
    linked_files_by_item_id = _linked_files_by_item_id(usage_statement_id)
    legal_basis_by_source_id = _legal_basis_by_citation(_legal_citations_from_results(item_results))
    _legal_apply_generated_item_reasons(
        item_results=item_results,
        legal_basis_by_source_id=legal_basis_by_source_id,
    )
    frontend_categories = _legal_frontend_categories(
        item_results=item_results,
        category_rows=category_rows,
        linked_files_by_item_id=linked_files_by_item_id,
        legal_basis_by_source_id=legal_basis_by_source_id,
    )
    return {
        "results": _legal_payload_item_results(item_results),
        "categories": frontend_categories,
        "item_results": item_results,
    }


def test_usage_statement_3_legal_dry_run_matches_cloud_contract():
    """Run the real legal chain against port-forwarded DB/Qdrant without writing agent_logs."""
    project_id, usage_statement_id = _target_ids()

    document, category_rows = _build_validator_document(project_id, usage_statement_id)
    assert document["사용내역서ID"] == usage_statement_id
    assert document["카테고리별데이터"]

    input_item_count = sum(len(rows) for rows in category_rows.values())
    assert input_item_count > 0

    audit_response = validate_usage_statement(document=document)
    assert audit_response.base_amount > 0
    assert audit_response.categories

    audit_items = [
        item
        for category_result in audit_response.categories.values()
        for item in category_result.items
    ]
    assert len(audit_items) == input_item_count
    assert all(item.category for item in audit_items)
    assert all(item.judgment_source for item in audit_items)

    summary_response = summarize_audit_response(
        response=audit_response,
        usage_statement_id=usage_statement_id,
    )
    assert summary_response.usage_statement_id == usage_statement_id
    assert summary_response.results

    legal_payload = _build_legal_payload(
        audit_response=audit_response,
        summary_response=summary_response,
        category_rows=category_rows,
        usage_statement_id=usage_statement_id,
    )
    assert len(legal_payload["results"]) == input_item_count
    assert legal_payload["categories"]
    assert all(row.get("item_name") for row in legal_payload["results"])
    assert all(row.get("status") in {"적절", "부적절", "검토필요"} for row in legal_payload["results"])
    assert all(row.get("reason") for row in legal_payload["results"])


@pytest.mark.skipif(
    not _enabled("RUN_LEGAL_REAL_WRITE_TEST"),
    reason="set RUN_LEGAL_REAL_WRITE_TEST=1 to execute run_legal_review and write agent_logs",
)
def test_usage_statement_3_legal_review_writes_agent_log():
    """Run the same path as the FastAPI legal endpoint, including agent_logs upsert."""
    project_id, usage_statement_id = _target_ids()

    response = run_legal_review(project_id, usage_statement_id)
    legal_result = response.result.get("legal") or {}

    assert response.status == "success"
    assert response.usage_statement_id == usage_statement_id
    assert response.target_agents == ["legal"]
    # _run_legal_agent()의 반환 dict는 status_code/result_code/reason/result_count/todos만
    # 가진다 — "event": "legal_completed"는 upsert_agent_log(details=...)를 통해
    # agent_logs 테이블에만 기록되고, 호출자에게 돌아오는 응답에는 details 키 자체가 없다.
    # (orchestrator_service.py의 _run_legal_agent, _run_safety_doc_agent, _run_link_agent
    # 모두 동일한 패턴이므로 "details.event" 식 단언은 이 응답 dict에 절대 성립하지 않는다.)
    assert legal_result.get("status_code") == "success"
    assert legal_result.get("result_code") in {"success", "hil"}
    assert isinstance(legal_result.get("reason"), str) and legal_result["reason"]
    assert isinstance(legal_result.get("result_count"), int) and legal_result["result_count"] >= 0
    assert isinstance(legal_result.get("todos"), list)
    assert "details" not in legal_result
