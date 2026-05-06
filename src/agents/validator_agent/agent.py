# --------------------------------------------------------------------------
# 작성자   : 송상민(ss19801)
# 작성일   : 2026-05-04
#
# [ 주요 클래스 및 함수 정의 ]
#
# 1. run_audit() : 카테고리 묶음 검토 진입점
# 2. validate_document() : 단일 카테고리 검토 진입점
# 3. validate_usage_statement() : 사용내역서 전체 검토 진입점
# 4. to_summary_response() : 요약 응답 구조 변환
# --------------------------------------------------------------------------

try:
    from langsmith import traceable
except ImportError:  # pragma: no cover
    def traceable(*args, **kwargs):  # type: ignore
        def decorator(func):
            return func
        return decorator

from src.agents.validator_agent.presenter import summarize_audit_response, to_validator_response
from src.core.storage import DEFAULT_COLLECTION
from src.schemas.validator import (
    AuditResponse,
    CategoryAuditResult,
    UsageStatementAuditSummaryResponse,
    ValidatorAuditResponse,
)
from src.services.validator_service import (
    run_audit_service,
    validate_document_service,
    validate_usage_statement_service,
)


@traceable(name="validator.run_audit")
def run_audit(
    base_amount: float,
    categories: dict[str, dict[str, float]],
    collection: str = DEFAULT_COLLECTION,
    basic_info_by_category: dict[str, dict] | None = None,
    summaries_by_category: dict[str, dict] | None = None,
    progress_rate: float | None = None,
) -> AuditResponse:
    return run_audit_service(
        base_amount=base_amount,
        categories=categories,
        collection=collection,
        basic_info_by_category=basic_info_by_category,
        summaries_by_category=summaries_by_category,
        progress_rate=progress_rate,
    )


@traceable(name="validator.validate_document")
def validate_document(
    *,
    category: str,
    items: dict[str, float] | None = None,
    basic_info: dict | None = None,
    base_amount: float | None = None,
    document: dict | None = None,
    collection: str = DEFAULT_COLLECTION,
) -> CategoryAuditResult:
    return validate_document_service(
        category=category,
        items=items,
        basic_info=basic_info,
        base_amount=base_amount,
        document=document,
        collection=collection,
    )


@traceable(name="validator.validate_usage_statement")
def validate_usage_statement(
    *,
    document: dict,
    collection: str = DEFAULT_COLLECTION,
) -> AuditResponse:
    return validate_usage_statement_service(document=document, collection=collection)


def to_summary_response(
    *,
    response: AuditResponse,
    usage_statement_id: int | str | None = None,
) -> UsageStatementAuditSummaryResponse:
    return summarize_audit_response(response=response, usage_statement_id=usage_statement_id)


__all__ = [
    "run_audit",
    "validate_document",
    "validate_usage_statement",
    "summarize_audit_response",
    "to_validator_response",
    "to_summary_response",
]
