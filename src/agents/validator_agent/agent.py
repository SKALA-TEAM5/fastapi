# --------------------------------------------------------------------------
# 작성자   : 송상민(ss19801)
# 작성일   : 2026-05-04
# 수정일   : 2026-06-18
#
# [ 주요 클래스 및 함수 정의 ]
#
# 1. validate_usage_statement() : 사용내역서 전체 검토 진입점
# 2. summarize_audit_response() : orchestrator용 legal 요약 응답 변환
# --------------------------------------------------------------------------

try:
    from langsmith import traceable
except ImportError:  # pragma: no cover
    def traceable(*args, **kwargs):  # type: ignore
        def decorator(func):
            return func
        return decorator

from src.agents.validator_agent.presenter import summarize_audit_response
from src.core.storage import DEFAULT_COLLECTION
from src.schemas.validator import AuditResponse
from src.services.validator_service import validate_usage_statement_service


@traceable(name="validator.validate_usage_statement")
def validate_usage_statement(
    *,
    document: dict,
    collection: str = DEFAULT_COLLECTION,
) -> AuditResponse:
    """Validate a usage statement against legal rules.

    Args:
        document: Usage-statement legal validator input.
        collection: Qdrant collection name used for legal context retrieval.

    Returns:
        Category-level audit response consumed by orchestrator.
    """
    return validate_usage_statement_service(document=document, collection=collection)


__all__ = [
    "validate_usage_statement",
    "summarize_audit_response",
]
