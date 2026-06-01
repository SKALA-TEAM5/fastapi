"""
AI Review Orchestrator API 라우터
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Spring Backend가 호출하는 `/api/v1/orchestrator` 엔드포인트를 제공한다.

역할:
  - 사용내역서 업로드 후 OCR/Parse 및 classi 실행 진입점 제공
  - 증빙 검증, 법령 검토, 보고서 생성 단계의 실행 조건 확인
  - 화면 상태와 SHE 대시보드 요약 조회 API 제공
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from src.schemas.orchestrator import (
    LegalReviewRequest,
    OrchestratorActionResponse,
    OrchestratorDashboardResponse,
    OrchestratorStatusResponse,
    EvidenceReviewRequest,
    ReportDraftRequest,
    UsageStatementClassifyRequest,
    UsageStatementParseRequest,
)
from src.services.orchestrator_service import (
    classify_existing_usage_statement,
    get_orchestrator_dashboard,
    get_orchestrator_status,
    parse_and_classify_usage_statement,
    run_evidence_review,
    run_legal_review,
    run_report_draft,
)


router = APIRouter(prefix="/orchestrator", tags=["Orchestrator"])


@router.post(
    "/usage-statements/parse",
    response_model=OrchestratorActionResponse,
    summary="사용내역서 파싱 및 classi 실행",
)
async def parse_usage_statement(request: UsageStatementParseRequest) -> OrchestratorActionResponse:
    try:
        return parse_and_classify_usage_statement(request.file_id)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"사용내역서 파싱/classi 실행 실패: {type(exc).__name__}: {exc}",
        ) from exc


@router.post(
    "/usage-statements/classify",
    response_model=OrchestratorActionResponse,
    summary="저장된 사용내역서 세부항목 classi 재분류",
)
async def classify_usage_statement(request: UsageStatementClassifyRequest) -> OrchestratorActionResponse:
    try:
        return classify_existing_usage_statement(request)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"사용내역서 세부항목 classi 재분류 실패: {type(exc).__name__}: {exc}",
        ) from exc


@router.post(
    "/usage-statements/evidence",
    response_model=OrchestratorActionResponse,
    summary="증빙 검증 대상 Agent 결정",
)
async def evidence_review(request: EvidenceReviewRequest) -> OrchestratorActionResponse:
    return run_evidence_review(
        request.project_id,
        request.usage_statement_id,
        requested_by_user_id=request.requested_by_user_id,
    )


@router.post(
    "/usage-statements/legal",
    response_model=OrchestratorActionResponse,
    summary="SHE 법령 검토 대상 Agent 결정",
)
async def legal_review(request: LegalReviewRequest) -> OrchestratorActionResponse:
    return run_legal_review(
        request.project_id,
        request.usage_statement_id,
        she_user_id=request.she_user_id,
    )


@router.post(
    "/usage-statements/report",
    response_model=OrchestratorActionResponse,
    summary="보고서 초안 생성 대상 Agent 결정",
)
async def report_draft(request: ReportDraftRequest) -> OrchestratorActionResponse:
    return run_report_draft(
        request.project_id,
        request.usage_statement_id,
        she_user_id=request.she_user_id,
    )


@router.get(
    "/projects/{project_id}/usage-statements/{usage_statement_id}/status",
    response_model=OrchestratorStatusResponse,
    summary="Orchestrator 상태 조회",
)
async def status_view(project_id: int, usage_statement_id: int) -> OrchestratorStatusResponse:
    return get_orchestrator_status(project_id, usage_statement_id)


@router.get(
    "/projects/{project_id}/dashboard",
    response_model=OrchestratorDashboardResponse,
    summary="SHE 대시보드용 Orchestrator 요약 조회",
)
async def dashboard_view(
    project_id: int,
    usage_statement_id: int | None = None,
) -> OrchestratorDashboardResponse:
    return get_orchestrator_dashboard(project_id, usage_statement_id)
