from __future__ import annotations

"""보고서 생성 agent 단독 실행 라우터.

운영 report 실행은 orchestrator의 `/api/v1/orchestrator/usage-statements/report`가 담당합니다.
이 라우터는 테스트나 단독 디버깅에서 ReportContext -> ReportDraft 경로를 직접 확인할 때 사용합니다.
"""

from datetime import date
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.agents.report_agent.agent import ReportAgent
from src.agents.report_agent.context_builder import build_report_context
from src.agents.report_agent.llm import ReportLLMError
from src.agents.report_agent.schemas import ReportContext, ReviewerContext
from src.repositories.db import get_connection
from src.repositories.report_repository import PostgresReportRepository, default_report_no
from src.repositories.usage_statement_pipeline_repository import insert_agent_log, update_agent_log_status


router = APIRouter(prefix="/agents/report", tags=["보고서 Agent"])


class ReportAgentRunRequest(BaseModel):
    run_id: UUID
    project_id: int
    usage_statement_id: int
    report_no: str | None = None
    report_written_date: date | None = None
    report_period_label: str | None = None
    reviewer: ReviewerContext | None = None
    context: ReportContext | None = Field(
        default=None,
        description="DB repository 구현 전 또는 테스트에서 직접 전달하는 ReportContext.",
    )


class ReportAgentRunResponse(BaseModel):
    run_id: UUID
    agent_type: str = "report"
    status: str
    log_ids: list[int] = Field(default_factory=list)
    result: dict[str, Any] = Field(default_factory=dict)


@router.post(
    "/run",
    response_model=ReportAgentRunResponse,
    status_code=status.HTTP_200_OK,
    summary="보고서 Agent 실행",
    description="""
ReportContext를 ReportDraft JSON으로 변환합니다.

`context`가 없으면 `project_id`와 `usage_statement_id`로 DB에서 ReportContext를 조립합니다.
실행 결과는 `agent_logs`에 service 스키마의 `success/fail` 상태값으로 기록합니다.
    """,
)
async def run_report_agent(request: ReportAgentRunRequest) -> ReportAgentRunResponse:
    log_id: int | None = None
    written_date = request.report_written_date or date.today()

    try:
        if request.context is None:
            with get_connection() as conn:
                repo = PostgresReportRepository(conn)
                project = repo.get_project(request.project_id)
                report_no = request.report_no or default_report_no(project.get("contract_no") or request.project_id, request.usage_statement_id, written_date)
                context = build_report_context(
                    repo,
                    project_id=request.project_id,
                    usage_statement_id=request.usage_statement_id,
                    report_no=report_no,
                    report_written_date=written_date,
                    report_period_label=request.report_period_label or _default_period_label(repo.get_usage_statement(request.usage_statement_id)["report_month"]),
                    reviewer=request.reviewer,
                )
                log_id = insert_agent_log(
                    conn,
                    project_id=request.project_id,
                    usage_statement_id=request.usage_statement_id,
                    details={"report_no": report_no},
                    agent_type_code="report",
                    model_name="report_agent",
                    run_id=str(request.run_id),
                )
        else:
            context = request.context

        draft = ReportAgent().generate(context)

        if log_id is not None:
            with get_connection() as conn:
                update_agent_log_status(
                    conn,
                    log_id=log_id,
                    status_code="success",
                    details={
                        "report_no": draft.report_no,
                        "site_name": draft.site_name,
                        "needs_human_review": draft.needs_human_review,
                        "event": "report_completed",
                    },
                )
    except ReportLLMError as exc:
        _mark_failed(log_id, str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except KeyError as exc:
        _mark_failed(log_id, str(exc))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        _mark_failed(log_id, f"{type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"보고서 생성 실패: {type(exc).__name__}",
        ) from exc

    return ReportAgentRunResponse(
        run_id=request.run_id,
        status="success",
        log_ids=[log_id] if log_id is not None else [],
        result={"reportDraft": draft.model_dump(mode="json")},
    )


def _default_period_label(report_month: date) -> str:
    return f"{report_month:%Y년 %m월}"


def _mark_failed(log_id: int | None, message: str) -> None:
    if log_id is None:
        return
    with get_connection() as conn:
        update_agent_log_status(
            conn,
            log_id=log_id,
            status_code="fail",
            details={"event": "report_failed", "error": message},
        )
