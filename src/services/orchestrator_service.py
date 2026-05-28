"""
AI Review Orchestrator 서비스
━━━━━━━━━━━━━━━━━━━━━━━━━━
사용자 버튼 클릭 또는 업로드 완료 시점마다 DB 상태를 스캔하고
실행 가능한 Agent와 다음 화면 상태를 결정한다.

Orchestrator는 장시간 대기하지 않는다.
각 단계의 현재 상태는 `agent_logs`를 기준으로 계산하고,
실제 Agent 연결 전에는 실행 대상 로그를 pending 상태로 기록한다.
"""

from __future__ import annotations

from collections import Counter
from datetime import date
from typing import Any
from uuid import uuid4

from src.agents.report_agent.agent import ReportAgent
from src.agents.report_agent.context_builder import build_report_context
from src.agents.safety_doc_agent.agent import check_missing_evidence
from src.repositories.orchestrator_repository import (
    close_supplement_todos,
    create_supplement_todos,
    list_evidence_file_ids_by_type,
    list_latest_agent_logs,
    list_usage_statement_item_ids,
    mark_orchestrator,
    scan_orchestrator_state,
    select_evidence_agents,
    upsert_agent_log,
)
from src.repositories.db import get_connection
from src.repositories.report_repository import PostgresReportRepository, default_report_no
from src.schemas.orchestrator import (
    AgentDashboardSummary,
    AgentLogSnapshot,
    OrchestratorActionResponse,
    OrchestratorDashboardResponse,
    OrchestratorStatusResponse,
)
from src.services.usage_statement_pipeline_service import parse_usage_statement, run_link_pipeline


def parse_and_classify_usage_statement(file_id: int) -> OrchestratorActionResponse:
    result = parse_usage_statement(file_id)
    return OrchestratorActionResponse(
        status="success",
        message="사용내역서 파싱 및 classi 실행 요청이 완료되었습니다.",
        usage_statement_id=result.get("usage_statement_id"),
        target_agents=["classi"],
        result=result,
    )


def get_orchestrator_status(project_id: int, usage_statement_id: int) -> OrchestratorStatusResponse:
    state = scan_orchestrator_state(project_id, usage_statement_id)
    return OrchestratorStatusResponse(
        project_id=project_id,
        usage_statement_id=usage_statement_id,
        has_usage_statement_items=state.has_usage_statement_items,
        has_receipts_or_tax_invoices=state.has_receipts_or_tax_invoices,
        has_site_photos=state.has_site_photos,
        classi_ready=state.classi_ready,
        evidence_review_ready=state.evidence_review_ready,
        legal_ready=state.legal_ready,
        report_ready=state.report_ready,
        logs=[
            AgentLogSnapshot(
                agent_type_code=agent,
                status_code=log.get("status_code"),
                result_code=log.get("result_code"),
                reason=log.get("reason"),
                details=log.get("details"),
                token=log.get("token"),
            )
            for agent, log in sorted(state.logs.items())
        ],
    )


def get_orchestrator_dashboard(
    project_id: int,
    usage_statement_id: int | None = None,
) -> OrchestratorDashboardResponse:
    logs = list_latest_agent_logs(project_id=project_id, usage_statement_id=usage_statement_id)
    status_counts = Counter(str(log.get("status_code")) for log in logs if log.get("status_code"))
    result_counts = Counter(str(log.get("result_code")) for log in logs if log.get("result_code"))
    hil_agents = sorted(
        {
            str(log.get("agent_type_code"))
            for log in logs
            if log.get("result_code") == "hil" and log.get("agent_type_code")
        }
    )

    return OrchestratorDashboardResponse(
        project_id=project_id,
        usage_statement_id=usage_statement_id,
        total_logs=len(logs),
        total_token=sum(int(log.get("token") or 0) for log in logs),
        status_counts=dict(status_counts),
        result_counts=dict(result_counts),
        hil_agents=hil_agents,
        agents=[
            AgentDashboardSummary(
                agent_type_code=log.get("agent_type_code"),
                status_code=log.get("status_code"),
                result_code=log.get("result_code"),
                usage_statement_id=log.get("usage_statement_id"),
                token=int(log.get("token") or 0),
                reason=log.get("reason"),
            )
            for log in logs
        ],
    )


def run_evidence_review(
    project_id: int,
    usage_statement_id: int,
    requested_by_user_id: int | None = None,
) -> OrchestratorActionResponse:
    state = scan_orchestrator_state(project_id, usage_statement_id)
    if not state.classi_ready:
        mark_orchestrator(
            project_id=project_id,
            usage_statement_id=usage_statement_id,
            event="evidence_review_blocked",
            status_code="fail",
            result_code="fail",
            reason="classi가 success/success 상태가 아니어서 증빙 검증을 실행할 수 없습니다.",
        )
        return OrchestratorActionResponse(
            status="blocked",
            message="classi 성공 후 증빙 검증을 실행할 수 있습니다.",
            usage_statement_id=usage_statement_id,
        )

    target_agents = select_evidence_agents(state)
    mark_orchestrator(
        project_id=project_id,
        usage_statement_id=usage_statement_id,
        event="evidence_review_started",
        status_code="running",
        result_code=None,
        reason="증빙 검증을 시작했습니다.",
        payload={"target_agents": target_agents},
    )

    results: dict[str, Any] = {}
    for agent in target_agents:
        if agent == "safety-doc":
            results[agent] = _run_safety_doc_agent(project_id, usage_statement_id)
        elif agent == "link":
            results[agent] = _run_link_agent(project_id, usage_statement_id)
        elif agent == "vision":
            results[agent] = _mark_missing_agent(
                project_id=project_id,
                usage_statement_id=usage_statement_id,
                agent_type_code="vision",
                reason="vision Agent 구현체가 아직 FastAPI에 연결되어 있지 않습니다.",
            )
        _sync_supplement_todos(
            project_id=project_id,
            usage_statement_id=usage_statement_id,
            agent_type_code=agent,
            requested_by_user_id=requested_by_user_id,
            result=results[agent],
        )

    return OrchestratorActionResponse(
        status="success",
        message="증빙 검증 대상 Agent 실행을 완료했습니다.",
        usage_statement_id=usage_statement_id,
        target_agents=target_agents,
        hil_agents=[
            agent
            for agent, result in results.items()
            if result.get("result_code") == "hil"
        ],
        result=results,
    )


def run_legal_review(
    project_id: int,
    usage_statement_id: int,
    she_user_id: int | None = None,
) -> OrchestratorActionResponse:
    state = scan_orchestrator_state(project_id, usage_statement_id)
    if not state.evidence_review_ready:
        mark_orchestrator(
            project_id=project_id,
            usage_statement_id=usage_statement_id,
            event="legal_review_blocked",
            status_code="fail",
            result_code="fail",
            reason="증빙 검증 Agent가 모두 success/success 상태가 아니어서 legal을 실행할 수 없습니다.",
            payload={"hil_exists": state.evidence_has_hil},
        )
        return OrchestratorActionResponse(
            status="blocked",
            message="증빙 검증이 모두 성공한 뒤 SHE 담당자가 legal을 실행할 수 있습니다.",
            usage_statement_id=usage_statement_id,
            hil_agents=[
                agent
                for agent in ("safety-doc", "link", "vision")
                if (state.logs.get(agent) or {}).get("result_code") == "hil"
            ],
        )

    mark_orchestrator(
        project_id=project_id,
        usage_statement_id=usage_statement_id,
        event="legal_review_started",
        status_code="running",
        result_code=None,
        reason="SHE 담당자가 legal 검토를 시작했습니다.",
        payload={"she_user_id": she_user_id},
    )
    result = _mark_missing_agent(
        project_id=project_id,
        usage_statement_id=usage_statement_id,
        agent_type_code="legal",
        reason="legal Agent 구현체가 아직 FastAPI에 연결되어 있지 않습니다.",
        payload={"she_user_id": she_user_id},
    )
    return OrchestratorActionResponse(
        status="fail",
        message="legal 실행 조건은 통과했지만 실제 legal Agent 구현체가 아직 없습니다.",
        usage_statement_id=usage_statement_id,
        target_agents=["legal"],
        result={"legal": result},
    )


def run_report_draft(
    project_id: int,
    usage_statement_id: int,
    she_user_id: int | None = None,
) -> OrchestratorActionResponse:
    state = scan_orchestrator_state(project_id, usage_statement_id)
    if not state.report_ready:
        mark_orchestrator(
            project_id=project_id,
            usage_statement_id=usage_statement_id,
            event="report_draft_blocked",
            status_code="fail",
            result_code="fail",
            reason="legal이 success/success 상태가 아니어서 report를 실행할 수 없습니다.",
        )
        return OrchestratorActionResponse(
            status="blocked",
            message="legal 검토가 성공한 뒤 report 초안을 생성할 수 있습니다.",
            usage_statement_id=usage_statement_id,
            hil_agents=["legal"] if (state.logs.get("legal") or {}).get("result_code") == "hil" else [],
        )

    result = _run_report_agent(project_id, usage_statement_id, she_user_id=she_user_id)
    return OrchestratorActionResponse(
        status="success",
        message="report Agent 실행을 완료했습니다.",
        usage_statement_id=usage_statement_id,
        target_agents=["report"],
        result={"report": result},
    )


def _run_safety_doc_agent(project_id: int, usage_statement_id: int) -> dict[str, Any]:
    item_ids = list_usage_statement_item_ids(usage_statement_id)
    upsert_agent_log(
        project_id=project_id,
        usage_statement_id=usage_statement_id,
        agent_type_code="safety-doc",
        status_code="running",
        result_code=None,
        reason="safety-doc Agent를 실행 중입니다.",
        details={"event": "agent_running", "payload": {"item_ids": item_ids}},
        model_name="safety_doc_agent",
    )

    try:
        item_results = [
            {"item_id": item_id, "result": check_missing_evidence(item_id)}
            for item_id in item_ids
        ]
        hil_item_ids = [
            row["item_id"]
            for row in item_results
            if ((row["result"].get("evidence_status") or {}).get("missing_evidences") or [])
        ]
        result_code = "hil" if hil_item_ids else "success"
        reason = (
            f"필수 증빙 누락 항목 {len(hil_item_ids)}건"
            if hil_item_ids
            else "필수 증빙 누락 없음"
        )
        token = sum(
            int(((row["result"].get("ai_response") or {}).get("usage") or {}).get("total_tokens") or 0)
            for row in item_results
        )
        upsert_agent_log(
            project_id=project_id,
            usage_statement_id=usage_statement_id,
            agent_type_code="safety-doc",
            status_code="success",
            result_code=result_code,
            reason=reason,
            details={
                "event": "safety_doc_completed",
                "summary": reason,
                "payload": {
                    "item_count": len(item_ids),
                    "hil_item_ids": hil_item_ids,
                    "item_results": item_results,
                },
            },
            model_name="safety_doc_agent",
            token=token,
        )
        todos = [
            {
                "usage_statement_item_id": row["item_id"],
                "reason": "필수 증빙 누락: "
                + ", ".join((row["result"].get("evidence_status") or {}).get("missing_evidences") or []),
            }
            for row in item_results
            if ((row["result"].get("evidence_status") or {}).get("missing_evidences") or [])
        ]
        return {
            "status_code": "success",
            "result_code": result_code,
            "reason": reason,
            "todos": todos,
        }
    except Exception as exc:
        return _mark_agent_failed(project_id, usage_statement_id, "safety-doc", exc)


def _run_link_agent(project_id: int, usage_statement_id: int) -> dict[str, Any]:
    grouped_files = list_evidence_file_ids_by_type(project_id)
    receipt_file_ids = [
        file_id
        for evidence_type in ("receipt", "transaction_statement")
        for file_id in grouped_files.get(evidence_type, [])
    ]
    tax_invoice_file_ids = grouped_files.get("tax_invoice", [])

    if not receipt_file_ids and not tax_invoice_file_ids:
        return {"status_code": "skipped", "result_code": None, "reason": "영수증/세금계산서 파일 없음"}

    upsert_agent_log(
        project_id=project_id,
        usage_statement_id=usage_statement_id,
        agent_type_code="link",
        status_code="running",
        result_code=None,
        reason="link Agent를 실행 중입니다.",
        details={
            "event": "agent_running",
            "payload": {
                "receipt_file_ids": receipt_file_ids,
                "tax_invoice_file_ids": tax_invoice_file_ids,
            },
        },
        model_name="link_pipeline",
    )

    try:
        result = run_link_pipeline(
            usage_statement_id=usage_statement_id,
            receipt_file_ids=receipt_file_ids,
            tax_invoice_file_ids=tax_invoice_file_ids,
        )
        summary = result.get("summary") or {}
        issue_count = sum(
            int(summary.get(key) or 0)
            for key in ("review_needed", "unmatched", "rejected")
        )
        result_code = "hil" if issue_count else "success"
        reason = f"매칭 검토 필요 {issue_count}건" if issue_count else "증빙 파일 매칭 적정"
        upsert_agent_log(
            project_id=project_id,
            usage_statement_id=usage_statement_id,
            agent_type_code="link",
            status_code="success",
            result_code=result_code,
            reason=reason,
            details={
                "event": "link_completed",
                "summary": reason,
                "payload": result,
            },
            model_name="link_pipeline",
        )
        todos = [
            {
                "usage_statement_item_id": int(row.get("line_id")),
                "reason": f"증빙 매칭 검토 필요: {row.get('match_status')}",
            }
            for row in (result.get("match_results") or [])
            if str(row.get("line_id") or "").isdigit()
            and row.get("match_status") in {"review_needed", "unmatched", "rejected"}
        ]
        return {
            "status_code": "success",
            "result_code": result_code,
            "reason": reason,
            "todos": todos,
        }
    except Exception as exc:
        return _mark_agent_failed(project_id, usage_statement_id, "link", exc)


def _run_report_agent(
    project_id: int,
    usage_statement_id: int,
    *,
    she_user_id: int | None = None,
) -> dict[str, Any]:
    written_date = date.today()
    upsert_agent_log(
        project_id=project_id,
        usage_statement_id=usage_statement_id,
        agent_type_code="report",
        status_code="running",
        result_code=None,
        reason="report Agent를 실행 중입니다.",
        details={"event": "agent_running", "payload": {"she_user_id": she_user_id}},
        model_name="report_agent",
    )

    try:
        with get_connection() as conn:
            repo = PostgresReportRepository(conn)
            report_no = default_report_no(project_id, usage_statement_id, written_date)
            usage_statement = repo.get_usage_statement(usage_statement_id)
            context = build_report_context(
                repo,
                project_id=project_id,
                usage_statement_id=usage_statement_id,
                report_no=report_no,
                report_written_date=written_date,
                report_period_label=f"{usage_statement['report_month']:%Y년 %m월}",
            )
        draft = ReportAgent().generate(context)
        result = {"reportDraft": draft.model_dump(mode="json"), "run_id": str(uuid4())}
        upsert_agent_log(
            project_id=project_id,
            usage_statement_id=usage_statement_id,
            agent_type_code="report",
            status_code="success",
            result_code="success",
            reason="보고서 초안 생성 완료",
            details={
                "event": "report_completed",
                "summary": "보고서 초안 생성 완료",
                "payload": {
                    "report_no": draft.report_no,
                    "site_name": draft.site_name,
                    "needs_human_review": draft.needs_human_review,
                },
            },
            model_name="report_agent",
        )
        return {"status_code": "success", "result_code": "success", "reason": "보고서 초안 생성 완료", "result": result}
    except Exception as exc:
        return _mark_agent_failed(project_id, usage_statement_id, "report", exc)


def _mark_missing_agent(
    *,
    project_id: int,
    usage_statement_id: int,
    agent_type_code: str,
    reason: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    upsert_agent_log(
        project_id=project_id,
        usage_statement_id=usage_statement_id,
        agent_type_code=agent_type_code,
        status_code="fail",
        result_code="fail",
        reason=reason,
        details={
            "event": "agent_not_implemented",
            "summary": reason,
            "payload": payload or {},
        },
        model_name="not_connected",
    )
    return {"status_code": "fail", "result_code": "fail", "reason": reason}


def _sync_supplement_todos(
    *,
    project_id: int,
    usage_statement_id: int,
    agent_type_code: str,
    requested_by_user_id: int | None,
    result: dict[str, Any],
) -> None:
    if agent_type_code not in {"safety-doc", "link", "vision", "legal"}:
        return

    close_supplement_todos(
        project_id=project_id,
        usage_statement_id=usage_statement_id,
        agent_type_code=agent_type_code,
    )
    if result.get("result_code") != "hil":
        return

    create_supplement_todos(
        project_id=project_id,
        usage_statement_id=usage_statement_id,
        requested_by_user_id=requested_by_user_id,
        agent_type_code=agent_type_code,
        todos=result.get("todos") or [],
    )


def _mark_agent_failed(
    project_id: int,
    usage_statement_id: int,
    agent_type_code: str,
    exc: Exception,
) -> dict[str, Any]:
    reason = f"{agent_type_code} Agent 실행 실패: {type(exc).__name__}: {exc}"
    upsert_agent_log(
        project_id=project_id,
        usage_statement_id=usage_statement_id,
        agent_type_code=agent_type_code,
        status_code="fail",
        result_code="fail",
        reason=reason,
        details={
            "event": "agent_failed",
            "summary": reason,
            "payload": {"error_type": type(exc).__name__, "error": str(exc)},
        },
        model_name=agent_type_code,
    )
    return {"status_code": "fail", "result_code": "fail", "reason": reason}
