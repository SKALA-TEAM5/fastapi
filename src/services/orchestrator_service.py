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
from typing import Any, cast
from uuid import uuid4

import psycopg2.extras
import requests

from src.agents.report_agent.agent import ReportAgent
from src.agents.report_agent.context_builder import build_report_context
from src.agents.safety_doc_agent.agent import check_missing_evidence
from src.agents.classifier_agent.agent import review_usage_statement
from src.agents.validator_agent.agent import summarize_audit_response, validate_usage_statement
from src.core.config import (
    VISION_AGENT_BASE_URL,
    VISION_AGENT_REVIEW_PATH,
    VISION_AGENT_TIMEOUT_SECONDS,
)
from src.schemas.classifier import CATEGORIES
from src.repositories.orchestrator_repository import (
    SITE_PHOTO_TYPES,
    list_evidence_files_by_type,
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
    SupplementTodoSnapshot,
    UsageStatementClassifyRequest,
)
from src.services.minio_client import create_presigned_file_url
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


def classify_existing_usage_statement(
    request: UsageStatementClassifyRequest,
) -> OrchestratorActionResponse:
    try:
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
        upsert_agent_log(
            project_id=request.project_id,
            usage_statement_id=request.usage_statement_id,
            agent_type_code="classi",
            status_code="success",
            result_code="success",
            reason=summary,
            details=details,
            model_name="classifier_agent",
        )
        return OrchestratorActionResponse(
            status="success",
            message=summary,
            usage_statement_id=request.usage_statement_id,
            target_agents=["classi"],
            result=details,
        )
    except Exception as exc:
        reason = f"classi Agent 실행 실패: {type(exc).__name__}: {exc}"
        upsert_agent_log(
            project_id=request.project_id,
            usage_statement_id=request.usage_statement_id,
            agent_type_code="classi",
            status_code="fail",
            result_code="fail",
            reason=reason,
            details={
                "event": "classification_failed",
                "summary": reason,
                "payload": {"error_type": type(exc).__name__, "error": str(exc)},
            },
            model_name="classifier_agent",
        )
        return OrchestratorActionResponse(
            status="fail",
            message=reason,
            usage_statement_id=request.usage_statement_id,
            target_agents=["classi"],
            result={
                "classi": {
                    "status_code": "fail",
                    "result_code": "fail",
                    "reason": reason,
                }
            },
        )


def get_orchestrator_status(project_id: int, usage_statement_id: int) -> OrchestratorStatusResponse:
    state = scan_orchestrator_state(project_id, usage_statement_id)
    logs = [
        AgentLogSnapshot(
            agent_type_code=agent,
            status_code=log.get("status_code"),
            result_code=log.get("result_code"),
            reason=log.get("reason"),
            details=log.get("details"),
            token=log.get("token"),
        )
        for agent, log in sorted(state.logs.items())
    ]
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
        logs=logs,
        todos=_build_status_todos(state.logs),
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
                agent_type_code=str(log.get("agent_type_code") or ""),
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
            results[agent] = _run_vision_agent(project_id, usage_statement_id)

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
    if not state.legal_ready:
        mark_orchestrator(
            project_id=project_id,
            usage_statement_id=usage_statement_id,
            event="legal_review_blocked",
            status_code="fail",
            result_code="fail",
            reason="safety-doc 로그가 없어 legal을 실행할 수 없습니다.",
            payload={"hil_exists": state.evidence_has_hil},
        )
        return OrchestratorActionResponse(
            status="blocked",
            message="validate를 먼저 실행해야 legal을 실행할 수 있습니다.",
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
    result = _run_legal_agent(project_id, usage_statement_id, she_user_id=she_user_id)
    return OrchestratorActionResponse(
        status=result["status_code"],
        message=result["reason"],
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
            reason="legal 실행이 정상 완료되지 않아 report를 실행할 수 없습니다.",
        )
        return OrchestratorActionResponse(
            status="blocked",
            message="legal 실행이 완료된 뒤 report 초안을 생성할 수 있습니다.",
            usage_statement_id=usage_statement_id,
            hil_agents=[],
        )

    result = _run_report_agent(project_id, usage_statement_id, she_user_id=she_user_id)
    report_result = cast(dict[str, Any], result.get("result")) if isinstance(result.get("result"), dict) else {}
    return OrchestratorActionResponse(
        status=result.get("status_code", "fail"),
        message=result.get("reason", "report Agent 실행을 완료했습니다."),
        usage_statement_id=usage_statement_id,
        target_agents=["report"],
        result={
            "report": result,
            "reportDraft": report_result.get("reportDraft"),
            "run_id": report_result.get("run_id"),
        },
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
        item_results: list[dict[str, Any]] = [
            {"item_id": item_id, "result": check_missing_evidence(item_id)}
            for item_id in item_ids
        ]
        hil_item_ids = [
            row["item_id"]
            for row in item_results
            if ((_dict_or_empty(row.get("result")).get("evidence_status") or {}).get("missing_evidences") or [])
        ]
        result_code = "hil" if hil_item_ids else "success"
        reason = (
            f"필수 증빙 누락 항목 {len(hil_item_ids)}건"
            if hil_item_ids
            else "필수 증빙 누락 없음"
        )
        token = sum(
            int(((_dict_or_empty(_dict_or_empty(row.get("result")).get("ai_response")).get("usage") or {}).get("total_tokens")) or 0)
            for row in item_results
        )
        todos = [
            {
                "usage_statement_item_id": row["item_id"],
                "reason": "필수 증빙 누락: "
                + ", ".join((_dict_or_empty(row.get("result")).get("evidence_status") or {}).get("missing_evidences") or []),
            }
            for row in item_results
            if ((_dict_or_empty(row.get("result")).get("evidence_status") or {}).get("missing_evidences") or [])
        ]
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
                    "todos": todos,
                },
            },
            model_name="safety_doc_agent",
            token=token,
        )
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
        todos = [
            {
                "usage_statement_item_id": int(row.get("line_id")),
                "reason": f"증빙 매칭 검토 필요: {row.get('match_status')}",
            }
            for row in (result.get("match_results") or [])
            if str(row.get("line_id") or "").isdigit()
            and row.get("match_status") in {"review_needed", "unmatched", "rejected"}
        ]
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
                "payload": {**result, "todos": todos},
            },
            model_name="link_pipeline",
        )
        return {
            "status_code": "success",
            "result_code": result_code,
            "reason": reason,
            "todos": todos,
        }
    except Exception as exc:
        return _mark_agent_failed(project_id, usage_statement_id, "link", exc)


def _vision_agent_review_url() -> str:
    base_url = VISION_AGENT_BASE_URL.rstrip("/")
    path = VISION_AGENT_REVIEW_PATH
    if not path.startswith("/"):
        path = f"/{path}"
    return f"{base_url}{path}"


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _run_vision_agent(project_id: int, usage_statement_id: int) -> dict[str, Any]:
    photo_files = list_evidence_files_by_type(project_id, SITE_PHOTO_TYPES)
    if not photo_files:
        return {"status_code": "skipped", "result_code": None, "reason": "현장사진 파일 없음"}

    photos = [
        {
            "file_id": file_info.get("id"),
            "original_filename": file_info.get("original_filename"),
            "storage_key": file_info.get("storage_key"),
            "evidence_type_code": file_info.get("uploaded_evidence_type_code"),
            "mime_type": file_info.get("mime_type"),
            "size_bytes": file_info.get("size_bytes"),
            "presigned_url": create_presigned_file_url(file_info["storage_key"]),
        }
        for file_info in photo_files
    ]

    upsert_agent_log(
        project_id=project_id,
        usage_statement_id=usage_statement_id,
        agent_type_code="vision",
        status_code="running",
        result_code=None,
        reason="vision Agent를 실행 중입니다.",
        details={
            "event": "agent_running",
            "payload": {
                "vision_agent_url": _vision_agent_review_url(),
                "photos": photos,
            },
        },
        model_name="vision_agent",
    )

    if not VISION_AGENT_BASE_URL:
        reason = "VISION_AGENT_BASE_URL 환경변수가 설정되지 않아 vision Agent를 호출할 수 없습니다."
        upsert_agent_log(
            project_id=project_id,
            usage_statement_id=usage_statement_id,
            agent_type_code="vision",
            status_code="fail",
            result_code="fail",
            reason=reason,
            details={
                "event": "agent_config_missing",
                "summary": reason,
                "payload": {"photos": photos},
            },
            model_name="vision_agent",
        )
        return {"status_code": "fail", "result_code": "fail", "reason": reason}

    payload = {
        "project_id": project_id,
        "usage_statement_id": usage_statement_id,
        "photos": photos,
    }

    try:
        response = requests.post(
            _vision_agent_review_url(),
            json=payload,
            timeout=VISION_AGENT_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        body = response.json()

        todos = body.get("todos") or []
        status_code = str(body.get("status_code") or "success").lower()
        if status_code == "failed":
            status_code = "fail"
        default_result_code = "fail" if status_code in {"fail", "canceled"} else ("hil" if todos else "success")
        result_code = str(body.get("result_code") or default_result_code).lower()
        reason = str(
            body.get("reason")
            or ("현장사진 검토 보완 필요" if result_code == "hil" else "현장사진 검토 적정")
        )
        token = _int_or_none(body.get("token") or body.get("token_usage"))
        details = body.get("details") if isinstance(body.get("details"), dict) else {}
        details.setdefault("event", "vision_completed")
        details.setdefault("summary", reason)
        details.setdefault("payload", {})
        details["payload"].update(
            {
                "photos": photos,
                "vision_response": body,
                "todos": todos,
            }
        )

        upsert_agent_log(
            project_id=project_id,
            usage_statement_id=usage_statement_id,
            agent_type_code="vision",
            status_code=status_code,
            result_code=result_code,
            reason=reason,
            details=details,
            model_name=str(body.get("model_name") or "vision_agent"),
            token=token,
        )
        return {
            "status_code": status_code,
            "result_code": result_code,
            "reason": reason,
            "todos": todos,
            "token": token,
            "details": details,
        }
    except Exception as exc:
        return _mark_agent_failed(project_id, usage_statement_id, "vision", exc)


def _run_legal_agent(
    project_id: int,
    usage_statement_id: int,
    *,
    she_user_id: int | None = None,
) -> dict[str, Any]:
    upsert_agent_log(
        project_id=project_id,
        usage_statement_id=usage_statement_id,
        agent_type_code="legal",
        status_code="running",
        result_code=None,
        reason="legal Agent를 실행 중입니다.",
        details={"event": "agent_running", "payload": {"she_user_id": she_user_id}},
        model_name="validator_agent",
    )

    try:
        document, category_rows = _build_validator_document(project_id, usage_statement_id)
        audit_response = validate_usage_statement(document=document)
        summary_response = summarize_audit_response(
            response=audit_response,
            usage_statement_id=usage_statement_id,
        )
        item_results = _legal_item_results_from_audit(
            audit_response=audit_response,
            summary_response=summary_response,
            category_rows=category_rows,
        )
        category_results = [
            result.model_dump(mode="json", by_alias=False)
            for result in summary_response.results
        ]
        review_count = sum(1 for row in item_results if row["status"] != "적절")
        result_code = "hil" if review_count else "success"
        reason = "법령 검토 결과 특이사항 없음" if review_count == 0 else f"법령 검토 결과 보고서 반영 대상 {review_count}건"
        todos = [
            {
                "usage_statement_item_id": row.get("item_id"),
                "reason": f"법령 검토 필요: {row.get('reason') or row.get('status')}",
            }
            for row in item_results
            if row["status"] != "적절"
        ]
        upsert_agent_log(
            project_id=project_id,
            usage_statement_id=usage_statement_id,
            agent_type_code="legal",
            status_code="success",
            result_code=result_code,
            reason=reason,
            details={
                "event": "legal_completed",
                "summary": reason,
                "payload": {
                    "she_user_id": she_user_id,
                    "usage_statement_id": usage_statement_id,
                    "category_results": category_results,
                    "results": item_results,
                    "todos": todos,
                },
            },
            model_name="validator_agent",
        )
        return {
            "status_code": "success",
            "result_code": result_code,
            "reason": reason,
            "result_count": len(item_results),
            "todos": todos,
        }
    except Exception as exc:
        return _mark_agent_failed(project_id, usage_statement_id, "legal", exc)


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
        report_draft = draft.model_dump(mode="json")
        result = {"reportDraft": report_draft, "run_id": str(uuid4())}
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
                    "reportDraft": report_draft,
                },
            },
            model_name="report_agent",
        )
        return {
            "agent_type_code": "report",
            "status_code": "success",
            "result_code": "success",
            "reason": "보고서 초안 생성 완료",
            "reportDraft": report_draft,
            "result": result,
        }
    except Exception as exc:
        return _mark_agent_failed(project_id, usage_statement_id, "report", exc)


def _build_validator_document(project_id: int, usage_statement_id: int) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT p.appropriated_amount,
                       us.cumulative_progress_rate
                FROM usage_statements us
                JOIN projects p ON p.id = us.project_id
                WHERE us.id = %(usage_statement_id)s
                  AND us.project_id = %(project_id)s
                """,
                {"project_id": project_id, "usage_statement_id": usage_statement_id},
            )
            header = cur.fetchone()
            if header is None:
                raise KeyError(f"usage_statement not found: {usage_statement_id}")

            cur.execute(
                """
                SELECT category_code, previous_amount, current_amount, cumulative_amount
                FROM usage_statement_summaries
                WHERE usage_statement_id = %(usage_statement_id)s
                """,
                {"usage_statement_id": usage_statement_id},
            )
            summaries = {row["category_code"]: row for row in cur.fetchall()}

            cur.execute(
                """
                SELECT id, category_code, used_on, item_name, unit, quantity,
                       unit_price, total_amount, remark
                FROM usage_statement_items
                WHERE usage_statement_id = %(usage_statement_id)s
                ORDER BY category_code, id
                """,
                {"usage_statement_id": usage_statement_id},
            )
            items = [dict(row) for row in cur.fetchall()]

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        grouped.setdefault(str(item["category_code"]), []).append(item)

    categories = []
    category_rows: dict[str, list[dict[str, Any]]] = {}
    for category_code, rows in grouped.items():
        summary = summaries.get(category_code) or {}
        item_rows = [
            {
                "행ID": row["id"],
                "사용일자": row["used_on"].isoformat() if row.get("used_on") else None,
                "항목명": row["item_name"],
                "단위": row.get("unit"),
                "수량": _number_or_none(row.get("quantity")),
                "단가": _number_or_none(row.get("unit_price")),
                "금액": _number_or_none(row.get("total_amount")) or 0,
                "비고": row.get("remark") or "",
            }
            for row in rows
        ]
        category_rows[category_code] = item_rows
        categories.append(
            {
                "카테고리코드": category_code,
                "집계정보": {
                    "전회사용금액": _number_or_none(summary.get("previous_amount")) or 0,
                    "금회사용금액": _number_or_none(summary.get("current_amount")) or 0,
                    "누적사용금액": _number_or_none(summary.get("cumulative_amount")) or 0,
                },
                "항목목록": item_rows,
            }
        )

    return (
        {
            "사용내역서ID": usage_statement_id,
            "기본정보": {
                "산안비총액": _number_or_none(header.get("appropriated_amount")) or 0,
                "누계공정률": _number_or_none(header.get("cumulative_progress_rate")),
            },
            "카테고리별데이터": categories,
        },
        category_rows,
    )


def _legal_item_results_from_audit(
    *,
    audit_response,
    summary_response,
    category_rows: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    summary_by_category = {summary.category_code: summary for summary in summary_response.results}
    category_name_to_code = {name: code for code, name in CATEGORIES.items()}
    results: list[dict[str, Any]] = []
    for category_name, category_result in audit_response.categories.items():
        category_name_text = str(category_name or "")
        category_code = category_name_to_code.get(category_name_text, category_name_text)
        summary = summary_by_category.get(category_code)
        source_citations = [
            {"legal_basis": source.law, "summary": source.summary}
            for source in (summary.sources if summary else [])
        ]
        item_rows = category_rows.get(category_code) or []
        for raw_item, judgment in zip(item_rows, category_result.items):
            status = _legal_status_from_judgment(judgment, summary.status if summary else category_result.status)
            citations = (
                [{"legal_basis": law, "summary": None} for law in judgment.referenced_laws]
                or source_citations
            )
            results.append(
                {
                    "item_id": raw_item["행ID"],
                    "category_code": category_code,
                    "status": status,
                    "reason": judgment.review_reason or judgment.reasoning or (summary.reason if summary else ""),
                    "citations": citations,
                }
            )
    return results


def _legal_status_from_judgment(judgment, category_status: str) -> str:
    if judgment.needs_human_review:
        return "검토필요"
    if not judgment.allowed:
        return "부적절"
    if category_status in {"부적절", "검토필요"}:
        return category_status
    return "적절"


def _number_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_status_todos(logs: dict[str, dict[str, Any]]) -> list[SupplementTodoSnapshot]:
    todos: list[SupplementTodoSnapshot] = []
    for agent_type_code, log in sorted(logs.items()):
        if log.get("result_code") != "hil":
            continue

        details = log.get("details") or {}
        payload = details.get("payload") or {}
        raw_todos = payload.get("todos") or []
        for todo in raw_todos:
            reason = todo.get("reason")
            if not reason:
                continue
            todos.append(
                SupplementTodoSnapshot(
                    agent_type_code=agent_type_code,
                    usage_statement_item_id=todo.get("usage_statement_item_id"),
                    file_id=todo.get("file_id"),
                    reason=reason,
                    status_code=todo.get("status_code") or "open",
                )
            )
    return todos


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
