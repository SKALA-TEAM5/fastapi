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
import os
import re
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
    LEGAL_DATABASE_URL,
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
    insert_agent_usage_record,
    scan_orchestrator_state,
    select_evidence_agents,
    update_file_details,
    update_file_statuses,
    update_file_statuses_by_id,
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

try:
    from langchain_community.callbacks import get_openai_callback
except ImportError:
    get_openai_callback = None  # type: ignore

_LEGAL_ORIGINAL_TEXT_MAX_LENGTH = 600
_TARGET_EQUIPMENT_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("safety_helmet", ("안전모", "헬멧")),
    ("safety_shoes", ("안전화",)),
    ("safety_belt", ("안전벨트", "안전대", "안전띠")),
    ("safety_net", ("안전망",)),
)
_VISION_ALLOWED_CATEGORY_CODES: set[str] | None = {"CAT_02", "CAT_03"}
# 모든 카테고리의 연결된 사진을 Vision 대상으로 실행하려면 아래 설정을 사용합니다.
# _VISION_ALLOWED_CATEGORY_CODES = None


def parse_and_classify_usage_statement(file_id: int) -> OrchestratorActionResponse:
    if get_openai_callback is not None:
        with get_openai_callback() as _classi_cb:
            result = parse_usage_statement(file_id)
        _classi_usage = {"input_tokens": _classi_cb.prompt_tokens, "output_tokens": _classi_cb.completion_tokens}
    else:
        result = parse_usage_statement(file_id)
        _classi_usage = {"input_tokens": None, "output_tokens": None}
    usage_statement_id = _int_or_none(result.get("usage_statement_id"))
    project_id = _int_or_none(result.get("project_id"))
    if project_id is not None and usage_statement_id is not None:
        classifier_details = result.get("classifier_details") if isinstance(result.get("classifier_details"), dict) else {}
        classifier_changed_count = int(result.get("classifier_changed_count") or 0)
        summary = (
            f"세부내역 {classifier_changed_count}건을 올바른 항목으로 이동했습니다."
            if classifier_changed_count
            else "세부내역 분류 이동 없음"
        )
        _classi_token = (_classi_usage["input_tokens"] or 0) + (_classi_usage["output_tokens"] or 0)
        upsert_agent_log(
            project_id=project_id,
            usage_statement_id=usage_statement_id,
            agent_type_code="classi",
            status_code="success",
            result_code="success",
            reason=summary,
            details=classifier_details,
            model_name="classifier_agent",
            token=_classi_token or None,
        )
        _record_agent_usage(
            project_id=project_id,
            usage_statement_id=usage_statement_id,
            agent_type_code="classi",
            model_name=_openai_model_name(),
            input_tokens=_classi_usage["input_tokens"],
            output_tokens=_classi_usage["output_tokens"],
        )
        mark_orchestrator(
            project_id=project_id,
            usage_statement_id=usage_statement_id,
            event="classi_completed",
            status_code="success",
            result_code="success",
            reason=summary,
            payload={"file_id": file_id, "result": result},
        )
    return OrchestratorActionResponse(
        status="success",
        message="사용내역서 파싱 및 classi 실행 요청이 완료되었습니다.",
        usage_statement_id=usage_statement_id,
        target_agents=["classi"],
        result=result,
    )


def classify_existing_usage_statement(
    request: UsageStatementClassifyRequest,
) -> OrchestratorActionResponse:
    try:
        mark_orchestrator(
            project_id=request.project_id,
            usage_statement_id=request.usage_statement_id,
            event="classi_started",
            status_code="running",
            result_code=None,
            reason="classi 재분류를 시작했습니다.",
            payload={"item_id": request.item_id, "item_name": request.item_name},
        )
        submitted_category = request.category_code or ""
        if get_openai_callback is not None:
            with get_openai_callback() as _classi_cb:
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
            _classi_usage = {"input_tokens": _classi_cb.prompt_tokens, "output_tokens": _classi_cb.completion_tokens}
        else:
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
            _classi_usage = {"input_tokens": None, "output_tokens": None}
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
        _classi_token = (_classi_usage["input_tokens"] or 0) + (_classi_usage["output_tokens"] or 0)
        upsert_agent_log(
            project_id=request.project_id,
            usage_statement_id=request.usage_statement_id,
            agent_type_code="classi",
            status_code="success",
            result_code="success",
            reason=summary,
            details=details,
            model_name="classifier_agent",
            token=_classi_token or None,
        )
        _record_agent_usage(
            project_id=request.project_id,
            usage_statement_id=request.usage_statement_id,
            agent_type_code="classi",
            model_name=_openai_model_name(),
            input_tokens=_classi_usage["input_tokens"],
            output_tokens=_classi_usage["output_tokens"],
        )
        mark_orchestrator(
            project_id=request.project_id,
            usage_statement_id=request.usage_statement_id,
            event="classi_completed",
            status_code="success",
            result_code="success",
            reason=summary,
            payload={"item_id": request.item_id, "details": details},
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
        mark_orchestrator(
            project_id=request.project_id,
            usage_statement_id=request.usage_statement_id,
            event="classi_failed",
            status_code="fail",
            result_code="fail",
            reason=reason,
            payload={
                "item_id": request.item_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
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
            results[agent] = _run_safety_doc_agent(project_id, usage_statement_id, requested_by_user_id=requested_by_user_id)
        elif agent == "link":
            results[agent] = _run_link_agent(project_id, usage_statement_id, requested_by_user_id=requested_by_user_id)
        elif agent == "vision":
            results[agent] = _run_vision_agent(project_id, usage_statement_id, requested_by_user_id=requested_by_user_id)

    result_code = _aggregate_orchestrator_result_code(results)
    mark_orchestrator(
        project_id=project_id,
        usage_statement_id=usage_statement_id,
        event="evidence_review_completed",
        status_code="success" if result_code != "fail" else "fail",
        result_code=result_code,
        reason="증빙 검증 대상 Agent 실행을 완료했습니다.",
        payload={"target_agents": target_agents, "results": results},
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
    mark_orchestrator(
        project_id=project_id,
        usage_statement_id=usage_statement_id,
        event="legal_review_completed",
        status_code=result.get("status_code", "fail"),
        result_code=result.get("result_code", "fail"),
        reason=result.get("reason", "legal Agent 실행을 완료했습니다."),
        payload={"she_user_id": she_user_id, "result": result},
    )
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

    mark_orchestrator(
        project_id=project_id,
        usage_statement_id=usage_statement_id,
        event="report_draft_started",
        status_code="running",
        result_code=None,
        reason="report 초안 생성을 시작했습니다.",
        payload={"she_user_id": she_user_id},
    )
    result = _run_report_agent(project_id, usage_statement_id, she_user_id=she_user_id)
    report_result = cast(dict[str, Any], result.get("result")) if isinstance(result.get("result"), dict) else {}
    mark_orchestrator(
        project_id=project_id,
        usage_statement_id=usage_statement_id,
        event="report_draft_completed",
        status_code=result.get("status_code", "fail"),
        result_code=result.get("result_code", "fail"),
        reason=result.get("reason", "report Agent 실행을 완료했습니다."),
        payload={"she_user_id": she_user_id, "result": result},
    )
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


def _run_safety_doc_agent(
    project_id: int,
    usage_statement_id: int,
    *,
    requested_by_user_id: int | None = None,
) -> dict[str, Any]:
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
            {"item_id": item_id, "result": check_missing_evidence(item_id, persist_log=False)}
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
        usage_tokens = _sum_usage_tokens(
            _dict_or_empty(_dict_or_empty(row.get("result")).get("ai_response")).get("usage")
            for row in item_results
        )
        token = usage_tokens["input_tokens"] + usage_tokens["output_tokens"]
        model_name = _first_string(
            _dict_or_empty(row.get("result")).get("model_name")
            for row in item_results
        ) or "safety_doc_agent"
        todos = []
        for row in item_results:
            missing_evidences = (
                (_dict_or_empty(row.get("result")).get("evidence_status") or {}).get("missing_evidences")
                or []
            )
            if not missing_evidences:
                continue
            todo_context = _todo_context_from_safety_doc_result(row)
            missing_text = ", ".join(missing_evidences)
            item_name = _first_string([todo_context.get("usage_statement_item_name")])
            reason_prefix = f"{item_name} 필수 증빙 누락" if item_name else "필수 증빙 누락"
            todos.append(
                {
                    "usage_statement_item_id": row["item_id"],
                    **todo_context,
                    "title": missing_text,
                    "evidence_type_codes": missing_evidences,
                    "reason": f"{reason_prefix}: {missing_text}",
                }
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
                    "todos": todos,
                },
            },
            model_name=model_name,
            token=token,
        )
        _record_agent_usage(
            project_id=project_id,
            usage_statement_id=usage_statement_id,
            agent_type_code="safety-doc",
            model_name=model_name,
            token=token,
            input_tokens=usage_tokens["input_tokens"],
            output_tokens=usage_tokens["output_tokens"],
            cached_input_tokens=usage_tokens["cached_input_tokens"],
            requested_by_user_id=requested_by_user_id,
        )
        return {
            "status_code": "success",
            "result_code": result_code,
            "reason": reason,
            "todos": todos,
        }
    except Exception as exc:
        return _mark_agent_failed(project_id, usage_statement_id, "safety-doc", exc)


def _run_link_agent(
    project_id: int,
    usage_statement_id: int,
    *,
    requested_by_user_id: int | None = None,
) -> dict[str, Any]:
    grouped_files = list_evidence_file_ids_by_type(project_id)
    receipt_file_ids = [
        file_id
        for evidence_type in ("receipt", "transaction_statement")
        for file_id in grouped_files.get(evidence_type, [])
    ]
    tax_invoice_file_ids = grouped_files.get("tax_invoice", [])
    target_file_ids = receipt_file_ids + tax_invoice_file_ids

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
        item_contexts = _usage_statement_item_context_index(usage_statement_id)
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
        update_file_statuses(
            project_id=project_id,
            file_ids=target_file_ids,
            status_code="success" if result_code == "success" else "fail",
        )
        todos = []
        for row in result.get("match_results") or []:
            if not str(row.get("line_id") or "").isdigit():
                continue
            if row.get("match_status") not in {"review_needed", "unmatched", "rejected"}:
                continue
            item_id = int(row.get("line_id"))
            todo_context = item_contexts.get(item_id, {})
            item_name = _first_string([todo_context.get("usage_statement_item_name")])
            status_text = str(row.get("match_status") or "review_needed")
            reason_prefix = f"{item_name} 증빙 매칭 검토 필요" if item_name else "증빙 매칭 검토 필요"
            todos.append(
                {
                    "usage_statement_item_id": item_id,
                    **todo_context,
                    "title": "영수증/세금계산서",
                    "match_status": status_text,
                    "reason": f"{reason_prefix}: {status_text}",
                }
            )
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
        _record_agent_usage(
            project_id=project_id,
            usage_statement_id=usage_statement_id,
            agent_type_code="link",
            model_name="link_pipeline",
            requested_by_user_id=requested_by_user_id,
        )
        return {
            "status_code": "success",
            "result_code": result_code,
            "reason": reason,
            "todos": todos,
        }
    except Exception as exc:
        update_file_statuses(project_id=project_id, file_ids=target_file_ids, status_code="fail")
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


def _aggregate_orchestrator_result_code(results: dict[str, Any]) -> str:
    result_codes = [
        str(result.get("result_code") or "")
        for result in results.values()
        if isinstance(result, dict)
    ]
    status_codes = [
        str(result.get("status_code") or "")
        for result in results.values()
        if isinstance(result, dict)
    ]
    if "fail" in result_codes or "fail" in status_codes:
        return "fail"
    if "hil" in result_codes:
        return "hil"
    return "success"


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _vision_result_rows(body: dict[str, Any], details: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[Any] = [
        *(_as_list(body.get("results"))),
        *(_as_list(_as_dict(body.get("result")).get("results"))),
        *(_as_list(_as_dict(body.get("details")).get("results"))),
        *(_as_list(details.get("results"))),
        *(_as_list(_as_dict(details.get("result")).get("results"))),
    ]
    payload = _as_dict(details.get("payload"))
    vision_response = _as_dict(payload.get("vision_response")) or _as_dict(payload.get("visionResponse"))
    candidates.extend(_as_list(vision_response.get("results")))
    candidates.extend(_as_list(_as_dict(vision_response.get("result")).get("results")))

    if not candidates and body.get("file_id") is not None:
        candidates.append(body)

    return [row for row in candidates if isinstance(row, dict)]


def _vision_file_details(
    *,
    body: dict[str, Any],
    details: dict[str, Any],
    photos: list[dict[str, Any]],
    usage_statement_id: int,
    reason: str,
    result_code: str,
) -> dict[int, dict[str, Any]]:
    photo_by_file_id = {
        int(photo["file_id"]): photo
        for photo in photos
        if photo.get("file_id") is not None
    }
    rows = _vision_result_rows(body, details)

    if not rows and len(photo_by_file_id) == 1 and body.get("detections"):
        only_file_id = next(iter(photo_by_file_id))
        rows = [{**body, "file_id": only_file_id}]

    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        raw_file_id = row.get("file_id") or row.get("fileId")
        if raw_file_id is None:
            continue
        try:
            file_id = int(raw_file_id)
        except (TypeError, ValueError):
            continue

        nested_result = _as_dict(row.get("result")) or row
        detections = _as_list(nested_result.get("detections"))
        is_appropriate = (
            row.get("is_appropriate")
            if row.get("is_appropriate") is not None
            else row.get("isAppropriate", nested_result.get("is_appropriate", nested_result.get("isAppropriate")))
        )
        result[file_id] = {
            "vision_validation": {
                "usage_statement_id": usage_statement_id,
                "status_code": str(body.get("status_code") or "success").lower(),
                "result_code": "success" if is_appropriate is True else result_code,
                "reason": str(row.get("reason") or row.get("message") or reason),
                "original_filename": (
                    row.get("original_filename")
                    or row.get("originalFilename")
                    or photo_by_file_id.get(file_id, {}).get("original_filename")
                ),
                "image_width": nested_result.get("image_width") or nested_result.get("imageWidth"),
                "image_height": nested_result.get("image_height") or nested_result.get("imageHeight"),
                "is_appropriate": is_appropriate,
                "detections": detections,
            }
        }

    return result


def _vision_file_statuses(
    *,
    body: dict[str, Any],
    details: dict[str, Any],
) -> dict[int, str]:
    statuses: dict[int, str] = {}
    for row in _vision_result_rows(body, details):
        raw_file_id = row.get("file_id") or row.get("fileId")
        file_id = _int_or_none(raw_file_id)
        if file_id is None:
            continue
        nested_result = _as_dict(row.get("result")) or row
        is_appropriate = (
            row.get("is_appropriate")
            if row.get("is_appropriate") is not None
            else row.get("isAppropriate", nested_result.get("is_appropriate", nested_result.get("isAppropriate")))
        )
        statuses[file_id] = "success" if is_appropriate is True else "fail"
    return statuses


def _is_vision_allowed_file_context(file_context: dict[str, Any] | None) -> bool:
    if not file_context:
        return False
    if _VISION_ALLOWED_CATEGORY_CODES is None:
        return True
    category_code = _first_string([file_context.get("category_code")])
    return category_code in _VISION_ALLOWED_CATEGORY_CODES


def _run_vision_agent(
    project_id: int,
    usage_statement_id: int,
    *,
    requested_by_user_id: int | None = None,
) -> dict[str, Any]:
    photo_files = list_evidence_files_by_type(project_id, SITE_PHOTO_TYPES)
    if not photo_files:
        return {"status_code": "skipped", "result_code": None, "reason": "현장사진 파일 없음"}
    file_contexts = _evidence_file_todo_context_index(usage_statement_id)
    vision_photo_files = [
        file_info
        for file_info in photo_files
        if _is_vision_allowed_file_context(file_contexts.get(_int_or_none(file_info.get("id")) or -1))
    ]
    if not vision_photo_files:
        return {"status_code": "skipped", "result_code": None, "reason": "Vision 검증 대상 현장사진 없음"}
    photo_file_ids = [int(file_info["id"]) for file_info in vision_photo_files if file_info.get("id") is not None]

    photos = [
        {
            "file_id": file_info.get("id"),
            "original_filename": file_info.get("original_filename"),
            "storage_key": file_info.get("storage_key"),
            "evidence_type_code": file_info.get("uploaded_evidence_type_code"),
            "mime_type": file_info.get("mime_type"),
            "size_bytes": file_info.get("size_bytes"),
            "presigned_url": create_presigned_file_url(file_info["storage_key"]),
            **file_contexts.get(_int_or_none(file_info.get("id")) or -1, {}),
        }
        for file_info in vision_photo_files
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
        update_file_statuses(project_id=project_id, file_ids=photo_file_ids, status_code="fail")
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

        todos = [
            {
                **todo,
                **file_contexts.get(_int_or_none(todo.get("file_id")) or -1, {}),
            }
            for todo in (body.get("todos") or [])
            if isinstance(todo, dict)
        ]
        status_code = str(body.get("status_code") or "success").lower()
        if status_code == "failed":
            status_code = "fail"
        default_result_code = "fail" if status_code in {"fail", "canceled"} else ("hil" if todos else "success")
        result_code = str(body.get("result_code") or default_result_code).lower()
        reason = str(
            body.get("reason")
            or ("현장사진 검토 보완 필요" if result_code == "hil" else "현장사진 검토 적정")
        )
        usage_tokens = _usage_tokens_from_usage(body.get("usage") or body.get("token_usage") or body)
        token = _int_or_none(body.get("token") or body.get("token_usage"))
        if token is None:
            token = usage_tokens["input_tokens"] + usage_tokens["output_tokens"]
        source_details = body.get("details")
        details = dict(source_details) if isinstance(source_details, dict) else {}
        details.setdefault("event", "vision_completed")
        details.setdefault("summary", reason)
        details["payload"] = dict(details.get("payload") or {})
        vision_response = {
            key: value
            for key, value in body.items()
            if key != "details"
        }
        details["payload"].update(
            {
                "photos": photos,
                "vision_response": vision_response,
                "todos": todos,
            }
        )
        update_file_statuses_by_id(
            project_id=project_id,
            statuses_by_file_id=_vision_file_statuses(body=body, details=details),
        )
        update_file_details(
            project_id=project_id,
            details_by_file_id=_vision_file_details(
                body=body,
                details=details,
                photos=photos,
                usage_statement_id=usage_statement_id,
                reason=reason,
                result_code=result_code,
            ),
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
        _record_agent_usage(
            project_id=project_id,
            usage_statement_id=usage_statement_id,
            agent_type_code="vision",
            model_name=str(body.get("model_name") or "vision_agent"),
            token=token,
            input_tokens=usage_tokens["input_tokens"],
            output_tokens=usage_tokens["output_tokens"],
            cached_input_tokens=usage_tokens["cached_input_tokens"],
            requested_by_user_id=requested_by_user_id,
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
        update_file_statuses(project_id=project_id, file_ids=photo_file_ids, status_code="fail")
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
        if get_openai_callback is not None:
            with get_openai_callback() as _legal_cb:
                audit_response = validate_usage_statement(document=document)
            _legal_usage = {"input_tokens": _legal_cb.prompt_tokens, "output_tokens": _legal_cb.completion_tokens}
        else:
            audit_response = validate_usage_statement(document=document)
            _legal_usage = {"input_tokens": None, "output_tokens": None}
        summary_response = summarize_audit_response(
            response=audit_response,
            usage_statement_id=usage_statement_id,
        )
        item_results = _legal_item_results_from_audit(
            audit_response=audit_response,
            summary_response=summary_response,
            category_rows=category_rows,
        )
        linked_files_by_item_id = _linked_files_by_item_id(usage_statement_id)
        legal_basis_by_source_id = _legal_basis_by_citation(_legal_citations_from_results(item_results))
        frontend_categories = _legal_frontend_categories(
            item_results=item_results,
            category_rows=category_rows,
            linked_files_by_item_id=linked_files_by_item_id,
            legal_basis_by_source_id=legal_basis_by_source_id,
        )
        payload_item_results = _legal_payload_item_results(item_results)
        review_count = sum(1 for row in item_results if row["status"] != "적절")
        result_code = "hil" if review_count else "success"
        reason = "법령 검토 결과 특이사항 없음" if review_count == 0 else f"법령 검토 결과 보고서 반영 대상 {review_count}건"
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
        _legal_token = (_legal_usage["input_tokens"] or 0) + (_legal_usage["output_tokens"] or 0)
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
                    "usage_statement_id": usage_statement_id,
                    "results": payload_item_results,
                    "todos": todos,
                    "categories": frontend_categories,
                },
            },
            model_name="validator_agent",
            token=_legal_token or None,
        )
        _record_agent_usage(
            project_id=project_id,
            usage_statement_id=usage_statement_id,
            agent_type_code="legal",
            model_name=_openai_model_name(),
            input_tokens=_legal_usage["input_tokens"],
            output_tokens=_legal_usage["output_tokens"],
            requested_by_user_id=she_user_id,
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
        report_agent = ReportAgent()
        draft = report_agent.generate(context)
        usage_tokens = _usage_tokens_from_report_agent(report_agent)
        report_model_name = _report_agent_model_name(report_agent)
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
            model_name=report_model_name,
        )
        _record_agent_usage(
            project_id=project_id,
            usage_statement_id=usage_statement_id,
            agent_type_code="report",
            model_name=report_model_name,
            input_tokens=usage_tokens["input_tokens"],
            output_tokens=usage_tokens["output_tokens"],
            cached_input_tokens=usage_tokens["cached_input_tokens"],
            requested_by_user_id=she_user_id,
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
            citations = _legal_item_citations(judgment, source_citations)
            results.append(
                {
                    "item_id": raw_item["행ID"],
                    "category_code": category_code,
                    "status": status,
                    "reason": judgment.reason_text or judgment.review_reason or judgment.reasoning or "",
                    "citations": citations,
                }
            )
    return results


def _legal_frontend_categories(
    *,
    item_results: list[dict[str, Any]],
    category_rows: dict[str, list[dict[str, Any]]],
    linked_files_by_item_id: dict[int, list[dict[str, Any]]],
    legal_basis_by_source_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    item_results_by_id: dict[int, dict[str, Any]] = {}
    for item_result in item_results:
        item_id = _int_or_none(item_result.get("item_id"))
        if item_id is not None:
            item_results_by_id[item_id] = item_result

    categories: list[dict[str, Any]] = []
    for category_code, rows in category_rows.items():
        category_items: list[dict[str, Any]] = []
        category_decisions: list[str] = []
        rows = category_rows.get(category_code) or []
        for row in rows:
            item_id = _int_or_none(row.get("행ID"))
            item_result = item_results_by_id.get(item_id) if item_id is not None else None
            status = str((item_result or {}).get("status") or "적절")
            decision = _frontend_legal_decision(status)
            category_decisions.append(decision)
            amount = _number_or_none(row.get("금액")) or 0
            recognized_amount = amount if decision == "appropriate" else 0
            disputed_amount = 0 if decision == "appropriate" else amount
            review_reason = str((item_result or {}).get("reason") or "")
            category_items.append(
                {
                    "usageStatementItemId": item_id,
                    "itemName": row.get("항목명") or "",
                    "usedOn": row.get("사용일자"),
                    "amount": amount,
                    "recognizedAmount": recognized_amount,
                    "disputedAmount": disputed_amount,
                    "decision": decision,
                    "reviewReason": review_reason,
                    "problemFiles": linked_files_by_item_id.get(item_id, []) if item_id is not None else [],
                    "legalBasis": _legal_basis_payload(
                        (item_result or {}).get("citations") or [],
                        legal_basis_by_source_id,
                    ),
                }
            )

        categories.append(
            {
                "categoryCode": category_code,
                "categoryName": CATEGORIES.get(category_code, category_code),
                "decision": _aggregate_frontend_legal_decision(category_decisions),
                "items": category_items,
            }
        )
    return categories


def _legal_payload_item_results(item_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload_results: list[dict[str, Any]] = []
    for item_result in item_results:
        payload_results.append(
            {
                "item_id": item_result.get("item_id"),
                "category_code": item_result.get("category_code"),
                "status": item_result.get("status"),
                "reason": item_result.get("reason") or "",
                "citations": [
                    {"legal_basis": str(citation.get("legal_basis") or "")}
                    for citation in (item_result.get("citations") or [])
                    if isinstance(citation, dict) and str(citation.get("legal_basis") or "").strip()
                ],
            }
        )
    return payload_results


def _legal_item_citations(judgment, fallback_citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    qdrant_citations = [
        citation
        for citation in (getattr(judgment, "qdrant_citations", []) or [])
        if isinstance(citation, dict)
    ]
    if getattr(judgment, "judgment_source", "") == "llm_fallback" and qdrant_citations:
        return qdrant_citations

    source_ids = [
        str(source_id)
        for source_id in (getattr(judgment, "source_ids", []) or [])
        if source_id and str(source_id) != "llm_fallback"
    ]
    referenced_laws = [str(law) for law in (getattr(judgment, "referenced_laws", []) or []) if law]
    if source_ids:
        citations = [
            {
                "source_id": source_id,
                "legal_basis": referenced_laws[index] if index < len(referenced_laws) else "",
                "summary": None,
                "judgment_source": getattr(judgment, "judgment_source", ""),
            }
            for index, source_id in enumerate(source_ids)
        ]
        return citations + qdrant_citations
    if referenced_laws:
        return [
            {
                "legal_basis": law,
                "summary": None,
                "judgment_source": getattr(judgment, "judgment_source", ""),
            }
            for law in referenced_laws
        ]
    return fallback_citations


def _linked_files_by_item_id(usage_statement_id: int) -> dict[int, list[dict[str, Any]]]:
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT efl.usage_statement_item_id AS item_id,
                       f.id AS file_id,
                       f.original_filename
                FROM evidence_file_links efl
                JOIN usage_statement_items usi ON usi.id = efl.usage_statement_item_id
                JOIN files f ON f.id = efl.file_id
                WHERE usi.usage_statement_id = %(usage_statement_id)s
                ORDER BY efl.usage_statement_item_id, f.uploaded_at, f.id
                """,
                {"usage_statement_id": usage_statement_id},
            )
            rows = cur.fetchall()

    result: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        item_id = _int_or_none(row.get("item_id"))
        file_id = _int_or_none(row.get("file_id"))
        filename = str(row.get("original_filename") or "").strip()
        if item_id is None or file_id is None or not filename:
            continue
        result.setdefault(item_id, []).append(
            {
                "fileId": file_id,
                "originalFilename": filename,
            }
        )
    return result


def _legal_citations_from_results(item_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        citation
        for row in item_results
        for citation in (row.get("citations") or [])
        if isinstance(citation, dict)
    ]


def _legal_basis_by_citation(citations: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    source_ids = sorted(
        {
            str(citation.get("source_id"))
            for citation in citations
            if citation.get("source_id") and str(citation.get("source_id")) != "llm_fallback"
        }
    )
    legal_basis_values = sorted(
        {
            str(citation.get("legal_basis") or "").strip()
            for citation in citations
            if str(citation.get("legal_basis") or "").strip()
        }
    )
    if not source_ids and not legal_basis_values:
        return {}
    conn = psycopg2.connect(LEGAL_DATABASE_URL)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, source_name, article_no, paragraph_no, item_no, body, legal_basis
                FROM legal_rag.legal_master
                WHERE id = ANY(%(source_ids)s)
                   OR legal_basis = ANY(%(legal_basis_values)s)
                """,
                {
                    "source_ids": source_ids,
                    "legal_basis_values": legal_basis_values,
                },
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        basis = dict(row)
        aliases = {
            str(row.get("id") or ""),
            str(row.get("legal_basis") or ""),
            _join_clause_parts(row.get("source_name"), row.get("legal_basis")),
            _join_clause_parts(row.get("source_name"), row.get("article_no")),
            _join_clause_parts(row.get("source_name"), row.get("article_no"), row.get("paragraph_no"), row.get("item_no")),
        }
        for alias in aliases:
            if alias and alias not in result:
                result[alias] = basis
    return result


def _legal_basis_payload(
    citations: list[dict[str, Any]],
    legal_basis_by_source_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        source_id = str(citation.get("source_id") or "")
        legal_basis = str(citation.get("legal_basis") or "")
        is_qdrant_citation = source_id.startswith("qdrant:")
        basis = None if is_qdrant_citation else (
            legal_basis_by_source_id.get(source_id) or _lookup_legal_basis(legal_basis, legal_basis_by_source_id)
        )
        legal_basis = legal_basis or str((basis or {}).get("legal_basis") or "")
        law_name, article, clause = _split_legal_basis(legal_basis)
        if basis:
            law_name = str(basis.get("source_name") or law_name or "산업안전보건관리비 계상 및 사용기준")
            article = str(basis.get("article_no") or article or "")
            clause = _join_clause_parts(basis.get("paragraph_no"), basis.get("item_no")) or clause
        citation_original_text = _limit_original_text(str(citation.get("original_text") or ""))
        fallback_original_text = _limit_original_text(str((basis or {}).get("body") or ""))
        original_text = citation_original_text if is_qdrant_citation else (fallback_original_text or citation_original_text)
        if not original_text:
            continue
        key = source_id or legal_basis or str(citation)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        payload.append(
            {
                "lawName": law_name or "산업안전보건관리비 계상 및 사용기준",
                "article": article,
                "clause": clause,
                "originalText": original_text,
                "summary": str(citation.get("summary") or ""),
            }
        )
    return payload


def _lookup_legal_basis(legal_basis: str, candidates: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if not legal_basis:
        return None
    law_name, article, clause = _split_legal_basis(legal_basis)
    lookup_keys = [
        legal_basis,
        _join_clause_parts(law_name, legal_basis),
        _join_clause_parts(law_name, article),
        _join_clause_parts(law_name, article, clause),
    ]
    for key in lookup_keys:
        if key and key in candidates:
            return candidates[key]
    return None


def _limit_original_text(text: str) -> str:
    if len(text) <= _LEGAL_ORIGINAL_TEXT_MAX_LENGTH:
        return text
    return text[:_LEGAL_ORIGINAL_TEXT_MAX_LENGTH].rstrip() + "...(원문 일부)"


def _aggregate_frontend_legal_decision(decisions: list[str]) -> str:
    if any(decision == "inappropriate" for decision in decisions):
        return "inappropriate"
    if any(decision == "conditional" for decision in decisions):
        return "conditional"
    return "appropriate"


def _join_clause_parts(*parts: Any) -> str:
    return " ".join(str(part).strip() for part in parts if str(part or "").strip())


def _split_legal_basis(value: str) -> tuple[str, str, str]:
    text = " ".join(str(value or "").split())
    if not text:
        return "", "", ""

    article_match = re.search(r"제\s*\d+\s*조(?:의\s*\d+)?|별표\s*\d+(?:의\s*\d+)?", text)
    article = article_match.group(0).replace(" ", "") if article_match else ""
    article = re.sub(r"(별표)(\d)", r"\1 \2", article)

    clause_parts: list[str] = []
    paragraph_match = re.search(r"제\s*\d+\s*항", text)
    item_match = re.search(r"제\s*\d+\s*호", text)
    if paragraph_match:
        clause_parts.append(paragraph_match.group(0).replace(" ", ""))
    if item_match:
        clause_parts.append(item_match.group(0).replace(" ", ""))

    law_name = text
    if article_match:
        law_name = text[: article_match.start()].strip()
    law_name = law_name.strip("「」 ,")
    return law_name, article, " ".join(clause_parts)


def _category_id_number(category_code: str) -> int:
    try:
        return int(category_code.removeprefix("CAT_"))
    except ValueError:
        return 0


def _frontend_legal_decision(status: str) -> str:
    normalized = status.strip().lower()
    if normalized in {"부적절", "inappropriate", "invalid", "fail", "failed"}:
        return "inappropriate"
    if normalized in {"검토필요", "검토 필요", "needs_review", "review", "conditional", "hil"}:
        return "conditional"
    return "appropriate"


def _frontend_legal_risk_level(decision: str) -> str:
    if decision == "inappropriate":
        return "high"
    if decision == "conditional":
        return "medium"
    return "low"


def _frontend_legal_basis(summary, item_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    basis_by_key: dict[str, dict[str, Any]] = {}
    for source in getattr(summary, "sources", []) or []:
        law_name = str(getattr(source, "law", "") or "산업안전보건관리비 계상 및 사용기준")
        basis_by_key[law_name] = {
            "lawName": law_name,
            "article": "",
            "clause": "",
            "summary": str(getattr(source, "summary", "") or ""),
            "agentReasoning": str(getattr(summary, "reason", "") or ""),
        }
    for item_result in item_results:
        for citation in item_result.get("citations") or []:
            if not isinstance(citation, dict):
                continue
            law_name = str(citation.get("legal_basis") or "산업안전보건관리비 계상 및 사용기준")
            basis_by_key.setdefault(
                law_name,
                {
                    "lawName": law_name,
                    "article": "",
                    "clause": "",
                    "summary": str(citation.get("summary") or ""),
                    "agentReasoning": str(item_result.get("reason") or getattr(summary, "reason", "") or ""),
                },
            )
    if basis_by_key:
        return list(basis_by_key.values())
    return [
        {
            "lawName": "산업안전보건관리비 계상 및 사용기준",
            "article": "",
            "clause": "",
            "summary": "",
            "agentReasoning": str(getattr(summary, "reason", "") or ""),
        }
    ]


def _frontend_legal_issues(
    item_results: list[dict[str, Any]],
    fallback_reason: str,
    linked_file_names_by_item_id: dict[int, list[str]],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for item_result in item_results:
        status = str(item_result.get("status") or "")
        if status == "적절":
            continue
        item_id = _int_or_none(item_result.get("item_id"))
        reason = str(item_result.get("reason") or fallback_reason or "법령 검토가 필요합니다.")
        issues.append(
            {
                "title": "법령 검토 필요" if status == "검토필요" else "부적정 사용 가능성",
                "description": reason,
                "problemFileNames": linked_file_names_by_item_id.get(item_id, []) if item_id is not None else [],
                "requiredAction": reason,
                "recommendedFiles": [],
            }
        )
    return issues


def _legal_status_from_judgment(judgment, category_status: str) -> str:
    if not judgment.allowed:
        if judgment.needs_human_review:
            return "검토필요"
        return "부적절"
    # allowed=True이면 needs_human_review 포함 모두 적절.
    # 전담 여부 등 조건/주의사항은 reason_text에 남긴다.
    return "적절"


def _number_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _record_agent_usage(
    *,
    project_id: int,
    usage_statement_id: int,
    agent_type_code: str,
    model_name: str | None = None,
    token: int | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cached_input_tokens: int | None = None,
    requested_by_user_id: int | None = None,
) -> None:
    try:
        insert_agent_usage_record(
            project_id=project_id,
            usage_statement_id=usage_statement_id,
            agent_type_code=agent_type_code,
            model_name=model_name,
            token=token,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            requested_by_user_id=requested_by_user_id,
        )
    except Exception:
        return


def _usage_tokens_from_report_agent(report_agent: ReportAgent) -> dict[str, int]:
    llm_client = getattr(report_agent, "llm_client", None)
    return _usage_tokens_from_usage(getattr(llm_client, "last_usage", None))


def _report_agent_model_name(report_agent: ReportAgent) -> str:
    llm_client = getattr(report_agent, "llm_client", None)
    model_name = getattr(llm_client, "model", None)
    return model_name if isinstance(model_name, str) and model_name.strip() else "report_agent"


def _sum_usage_tokens(usages) -> dict[str, int]:
    total = {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0}
    for usage in usages:
        tokens = _usage_tokens_from_usage(usage)
        total["input_tokens"] += tokens["input_tokens"]
        total["output_tokens"] += tokens["output_tokens"]
        total["cached_input_tokens"] += tokens["cached_input_tokens"]
    return total


def _usage_tokens_from_usage(usage: Any) -> dict[str, int]:
    usage_dict = _usage_dict(usage)
    input_tokens = _first_int(
        usage_dict.get("input_tokens"),
        usage_dict.get("prompt_tokens"),
        usage_dict.get("inputTokenCount"),
        usage_dict.get("promptTokenCount"),
    )
    output_tokens = _first_int(
        usage_dict.get("output_tokens"),
        usage_dict.get("completion_tokens"),
        usage_dict.get("candidatesTokenCount"),
        usage_dict.get("outputTokenCount"),
        usage_dict.get("completionTokenCount"),
    )
    total_tokens = _first_int(
        usage_dict.get("total_tokens"),
        usage_dict.get("totalTokenCount"),
        usage_dict.get("totalToken"),
    )
    cached_input_tokens = _first_int(
        usage_dict.get("cached_tokens"),
        usage_dict.get("cached_input_tokens"),
        usage_dict.get("cachedContentTokenCount"),
        _usage_dict(usage_dict.get("input_tokens_details")).get("cached_tokens"),
        _usage_dict(usage_dict.get("prompt_tokens_details")).get("cached_tokens"),
    )
    if input_tokens == 0 and output_tokens == 0 and total_tokens > 0:
        input_tokens = total_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": min(cached_input_tokens, input_tokens),
    }


def _usage_dict(usage: Any) -> dict[str, Any]:
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "model_dump"):
        dumped = usage.model_dump()
        return dumped if isinstance(dumped, dict) else {}
    if hasattr(usage, "dict"):
        dumped = usage.dict()
        return dumped if isinstance(dumped, dict) else {}
    return {}


def _first_int(*values: Any) -> int:
    for value in values:
        parsed = _int_or_none(value)
        if parsed is not None:
            return max(parsed, 0)
    return 0


def _first_string(values) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _target_equipment_from_item_context(item_context: dict[str, Any]) -> str | None:
    haystack = " ".join(
        value
        for value in (
            _first_string([item_context.get("usage_statement_item_name"), item_context.get("item_name")]),
            _first_string([item_context.get("category_name")]),
        )
        if value
    )
    category_code = _first_string([item_context.get("category_code")])
    if category_code == "CAT_02":
        allowed_targets = {"safety_net"}
    elif category_code == "CAT_03":
        allowed_targets = {"safety_helmet", "safety_shoes", "safety_belt"}
    else:
        allowed_targets = None
    for target_equipment, keywords in _TARGET_EQUIPMENT_KEYWORDS:
        if allowed_targets is not None and target_equipment not in allowed_targets:
            continue
        if any(keyword in haystack for keyword in keywords):
            return target_equipment
    return None


def _openai_model_name() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"


def _todo_context_from_safety_doc_result(row: dict[str, Any]) -> dict[str, Any]:
    result = _dict_or_empty(row.get("result"))
    input_from_db_views = _dict_or_empty(result.get("input_from_db_views"))
    item_context = _dict_or_empty(input_from_db_views.get("item_context"))
    if not item_context:
        return {}

    category_code = _first_string([item_context.get("category_code")])
    return {
        "category_code": category_code,
        "category_name": _first_string([item_context.get("category_name")])
        or (CATEGORIES.get(category_code) if category_code else None),
        "usage_statement_item_name": _first_string([item_context.get("item_name")]),
    }


def _usage_statement_item_context_index(usage_statement_id: int) -> dict[int, dict[str, Any]]:
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    usi.id AS item_id,
                    usi.category_code,
                    uc.name AS category_name,
                    usi.item_name
                FROM usage_statement_items usi
                LEFT JOIN usage_categories uc
                  ON uc.code = usi.category_code
                WHERE usi.usage_statement_id = %(usage_statement_id)s
                ORDER BY usi.id
                """,
                {"usage_statement_id": usage_statement_id},
            )
            rows = cur.fetchall()

    contexts: dict[int, dict[str, Any]] = {}
    for row in rows:
        item_id = _int_or_none(row.get("item_id"))
        if item_id is None:
            continue
        category_code = _first_string([row.get("category_code")])
        contexts[item_id] = {
            "category_code": category_code,
            "category_name": _first_string([row.get("category_name")])
            or (CATEGORIES.get(category_code) if category_code else None),
            "usage_statement_item_name": _first_string([row.get("item_name")]),
        }
        target_equipment = _target_equipment_from_item_context(contexts[item_id])
        if target_equipment:
            contexts[item_id]["target_equipment"] = target_equipment
    return contexts


def _evidence_file_todo_context_index(usage_statement_id: int) -> dict[int, dict[str, Any]]:
    item_contexts = _usage_statement_item_context_index(usage_statement_id)
    if not item_contexts:
        return {}

    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    efl.file_id,
                    efl.usage_statement_item_id
                FROM evidence_file_links efl
                JOIN usage_statement_items usi
                  ON usi.id = efl.usage_statement_item_id
                WHERE usi.usage_statement_id = %(usage_statement_id)s
                ORDER BY efl.file_id, efl.usage_statement_item_id
                """,
                {"usage_statement_id": usage_statement_id},
            )
            rows = cur.fetchall()

    contexts: dict[int, dict[str, Any]] = {}
    for row in rows:
        file_id = _int_or_none(row.get("file_id"))
        item_id = _int_or_none(row.get("usage_statement_item_id"))
        if file_id is None or item_id is None or file_id in contexts:
            continue
        item_context = item_contexts.get(item_id)
        if not item_context:
            continue
        contexts[file_id] = {
            "usage_statement_item_id": item_id,
            **item_context,
        }
    return contexts


def _build_todo_context_index(logs: dict[str, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    context_by_item_id: dict[int, dict[str, Any]] = {}
    for log in logs.values():
        payload = _dict_or_empty(_dict_or_empty(log.get("details")).get("payload"))

        for row in payload.get("item_results") or []:
            if not isinstance(row, dict):
                continue
            item_id = _int_or_none(row.get("item_id"))
            if item_id is None:
                continue
            context = _todo_context_from_safety_doc_result(row)
            if context:
                context_by_item_id[item_id] = {**context_by_item_id.get(item_id, {}), **context}

        for row in payload.get("results") or []:
            if not isinstance(row, dict):
                continue
            item_id = _int_or_none(row.get("item_id"))
            if item_id is None:
                continue
            category_code = _first_string([row.get("category_code")])
            if category_code:
                context_by_item_id[item_id] = {
                    **context_by_item_id.get(item_id, {}),
                    "category_code": category_code,
                    "category_name": CATEGORIES.get(category_code),
                }

    return context_by_item_id


def _build_status_todos(logs: dict[str, dict[str, Any]]) -> list[SupplementTodoSnapshot]:
    todos: list[SupplementTodoSnapshot] = []
    context_by_item_id = _build_todo_context_index(logs)
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
            item_id = _int_or_none(todo.get("usage_statement_item_id"))
            context = context_by_item_id.get(item_id or -1, {})
            category_code = _first_string([todo.get("category_code"), context.get("category_code")])
            todos.append(
                SupplementTodoSnapshot(
                    agent_type_code=agent_type_code,
                    usage_statement_item_id=item_id,
                    category_code=category_code,
                    category_name=_first_string([todo.get("category_name"), context.get("category_name")])
                    or (CATEGORIES.get(category_code) if category_code else None),
                    usage_statement_item_name=_first_string(
                        [todo.get("usage_statement_item_name"), context.get("usage_statement_item_name")]
                    ),
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
