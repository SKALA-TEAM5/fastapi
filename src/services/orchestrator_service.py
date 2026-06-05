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

import psycopg2.extras

from src.agents.report_agent.agent import ReportAgent
from src.agents.report_agent.context_builder import build_report_context
from src.agents.safety_doc_agent.agent import check_missing_evidence
from src.agents.classifier_agent.agent import review_usage_statement
from src.agents.validator_agent.agent import summarize_audit_response, validate_usage_statement
from src.schemas.classifier import CATEGORIES
from src.repositories.orchestrator_repository import (
    close_supplement_todos,
    create_supplement_todos,
    list_evidence_file_ids_by_type,
    list_latest_agent_logs,
    list_usage_statement_item_ids,
    list_usage_statement_items_for_classi,
    mark_orchestrator,
    scan_orchestrator_state,
    select_evidence_agents,
    update_usage_statement_item_categories,
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


def classify_existing_usage_statement(
    project_id: int,
    usage_statement_id: int,
) -> OrchestratorActionResponse:
    items = list_usage_statement_items_for_classi(usage_statement_id)
    upsert_agent_log(
        project_id=project_id,
        usage_statement_id=usage_statement_id,
        agent_type_code="classi",
        status_code="running",
        result_code=None,
        reason="저장된 세부내역 기준 classi 재분류를 실행 중입니다.",
        details={
            "event": "classification_running",
            "summary": "저장된 세부내역 기준 classi 재분류를 실행 중입니다.",
            "payload": {"item_count": len(items)},
        },
        model_name="classifier_agent",
    )

    if not items:
        details = {
            "event": "classification_checked",
            "summary": "분류할 세부내역이 없습니다.",
            "payload": {
                "changed_count": 0,
                "kept_count": 0,
                "changes": [],
                "results": [],
            },
        }
        upsert_agent_log(
            project_id=project_id,
            usage_statement_id=usage_statement_id,
            agent_type_code="classi",
            status_code="success",
            result_code="success",
            reason=details["summary"],
            details=details,
            model_name="classifier_agent",
        )
        return OrchestratorActionResponse(
            status="success",
            message=details["summary"],
            usage_statement_id=usage_statement_id,
            target_agents=["classi"],
            result=details,
        )

    try:
        classifier_rows = [
            {
                "row_id": index,
                "given_category_code": item.get("category_code"),
                "item_name": item.get("item_name"),
            }
            for index, item in enumerate(items, start=1)
        ]
        review_response = review_usage_statement(
            usage_statement_id=usage_statement_id,
            rows=classifier_rows,
            basic_info={},
        )
        review_map = {result.row_id: result for result in review_response.results}

        kept_count = 0
        changes: list[dict[str, Any]] = []
        category_updates: list[dict[str, Any]] = []
        classifier_results: list[dict[str, Any]] = []

        for row_id, item in enumerate(items, start=1):
            original_category = item.get("category_code")
            review = review_map.get(row_id)
            if review is None:
                kept_count += 1
                classifier_results.append(
                    {
                        "row_id": row_id,
                        "item_id": item.get("id"),
                        "item_name": item.get("item_name"),
                        "original_category_code": original_category,
                        "final_category_code": original_category,
                        "status": "appropriate",
                        "reason": "classifier result was missing, so the current category was kept.",
                    }
                )
                continue

            updated_category = review.final_category_code or original_category
            status = "appropriate" if review.decision_status == "유지" else "inappropriate"
            if updated_category != original_category:
                changes.append(
                    {
                        "row_id": row_id,
                        "item_id": item.get("id"),
                        "item_name": review.item_name,
                        "before": {"category_code": original_category},
                        "after": {"category_code": updated_category},
                        "reason": review.reason,
                    }
                )
                category_updates.append(
                    {
                        "item_id": item["id"],
                        "category_code": updated_category,
                    }
                )
            else:
                kept_count += 1

            classifier_results.append(
                {
                    "row_id": row_id,
                    "item_id": item.get("id"),
                    "item_name": review.item_name,
                    "original_category_code": original_category,
                    "final_category_code": updated_category,
                    "status": status,
                    "reason": review.reason,
                }
            )

        updated_count = update_usage_statement_item_categories(
            usage_statement_id=usage_statement_id,
            changes=category_updates,
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
                "updated_count": updated_count,
                "kept_count": kept_count,
                "changes": changes,
                "results": classifier_results,
            },
        }
        upsert_agent_log(
            project_id=project_id,
            usage_statement_id=usage_statement_id,
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
            usage_statement_id=usage_statement_id,
            target_agents=["classi"],
            result=details,
        )
    except Exception as exc:
        result = _mark_agent_failed(project_id, usage_statement_id, "classi", exc)
        return OrchestratorActionResponse(
            status="fail",
            message=result["reason"],
            usage_statement_id=usage_statement_id,
            target_agents=["classi"],
            result={"classi": result},
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
        reason = "법령 검토 결과 특이사항 없음" if review_count == 0 else f"법령 검토 결과 보고서 반영 대상 {review_count}건"
        upsert_agent_log(
            project_id=project_id,
            usage_statement_id=usage_statement_id,
            agent_type_code="legal",
            status_code="success",
            result_code="success",
            reason=reason,
            details={
                "event": "legal_completed",
                "summary": reason,
                "payload": {
                    "she_user_id": she_user_id,
                    "usage_statement_id": usage_statement_id,
                    "category_results": category_results,
                    "results": item_results,
                },
            },
            model_name="validator_agent",
        )
        return {
            "status_code": "success",
            "result_code": "success",
            "reason": reason,
            "result_count": len(item_results),
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
        category_code = category_name_to_code.get(category_name, category_name)
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
