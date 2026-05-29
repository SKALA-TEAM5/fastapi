"""
AI Review Orchestrator DB 레포지토리
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Orchestrator가 실행 조건을 판단하는 데 필요한 DB 상태를 조회하고,
Agent별 최신 실행 상태를 `agent_logs`에 기록한다.

주요 책임:
  - 사용내역서 세부항목, 증빙 파일, Agent 로그 상태 스캔
  - 조건부 실행 대상 Agent 선택 보조
  - Orchestrator/Agent 로그 upsert 및 보완 TODO 생성
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import psycopg2.extras

from src.repositories.db import get_connection


RECEIPT_TYPES = {"receipt", "tax_invoice"}
SITE_PHOTO_TYPES = {"site_photo"}


@dataclass(frozen=True)
class OrchestratorState:
    project_id: int
    usage_statement_id: int
    has_usage_statement_items: bool
    has_receipts_or_tax_invoices: bool
    has_site_photos: bool
    logs: dict[str, dict[str, Any]]

    @property
    def classi_ready(self) -> bool:
        log = self.logs.get("classi") or {}
        return log.get("status_code") == "success" and log.get("result_code") == "success"

    @property
    def evidence_has_hil(self) -> bool:
        return any(
            (self.logs.get(agent) or {}).get("result_code") == "hil"
            for agent in ("safety-doc", "link", "vision")
        )

    @property
    def evidence_review_ready(self) -> bool:
        if not self.classi_ready:
            return False
        required_agents = ["safety-doc"]
        if self.has_receipts_or_tax_invoices:
            required_agents.append("link")
        if self.has_site_photos:
            required_agents.append("vision")
        return all(
            (self.logs.get(agent) or {}).get("status_code") == "success"
            and (self.logs.get(agent) or {}).get("result_code") == "success"
            for agent in required_agents
        )

    @property
    def legal_ready(self) -> bool:
        return self.evidence_review_ready

    @property
    def report_ready(self) -> bool:
        log = self.logs.get("legal") or {}
        return log.get("status_code") == "success" and log.get("result_code") == "success"


def scan_orchestrator_state(project_id: int, usage_statement_id: int) -> OrchestratorState:
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT COUNT(*) AS item_count
                FROM usage_statement_items
                WHERE usage_statement_id = %(usage_statement_id)s
                """,
                {"usage_statement_id": usage_statement_id},
            )
            item_count = int((cur.fetchone() or {}).get("item_count") or 0)

            cur.execute(
                """
                SELECT uploaded_evidence_type_code, COUNT(*) AS file_count
                FROM files
                WHERE project_id = %(project_id)s
                  AND deleted_at IS NULL
                GROUP BY uploaded_evidence_type_code
                """,
                {"project_id": project_id},
            )
            file_counts = {
                row["uploaded_evidence_type_code"]: int(row["file_count"])
                for row in cur.fetchall()
            }

            cur.execute(
                """
                SELECT DISTINCT ON (agent_type_code)
                    agent_type_code, status_code, result_code, reason, details, token, updated_at
                FROM agent_logs
                WHERE project_id = %(project_id)s
                  AND usage_statement_id = %(usage_statement_id)s
                ORDER BY agent_type_code, updated_at DESC, id DESC
                """,
                {
                    "project_id": project_id,
                    "usage_statement_id": usage_statement_id,
                },
            )
            logs = {row["agent_type_code"]: dict(row) for row in cur.fetchall()}

    return OrchestratorState(
        project_id=project_id,
        usage_statement_id=usage_statement_id,
        has_usage_statement_items=item_count > 0,
        has_receipts_or_tax_invoices=any(file_counts.get(code, 0) > 0 for code in RECEIPT_TYPES),
        has_site_photos=any(file_counts.get(code, 0) > 0 for code in SITE_PHOTO_TYPES),
        logs=logs,
    )


def select_evidence_agents(state: OrchestratorState) -> list[str]:
    agents = ["safety-doc"]
    if state.has_receipts_or_tax_invoices:
        agents.append("link")
    if state.has_site_photos:
        agents.append("vision")
    return agents


def list_usage_statement_item_ids(usage_statement_id: int) -> list[int]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id
                FROM usage_statement_items
                WHERE usage_statement_id = %(usage_statement_id)s
                ORDER BY id
                """,
                {"usage_statement_id": usage_statement_id},
            )
            return [int(row[0]) for row in cur.fetchall()]


def list_usage_statement_items_for_classi(usage_statement_id: int) -> list[dict[str, Any]]:
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, category_code, item_name, unit, quantity, unit_price,
                       total_amount, remark, page_no, used_on
                FROM usage_statement_items
                WHERE usage_statement_id = %(usage_statement_id)s
                ORDER BY id
                """,
                {"usage_statement_id": usage_statement_id},
            )
            return [dict(row) for row in cur.fetchall()]


def update_usage_statement_item_categories(
    *,
    usage_statement_id: int,
    changes: list[dict[str, Any]],
) -> int:
    if not changes:
        return 0

    with get_connection() as conn:
        with conn.cursor() as cur:
            updated_count = 0
            for change in changes:
                cur.execute(
                    """
                    UPDATE usage_statement_items
                    SET category_code = %(category_code)s
                    WHERE id = %(item_id)s
                      AND usage_statement_id = %(usage_statement_id)s
                    """,
                    {
                        "item_id": change["item_id"],
                        "usage_statement_id": usage_statement_id,
                        "category_code": change["category_code"],
                    },
                )
                updated_count += int(cur.rowcount or 0)
            return updated_count


def list_evidence_file_ids_by_type(project_id: int) -> dict[str, list[int]]:
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, uploaded_evidence_type_code
                FROM files
                WHERE project_id = %(project_id)s
                  AND deleted_at IS NULL
                  AND uploaded_evidence_type_code IS NOT NULL
                ORDER BY id
                """,
                {"project_id": project_id},
            )
            grouped: dict[str, list[int]] = {}
            for row in cur.fetchall():
                grouped.setdefault(row["uploaded_evidence_type_code"], []).append(int(row["id"]))
            return grouped


def list_latest_agent_logs(
    *,
    project_id: int,
    usage_statement_id: int | None = None,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"project_id": project_id}
    usage_statement_filter = ""
    distinct_columns = "usage_statement_id, agent_type_code"
    order_columns = "usage_statement_id, agent_type_code, updated_at DESC, id DESC"

    if usage_statement_id is not None:
        params["usage_statement_id"] = usage_statement_id
        usage_statement_filter = "AND usage_statement_id = %(usage_statement_id)s"
        distinct_columns = "agent_type_code"
        order_columns = "agent_type_code, updated_at DESC, id DESC"

    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT DISTINCT ON ({distinct_columns})
                    id,
                    project_id,
                    usage_statement_id,
                    agent_type_code,
                    status_code,
                    result_code,
                    reason,
                    details,
                    model_name,
                    COALESCE(token, 0) AS token,
                    updated_at
                FROM agent_logs
                WHERE project_id = %(project_id)s
                  {usage_statement_filter}
                ORDER BY {order_columns}
                """,
                params,
            )
            return [dict(row) for row in cur.fetchall()]


def upsert_agent_log(
    *,
    project_id: int,
    usage_statement_id: int,
    agent_type_code: str,
    status_code: str,
    result_code: str | None = None,
    reason: str | None = None,
    details: dict[str, Any] | None = None,
    model_name: str | None = None,
    token: int | None = None,
) -> int:
    payload = json.dumps(details or {}, ensure_ascii=False)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id
                FROM agent_logs
                WHERE project_id = %(project_id)s
                  AND usage_statement_id = %(usage_statement_id)s
                  AND agent_type_code = %(agent_type_code)s
                ORDER BY id DESC
                LIMIT 1
                """,
                {
                    "project_id": project_id,
                    "usage_statement_id": usage_statement_id,
                    "agent_type_code": agent_type_code,
                },
            )
            row = cur.fetchone()
            if row:
                log_id = int(row[0])
                cur.execute(
                    """
                    UPDATE agent_logs
                    SET status_code = %(status_code)s,
                        result_code = %(result_code)s,
                        reason = %(reason)s,
                        details = %(details)s::jsonb,
                        model_name = COALESCE(%(model_name)s, model_name),
                        token = COALESCE(%(token)s, token)
                    WHERE id = %(log_id)s
                    """,
                    {
                        "log_id": log_id,
                        "status_code": status_code,
                        "result_code": result_code,
                        "reason": reason,
                        "details": payload,
                        "model_name": model_name,
                        "token": token,
                    },
                )
                return log_id

            cur.execute(
                """
                INSERT INTO agent_logs
                    (project_id, usage_statement_id, agent_type_code,
                     status_code, result_code, reason, details, model_name, token)
                VALUES
                    (%(project_id)s, %(usage_statement_id)s, %(agent_type_code)s,
                     %(status_code)s, %(result_code)s, %(reason)s, %(details)s::jsonb,
                     %(model_name)s, %(token)s)
                RETURNING id
                """,
                {
                    "project_id": project_id,
                    "usage_statement_id": usage_statement_id,
                    "agent_type_code": agent_type_code,
                    "status_code": status_code,
                    "result_code": result_code,
                    "reason": reason,
                    "details": payload,
                    "model_name": model_name,
                    "token": token,
                },
            )
            inserted = cur.fetchone()
            if inserted is None:
                raise RuntimeError("agent_logs INSERT failed: RETURNING id returned no row")
            return int(inserted[0])


def mark_orchestrator(
    *,
    project_id: int,
    usage_statement_id: int,
    event: str,
    status_code: str,
    result_code: str | None,
    reason: str,
    payload: dict[str, Any] | None = None,
) -> int:
    return upsert_agent_log(
        project_id=project_id,
        usage_statement_id=usage_statement_id,
        agent_type_code="orchestrator",
        status_code=status_code,
        result_code=result_code,
        reason=reason,
        details={
            "event": event,
            "summary": reason,
            "payload": payload or {},
        },
        model_name="fastapi_orchestrator",
    )


def close_supplement_todos(
    *,
    project_id: int,
    usage_statement_id: int,
    agent_type_code: str,
) -> int:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE action_requests
                SET status_code = 'closed',
                    closed_at = now()
                WHERE project_id = %(project_id)s
                  AND usage_statement_id = %(usage_statement_id)s
                  AND status_code IN ('open', 'in_progress')
                  AND title LIKE %(title_prefix)s
                """,
                {
                    "project_id": project_id,
                    "usage_statement_id": usage_statement_id,
                    "title_prefix": f"[{agent_type_code}]%",
                },
            )
            return int(cur.rowcount or 0)


def create_supplement_todos(
    *,
    project_id: int,
    usage_statement_id: int,
    requested_by_user_id: int | None,
    agent_type_code: str,
    todos: list[dict[str, Any]],
) -> int:
    if not requested_by_user_id or not todos:
        return 0

    with get_connection() as conn:
        with conn.cursor() as cur:
            created_count = 0
            for todo in todos:
                cur.execute(
                    """
                    INSERT INTO action_requests
                        (project_id, usage_statement_id, usage_statement_item_id,
                         requested_by_user_id, title, reason, status_code)
                    VALUES
                        (%(project_id)s, %(usage_statement_id)s, %(usage_statement_item_id)s,
                         %(requested_by_user_id)s, %(title)s, %(reason)s, 'open')
                    """,
                    {
                        "project_id": project_id,
                        "usage_statement_id": usage_statement_id,
                        "usage_statement_item_id": todo.get("usage_statement_item_id"),
                        "requested_by_user_id": requested_by_user_id,
                        "title": f"[{agent_type_code}] 보완 요청",
                        "reason": todo.get("reason"),
                    },
                )
                created_count += 1
            return created_count
