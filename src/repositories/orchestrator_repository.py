"""
AI Review Orchestrator DB 레포지토리
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Orchestrator가 실행 조건을 판단하는 데 필요한 DB 상태를 조회하고,
Agent별 최신 실행 상태를 `agent_logs`에 기록한다.

주요 책임:
  - 사용내역서 세부항목, 증빙 파일, Agent 로그 상태 스캔
  - 조건부 실행 대상 Agent 선택 보조
  - Orchestrator/Agent 로그 upsert
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

import psycopg2.extras

from src.core.json_utils import json_dumps
from src.repositories.db import get_connection


RECEIPT_TYPES = {"receipt", "tax_invoice"}
SITE_PHOTO_TYPES = {
    "site_photo",
    "item_photo",
    "wearing_photo",
    "work_photo",
    "tech_guidance_photo",
}
DEFAULT_AGENT_USAGE_USD_PER_1K_TOKENS = Decimal(os.getenv("AGENT_USAGE_DEFAULT_USD_PER_1K_TOKENS", "0.005"))
DEFAULT_AGENT_USAGE_INPUT_USD_PER_1M_TOKENS = Decimal(os.getenv("AGENT_USAGE_DEFAULT_INPUT_USD_PER_1M_TOKENS", "0.40"))
DEFAULT_AGENT_USAGE_CACHED_INPUT_USD_PER_1M_TOKENS = Decimal(os.getenv("AGENT_USAGE_DEFAULT_CACHED_INPUT_USD_PER_1M_TOKENS", "0.10"))
DEFAULT_AGENT_USAGE_OUTPUT_USD_PER_1M_TOKENS = Decimal(os.getenv("AGENT_USAGE_DEFAULT_OUTPUT_USD_PER_1M_TOKENS", "1.60"))

MODEL_USAGE_PRICES_USD_PER_1M: dict[str, tuple[Decimal, Decimal, Decimal]] = {
    "gpt-4.1": (Decimal("2.00"), Decimal("0.50"), Decimal("8.00")),
    "gpt-4.1-mini": (Decimal("0.40"), Decimal("0.10"), Decimal("1.60")),
    "gpt-4o": (Decimal("2.50"), Decimal("1.25"), Decimal("10.00")),
    "gpt-4o-mini": (Decimal("0.15"), Decimal("0.075"), Decimal("0.60")),
    "gpt-5.2": (Decimal("1.75"), Decimal("0.175"), Decimal("14.00")),
    "gpt-5-mini": (Decimal("0.25"), Decimal("0.025"), Decimal("2.00")),
    "text-embedding-3-small": (Decimal("0.02"), Decimal("0.02"), Decimal("0")),
    "gemini-2.5-flash": (Decimal("0.30"), Decimal("0.03"), Decimal("2.50")),
    "gemini-2.5-flash-lite": (Decimal("0.10"), Decimal("0.01"), Decimal("0.40")),
    "claude-sonnet-4.6": (Decimal("3.00"), Decimal("0.30"), Decimal("15.00")),
    "claude-sonnet-4-6": (Decimal("3.00"), Decimal("0.30"), Decimal("15.00")),
}


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
        return "safety-doc" in self.logs

    @property
    def report_ready(self) -> bool:
        log = self.logs.get("legal") or {}
        return log.get("status_code") == "success" and log.get("result_code") in {"success", "hil"}


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
                  AND status_code IN ('draft', 'fail')
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


def list_evidence_file_ids_by_type(project_id: int) -> dict[str, list[int]]:
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, uploaded_evidence_type_code
                FROM files
                WHERE project_id = %(project_id)s
                  AND deleted_at IS NULL
                  AND status_code IN ('draft', 'fail')
                  AND uploaded_evidence_type_code IS NOT NULL
                ORDER BY id
                """,
                {"project_id": project_id},
            )
            grouped: dict[str, list[int]] = {}
            for row in cur.fetchall():
                grouped.setdefault(row["uploaded_evidence_type_code"], []).append(int(row["id"]))
            return grouped


def list_evidence_files_by_type(
    project_id: int,
    evidence_type_codes: list[str] | tuple[str, ...] | set[str],
) -> list[dict[str, Any]]:
    if not evidence_type_codes:
        return []

    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    id,
                    project_id,
                    storage_key,
                    original_filename,
                    uploaded_evidence_type_code,
                    mime_type,
                    size_bytes
                FROM files
                WHERE project_id = %(project_id)s
                  AND deleted_at IS NULL
                  AND status_code IN ('draft', 'fail')
                  AND uploaded_evidence_type_code = ANY(%(evidence_type_codes)s)
                ORDER BY id
                """,
                {
                    "project_id": project_id,
                    "evidence_type_codes": list(evidence_type_codes),
                },
            )
            return [dict(row) for row in cur.fetchall()]


def update_file_statuses(
    *,
    project_id: int,
    file_ids: list[int],
    status_code: str,
) -> None:
    if not file_ids:
        return

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE files
                SET status_code = %(status_code)s
                WHERE project_id = %(project_id)s
                  AND id = ANY(%(file_ids)s)
                  AND deleted_at IS NULL
                """,
                {
                    "project_id": project_id,
                    "file_ids": file_ids,
                    "status_code": status_code,
                },
            )


def update_file_statuses_by_id(
    *,
    project_id: int,
    statuses_by_file_id: dict[int, str],
) -> None:
    if not statuses_by_file_id:
        return

    with get_connection() as conn:
        with conn.cursor() as cur:
            for file_id, status_code in statuses_by_file_id.items():
                cur.execute(
                    """
                    UPDATE files
                    SET status_code = %(status_code)s
                    WHERE project_id = %(project_id)s
                      AND id = %(file_id)s
                      AND deleted_at IS NULL
                    """,
                    {
                        "project_id": project_id,
                        "file_id": file_id,
                        "status_code": status_code,
                    },
                )


def update_file_details(
    *,
    project_id: int,
    details_by_file_id: dict[int, dict[str, Any]],
) -> None:
    if not details_by_file_id:
        return

    with get_connection() as conn:
        with conn.cursor() as cur:
            for file_id, detail in details_by_file_id.items():
                cur.execute(
                    """
                    UPDATE files
                    SET detail = COALESCE(detail, '{}'::jsonb) || %(detail)s::jsonb
                    WHERE project_id = %(project_id)s
                      AND id = %(file_id)s
                      AND deleted_at IS NULL
                    """,
                    {
                        "project_id": project_id,
                        "file_id": file_id,
                        "detail": json_dumps(detail, ensure_ascii=False),
                    },
                )


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
    payload = json_dumps(details or {}, ensure_ascii=False)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id
                FROM agent_logs
                WHERE project_id = %(project_id)s
                  AND usage_statement_id = %(usage_statement_id)s
                  AND usage_statement_item_id IS NULL
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
                        token = COALESCE(%(token)s, token),
                        token_current = COALESCE(%(token)s, token_current),
                        token_cumulative = COALESCE(token_cumulative, 0) + COALESCE(%(token)s, 0)
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
                     status_code, result_code, reason, details, model_name, token,
                     token_current, token_cumulative)
                VALUES
                    (%(project_id)s, %(usage_statement_id)s, %(agent_type_code)s,
                     %(status_code)s, %(result_code)s, %(reason)s, %(details)s::jsonb,
                     %(model_name)s, %(token)s,
                     %(token)s, COALESCE(%(token)s, 0))
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


def insert_agent_usage_record(
    *,
    project_id: int,
    usage_statement_id: int | None,
    agent_type_code: str,
    model_name: str | None = None,
    token: int | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cached_input_tokens: int | None = None,
    requested_by_user_id: int | None = None,
) -> int | None:
    """실제 Agent 실행 1회를 사용량 집계 테이블에 기록합니다.

    `created_at`은 DB 기본값(now())을 사용하므로 실제 실행 시각 기준으로 집계됩니다.
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            user_id = _resolve_usage_record_user_id(
                cur,
                project_id=project_id,
                usage_statement_id=usage_statement_id,
                requested_by_user_id=requested_by_user_id,
            )
            if user_id is None:
                return None

            stored_input_tokens = _non_negative_int(input_tokens if input_tokens is not None else token)
            stored_output_tokens = _non_negative_int(output_tokens)
            stored_cached_input_tokens = min(_non_negative_int(cached_input_tokens), stored_input_tokens)
            cost_usd = _estimate_agent_usage_cost_usd(
                model_name=model_name,
                agent_type_code=agent_type_code,
                input_tokens=stored_input_tokens,
                output_tokens=stored_output_tokens,
                cached_input_tokens=stored_cached_input_tokens,
            )
            cur.execute(
                """
                INSERT INTO agent_usage_records
                    (user_id, project_id, usage_statement_id,
                     agent_type_code, model_name, input_tokens, output_tokens, cost_usd)
                VALUES
                    (%(user_id)s, %(project_id)s, %(usage_statement_id)s,
                     %(agent_type_code)s, %(model_name)s, %(input_tokens)s, %(output_tokens)s, %(cost_usd)s)
                RETURNING id
                """,
                {
                    "user_id": user_id,
                    "project_id": project_id,
                    "usage_statement_id": usage_statement_id,
                    "agent_type_code": agent_type_code,
                    "model_name": model_name,
                    "input_tokens": stored_input_tokens,
                    "output_tokens": stored_output_tokens,
                    "cost_usd": cost_usd,
                },
            )
            inserted = cur.fetchone()
            return int(inserted[0]) if inserted else None


def _resolve_usage_record_user_id(
    cur,
    *,
    project_id: int,
    usage_statement_id: int | None,
    requested_by_user_id: int | None,
) -> int | None:
    if requested_by_user_id is not None:
        cur.execute(
            """
            SELECT id
            FROM users
            WHERE id = %(user_id)s
            """,
            {"user_id": requested_by_user_id},
        )
        row = cur.fetchone()
        if row:
            return int(row[0])

    if usage_statement_id is None:
        return None

    cur.execute(
        """
        SELECT f.uploaded_by_user_id
        FROM usage_statements us
        JOIN files f ON f.id = us.source_file_id
        WHERE us.id = %(usage_statement_id)s
          AND us.project_id = %(project_id)s
        """,
        {"project_id": project_id, "usage_statement_id": usage_statement_id},
    )
    row = cur.fetchone()
    if row and row[0] is not None:
        return int(row[0])

    cur.execute(
        """
        SELECT user_id
        FROM project_user_assignments
        WHERE project_id = %(project_id)s
        ORDER BY created_at DESC, id DESC
        LIMIT 1
        """,
        {"project_id": project_id},
    )
    row = cur.fetchone()
    if row and row[0] is not None:
        return int(row[0])

    return None


def _estimate_agent_usage_cost_usd(
    *,
    model_name: str | None,
    agent_type_code: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int,
) -> Decimal:
    if input_tokens <= 0 and output_tokens <= 0:
        return Decimal("0")
    input_price, cached_input_price, output_price = _agent_usage_prices_per_1m_tokens(
        model_name=model_name,
        agent_type_code=agent_type_code,
    )
    billable_input_tokens = max(input_tokens - cached_input_tokens, 0)
    cost = (
        Decimal(billable_input_tokens) / Decimal(1_000_000) * input_price
        + Decimal(cached_input_tokens) / Decimal(1_000_000) * cached_input_price
        + Decimal(output_tokens) / Decimal(1_000_000) * output_price
    )
    return cost.quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP)


def _agent_usage_prices_per_1m_tokens(*, model_name: str | None, agent_type_code: str) -> tuple[Decimal, Decimal, Decimal]:
    normalized_model_name = (model_name or "").strip().lower()
    model_price = _known_model_usage_price(normalized_model_name)
    if model_price is not None:
        return model_price

    legacy_price = _legacy_agent_usage_price_per_1k_tokens(model_name=model_name, agent_type_code=agent_type_code)
    input_price = _decimal_env(
        _price_env_keys("INPUT", model_name=model_name, agent_type_code=agent_type_code),
        DEFAULT_AGENT_USAGE_INPUT_USD_PER_1M_TOKENS,
    )
    cached_input_price = _decimal_env(
        _price_env_keys("CACHED_INPUT", model_name=model_name, agent_type_code=agent_type_code),
        DEFAULT_AGENT_USAGE_CACHED_INPUT_USD_PER_1M_TOKENS,
    )
    output_price = _decimal_env(
        _price_env_keys("OUTPUT", model_name=model_name, agent_type_code=agent_type_code),
        DEFAULT_AGENT_USAGE_OUTPUT_USD_PER_1M_TOKENS,
    )
    if legacy_price != DEFAULT_AGENT_USAGE_USD_PER_1K_TOKENS:
        legacy_per_1m = legacy_price * Decimal(1000)
        return legacy_per_1m, legacy_per_1m, legacy_per_1m
    return input_price, cached_input_price, output_price


def _known_model_usage_price(model_name: str) -> tuple[Decimal, Decimal, Decimal] | None:
    if model_name in MODEL_USAGE_PRICES_USD_PER_1M:
        return MODEL_USAGE_PRICES_USD_PER_1M[model_name]
    for known_model in sorted(MODEL_USAGE_PRICES_USD_PER_1M, key=len, reverse=True):
        if model_name.startswith(f"{known_model}-"):
            return MODEL_USAGE_PRICES_USD_PER_1M[known_model]
    return None


def _legacy_agent_usage_price_per_1k_tokens(*, model_name: str | None, agent_type_code: str) -> Decimal:
    env_keys: list[str] = []
    if model_name:
        env_keys.append(f"AGENT_USAGE_USD_PER_1K_TOKENS_{_env_key_suffix(model_name)}")
    env_keys.append(f"AGENT_USAGE_USD_PER_1K_TOKENS_{_env_key_suffix(agent_type_code)}")
    return _decimal_env(env_keys, DEFAULT_AGENT_USAGE_USD_PER_1K_TOKENS)


def _price_env_keys(kind: str, *, model_name: str | None, agent_type_code: str) -> list[str]:
    env_keys: list[str] = []
    if model_name:
        env_keys.append(f"AGENT_USAGE_{kind}_USD_PER_1M_TOKENS_{_env_key_suffix(model_name)}")
    env_keys.append(f"AGENT_USAGE_{kind}_USD_PER_1M_TOKENS_{_env_key_suffix(agent_type_code)}")
    return env_keys


def _decimal_env(env_keys: list[str], default: Decimal) -> Decimal:
    for key in env_keys:
        raw = os.getenv(key)
        if raw:
            try:
                return Decimal(raw)
            except Exception:
                continue
    return default


def _non_negative_int(value: int | None) -> int:
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


def _env_key_suffix(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value.upper()).strip("_")


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
