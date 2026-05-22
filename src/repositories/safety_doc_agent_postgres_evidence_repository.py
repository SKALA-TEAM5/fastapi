from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Iterator

import psycopg
from psycopg.rows import dict_row
from psycopg.errors import InsufficientPrivilege, UndefinedTable

from src.agents.safety_doc_agent.config import Settings, app_root_dir
from src.repositories.safety_doc_agent_evidence_repository import EvidenceRepository
from src.repositories.safety_doc_agent_postgres_queries import (
    DEACTIVATE_ACTIVE_REQUIREMENTS,
    GET_EVIDENCE_REQUIREMENT_ITEM_CONTEXT,
    GET_EVIDENCE_REQUIREMENT_ITEM_CONTEXT_FALLBACK,
    INSERT_AGENT_LOG,
    INSERT_REQUIREMENT,
    LIST_ACTIVE_REQUIREMENTS,
    LIST_EVIDENCE_LINKS,
    LIST_EVIDENCE_TYPES,
    LIST_ITEM_CONTEXT_TARGETS_FALLBACK,
    LIST_LINKED_FILE_CONTEXTS,
    LIST_LINKED_FILE_CONTEXTS_FALLBACK,
    MARK_SATISFIED_REQUIREMENTS,
)
from src.schemas.safety_doc_agent_evidence import (
    EvidenceRequirementItemContext,
    EvidenceFileLink,
    EvidenceRequirement,
    EvidenceType,
    LinkedEvidenceFileContext,
)


class PostgresEvidenceRepository(EvidenceRepository):
    """`service` 스키마에 직접 붙어 evidence requirement 흐름을 실험하는 저장소."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def get_item_context(self, item_id: int) -> EvidenceRequirementItemContext:
        row = self._fetch_one_with_fallback(
            primary_query=GET_EVIDENCE_REQUIREMENT_ITEM_CONTEXT,
            fallback_query=GET_EVIDENCE_REQUIREMENT_ITEM_CONTEXT_FALLBACK,
            params={"item_id": item_id},
        )
        if row is None:
            raise KeyError(
                "item_context not found. "
                "Run the draft view SQL first, and check that the item_id exists."
            )
        return EvidenceRequirementItemContext(**row)

    def list_evidence_types(self) -> list[EvidenceType]:
        rows = self._fetch_all(LIST_EVIDENCE_TYPES)
        return [EvidenceType(**row) for row in rows]

    def list_linked_file_contexts(self, item_id: int) -> list[LinkedEvidenceFileContext]:
        rows = self._fetch_all_with_fallback(
            primary_query=LIST_LINKED_FILE_CONTEXTS,
            fallback_query=LIST_LINKED_FILE_CONTEXTS_FALLBACK,
            params={"item_id": item_id},
        )
        return [LinkedEvidenceFileContext(**row) for row in rows]

    def replace_active_requirements(
        self,
        item_id: int,
        evidence_type_codes: list[str],
    ) -> list[EvidenceRequirement]:
        with self._connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(DEACTIVATE_ACTIVE_REQUIREMENTS, {"item_id": item_id})
            inserted: list[EvidenceRequirement] = []
            for code in evidence_type_codes:
                cur.execute(INSERT_REQUIREMENT, {"item_id": item_id, "evidence_type_code": code})
                row = cur.fetchone()
                if row is not None:
                    inserted.append(EvidenceRequirement(**row))
            conn.commit()
            return inserted

    def list_active_requirements(self, item_id: int) -> list[EvidenceRequirement]:
        rows = self._fetch_all(LIST_ACTIVE_REQUIREMENTS, {"item_id": item_id})
        return [EvidenceRequirement(**row) for row in rows]

    def list_evidence_links(self, item_id: int) -> list[EvidenceFileLink]:
        rows = self._fetch_all(LIST_EVIDENCE_LINKS, {"item_id": item_id})
        return [EvidenceFileLink(**row) for row in rows]

    def update_requirement_satisfaction(
        self,
        item_id: int,
        satisfied_codes: list[str],
    ) -> None:
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute(
                MARK_SATISFIED_REQUIREMENTS,
                {"item_id": item_id, "submitted_codes": satisfied_codes},
            )
            conn.commit()

    def append_agent_log(
        self,
        *,
        project_id: int,
        usage_statement_id: int | None,
        usage_statement_item_id: int | None,
        status_code: str,
        result_code: str,
        reason: str,
        details: dict,
        model_name: str | None,
        token: int | None,
    ) -> None:
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute(
                INSERT_AGENT_LOG,
                {
                    "project_id": project_id,
                    "usage_statement_id": usage_statement_id,
                    "usage_statement_item_id": usage_statement_item_id,
                    "status_code": status_code,
                    "result_code": result_code,
                    "reason": reason,
                    "details": json.dumps(details, ensure_ascii=False),
                    "model_name": model_name,
                    "token": token,
                },
            )
            conn.commit()

    def apply_draft_views(self) -> None:
        """`db/migrations/V5__views.sql`의 safety-doc-agent 조회 뷰를 현재 DB에 반영한다."""

        sql_path = app_root_dir().parent / "db" / "migrations" / "V5__views.sql"
        sql = sql_path.read_text(encoding="utf-8")
        with self._connection() as conn, conn.cursor() as cur:
            try:
                cur.execute(sql)
                conn.commit()
            except InsufficientPrivilege as exc:
                raise PermissionError(
                    "Current DB user cannot create views in schema 'service'. "
                    "Use a higher-privileged account or apply the SQL through db migrations."
                ) from exc

    def list_targets(self, limit: int = 20) -> list[dict]:
        """실험에 쓸 item_id를 빠르게 찾기 위한 간단한 대상 목록."""

        return self._fetch_all(LIST_ITEM_CONTEXT_TARGETS_FALLBACK, {"limit": limit})

    def _fetch_one(self, query: str, params: dict | None = None) -> dict | None:
        with self._connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(query, params or {})
            return cur.fetchone()

    def _fetch_all(self, query: str, params: dict | None = None) -> list[dict]:
        with self._connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(query, params or {})
            return list(cur.fetchall())

    def _fetch_one_with_fallback(
        self,
        *,
        primary_query: str,
        fallback_query: str,
        params: dict | None = None,
    ) -> dict | None:
        try:
            return self._fetch_one(primary_query, params)
        except (UndefinedTable, InsufficientPrivilege):
            return self._fetch_one(fallback_query, params)

    def _fetch_all_with_fallback(
        self,
        *,
        primary_query: str,
        fallback_query: str,
        params: dict | None = None,
    ) -> list[dict]:
        try:
            return self._fetch_all(primary_query, params)
        except (UndefinedTable, InsufficientPrivilege):
            return self._fetch_all(fallback_query, params)

    @contextmanager
    def _connection(self) -> Iterator[psycopg.Connection]:
        conn = psycopg.connect(
            host=self.settings.db_host,
            port=self.settings.db_port,
            dbname=self.settings.db_name,
            user=self.settings.db_user,
            password=self.settings.db_password,
        )
        try:
            yield conn
        finally:
            conn.close()
