from __future__ import annotations

"""보고서 생성용 Postgres repository.

기존 service 스키마의 프로젝트, 사용내역서, 증빙, 검증 로그 row를 읽어
ReportAgent의 입력인 ReportContext를 조립할 수 있는 dict 형태로 정규화합니다.
"""

import json
from collections import defaultdict
from datetime import date
from typing import Any

import psycopg2.extras
from psycopg2.extensions import connection as PgConnection

from src.agents.report_agent.context_builder import ReportRepository


class PostgresReportRepository(ReportRepository):
    def __init__(self, conn: PgConnection) -> None:
        self.conn = conn

    def get_project(self, project_id: int) -> dict:
        row = self._fetch_one(
            """
            SELECT id, contract_no, construction_company, project_name, site_location,
                   representative_name, contract_amount, construction_start_date,
                   construction_end_date, client_name, appropriated_amount
            FROM projects
            WHERE id = %(project_id)s
            """,
            {"project_id": project_id},
        )
        if row is None:
            raise KeyError(f"project not found: {project_id}")
        return row

    def get_usage_statement(self, usage_statement_id: int) -> dict:
        row = self._fetch_one(
            """
            SELECT id, report_month, revision_no, document_written_date,
                   cumulative_progress_rate, source_file_id
            FROM usage_statements
            WHERE id = %(usage_statement_id)s
            """,
            {"usage_statement_id": usage_statement_id},
        )
        if row is None:
            raise KeyError(f"usage_statement not found: {usage_statement_id}")
        return row

    def list_usage_categories(self) -> list[dict]:
        return self._fetch_all(
            """
            SELECT code, name
            FROM usage_categories
            ORDER BY code
            """
        )

    def list_usage_statement_summaries(self, usage_statement_id: int) -> list[dict]:
        return self._fetch_all(
            """
            SELECT category_code, previous_amount, current_amount, cumulative_amount
            FROM usage_statement_summaries
            WHERE usage_statement_id = %(usage_statement_id)s
            ORDER BY category_code
            """,
            {"usage_statement_id": usage_statement_id},
        )

    def list_usage_statement_items(self, usage_statement_id: int) -> list[dict]:
        rows = self._fetch_all(
            """
            SELECT id, category_code, used_on, item_name, unit, quantity,
                   unit_price, total_amount, remark, page_no
            FROM usage_statement_items
            WHERE usage_statement_id = %(usage_statement_id)s
            ORDER BY used_on, id
            """,
            {"usage_statement_id": usage_statement_id},
        )
        classification_by_item, legal_by_item, legal_by_category = self._list_agent_results(usage_statement_id)
        for row in rows:
            item_id = row["id"]
            row["classification_result"] = classification_by_item.get(item_id)
            row["legal_validation_result"] = legal_by_item.get(item_id) or legal_by_category.get(row["category_code"])
        return rows

    def list_evidence_files_by_item(self, usage_statement_id: int) -> dict[int, list[dict]]:
        rows = self._fetch_all(
            """
            SELECT efl.usage_statement_item_id AS item_id,
                   f.id AS file_id,
                   f.original_filename,
                   efl.evidence_type_code,
                   NULL::text AS evidence_detail_name,
                   f.mime_type,
                   f.captured_at,
                   f.uploaded_at
            FROM evidence_file_links efl
            JOIN usage_statement_items usi
              ON usi.id = efl.usage_statement_item_id
            JOIN files f
              ON f.id = efl.file_id
             AND f.deleted_at IS NULL
            WHERE usi.usage_statement_id = %(usage_statement_id)s
            ORDER BY efl.usage_statement_item_id, f.uploaded_at, f.id
            """,
            {"usage_statement_id": usage_statement_id},
        )
        return _group_by_item(rows)

    def list_evidence_requirements_by_item(self, usage_statement_id: int) -> dict[int, list[dict]]:
        rows = self._fetch_all(
            """
            SELECT er.usage_statement_item_id AS item_id,
                   er.evidence_type_code,
                   er.is_satisfied
            FROM evidence_requirements er
            JOIN usage_statement_items usi
              ON usi.id = er.usage_statement_item_id
            WHERE usi.usage_statement_id = %(usage_statement_id)s
              AND er.is_active = true
            ORDER BY er.usage_statement_item_id, er.evidence_type_code
            """,
            {"usage_statement_id": usage_statement_id},
        )
        return _group_by_item(rows)

    def list_validation_logs_by_item(self, usage_statement_id: int) -> dict[int, list[dict]]:
        rows = self._fetch_all(
            """
            SELECT id, agent_type_code, result_code, reason, details,
                   model_name, created_at
            FROM agent_logs
            WHERE usage_statement_id = %(usage_statement_id)s
              AND agent_type_code IN ('safety-doc', 'link', 'vision', 'legal')
              AND status_code = 'success'
            ORDER BY created_at, id
            """,
            {"usage_statement_id": usage_statement_id},
        )
        return _group_agent_logs_by_item(rows)

    def list_action_requests(self, project_id: int, usage_statement_id: int) -> list[dict]:
        rows = self._fetch_all(
            """
            SELECT id, agent_type_code, reason, details, created_at
            FROM agent_logs
            WHERE project_id = %(project_id)s
              AND usage_statement_id = %(usage_statement_id)s
              AND status_code = 'success'
              AND result_code = 'hil'
              AND details ? 'payload'
            ORDER BY created_at, id
            """,
            {"project_id": project_id, "usage_statement_id": usage_statement_id},
        )
        action_requests: list[dict] = []
        for row in rows:
            payload = _json_dict(row.get("details")).get("payload")
            if not isinstance(payload, dict):
                continue
            todos = payload.get("todos")
            if not isinstance(todos, list):
                continue
            for index, todo in enumerate(todos, start=1):
                if not isinstance(todo, dict):
                    continue
                reason = str(todo.get("reason") or row.get("reason") or "보완 요청")
                action_requests.append(
                    {
                        "id": int(row["id"]) * 1000 + index,
                        "usage_statement_item_id": _item_id(todo),
                        "title": str(todo.get("title") or todo.get("usage_statement_item_name") or "보완 요청"),
                        "reason": reason,
                        "status_code": str(todo.get("status_code") or "open"),
                        "due_date": None,
                        "assignee_name": todo.get("assignee_name") or todo.get("assignee"),
                        "created_at": row["created_at"],
                        "resolved_at": None,
                    }
                )
        return action_requests

    def _list_agent_results(self, usage_statement_id: int) -> tuple[dict[int, dict], dict[int, dict], dict[str, dict]]:
        rows = self._fetch_all(
            """
            SELECT agent_type_code, details
            FROM agent_logs
            WHERE usage_statement_id = %(usage_statement_id)s
              AND agent_type_code IN ('classi', 'legal')
              AND status_code = 'success'
            ORDER BY created_at, id
            """,
            {"usage_statement_id": usage_statement_id},
        )
        classification_by_item: dict[int, dict] = {}
        legal_by_item: dict[int, dict] = {}
        legal_by_category: dict[str, dict] = {}
        for row in rows:
            details = _json_dict(row.get("details"))
            if row["agent_type_code"] == "classi":
                classification_by_item.update(_extract_classification_results(details))
            elif row["agent_type_code"] == "legal":
                item_results, category_results = _extract_legal_results(details)
                legal_by_item.update(item_results)
                legal_by_category.update(category_results)
        return classification_by_item, legal_by_item, legal_by_category

    def _fetch_one(self, query: str, params: dict | None = None) -> dict | None:
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params or {})
            row = cur.fetchone()
            return _normalize_row(row) if row is not None else None

    def _fetch_all(self, query: str, params: dict | None = None) -> list[dict]:
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params or {})
            return [_normalize_row(row) for row in cur.fetchall()]


def _group_by_item(rows: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        item_id = row.pop("item_id")
        grouped[item_id].append(row)
    return dict(grouped)


def _group_agent_logs_by_item(rows: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        details = _json_dict(row.get("details"))
        for item_id, item_details in _iter_agent_log_item_details(details):
            grouped[item_id].append(
                {
                    "id": row["id"],
                    "validation_type_code": row["agent_type_code"],
                    "result_code": row.get("result_code") or "success",
                    "details": {
                        "agent_type_code": row["agent_type_code"],
                        "reason": row.get("reason"),
                        **item_details,
                    },
                    "model_name": row.get("model_name"),
                    "created_at": row["created_at"],
                }
            )
    return dict(grouped)


def _iter_agent_log_item_details(details: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    item_details: list[tuple[int, dict[str, Any]]] = []
    for row in _iter_result_rows(details):
        item_id = _item_id(row)
        if item_id is not None:
            item_details.append((item_id, row))

    raw_payload = details.get("payload")
    payload: dict[str, Any] = raw_payload if isinstance(raw_payload, dict) else {}
    todos = payload.get("todos")
    if isinstance(todos, list):
        for todo in todos:
            if not isinstance(todo, dict):
                continue
            item_id = _item_id(todo)
            if item_id is not None:
                item_details.append((item_id, todo))
    return item_details


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    details = normalized.get("details")
    if details is not None:
        normalized["details"] = _json_dict(details)
    return normalized


def _json_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _extract_classification_results(details: dict[str, Any]) -> dict[int, dict]:
    results: dict[int, dict] = {}
    for row in _iter_result_rows(details):
        item_id = _item_id(row)
        if item_id is None:
            continue
        status = row.get("status") or row.get("decision_status") or row.get("classification_status")
        results[item_id] = {
            "original_category_code": str(row.get("original_category_code") or row.get("given_category_code") or row.get("category_code") or ""),
            "final_category_code": str(row.get("final_category_code") or row.get("category_code") or ""),
            "status": _classification_status(str(status or "")),
            "needs_review": bool(row.get("needs_review") or status in ("검토필요", "needs_review")),
            "reason": str(row.get("reason") or row.get("summary") or ""),
        }
    return results


def _extract_legal_results(details: dict[str, Any]) -> tuple[dict[int, dict], dict[str, dict]]:
    item_results: dict[int, dict] = {}
    category_results: dict[str, dict] = {}
    for row in _iter_result_rows(details):
        result = _legal_result_from_row(row)
        category_code = result["category_code"]
        item_id = _item_id(row)
        if item_id is not None:
            item_results[item_id] = result
        elif category_code:
            category_results[category_code] = result
    return item_results, category_results


def _iter_result_rows(details: dict[str, Any]) -> list[dict[str, Any]]:
    raw_payload = details.get("payload")
    payload: dict[str, Any] = raw_payload if isinstance(raw_payload, dict) else {}
    rows: list[dict[str, Any]] = []
    sources: tuple[dict[str, Any], dict[str, Any]] = (details, payload)
    for source in sources:
        for key in ("results", "items", "item_results", "category_results", "result"):
            candidates = source.get(key)
            if isinstance(candidates, dict):
                candidates = candidates.get("results") or candidates.get("items") or candidates.get("item_results") or candidates.get("category_results")
            if isinstance(candidates, list):
                rows.extend(row for row in candidates if isinstance(row, dict))
    return rows


def _legal_result_from_row(row: dict[str, Any]) -> dict[str, Any]:
    citations = row.get("citations") or row.get("sources") or row.get("출처") or []
    return {
        "category_code": str(row.get("category_code") or row.get("카테고리코드") or ""),
        "status": _legal_status(str(row.get("status") or row.get("result_code") or row.get("decision") or row.get("판정상태") or "")),
        "reason": str(row.get("reason") or row.get("summary") or row.get("agent_conclusion") or row.get("사유") or ""),
        "citations": [_legal_citation(source) for source in citations if isinstance(source, dict)],
    }


def _item_id(row: dict[str, Any]) -> int | None:
    for key in ("usage_statement_item_id", "item_id", "id"):
        value = row.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _classification_status(value: str) -> str:
    if value in {"유지", "카테고리변경", "검토필요"}:
        return value
    if value in {"needs_review", "review", "warning"}:
        return "검토필요"
    if value in {"changed", "category_changed"}:
        return "카테고리변경"
    return "유지"


def _legal_status(value: str) -> str:
    if value in {"적절", "부적절", "검토필요"}:
        return value
    if value in {"ok", "appropriate", "pass", "matched", "approved"}:
        return "적절"
    if value in {"error", "fail", "inappropriate", "rejected"}:
        return "부적절"
    return "검토필요"


def _legal_citation(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "legal_basis": str(source.get("legal_basis") or source.get("article_no") or source.get("조항") or ""),
        "summary": source.get("summary") or source.get("요지"),
        "citation_text": source.get("citation_text") or source.get("인용원문"),
        "source_id": source.get("source_id"),
        "source_name": source.get("source_name"),
        "article_no": source.get("article_no"),
        "paragraph_no": source.get("paragraph_no"),
        "item_no": source.get("item_no"),
    }


def default_report_no(project_ref: int | str, usage_statement_id: int, written_date: date) -> str:
    normalized_project_ref = str(project_ref or "").strip()
    return f"AR-{written_date:%Y%m%d}-{normalized_project_ref}-{usage_statement_id}"
