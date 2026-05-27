"""Diff calculation for refreshable legal sources.

Refresh scope is intentionally limited to deterministic external sources:
law.go.kr Open API rows (``law_api:%``) and usage standard rows
(``usage_standard:%``). PDF and operational profiles are not part of refresh.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Literal

import psycopg
from dotenv import load_dotenv
from langchain_core.documents import Document
from psycopg.rows import dict_row

from src.core.storage import make_chunk_id
from src.services.ingestion import law_api_scraper
from src.services.ingestion import usage_standard_scraper

load_dotenv()

DEFAULT_DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://safety_user:safety_password@localhost:5432/safety",
)

RefreshSource = Literal["law_api", "usage_standard"]


@dataclass(frozen=True)
class CurrentDoc:
    id: str
    hash: str
    chunk_id: str
    source_name: str
    article_no: str | None = None
    paragraph_no: str | None = None
    item_no: str | None = None


@dataclass(frozen=True)
class IncomingDoc:
    row: dict[str, Any]
    document: Document

    @property
    def id(self) -> str:
        return str(self.row["id"])

    @property
    def hash(self) -> str:
        return str(self.row["hash"])

    @property
    def chunk_id(self) -> str:
        return str(self.row["chunk_id"])


@dataclass(frozen=True)
class SourceDiff:
    source: RefreshSource
    added: list[IncomingDoc]
    updated: list[IncomingDoc]
    deleted: list[CurrentDoc]
    unchanged_count: int

    @property
    def changed_count(self) -> int:
        return len(self.added) + len(self.updated) + len(self.deleted)


def load_current_snapshot(
    source: RefreshSource,
    *,
    database_url: str = DEFAULT_DATABASE_URL,
) -> dict[str, CurrentDoc]:
    """Load current DB rows for one refreshable source."""
    if source == "law_api":
        where_sql = "source_type = 'law' AND id LIKE %s"
        params: tuple[Any, ...] = ("law_api:%",)
    elif source == "usage_standard":
        where_sql = "source_type = 'law' AND id LIKE %s"
        params = ("usage_standard:%",)
    else:
        raise ValueError(f"Unknown refresh source: {source}")

    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    id, hash, chunk_id, source_name,
                    article_no, paragraph_no, item_no
                FROM legal_rag.legal_master
                WHERE {where_sql}
                """,
                params,
            )
            rows = cur.fetchall()

    return {
        row["id"]: CurrentDoc(
            id=row["id"],
            hash=row["hash"],
            chunk_id=row["chunk_id"] or make_chunk_id(row["id"]),
            source_name=row["source_name"],
            article_no=row["article_no"],
            paragraph_no=row["paragraph_no"],
            item_no=row["item_no"],
        )
        for row in rows
    }


def collect_law_api_incoming(
    *,
    target_laws: dict[str, list[str]] | None = None,
) -> list[IncomingDoc]:
    """Collect Open API articles and convert them to legal_master rows/docs."""
    articles = law_api_scraper._dedupe_law_articles(
        law_api_scraper.fetch_law_articles(target_laws=target_laws)
    )
    if not articles:
        raise RuntimeError("법제처 Open API 수집 결과가 0건입니다. RDB/Qdrant refresh를 중단합니다.")

    docs = law_api_scraper.articles_to_documents(articles)
    docs_by_id = {doc.metadata["master_id"]: doc for doc in docs}

    incoming: list[IncomingDoc] = []
    for article in articles:
        row = law_api_scraper.law_article_to_row(article)
        doc = docs_by_id.get(row["id"])
        if doc is None:
            continue
        incoming.append(IncomingDoc(row=row, document=doc))

    return _dedupe_incoming(incoming)


def collect_usage_standard_incoming(
    *,
    force_refresh: bool = True,
) -> list[IncomingDoc]:
    """Collect usage standard HTML and convert it to legal_master rows/docs."""
    raw_text = usage_standard_scraper.fetch_usage_standard(force_refresh=force_refresh)
    rows = (
        usage_standard_scraper._build_corpus_rows(raw_text)
        + usage_standard_scraper._build_rules_rows(raw_text)
    )
    if not rows:
        raise RuntimeError("산안비 사용기준 파싱 결과가 0건입니다. RDB/Qdrant refresh를 중단합니다.")

    incoming: list[IncomingDoc] = []
    for row in rows:
        breadcrumb = row.get("section_path") or row["source_name"]
        incoming.append(
            IncomingDoc(
                row=row,
                document=Document(
                    page_content=f"{breadcrumb}\n\n{row['body']}",
                    metadata={
                        "source": row["source_name"],
                        "source_type": "usage_standard",
                        "header_1": usage_standard_scraper._LAW_PREFIX,
                        "article_no": row.get("article_no"),
                        "record_type": row["record_type"],
                        "master_id": row["id"],
                        "chunk_id": row["chunk_id"],
                    },
                ),
            )
        )

    return _dedupe_incoming(incoming)


def calculate_diff(
    source: RefreshSource,
    incoming: list[IncomingDoc],
    *,
    database_url: str = DEFAULT_DATABASE_URL,
) -> SourceDiff:
    """Compare incoming rows with the current DB snapshot."""
    current = load_current_snapshot(source, database_url=database_url)
    incoming_by_id = {doc.id: doc for doc in incoming}

    current_ids = set(current)
    incoming_ids = set(incoming_by_id)

    added = sorted(
        (incoming_by_id[row_id] for row_id in incoming_ids - current_ids),
        key=lambda doc: doc.id,
    )
    updated = sorted(
        (
            incoming_by_id[row_id]
            for row_id in incoming_ids & current_ids
            if incoming_by_id[row_id].hash != current[row_id].hash
        ),
        key=lambda doc: doc.id,
    )
    deleted = sorted(
        (current[row_id] for row_id in current_ids - incoming_ids),
        key=lambda doc: doc.id,
    )
    unchanged_count = len(current_ids & incoming_ids) - len(updated)

    return SourceDiff(
        source=source,
        added=added,
        updated=updated,
        deleted=deleted,
        unchanged_count=unchanged_count,
    )


def collect_and_diff_law_api(
    *,
    database_url: str = DEFAULT_DATABASE_URL,
    target_laws: dict[str, list[str]] | None = None,
) -> SourceDiff:
    incoming = collect_law_api_incoming(target_laws=target_laws)
    return calculate_diff("law_api", incoming, database_url=database_url)


def collect_and_diff_usage_standard(
    *,
    database_url: str = DEFAULT_DATABASE_URL,
    force_refresh: bool = True,
) -> SourceDiff:
    incoming = collect_usage_standard_incoming(force_refresh=force_refresh)
    return calculate_diff("usage_standard", incoming, database_url=database_url)


def _dedupe_incoming(incoming: list[IncomingDoc]) -> list[IncomingDoc]:
    deduped: dict[str, IncomingDoc] = {}
    for doc in incoming:
        deduped[doc.id] = doc
    return list(deduped.values())
