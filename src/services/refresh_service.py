"""Public service entrypoint for legal refresh."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import psycopg
from dotenv import load_dotenv

from src.core.storage import DEFAULT_COLLECTION, _get_qdrant_client, _sanitize_name, upsert_with_ids
from src.services.refresh.diff_engine import (
    DEFAULT_DATABASE_URL,
    IncomingDoc,
    SourceDiff,
    collect_and_diff_law_api,
    collect_and_diff_usage_standard,
)
from src.services.refresh.law_log_writer import build_log_row, insert_law_log, new_run_id

log = logging.getLogger(__name__)

load_dotenv()


@dataclass(frozen=True)
class RefreshSourceSummary:
    source: str
    added: int
    updated: int
    deleted: int
    unchanged: int

    @property
    def changed(self) -> int:
        return self.added + self.updated + self.deleted


@dataclass(frozen=True)
class RefreshSummary:
    run_id: str
    sources: list[RefreshSourceSummary]

    @property
    def changed(self) -> int:
        return sum(source.changed for source in self.sources)


def run_refresh(
    *,
    collection_name: str = DEFAULT_COLLECTION,
    database_url: str | None = None,
    refresh_law_api: bool = True,
    refresh_usage_standard: bool = True,
    usage_force_refresh: bool = True,
    target_laws: dict[str, list[str]] | None = None,
) -> RefreshSummary:
    """Refresh Open API and usage-standard rows using hash diffs."""
    db_url = database_url or os.environ.get("DATABASE_URL") or DEFAULT_DATABASE_URL
    run_id = new_run_id()
    diffs: list[SourceDiff] = []

    if refresh_law_api:
        log.info("법제처 Open API refresh diff 계산 시작")
        diffs.append(
            collect_and_diff_law_api(
                database_url=db_url,
                target_laws=target_laws,
            )
        )

    if refresh_usage_standard:
        log.info("산안비 사용기준 refresh diff 계산 시작")
        diffs.append(
            collect_and_diff_usage_standard(
                database_url=db_url,
                force_refresh=usage_force_refresh,
            )
        )

    for diff in diffs:
        if diff.changed_count == 0:
            log.info("%s 변경 없음", diff.source)
            continue

        log.info(
            "%s 변경 감지: added=%d updated=%d deleted=%d unchanged=%d",
            diff.source,
            len(diff.added),
            len(diff.updated),
            len(diff.deleted),
            diff.unchanged_count,
        )
        _apply_qdrant_changes(collection_name, diff)
        _apply_rdb_changes(db_url, run_id, diff)

    return RefreshSummary(
        run_id=run_id,
        sources=[
            RefreshSourceSummary(
                source=diff.source,
                added=len(diff.added),
                updated=len(diff.updated),
                deleted=len(diff.deleted),
                unchanged=diff.unchanged_count,
            )
            for diff in diffs
        ],
    )


def _apply_qdrant_changes(collection_name: str, diff: SourceDiff) -> None:
    upserts = diff.added + diff.updated
    if upserts:
        upsert_with_ids(
            collection_name=collection_name,
            documents=[doc.document for doc in upserts],
            ids=[doc.chunk_id for doc in upserts],
        )

    delete_ids = [doc.chunk_id for doc in diff.deleted if doc.chunk_id]
    if delete_ids:
        from qdrant_client.models import PointIdsList

        client = _get_qdrant_client()
        client.delete(
            collection_name=_sanitize_name(collection_name),
            points_selector=PointIdsList(points=delete_ids),
        )


def _apply_rdb_changes(database_url: str, run_id: str, diff: SourceDiff) -> None:
    current_hashes = {doc.id: doc.hash for doc in diff.deleted}
    if diff.updated:
        current_hashes.update(_load_hashes(database_url, [doc.id for doc in diff.updated]))

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            for doc in diff.added:
                _upsert_legal_master(cur, doc)
                insert_law_log(
                    cur,
                    build_log_row(
                        run_id=run_id,
                        master_id=doc.id,
                        source_name=doc.row["source_name"],
                        article_no=doc.row.get("article_no"),
                        paragraph_no=doc.row.get("paragraph_no"),
                        item_no=doc.row.get("item_no"),
                        prev_hash=None,
                        new_hash=doc.hash,
                        change_type="added",
                    ),
                )

            for doc in diff.updated:
                _upsert_legal_master(cur, doc)
                insert_law_log(
                    cur,
                    build_log_row(
                        run_id=run_id,
                        master_id=doc.id,
                        source_name=doc.row["source_name"],
                        article_no=doc.row.get("article_no"),
                        paragraph_no=doc.row.get("paragraph_no"),
                        item_no=doc.row.get("item_no"),
                        prev_hash=current_hashes.get(doc.id),
                        new_hash=doc.hash,
                        change_type="updated",
                    ),
                )

            for doc in diff.deleted:
                insert_law_log(
                    cur,
                    build_log_row(
                        run_id=run_id,
                        master_id=doc.id,
                        source_name=doc.source_name,
                        article_no=doc.article_no,
                        paragraph_no=doc.paragraph_no,
                        item_no=doc.item_no,
                        prev_hash=doc.hash,
                        new_hash=None,
                        change_type="deleted",
                    ),
                )
                cur.execute(
                    "DELETE FROM legal_rag.legal_master WHERE id = %s",
                    (doc.id,),
                )


def _load_hashes(database_url: str, ids: list[str]) -> dict[str, str]:
    if not ids:
        return {}
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, hash
                FROM legal_rag.legal_master
                WHERE id = ANY(%s)
                """,
                (ids,),
            )
            return {row[0]: row[1] for row in cur.fetchall()}


def _upsert_legal_master(cur: Any, doc: IncomingDoc) -> None:
    cur.execute(
        """
        INSERT INTO legal_rag.legal_master
          (id, source_name, source_type, source_path,
           article_no, paragraph_no, item_no, section_path,
           chunk_id, body, record_type, content_type, rule_type,
           category_code, category_name, allowed, limit_pct,
           keyword, item_pattern, legal_basis,
           cited_laws, keywords, hash, metadata)
        VALUES
          (%(id)s, %(source_name)s, %(source_type)s, %(source_path)s,
           %(article_no)s, %(paragraph_no)s, %(item_no)s, %(section_path)s,
           %(chunk_id)s, %(body)s, %(record_type)s, %(content_type)s, %(rule_type)s,
           %(category_code)s, %(category_name)s, %(allowed)s, %(limit_pct)s,
           %(keyword)s, %(item_pattern)s, %(legal_basis)s,
           %(cited_laws)s, %(keywords)s, %(hash)s, %(metadata)s::jsonb)
        ON CONFLICT (id) DO UPDATE SET
          source_name   = EXCLUDED.source_name,
          source_type   = EXCLUDED.source_type,
          source_path   = EXCLUDED.source_path,
          article_no    = EXCLUDED.article_no,
          paragraph_no  = EXCLUDED.paragraph_no,
          item_no       = EXCLUDED.item_no,
          section_path  = EXCLUDED.section_path,
          chunk_id      = EXCLUDED.chunk_id,
          body          = EXCLUDED.body,
          record_type   = EXCLUDED.record_type,
          content_type  = EXCLUDED.content_type,
          rule_type     = EXCLUDED.rule_type,
          category_code = EXCLUDED.category_code,
          category_name = EXCLUDED.category_name,
          allowed       = EXCLUDED.allowed,
          limit_pct     = EXCLUDED.limit_pct,
          keyword       = EXCLUDED.keyword,
          item_pattern  = EXCLUDED.item_pattern,
          legal_basis   = EXCLUDED.legal_basis,
          cited_laws    = EXCLUDED.cited_laws,
          keywords      = EXCLUDED.keywords,
          hash          = EXCLUDED.hash,
          metadata      = EXCLUDED.metadata
        """,
        doc.row,
    )


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Refresh legal_master/Qdrant from external legal sources")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--skip-law-api", action="store_true")
    parser.add_argument("--skip-usage-standard", action="store_true")
    parser.add_argument("--use-usage-cache", action="store_true")
    args = parser.parse_args()

    summary = run_refresh(
        collection_name=args.collection,
        refresh_law_api=not args.skip_law_api,
        refresh_usage_standard=not args.skip_usage_standard,
        usage_force_refresh=not args.use_usage_cache,
    )
    print(
        f"refresh complete: run_id={summary.run_id} changed={summary.changed} "
        f"sources={summary.sources}"
    )
