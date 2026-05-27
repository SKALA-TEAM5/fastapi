from langchain_core.documents import Document

from src.services.ingestion import law_api_scraper
from src.services.refresh import diff_engine
from src.services.refresh.diff_engine import CurrentDoc, IncomingDoc


def _incoming(row_id: str, body_hash: str) -> IncomingDoc:
    return IncomingDoc(
        row={
            "id": row_id,
            "hash": body_hash,
            "chunk_id": f"chunk-{row_id}",
        },
        document=Document(page_content=row_id),
    )


def test_calculate_diff_splits_added_updated_deleted(monkeypatch):
    monkeypatch.setattr(
        diff_engine,
        "load_current_snapshot",
        lambda source, database_url: {
            "same": CurrentDoc(id="same", hash="h1", chunk_id="chunk-same", source_name="src"),
            "changed": CurrentDoc(id="changed", hash="old", chunk_id="chunk-changed", source_name="src"),
            "removed": CurrentDoc(id="removed", hash="gone", chunk_id="chunk-removed", source_name="src"),
        },
    )

    result = diff_engine.calculate_diff(
        "law_api",
        [_incoming("same", "h1"), _incoming("changed", "new"), _incoming("added", "fresh")],
        database_url="postgresql://unused",
    )

    assert [doc.id for doc in result.added] == ["added"]
    assert [doc.id for doc in result.updated] == ["changed"]
    assert [doc.id for doc in result.deleted] == ["removed"]
    assert result.unchanged_count == 1
    assert result.changed_count == 3


def test_calculate_diff_no_changes(monkeypatch):
    monkeypatch.setattr(
        diff_engine,
        "load_current_snapshot",
        lambda source, database_url: {
            "same": CurrentDoc(id="same", hash="h1", chunk_id="chunk-same", source_name="src"),
        },
    )

    result = diff_engine.calculate_diff(
        "usage_standard",
        [_incoming("same", "h1")],
        database_url="postgresql://unused",
    )

    assert result.added == []
    assert result.updated == []
    assert result.deleted == []
    assert result.unchanged_count == 1
    assert result.changed_count == 0


def test_law_api_row_id_is_stable_across_mst_versions():
    base = {
        "source_kind": "law",
        "law_name": "산업안전보건법 시행규칙",
        "article_no": "89",
        "article_title": "기술지도계약서 등",
        "content": "본문",
        "content_type": "article",
        "version_id": "old-mst",
        "mst": "old-mst",
        "admrul_seq": None,
        "stable_source_key": "산업안전보건법_시행규칙",
        "source_path": "https://example.test/old",
        "metadata": {"mst": "old-mst"},
    }
    changed_version = {**base, "version_id": "new-mst", "mst": "new-mst", "metadata": {"mst": "new-mst"}}

    old_row = law_api_scraper.law_article_to_row(base)
    new_row = law_api_scraper.law_article_to_row(changed_version)

    assert old_row["id"] == "law_api:산업안전보건법_시행규칙:article:89"
    assert new_row["id"] == old_row["id"]
    assert new_row["chunk_id"] == old_row["chunk_id"]


def test_law_api_appendix_row_id_uses_stable_appendix_key():
    article = {
        "source_kind": "law",
        "law_name": "산업안전보건법 시행규칙",
        "article_no": "별표 4",
        "article_title": "안전보건교육 교육과정별 교육시간",
        "content": "별표 본문",
        "content_type": "guideline",
        "version_id": "271485",
        "mst": "271485",
        "admrul_seq": None,
        "stable_source_key": "산업안전보건법_시행규칙",
        "source_path": "https://example.test",
        "metadata": {},
    }

    row = law_api_scraper.law_article_to_row(article)

    assert row["id"] == "law_api:산업안전보건법_시행규칙:appendix:별표4"
    assert row["content_type"] == "guideline"
    assert row["section_path"] == "산업안전보건법 시행규칙 > 별표 4 (안전보건교육 교육과정별 교육시간)"
