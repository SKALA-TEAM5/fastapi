"""Pytest smoke tests for the live classi review helpers."""

from types import SimpleNamespace

from langchain_core.documents import Document

from src.agents.classifier_agent.agent import (
    _ClassificationSignals,
    _classify_document_with_signals,
    _coerce_usage_statement_input,
    _confidence_from_candidates,
    _item_status,
)
from src.schemas.classifier import DocumentClassification


def test_coerce_usage_statement_input_accepts_alias_rows():
    """Normalize alias-based rows into the review request DTO."""
    request = _coerce_usage_statement_input(
        usage_statement_id=10,
        rows=[
            {
                "행ID": 1,
                "기존카테고리코드": "CAT_03",
                "항목명": "안전모",
                "금액": 10000,
            }
        ],
        basic_info={"공사명": "테스트 공사"},
    )

    assert request.usage_statement_id == 10
    assert request.basic_info == {"공사명": "테스트 공사"}
    assert request.rows[0].row_id == 1
    assert request.rows[0].given_category_code == "CAT_03"
    assert request.rows[0].item_name == "안전모"
    assert request.rows[0].total_amount == 10000


def test_item_status_keeps_matching_category():
    """Keep the existing category when prediction and submitted code agree."""
    predicted = DocumentClassification(
        category_id="CAT_03",
        category_name="보호구 등",
        confidence=0.9,
        total_amount=10000,
        items={"안전모": 10000},
    )
    candidates = [SimpleNamespace(category_code="CAT_03", score=5.0)]

    status, final_code, reason, path = _item_status(
        predicted=predicted,
        given_code="CAT_03",
        candidates=candidates,
        item_name="안전모",
        basic_info={},
    )

    assert status == "유지"
    assert final_code == "CAT_03"
    assert reason == ""
    assert path == "rule(rdb_agree_keep)"


def test_confidence_from_candidates_is_bounded():
    """Return a stable bounded confidence for RDB candidate scores."""
    candidates = [
        SimpleNamespace(category_code="CAT_03", score=6.0),
        SimpleNamespace(category_code="CAT_02", score=2.0),
    ]

    assert _confidence_from_candidates(candidates) == 0.72


def test_classify_document_with_signals_uses_rdb_clear_path():
    """Prefer a clear RDB candidate before considering Qdrant signals."""
    candidates = [
        SimpleNamespace(category_code="CAT_03", score=8.0),
        SimpleNamespace(category_code="CAT_02", score=1.0),
    ]

    outcome = _classify_document_with_signals(
        items={"안전모": 10000},
        basic_info={},
        collection="unused",
        signals=_signals(candidates=candidates),
    )

    assert outcome.classification.category_id == "CAT_03"
    assert outcome.signal_path == "rdb-clear (score=8.0, margin=7.0)"


def test_classify_document_with_signals_uses_rdb_vote_agree_path():
    """Boost confidence when RDB and citation vote agree."""
    candidates = [SimpleNamespace(category_code="CAT_03", score=3.0)]

    outcome = _classify_document_with_signals(
        items={"안전모": 10000},
        basic_info={},
        collection="unused",
        signals=_signals(candidates=candidates, vote_scores={"CAT_03": 3.0}),
    )

    assert outcome.classification.category_id == "CAT_03"
    assert outcome.classification.confidence == 0.66
    assert outcome.signal_path == "rdb+vote-agree (rdb=3.0, vote_ratio=1.00)"


def test_classify_document_with_signals_uses_vote_only_path():
    """Use a clear citation vote when RDB is absent."""
    outcome = _classify_document_with_signals(
        items={"안전망": 10000},
        basic_info={},
        collection="unused",
        signals=_signals(vote_scores={"CAT_02": 3.0, "CAT_03": 1.0}),
    )

    assert outcome.classification.category_id == "CAT_02"
    assert outcome.signal_path == "citation-vote (top=CAT_02, ratio=0.75)"


def test_classify_document_with_signals_uses_rdb_only_path():
    """Fall back to RDB when citation vote is not clear."""
    candidates = [SimpleNamespace(category_code="CAT_05", score=3.0)]

    outcome = _classify_document_with_signals(
        items={"정기 안전 교육": 10000},
        basic_info={},
        collection="unused",
        signals=_signals(candidates=candidates, vote_scores={}),
    )

    assert outcome.classification.category_id == "CAT_05"
    assert outcome.classification.needs_human_review is True
    assert outcome.signal_path == "rdb-only (score=3.0)"


def test_classify_document_with_signals_uses_header_hint_path():
    """Use chunk header hints as the final classified fallback."""
    outcome = _classify_document_with_signals(
        items={"방지망": 10000},
        basic_info={},
        collection="unused",
        signals=_signals(
            docs=[
                Document(
                    page_content="관련 내용",
                    metadata={"header_2": "안전시설비", "breadcrumb": ""},
                )
            ]
        ),
    )

    assert outcome.classification.category_id == "CAT_02"
    assert outcome.classification.needs_human_review is True
    assert outcome.signal_path == "header-hint"


def test_classify_document_with_signals_returns_unclassified():
    """Return UNCLASSIFIED when every signal is empty."""
    outcome = _classify_document_with_signals(
        items={"불명확 항목": 10000},
        basic_info={},
        collection="unused",
        signals=_signals(),
    )

    assert outcome.classification.category_id == "미분류"
    assert outcome.classification.needs_human_review is True
    assert outcome.signal_path == "unclassified"


def _signals(
    *,
    docs: list[Document] | None = None,
    candidates: list | None = None,
    vote_scores: dict[str, float] | None = None,
) -> _ClassificationSignals:
    """Build precomputed signals for branch-level characterization tests."""
    return _ClassificationSignals(
        docs=docs or [],
        candidates=candidates or [],
        vote_scores=vote_scores or {},
        item_names=[],
        total_amount=0.0,
    )
