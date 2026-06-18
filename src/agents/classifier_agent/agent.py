# --------------------------------------------------------------------------
# 작성자   : 송상민(ss19801)
# 작성일   : 2026-05-04
# 수정일   : 2026-06-18
#
# [ 주요 클래스 및 함수 정의 ]
#
# 1. review_usage_statement() : 사용내역서 행 단위 카테고리 검토
# 2. _classify_document_with_signals() : RDB/Qdrant 신호 기반 카테고리 후보 산출
# 3. _review_single_usage_statement_row() : 단일 사용내역서 행 최종 분류 판정
# 4. _write_classi_agent_log() : classi 행 단위 agent_logs 기록
# --------------------------------------------------------------------------
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

try:
    from langchain_community.callbacks import get_openai_callback
except ImportError:  # pragma: no cover
    get_openai_callback = None  # type: ignore

from langchain_core.documents import Document
try:
    from langsmith import traceable
except ImportError:  # pragma: no cover
    def traceable(*args, **kwargs):  # type: ignore
        def decorator(func):
            return func
        return decorator

from pydantic import BaseModel, Field

from src.core.rag import (
    MAX_RETRY,
    build_retriever,
    retrieve,
    rewrite_query,
)
from src.core.storage import DEFAULT_COLLECTION, load_vectorstore
import src.core.llm_config as llm_config
from src.repositories import LegalRulesRepository
from src.prompts import CLASSIFIER_CATEGORY_PROMPT
from src.schemas.classifier import (
    CATEGORIES,
    UNCLASSIFIED,
    DocumentClassification,
    RowReviewResult,
    UsageStatementReviewRequest,
    UsageStatementReviewResponse,
    UsageStatementRow,
)

log = logging.getLogger(__name__)

_rules_repo = LegalRulesRepository()
_HUMAN_REVIEW_MIN_CONFIDENCE = 0.7

# ── RDB 신뢰 임계값 (Validator 구조와 동일한 철학) ─────────────────────────
# RDB 단독 확정: top score가 이 값 이상 + margin이 _RDB_CLEAR_MARGIN 이상이면
# Qdrant를 조회하지 않고 바로 카테고리 확정한다.
_RDB_CLEAR_SCORE = 5.0
_RDB_CLEAR_MARGIN = 2.0
# RDB가 약한 신호라도 후보로 인정하는 최소 점수
_RDB_MIN_SCORE = 2.0
# predicted == given일 때 LLM 검증을 생략할 최소 점수
# ※ predicted == given 케이스에서는 LLM 재검증 자체를 제거하여 상수 미사용
_LLM_VERIFY_SKIP_SCORE = 8.0  # noqa: F841 — 하위 호환 보존, 실제 분기 제거됨

# 인용 투표 비율이 이 값 이상이어야 신호로 인정
_VOTE_RATIO_MIN = 0.6
_RECLASSIFY_MARGIN_MIN = 4.0
_RECLASSIFY_CONFIDENCE_MIN = 0.72

# Citation-based voting constants
_CITE_TAG_RE = re.compile(r"\[LEGAL_CITE:\s*([^\]]+)\]")
_ART7_ITEM_RE = re.compile(r"제7조제1항제(\d+)호")
_ITEM_NO_TO_CAT: dict[str, str] = {str(i): f"CAT_0{i}" for i in range(1, 10)}

# 청크 헤더에서 카테고리를 추론할 키워드 힌트 (마지막 폴백용)
_HEADER_CAT_HINTS: dict[str, str] = {
    "보호구": "CAT_03",
    "안전시설비": "CAT_02",
    "안전난간": "CAT_02",
    "인건비": "CAT_01",
    "업무수당": "CAT_01",
    "교육비": "CAT_05",
    "진단비": "CAT_04",
    "건강장해": "CAT_06",
    "기술지도": "CAT_07",
    "위험성평가": "CAT_09",
}


@dataclass
class _ClassificationSignals:
    docs: list[Document]
    candidates: list
    vote_scores: dict[str, float]
    item_names: list[str]
    total_amount: float


@dataclass
class _ClassificationOutcome:
    classification: DocumentClassification
    signals: _ClassificationSignals
    signal_path: str = ""  # 분류 경로: rdb-clear / rdb+vote-agree / citation-vote / rdb-only / header-hint / unclassified


@dataclass
class _ClassificationContext:
    items: dict[str, float]
    basic_info: dict[str, Any]
    collection: str
    item_names: list[str]
    representative: str
    total_amount: float
    query: str


@dataclass
class _RdbStats:
    top: float
    second: float
    margin: float


class _ClassifierLLMOutput(BaseModel):
    """LLM 카테고리 분류 결과."""
    is_safety_item: bool = Field(
        default=True,
        description="산안비(산업안전보건관리비) 지출 가능 항목이면 true, 사람 이름·관련 없는 항목이면 false",
    )
    category_code: str = Field(description="CAT_01~CAT_09 중 하나 (is_safety_item=false여도 반드시 입력)")
    reasoning: str = Field(default="", description="분류 근거 한 문장")


_RULE_CONTEXT_CACHE: str | None = None

def _build_rdb_rule_context() -> str:
    """Build cached RDB rule context for classifier LLM prompts.

    Returns:
        Category examples grouped as a prompt context string.
    """
    global _RULE_CONTEXT_CACHE
    if _RULE_CONTEXT_CACHE is not None:
        return _RULE_CONTEXT_CACHE

    _INFORMATIVE_TYPES = {"allowed", "category", "qa_allowed", "rule_like_allowed"}
    lines: list[str] = []
    for cat_code, cat_name in CATEGORIES.items():
        examples: list[str] = []
        for rule in _rules_repo.rules:
            if rule.get("category_code") != cat_code:
                continue
            if rule.get("rule_type") not in _INFORMATIVE_TYPES:
                continue
            kw = (rule.get("item_key") or rule.get("keyword") or "").strip()
            if kw and kw not in examples:
                examples.append(kw)
            if len(examples) >= 5:
                break
        if examples:
            lines.append(f"[{cat_code} {cat_name}]\n  예시: {' / '.join(examples)}")

    _RULE_CONTEXT_CACHE = "\n".join(lines)
    return _RULE_CONTEXT_CACHE


def _llm_classify_item(
    *,
    item_name: str,
    given_code: str,
    basic_info: dict,
    candidates: list,
) -> str | None:
    """Classify one item with the LLM using RDB rule context.

    Args:
        item_name: Usage-statement item name to classify.
        given_code: Category code submitted by OCR or the caller.
        basic_info: Usage-statement header fields used as prompt context.
        candidates: RDB category candidates used as prompt hints.

    Returns:
        A valid category code, ``UNCLASSIFIED``, or ``None`` when LLM
        classification is unavailable or invalid.
    """
    try:
        llm = llm_config.get()
    except RuntimeError:
        return None

    # RDB 법령 규칙을 컨텍스트로 사용 (Qdrant 청크 대신)
    rdb_context = _build_rdb_rule_context()

    # RDB 후보 점수 (참고용)
    candidate_lines = "\n".join(
        f"- {c.category_code} {c.category_name}: {c.score:.1f}점"
        for c in (candidates or [])[:5]
    ) or "(없음)"

    given_name = CATEGORIES.get(given_code, given_code)
    basic_str = ", ".join(f"{k}: {v}" for k, v in (basic_info or {}).items()) or "없음"

    try:
        result: _ClassifierLLMOutput = (
            CLASSIFIER_CATEGORY_PROMPT
            | llm.with_structured_output(_ClassifierLLMOutput)
        ).invoke(
            {
                "item_name": item_name,
                "given_code": given_code,
                "given_name": given_name,
                "basic_info": basic_str,
                "context": rdb_context,
                "candidates": candidate_lines,
            }
        )
        if not result.is_safety_item:
            return UNCLASSIFIED
        code = (result.category_code or "").strip().upper()
        return code if code in CATEGORIES else None
    except Exception:
        log.debug("LLM classifier failed for item=%s", item_name, exc_info=True)
        return None


def _get_generic_item_policy(item_name: str, category_id: str | None = None) -> dict | None:
    """Find a generic item policy matching the item and category.

    Args:
        item_name: Item name normalized against policy aliases.
        category_id: Optional category code used to filter conditional policies.

    Returns:
        Matching policy dictionary, or ``None`` when no policy applies.
    """
    normalized = re.sub(r"\s+", " ", (item_name or "").strip().lower())
    for policy in _rules_repo.generic_item_policies.values():
        categories = policy.get("conditional_categories") or []
        if categories and category_id and category_id not in categories:
            continue
        aliases = [str(alias).lower() for alias in policy.get("aliases") or []]
        if normalized in aliases:
            return policy
    return None


def _apply_generic_review_policy(
    *,
    item_names: list[str],
    category_id: str,
    needs_human_review: bool,
    review_reason: str,
) -> tuple[bool, str]:
    """Apply generic item review policy to the current review decision.

    Args:
        item_names: Item names included in the classification request.
        category_id: Candidate category code.
        needs_human_review: Current human-review flag.
        review_reason: Current review reason.

    Returns:
        Updated ``(needs_human_review, review_reason)`` tuple.
    """
    if len(item_names) != 1:
        return needs_human_review, review_reason
    policy = _get_generic_item_policy(item_names[0], category_id=category_id)
    if not policy:
        return needs_human_review, review_reason
    if not bool(policy.get("classifier_review_required")):
        return needs_human_review, review_reason
    warning = str(policy.get("warning_template") or "").strip()
    if not warning:
        return True, review_reason or "예외형 generic 품목으로 분류되어 추가 확인이 필요합니다."
    return True, f"{review_reason} / {warning}" if review_reason else warning


def _retrieve_docs(question: str, collection: str) -> list[Document]:
    """Retrieve legal context documents from Qdrant with retry rewriting.

    Args:
        question: Retrieval query.
        collection: Qdrant collection name.

    Returns:
        Retrieved LangChain documents, or an empty list.
    """
    vectorstore = load_vectorstore(collection_name=collection)
    retriever = build_retriever(vectorstore, collection_name=collection)
    state = {
        "question": question,
        "retrieved_docs": [],
        "judgment": None,
        "retry_count": 0,
    }
    state = retrieve(state, retriever)
    while not state["retrieved_docs"] and state.get("retry_count", 0) < MAX_RETRY:
        state = rewrite_query(state)
        state = retrieve(state, retriever)
    return state["retrieved_docs"]


def _base_query(items: dict[str, float], basic_info: dict[str, Any]) -> str:
    """Build the base retrieval and RDB matching query.

    Args:
        items: Item names mapped to amounts.
        basic_info: Usage-statement header fields.

    Returns:
        Compact query text combining representative items and header values.
    """
    representative = ", ".join(list(items.keys())[:3])
    basic_info_query = " ".join(f"{k} {v}" for k, v in basic_info.items())
    return f"{representative} {basic_info_query}".strip()


def _qdrant_query(base_query: str, candidates: list) -> str:
    """Build an enriched Qdrant query from RDB category candidates.

    Args:
        base_query: Base query built from items and usage-statement metadata.
        candidates: RDB category candidates used as hints.

    Returns:
        Query string including candidate categories, legal hints, and keywords.
    """
    hint_parts: list[str] = []
    law_hints: list[str] = []
    keyword_hints: list[str] = []
    if candidates:
        category_hints = _rules_repo.find_category_hints(
            category_codes=[c.category_code for c in candidates],
            limit_per_category=3,
        )
        for c in candidates:
            hint_parts.extend([c.category_code, c.category_name])
            hints = category_hints.get(c.category_code, {})
            law_hints.extend(hints.get("cited_laws", []))
            keyword_hints.extend(hints.get("keywords", []))

    parts = [
        f"산업안전보건관리비 항목 '{base_query}'의 분류 기준",
        f"우선 검토 카테고리 {', '.join(dict.fromkeys(p.strip() for p in hint_parts if p.strip()))}" if hint_parts else "",
        f"관련 조항 {', '.join(dict.fromkeys(p.strip() for p in law_hints if p.strip()))}" if law_hints else "",
        f"관련 키워드 {', '.join(dict.fromkeys(p.strip() for p in keyword_hints if p.strip()))}" if keyword_hints else "",
    ]
    return " | ".join(p for p in parts if p).strip()


def _coerce_usage_statement_input(
    *,
    payload: dict[str, Any] | None = None,
    usage_statement_id: int | str | None = None,
    rows: list[dict[str, Any]] | list[UsageStatementRow] | None = None,
    basic_info: dict[str, Any] | None = None,
) -> UsageStatementReviewRequest:
    """Normalize supported classifier inputs into a review request DTO.

    Args:
        payload: Full request payload using schema aliases or field names.
        usage_statement_id: Usage-statement identifier when no payload is given.
        rows: Usage-statement rows to review.
        basic_info: Header metadata used as classifier context.

    Returns:
        Validated ``UsageStatementReviewRequest``.
    """
    if payload is not None:
        return UsageStatementReviewRequest.model_validate(payload)
    data = {
        "사용내역서ID": usage_statement_id,
        "항목목록": rows or [],
        "기본정보": basic_info or {},
    }
    return UsageStatementReviewRequest.model_validate(data)


def _vote_category_from_chunks(docs: list[Document]) -> dict[str, float]:
    """Vote category candidates from LEGAL_CITE tags in retrieved chunks.

    law_article 타입 청크(법제처 Open API 조문)는 citation vote에서 제외하고
    LLM 컨텍스트 보강에만 활용한다. 해당 청크에는 법령 자체의 LEGAL_CITE 태그가
    있어 vote 신호를 오염시킬 수 있다.

    Args:
        docs: Retrieved legal-context documents.

    Returns:
        Category code to weighted citation-vote score mapping.
    """
    scores: dict[str, float] = {}
    for rank, doc in enumerate(docs):
        # 법령 조문 청크는 citation vote 대상에서 제외
        if doc.metadata.get("source_type") == "law_article":
            continue
        weight = 1.0 / (rank + 1)
        found: set[str] = set()
        for raw in _CITE_TAG_RE.findall(doc.page_content):
            for part in raw.split("|"):
                m = _ART7_ITEM_RE.search(part)
                if m:
                    cat = _ITEM_NO_TO_CAT.get(m.group(1))
                    if cat:
                        found.add(cat)
        for cat in found:
            scores[cat] = scores.get(cat, 0.0) + weight
    return scores


def _hint_from_chunk_headers(docs: list[Document]) -> str | None:
    """Infer a final fallback category from retrieved chunk headers.

    Args:
        docs: Retrieved legal-context documents.

    Returns:
        Category code inferred from header keywords, or ``None``.
    """
    for doc in docs:
        meta = doc.metadata
        header_text = " ".join(filter(None, [
            meta.get("header_2", ""),
            meta.get("header_3", ""),
            meta.get("breadcrumb", ""),
        ]))
        for keyword, cat in _HEADER_CAT_HINTS.items():
            if keyword in header_text:
                return cat
    return None


def _confidence_from_votes(scores: dict[str, float]) -> float:
    """Convert citation-vote scores into a bounded confidence value.

    Args:
        scores: Category code to weighted vote score mapping.

    Returns:
        Confidence between 0.0 and 0.95.
    """
    if not scores:
        return 0.0
    total = sum(scores.values())
    top = max(scores.values())
    ratio = top / total
    return round(max(0.0, min(0.35 + 0.60 * ratio, 0.95)), 2)


def _review_from_votes(
    scores: dict[str, float],
    top_cat: str,
    confidence: float,
) -> tuple[bool, str]:
    """Decide whether citation-vote output requires human review.

    Args:
        scores: Category code to weighted vote score mapping.
        top_cat: Category code with the highest vote score.
        confidence: Confidence calculated from vote scores.

    Returns:
        ``(needs_human_review, review_reason)`` tuple.
    """
    if not scores:
        return True, "청크에서 법령 조항 인용이 발견되지 않았습니다."
    total = sum(scores.values())
    ratio = scores[top_cat] / total
    if ratio < 0.5:
        return True, f"상위 카테고리 투표 비율이 낮습니다. ratio={ratio:.2f}"
    if confidence < _HUMAN_REVIEW_MIN_CONFIDENCE:
        return True, f"인용 투표 신뢰도가 낮습니다. confidence={confidence:.2f}"
    return False, ""


def _confidence_from_candidates(candidates: list) -> float:
    """Convert RDB candidate score separation into confidence.

    Args:
        candidates: Ordered RDB category candidates.

    Returns:
        Confidence between 0.0 and 0.95.
    """
    if not candidates:
        return 0.0
    top = candidates[0].score
    second = candidates[1].score if len(candidates) > 1 else 0.0
    ratio = top / (top + second + 1.0)
    score_factor = min(top / 6.0, 1.0)
    return round(max(0.0, min(0.35 + 0.55 * ratio * score_factor, 0.95)), 2)


def _rdb_stats(candidates: list) -> _RdbStats:
    """Return top, second, and margin scores from ordered RDB candidates."""
    top = candidates[0].score if candidates else 0.0
    second = candidates[1].score if len(candidates) > 1 else 0.0
    return _RdbStats(top=top, second=second, margin=top - second)


def _classification_context(
    *,
    items: dict[str, float],
    basic_info: dict[str, Any],
    collection: str,
) -> _ClassificationContext:
    """Build shared values used across the classi decision stages."""
    item_names = list(items.keys())
    return _ClassificationContext(
        items=items,
        basic_info=basic_info,
        collection=collection,
        item_names=item_names,
        representative=", ".join(item_names[:3]),
        total_amount=sum(items.values()),
        query=_base_query(items, basic_info),
    )


def _signals_from_rdb_and_qdrant(
    *,
    context: _ClassificationContext,
    initial_candidates: list,
    signals: _ClassificationSignals | None,
) -> tuple[_ClassificationSignals, _RdbStats]:
    """Collect Qdrant vote signals and refresh RDB candidates when needed."""
    candidates = initial_candidates
    stats = _rdb_stats(candidates)

    if signals is not None:
        docs = signals.docs
        vote_scores = signals.vote_scores
    else:
        docs = _retrieve_docs(
            question=_qdrant_query(context.query, candidates),
            collection=context.collection,
        )
        vote_scores = _vote_category_from_chunks(docs)
        if docs:
            retrieved_context = "\n\n---\n\n".join(
                f"[출처: {d.metadata.get('source', '알 수 없음')}]\n{d.page_content}"
                for d in docs
            )
            candidates = _rules_repo.find_category_candidates(
                query_text=context.query,
                retrieved_context=retrieved_context,
                limit=5,
            )
            stats = _rdb_stats(candidates)

    return (
        _ClassificationSignals(
            docs=docs,
            candidates=candidates,
            vote_scores=vote_scores,
            item_names=context.item_names,
            total_amount=context.total_amount,
        ),
        stats,
    )


def _resolve_rdb_clear(
    *,
    context: _ClassificationContext,
    candidates: list,
    signals: _ClassificationSignals | None,
    stats: _RdbStats,
) -> _ClassificationOutcome | None:
    """Resolve the high-confidence RDB-only path before Qdrant lookup."""
    if not candidates or stats.top < _RDB_CLEAR_SCORE or stats.margin < _RDB_CLEAR_MARGIN:
        return None

    category_id = candidates[0].category_code
    confidence = _confidence_from_candidates(candidates)
    needs_review = confidence < _HUMAN_REVIEW_MIN_CONFIDENCE
    review_reason = f"결정 신뢰도 확인 필요. confidence={confidence:.2f}" if needs_review else ""
    needs_review, review_reason = _apply_generic_review_policy(
        item_names=context.item_names,
        category_id=category_id,
        needs_human_review=needs_review,
        review_reason=review_reason,
    )
    log.debug(
        "rdb-clear: item=%s cat=%s score=%.1f margin=%.1f",
        context.representative,
        category_id,
        stats.top,
        stats.margin,
    )
    final_signals = signals or _ClassificationSignals(
        docs=[],
        candidates=candidates,
        vote_scores={},
        item_names=context.item_names,
        total_amount=context.total_amount,
    )
    return _ClassificationOutcome(
        classification=DocumentClassification(
            category_id=category_id,
            category_name=CATEGORIES.get(category_id, UNCLASSIFIED),
            confidence=confidence,
            total_amount=context.total_amount,
            items=context.items,
            needs_human_review=needs_review,
            review_reason=review_reason,
        ),
        signals=final_signals,
        signal_path=f"rdb-clear (score={stats.top:.1f}, margin={stats.margin:.1f})",
    )


def _resolve_vote_and_rdb(
    *,
    context: _ClassificationContext,
    final_signals: _ClassificationSignals,
    stats: _RdbStats,
) -> _ClassificationOutcome | None:
    """Resolve citation-vote and weak-RDB fallback paths."""
    candidates = final_signals.candidates
    vote_scores = final_signals.vote_scores

    top_vote_cat: str | None = None
    vote_ratio = 0.0
    if vote_scores:
        top_vote_cat = max(vote_scores, key=lambda k: vote_scores[k])
        vote_total = sum(vote_scores.values())
        vote_ratio = vote_scores[top_vote_cat] / vote_total

    vote_is_clear = bool(top_vote_cat and vote_ratio >= _VOTE_RATIO_MIN)
    rdb_is_present = bool(candidates and stats.top >= _RDB_MIN_SCORE)

    if vote_is_clear and rdb_is_present and candidates[0].category_code == top_vote_cat:
        category_id = top_vote_cat
        confidence = min(_confidence_from_candidates(candidates) + 0.1, 0.95)
        needs_review, review_reason = _apply_generic_review_policy(
            item_names=context.item_names,
            category_id=category_id,
            needs_human_review=False,
            review_reason="",
        )
        log.debug("rdb+vote-agree: item=%s cat=%s", context.representative, category_id)
        return _ClassificationOutcome(
            classification=DocumentClassification(
                category_id=category_id,
                category_name=CATEGORIES.get(category_id, UNCLASSIFIED),
                confidence=confidence,
                total_amount=context.total_amount,
                items=context.items,
                needs_human_review=needs_review,
                review_reason=review_reason,
            ),
            signals=final_signals,
            signal_path=f"rdb+vote-agree (rdb={stats.top:.1f}, vote_ratio={vote_ratio:.2f})",
        )

    if vote_is_clear:
        confidence = _confidence_from_votes(vote_scores)
        needs_review, review_reason = _review_from_votes(vote_scores, top_vote_cat, confidence)
        needs_review, review_reason = _apply_generic_review_policy(
            item_names=context.item_names,
            category_id=top_vote_cat,
            needs_human_review=needs_review,
            review_reason=review_reason,
        )
        log.debug("citation-vote: item=%s top=%s ratio=%.2f", context.representative, top_vote_cat, vote_ratio)
        return _ClassificationOutcome(
            classification=DocumentClassification(
                category_id=top_vote_cat,
                category_name=CATEGORIES.get(top_vote_cat, UNCLASSIFIED),
                confidence=confidence,
                total_amount=context.total_amount,
                items=context.items,
                needs_human_review=needs_review,
                review_reason=review_reason,
            ),
            signals=final_signals,
            signal_path=f"citation-vote (top={top_vote_cat}, ratio={vote_ratio:.2f})",
        )

    if rdb_is_present:
        category_id = candidates[0].category_code
        confidence = _confidence_from_candidates(candidates)
        needs_review = confidence < _HUMAN_REVIEW_MIN_CONFIDENCE or stats.top < _RDB_CLEAR_SCORE
        review_reason = f"RDB 규칙 점수가 낮습니다. score={stats.top:.1f}" if needs_review else ""
        needs_review, review_reason = _apply_generic_review_policy(
            item_names=context.item_names,
            category_id=category_id,
            needs_human_review=needs_review,
            review_reason=review_reason,
        )
        log.debug("rdb-only: item=%s cat=%s score=%.1f", context.representative, category_id, stats.top)
        return _ClassificationOutcome(
            classification=DocumentClassification(
                category_id=category_id,
                category_name=CATEGORIES.get(category_id, UNCLASSIFIED),
                confidence=confidence,
                total_amount=context.total_amount,
                items=context.items,
                needs_human_review=needs_review,
                review_reason=review_reason,
            ),
            signals=final_signals,
            signal_path=f"rdb-only (score={stats.top:.1f})",
        )

    return None


def _resolve_header_or_unclassified(
    *,
    context: _ClassificationContext,
    final_signals: _ClassificationSignals,
) -> _ClassificationOutcome:
    """Resolve the last fallback from chunk headers or UNCLASSIFIED."""
    hint_cat = _hint_from_chunk_headers(final_signals.docs)
    if hint_cat:
        log.debug("header-hint: item=%s cat=%s", context.representative, hint_cat)
        return _ClassificationOutcome(
            classification=DocumentClassification(
                category_id=hint_cat,
                category_name=CATEGORIES.get(hint_cat, UNCLASSIFIED),
                confidence=0.55,
                total_amount=context.total_amount,
                items=context.items,
                needs_human_review=True,
                review_reason="청크 헤더 키워드 기반 분류. 확인 권장.",
            ),
            signals=final_signals,
            signal_path="header-hint",
        )

    return _ClassificationOutcome(
        classification=DocumentClassification(
            category_id=UNCLASSIFIED,
            category_name=UNCLASSIFIED,
            confidence=0.0,
            total_amount=context.total_amount,
            items=context.items,
            needs_human_review=True,
            review_reason="RDB 규칙과 법령 청크 인용 모두 카테고리 후보를 찾지 못했습니다.",
        ),
        signals=final_signals,
        signal_path="unclassified",
    )


def _classify_document_with_signals(
    *,
    items: dict[str, float],
    basic_info: dict[str, Any],
    collection: str,
    signals: _ClassificationSignals | None = None,
) -> _ClassificationOutcome:
    """Classify items with staged RDB and Qdrant signals.

    1순위: RDB 단독 확정 — score >= _RDB_CLEAR_SCORE & margin >= _RDB_CLEAR_MARGIN
           → Qdrant 조회 없이 즉시 확정 (빠르고 신뢰도 높음)
    2순위: RDB + Qdrant citation vote 조합
           → 두 신호가 같은 카테고리 → confidence 상승
           → citation vote 단독으로 명확 → vote 결과 사용
    3순위: RDB 약한 신호 단독
           → Qdrant에서 citation vote 없을 때 RDB 후보 사용
    4순위: 청크 헤더 힌트 → UNCLASSIFIED

    Args:
        items: Item names mapped to amounts.
        basic_info: Usage-statement header fields.
        collection: Qdrant collection name.
        signals: Optional precomputed signals used by compatibility paths.

    Returns:
        Classification outcome containing the chosen classification, collected
        signals, and a human-readable decision path.
    """
    context = _classification_context(
        items=items,
        basic_info=basic_info,
        collection=collection,
    )

    # ── 1. RDB 먼저 (Qdrant 없이) ──────────────────────────────────────────
    if signals is None:
        candidates = _rules_repo.find_category_candidates(
            query_text=context.query,
            retrieved_context="",
            limit=5,
        )
    else:
        candidates = signals.candidates

    stats = _rdb_stats(candidates)

    # ── 2. RDB 단독 확정 조건 충족 시 Qdrant 스킵 ─────────────────────────
    outcome = _resolve_rdb_clear(
        context=context,
        candidates=candidates,
        signals=signals,
        stats=stats,
    )
    if outcome is not None:
        return outcome

    # ── 3. RDB 불확실 → Qdrant 조회 ───────────────────────────────────────
    final_signals, stats = _signals_from_rdb_and_qdrant(
        context=context,
        initial_candidates=candidates,
        signals=signals,
    )

    # ── 4~7. citation vote 및 RDB 약한 신호 확인 ───────────────────────────
    outcome = _resolve_vote_and_rdb(
        context=context,
        final_signals=final_signals,
        stats=stats,
    )
    if outcome is not None:
        return outcome

    # ── 8. 청크 헤더 힌트 (최후 폴백) ────────────────────────────────────
    return _resolve_header_or_unclassified(
        context=context,
        final_signals=final_signals,
    )


def _item_status(
    predicted: DocumentClassification,
    given_code: str | None,
    candidates: list | None = None,
    *,
    item_name: str = "",
    basic_info: dict | None = None,
) -> tuple[str, str, str, str]:
    """Decide the final item status from classifier output and original code.

    판정상태는 반드시 '유지' 또는 '카테고리변경' 중 하나. '검토필요' 없음.
    애매한 경우 LLM이 최종 결정한다.

    Args:
        predicted: Classifier-predicted category and confidence.
        given_code: Original category code supplied by the caller.
        candidates: RDB category candidates used by the classifier.
        item_name: Item name used when LLM fallback is needed.
        basic_info: Usage-statement header fields used for LLM fallback.

    Returns:
        ``(decision_status, final_category_code, reason, decision_path)``.
    """
    candidates = candidates or []
    score_by_category = {c.category_code: c.score for c in candidates}
    predicted_score = score_by_category.get(predicted.category_id, 0.0)
    given_score = score_by_category.get(given_code or "", 0.0)
    top_score = candidates[0].score if candidates else 0.0

    def _fallback_to_llm(reason_tag: str) -> tuple[str, str, str, str]:
        """LLM에게 최종 분류를 위임한다. LLM 불가 시 '유지'로 안전하게 처리.

        LLM이 성공적으로 분류하면 path 태그를 'llm(classified)'로 남겨
        오케스트레이터의 llm(unclassified) 차단 조건에 걸리지 않도록 한다.
        LLM도 분류에 실패한 경우에만 'llm(unclassified)' 태그를 사용한다.
        """
        llm_code = _llm_classify_item(
            item_name=item_name,
            given_code=given_code or UNCLASSIFIED,
            basic_info=basic_info or {},
            candidates=candidates,
        )
        # LLM이 유효한 카테고리를 찾은 경우 → 성공 태그 사용 (차단 안 됨)
        if llm_code and llm_code != UNCLASSIFIED:
            llm_tag = "llm(classified)"
            if llm_code != given_code:
                cat_name = CATEGORIES.get(llm_code, llm_code)
                return "카테고리변경", llm_code, f"{cat_name} 카테고리로 변경이 필요함.", llm_tag
            return "유지", given_code or llm_code, "", llm_tag
        # LLM도 분류 실패 → unclassified 태그 유지 (오케스트레이터가 차단)
        return "유지", given_code or (predicted.category_id if predicted.category_id != UNCLASSIFIED else ""), "", f"llm({reason_tag})"

    # ── 1. 분류 불가 → LLM 위임
    if predicted.category_id == UNCLASSIFIED:
        return _fallback_to_llm("unclassified")

    # ── 2. 예측 카테고리가 기존과 다를 때
    if given_code and predicted.category_id != given_code:
        # 후보 없고 확신도 낮음 → 유지
        if not candidates and predicted.confidence < 0.8:
            return "유지", given_code, "", "rule(no_candidates_low_conf)"

        # 기존 카테고리 점수가 있고 예측 점수가 없음 → 유지
        if given_score > 0 and predicted_score <= 0:
            if given_score >= top_score:
                return "유지", given_code, "", "rule(given_score_top)"
            # 기존이 top도 아니고 예측 점수도 없는 경우 → 유지 (예측 근거 없음)
            return "유지", given_code, "", "rule(given_not_top_no_predicted_score)"

        margin = predicted_score - given_score
        # 마진이 충분히 크지 않은 경우 → 유지 우선 (LLM은 마진 작을 때 신뢰 어려움)
        if given_score > 0 and margin < _RECLASSIFY_MARGIN_MIN:
            return "유지", given_code, "", "rule(low_margin_keep)"

        # 신뢰도 미달 → 유지 (확신 없을 때 기존 카테고리 보수적 유지)
        if predicted.confidence < _RECLASSIFY_CONFIDENCE_MIN:
            return "유지", given_code, "", "rule(low_confidence_keep)"

    # ── 3. 예측 == 기존 → 두 신호가 일치하므로 LLM 재검증 없이 유지
    # score가 너무 낮으면 (<= 3.0) 신호 자체가 없는 것이므로 그냥 유지
    # predicted == given이면 RDB 강도·신뢰도와 무관하게 유지
    # (분류 결과와 기존 카테고리가 일치하는 경우 LLM 재검증 불필요)
    if predicted.category_id == given_code:
        top_score = candidates[0].score if candidates else 0.0
        if top_score <= 3.0:
            return "유지", predicted.category_id, "", "rule(too_weak_signal_keep)"
        return "유지", predicted.category_id, "", "rule(rdb_agree_keep)"

    # ── 4. 사람 검토 필요 신호 → LLM 위임
    if predicted.needs_human_review:
        return _fallback_to_llm("needs_human_review")

    # ── 5. 확실한 재분류
    return "카테고리변경", predicted.category_id, f"{predicted.category_name} 카테고리로 변경이 필요함.", "rule(confident_reclass)"


def _review_single_usage_statement_row(
    *,
    row: UsageStatementRow,
    basic_info: dict[str, Any],
    collection: str,
) -> RowReviewResult:
    """Review one usage-statement row and return its final classifier result.

    Args:
        row: Usage-statement row to classify.
        basic_info: Usage-statement header fields.
        collection: Qdrant collection name.

    Returns:
        Row-level review result consumed by orchestrator and pipeline services.
    """
    outcome = _classify_document_with_signals(
        items={row.item_name: row.total_amount},
        basic_info=basic_info,
        collection=collection,
    )
    predicted = outcome.classification
    signals = outcome.signals
    decision_status, final_category_code, reason, decision_path = _item_status(
        predicted=predicted,
        given_code=row.given_category_code,
        candidates=signals.candidates,
        item_name=row.item_name,
        basic_info=basic_info,
    )
    full_path = f"{outcome.signal_path} → {decision_path}"
    full_reason = reason if decision_status != "유지" else ""
    if full_reason:
        full_reason = f"{full_reason} [{full_path}]"
    else:
        full_reason = f"[{full_path}]"
    return RowReviewResult(
        row_id=row.row_id,
        item_name=row.item_name,
        given_category_code=row.given_category_code,
        final_category_code=final_category_code,
        decision_status=decision_status,
        needs_human_review=False,
        reason=full_reason,
    )


_INSERT_CLASSI_LOG_SQL = """
    INSERT INTO agent_logs (
        project_id, usage_statement_id, usage_statement_item_id,
        agent_type_code, status_code, result_code,
        reason, details, model_name, token
    )
    VALUES (
        %(project_id)s, %(usage_statement_id)s, %(item_db_id)s,
        'classi', 'success', %(result_code)s,
        %(reason)s, %(details)s::jsonb, %(model_name)s, %(token)s
    )
    RETURNING id
"""


def _write_classi_agent_log(
    *,
    project_id: int,
    usage_statement_id: int,
    results: list[RowReviewResult],
    model_name: str,
    token: int | None,
) -> None:
    """Insert row-level classi results into ``agent_logs``.

    Args:
        project_id: Project identifier.
        usage_statement_id: Usage statement identifier.
        results: Row-level classifier results.
        model_name: Model name to store in the log row.
        token: Total token count, if available.

    Returns:
        None.
    """
    from src.repositories.db import get_connection

    try:
        with get_connection() as conn:
            # item_name → usage_statement_items.id 맵 선조회
            item_id_map: dict[str, int] = {}
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, item_name FROM usage_statement_items WHERE usage_statement_id = %s ORDER BY id",
                    (usage_statement_id,),
                )
                for row_id, item_name in cur.fetchall():
                    if item_name not in item_id_map:
                        item_id_map[item_name] = row_id

            inserted = 0
            with conn.cursor() as cur:
                for result in results:
                    item_db_id = item_id_map.get(result.item_name)
                    result_code = (
                        "hil"
                        if result.decision_status == "카테고리변경" or result.needs_human_review
                        else "success"
                    )
                    details = {
                        "item": {
                            "item_name": result.item_name,
                            "given_category_code": result.given_category_code,
                            "final_category_code": result.final_category_code,
                            "decision_status": result.decision_status,
                        }
                    }
                    cur.execute(_INSERT_CLASSI_LOG_SQL, {
                        "project_id":         project_id,
                        "usage_statement_id": usage_statement_id,
                        "item_db_id":         item_db_id,
                        "result_code":        result_code,
                        "reason":             result.reason[:1000] if result.reason else None,
                        "details":            json.dumps(details, ensure_ascii=False),
                        "model_name":         model_name,
                        "token":              token,
                    })
                    inserted += 1

        log.info("[agent_log] classi INSERT %d건 완료", inserted)
    except Exception as exc:
        log.warning("[agent_log] classi INSERT 실패 (로그 생략 후 계속): %s", exc)


@traceable(name="classifier.review_usage_statement")
def review_usage_statement(
    payload: dict[str, Any] | None = None,
    *,
    usage_statement_id: int | str | None = None,
    rows: list[dict[str, Any]] | list[UsageStatementRow] | None = None,
    basic_info: dict[str, Any] | None = None,
    collection: str = DEFAULT_COLLECTION,
    project_id: int | None = None,
    model_name: str = "gpt-4o-mini",
) -> UsageStatementReviewResponse:
    """Review usage-statement rows and return final category decisions.

    project_id, usage_statement_id(int)가 모두 제공되면 agent_logs에 INSERT한다.

    Args:
        payload: Full request payload using schema aliases or field names.
        usage_statement_id: Usage-statement identifier when no payload is given.
        rows: Usage-statement rows to review.
        basic_info: Header metadata used as classifier context.
        collection: Qdrant collection name.
        project_id: Project identifier used for optional agent log insertion.
        model_name: Model name to store in optional agent log rows.

    Returns:
        Usage-statement review response with row-level classification results.
    """
    request = _coerce_usage_statement_input(
        payload=payload,
        usage_statement_id=usage_statement_id,
        rows=rows,
        basic_info=basic_info,
    )

    total_tokens: int | None = None

    if not request.rows:
        results: list[RowReviewResult] = []
    elif get_openai_callback is not None:
        with get_openai_callback() as _cb:
            results = [
                _review_single_usage_statement_row(
                    row=row,
                    basic_info=request.basic_info,
                    collection=collection,
                )
                for row in request.rows
            ]
        total_tokens = _cb.total_tokens or None
    else:
        results = [
            _review_single_usage_statement_row(
                row=row,
                basic_info=request.basic_info,
                collection=collection,
            )
            for row in request.rows
        ]

    # agent_logs INSERT (project_id + int usage_statement_id 있을 때만)
    _uid = request.usage_statement_id
    if project_id is not None and isinstance(_uid, int):
        _write_classi_agent_log(
            project_id=project_id,
            usage_statement_id=_uid,
            results=results,
            model_name=model_name,
            token=total_tokens,
        )

    return UsageStatementReviewResponse(
        usage_statement_id=request.usage_statement_id,
        results=results,
    )
