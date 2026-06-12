# --------------------------------------------------------------------------
# 작성자   : 송상민(ss19801)
# 작성일   : 2026-05-04
#
# [ 주요 클래스 및 함수 정의 ]
#
# 1. classify_document() : 단일 항목을 산안비 카테고리로 분류
# 2. review_usage_statement() : 사용내역서 행 단위 카테고리 검토
# 3. verify_categories() : 분류 결과 검증 응답 생성
# --------------------------------------------------------------------------
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
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
    ClassifiedUsageStatementRow,
    DocumentClassification,
    RowReviewResult,
    UsageStatementReviewRequest,
    UsageStatementReviewResponse,
    UsageStatementRow,
    UsageStatementItemsResponse,
    UsageStatementSingleItemRequest,
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


class _ClassifierLLMOutput(BaseModel):
    """LLM 카테고리 분류 결과."""
    category_code: str = Field(description="CAT_01~CAT_09 중 하나")
    reasoning: str = Field(default="", description="분류 근거 한 문장")


_RULE_CONTEXT_CACHE: str | None = None

def _build_rdb_rule_context() -> str:
    """카테고리별 RDB 규칙(허용·QA 예시)을 LLM 컨텍스트 문자열로 조립한다. 캐시 사용."""
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
    """LLM으로 항목 카테고리를 분류한다. RDB 규칙을 컨텍스트로 사용. 실패 시 None 반환."""
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
        code = (result.category_code or "").strip().upper()
        return code if code in CATEGORIES else None
    except Exception:
        log.debug("LLM classifier failed for item=%s", item_name, exc_info=True)
        return None


def _get_generic_item_policy(item_name: str, category_id: str | None = None) -> dict | None:
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
    representative = ", ".join(list(items.keys())[:3])
    basic_info_query = " ".join(f"{k} {v}" for k, v in basic_info.items())
    return f"{representative} {basic_info_query}".strip()


def _qdrant_query(base_query: str, candidates: list) -> str:
    """RDB 후보를 힌트로 삼아 Qdrant 검색 쿼리를 풍부하게 만든다."""
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


def _build_classification_signals(
    *,
    items: dict[str, float],
    basic_info: dict[str, Any],
    collection: str,
) -> _ClassificationSignals:
    """하위 호환용 — _classify_document_with_signals 내부에서 직접 제어하므로 얇은 래퍼."""
    item_names = list(items.keys())
    total_amount = sum(items.values())
    query = _base_query(items, basic_info)

    # RDB 우선 조회
    candidates = _rules_repo.find_category_candidates(
        query_text=query, retrieved_context="", limit=5,
    )
    # Qdrant는 RDB가 불확실할 때만 조회
    rdb_top = candidates[0].score if candidates else 0.0
    rdb_margin = (candidates[0].score - candidates[1].score) if len(candidates) > 1 else rdb_top
    if rdb_top >= _RDB_CLEAR_SCORE and rdb_margin >= _RDB_CLEAR_MARGIN:
        docs: list[Document] = []
        vote_scores: dict[str, float] = {}
    else:
        docs = _retrieve_docs(question=_qdrant_query(query, candidates), collection=collection)
        vote_scores = _vote_category_from_chunks(docs)
        if docs:
            context = "\n\n---\n\n".join(
                f"[출처: {d.metadata.get('source', '알 수 없음')}]\n{d.page_content}"
                for d in docs
            )
            candidates = _rules_repo.find_category_candidates(
                query_text=query, retrieved_context=context, limit=5,
            )

    return _ClassificationSignals(
        docs=docs,
        candidates=candidates,
        vote_scores=vote_scores,
        item_names=item_names,
        total_amount=total_amount,
    )


def _coerce_input(
    items: dict[str, float] | None = None,
    basic_info: dict[str, Any] | None = None,
    document: dict[str, Any] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    if document:
        if items is None:
            items = document.get("items") or document.get("data") or {}
        if basic_info is None:
            basic_info = document.get("basic_info") or document.get("meta") or {}

    items = items or {}
    basic_info = basic_info or {}

    normalized_items: dict[str, float] = {}
    for key, value in items.items():
        try:
            normalized_items[str(key)] = float(value)
        except (TypeError, ValueError):
            continue

    if not normalized_items:
        raise ValueError("분류할 items 데이터가 비어 있습니다.")

    return normalized_items, basic_info


def _coerce_usage_statement_input(
    *,
    payload: dict[str, Any] | None = None,
    usage_statement_id: int | str | None = None,
    rows: list[dict[str, Any]] | list[UsageStatementRow] | None = None,
    basic_info: dict[str, Any] | None = None,
) -> UsageStatementReviewRequest:
    if payload is not None:
        return UsageStatementReviewRequest.model_validate(payload)
    data = {
        "사용내역서ID": usage_statement_id,
        "항목목록": rows or [],
        "기본정보": basic_info or {},
    }
    return UsageStatementReviewRequest.model_validate(data)


def _vote_category_from_chunks(docs: list[Document]) -> dict[str, float]:
    """청크별 LEGAL_CITE 태그에서 제7조제1항제X호 인용을 추출해 역순위 가중 투표.

    law_article 타입 청크(법제처 Open API 조문)는 citation vote에서 제외하고
    LLM 컨텍스트 보강에만 활용한다. 해당 청크에는 법령 자체의 LEGAL_CITE 태그가
    있어 vote 신호를 오염시킬 수 있다.
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
    """RDB·인용투표 모두 실패 시 청크 헤더 키워드로 카테고리 힌트 추출 (최후 폴백)."""
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
    if not scores:
        return True, "청크에서 법령 조항 인용이 발견되지 않았습니다."
    total = sum(scores.values())
    ratio = scores[top_cat] / total
    if ratio < 0.5:
        return True, f"상위 카테고리 투표 비율이 낮습니다. ratio={ratio:.2f}"
    if confidence < _HUMAN_REVIEW_MIN_CONFIDENCE:
        return True, f"인용 투표 신뢰도가 낮습니다. confidence={confidence:.2f}"
    return False, ""


def _human_review_from_rdb(
    *,
    candidates: list,
    docs: list[Document],
    confidence: float,
) -> tuple[bool, str]:
    """RDB 후보 기반 검토 필요 여부 (인용 투표 실패 시 폴백)."""
    if not docs:
        return True, "관련 법령 문맥 검색 결과가 없습니다."
    if not candidates:
        return True, "RDB에서 직접 매칭되는 카테고리 후보가 없습니다."

    top = candidates[0]
    second_score = candidates[1].score if len(candidates) > 1 else 0.0
    margin = top.score - second_score

    if top.score < 4.0:
        return True, f"RDB 후보 근거 점수가 낮습니다. top_score={top.score:.1f}"
    if margin < 1.0 and second_score > 0:
        return True, f"상위 카테고리 후보 간 점수 차가 작습니다. margin={margin:.1f}"
    if confidence < _HUMAN_REVIEW_MIN_CONFIDENCE:
        return True, f"결정 신뢰도가 낮아 사람 검토가 필요합니다. confidence={confidence:.2f}"
    return False, ""


def _confidence_from_candidates(candidates: list) -> float:
    if not candidates:
        return 0.0
    top = candidates[0].score
    second = candidates[1].score if len(candidates) > 1 else 0.0
    ratio = top / (top + second + 1.0)
    score_factor = min(top / 6.0, 1.0)
    return round(max(0.0, min(0.35 + 0.55 * ratio * score_factor, 0.95)), 2)


@traceable(name="classifier.classify_document")
def classify_document(
    items: dict[str, float] | None = None,
    basic_info: dict[str, Any] | None = None,
    document: dict[str, Any] | None = None,
    collection: str = DEFAULT_COLLECTION,
) -> DocumentClassification:
    """
    증빙자료 1건(항목:금액 딕셔너리)을 받아 산안비 카테고리 하나로 분류한다.

    1순위: 검색된 청크의 LEGAL_CITE 인용 투표
    2순위: RDB TF-IDF 규칙 매칭 (폴백)
    """
    items, basic_info = _coerce_input(items=items, basic_info=basic_info, document=document)
    return _classify_document_with_signals(
        items=items,
        basic_info=basic_info,
        collection=collection,
    ).classification


def _classify_document_with_signals(
    *,
    items: dict[str, float],
    basic_info: dict[str, Any],
    collection: str,
    signals: _ClassificationSignals | None = None,
) -> _ClassificationOutcome:
    """
    Validator와 동일한 철학의 우선순위 구조.

    1순위: RDB 단독 확정 — score >= _RDB_CLEAR_SCORE & margin >= _RDB_CLEAR_MARGIN
           → Qdrant 조회 없이 즉시 확정 (빠르고 신뢰도 높음)
    2순위: RDB + Qdrant citation vote 조합
           → 두 신호가 같은 카테고리 → confidence 상승
           → citation vote 단독으로 명확 → vote 결과 사용
    3순위: RDB 약한 신호 단독
           → Qdrant에서 citation vote 없을 때 RDB 후보 사용
    4순위: 청크 헤더 힌트 → UNCLASSIFIED
    """
    item_names = list(items.keys())
    representative = ", ".join(item_names[:3])
    total_amount = sum(items.values())
    query = _base_query(items, basic_info)

    # ── 1. RDB 먼저 (Qdrant 없이) ──────────────────────────────────────────
    if signals is None:
        candidates = _rules_repo.find_category_candidates(
            query_text=query, retrieved_context="", limit=5,
        )
    else:
        candidates = signals.candidates

    rdb_top = candidates[0].score if candidates else 0.0
    rdb_second = candidates[1].score if len(candidates) > 1 else 0.0
    rdb_margin = rdb_top - rdb_second

    # ── 2. RDB 단독 확정 조건 충족 시 Qdrant 스킵 ─────────────────────────
    if candidates and rdb_top >= _RDB_CLEAR_SCORE and rdb_margin >= _RDB_CLEAR_MARGIN:
        category_id = candidates[0].category_code
        confidence = _confidence_from_candidates(candidates)
        needs_review = confidence < _HUMAN_REVIEW_MIN_CONFIDENCE
        review_reason = f"결정 신뢰도 확인 필요. confidence={confidence:.2f}" if needs_review else ""
        needs_review, review_reason = _apply_generic_review_policy(
            item_names=item_names,
            category_id=category_id,
            needs_human_review=needs_review,
            review_reason=review_reason,
        )
        log.debug("rdb-clear: item=%s cat=%s score=%.1f margin=%.1f", representative, category_id, rdb_top, rdb_margin)
        final_signals = signals or _ClassificationSignals(
            docs=[], candidates=candidates, vote_scores={},
            item_names=item_names, total_amount=total_amount,
        )
        return _ClassificationOutcome(
            classification=DocumentClassification(
                category_id=category_id,
                category_name=CATEGORIES.get(category_id, UNCLASSIFIED),
                confidence=confidence,
                total_amount=total_amount,
                items=items,
                needs_human_review=needs_review,
                review_reason=review_reason,
            ),
            signals=final_signals,
            signal_path=f"rdb-clear (score={rdb_top:.1f}, margin={rdb_margin:.1f})",
        )

    # ── 3. RDB 불확실 → Qdrant 조회 ───────────────────────────────────────
    if signals is not None:
        docs = signals.docs
        vote_scores = signals.vote_scores
    else:
        docs = _retrieve_docs(
            question=_qdrant_query(query, candidates), collection=collection,
        )
        vote_scores = _vote_category_from_chunks(docs)
        if docs:
            context = "\n\n---\n\n".join(
                f"[출처: {d.metadata.get('source', '알 수 없음')}]\n{d.page_content}"
                for d in docs
            )
            candidates = _rules_repo.find_category_candidates(
                query_text=query, retrieved_context=context, limit=5,
            )
            rdb_top = candidates[0].score if candidates else 0.0
            rdb_second = candidates[1].score if len(candidates) > 1 else 0.0
            rdb_margin = rdb_top - rdb_second

    final_signals = _ClassificationSignals(
        docs=docs, candidates=candidates, vote_scores=vote_scores,
        item_names=item_names, total_amount=total_amount,
    )

    # ── 4. citation vote 신호 확인 ─────────────────────────────────────────
    top_vote_cat: str | None = None
    vote_ratio = 0.0
    if vote_scores:
        top_vote_cat = max(vote_scores, key=lambda k: vote_scores[k])
        vote_total = sum(vote_scores.values())
        vote_ratio = vote_scores[top_vote_cat] / vote_total

    vote_is_clear = bool(top_vote_cat and vote_ratio >= _VOTE_RATIO_MIN)
    rdb_is_present = bool(candidates and rdb_top >= _RDB_MIN_SCORE)

    # ── 5. RDB + vote 일치 → 가장 높은 신뢰도 ────────────────────────────
    if vote_is_clear and rdb_is_present and candidates[0].category_code == top_vote_cat:
        category_id = top_vote_cat
        confidence = min(_confidence_from_candidates(candidates) + 0.1, 0.95)
        needs_review, review_reason = False, ""
        needs_review, review_reason = _apply_generic_review_policy(
            item_names=item_names, category_id=category_id,
            needs_human_review=needs_review, review_reason=review_reason,
        )
        log.debug("rdb+vote-agree: item=%s cat=%s", representative, category_id)
        return _ClassificationOutcome(
            classification=DocumentClassification(
                category_id=category_id,
                category_name=CATEGORIES.get(category_id, UNCLASSIFIED),
                confidence=confidence, total_amount=total_amount, items=items,
                needs_human_review=needs_review, review_reason=review_reason,
            ),
            signals=final_signals,
            signal_path=f"rdb+vote-agree (rdb={rdb_top:.1f}, vote_ratio={vote_ratio:.2f})",
        )

    # ── 6. vote 단독으로 명확 ─────────────────────────────────────────────
    if vote_is_clear:
        confidence = _confidence_from_votes(vote_scores)
        needs_review, review_reason = _review_from_votes(vote_scores, top_vote_cat, confidence)
        needs_review, review_reason = _apply_generic_review_policy(
            item_names=item_names, category_id=top_vote_cat,
            needs_human_review=needs_review, review_reason=review_reason,
        )
        log.debug("citation-vote: item=%s top=%s ratio=%.2f", representative, top_vote_cat, vote_ratio)
        return _ClassificationOutcome(
            classification=DocumentClassification(
                category_id=top_vote_cat,
                category_name=CATEGORIES.get(top_vote_cat, UNCLASSIFIED),
                confidence=confidence, total_amount=total_amount, items=items,
                needs_human_review=needs_review, review_reason=review_reason,
            ),
            signals=final_signals,
            signal_path=f"citation-vote (top={top_vote_cat}, ratio={vote_ratio:.2f})",
        )

    # ── 7. RDB 단독 (vote 없음, 약한 신호라도 사용) ───────────────────────
    if rdb_is_present:
        category_id = candidates[0].category_code
        confidence = _confidence_from_candidates(candidates)
        needs_review = confidence < _HUMAN_REVIEW_MIN_CONFIDENCE or rdb_top < _RDB_CLEAR_SCORE
        review_reason = f"RDB 규칙 점수가 낮습니다. score={rdb_top:.1f}" if needs_review else ""
        needs_review, review_reason = _apply_generic_review_policy(
            item_names=item_names, category_id=category_id,
            needs_human_review=needs_review, review_reason=review_reason,
        )
        log.debug("rdb-only: item=%s cat=%s score=%.1f", representative, category_id, rdb_top)
        return _ClassificationOutcome(
            classification=DocumentClassification(
                category_id=category_id,
                category_name=CATEGORIES.get(category_id, UNCLASSIFIED),
                confidence=confidence, total_amount=total_amount, items=items,
                needs_human_review=needs_review, review_reason=review_reason,
            ),
            signals=final_signals,
            signal_path=f"rdb-only (score={rdb_top:.1f})",
        )

    # ── 8. 청크 헤더 힌트 (최후 폴백) ────────────────────────────────────
    hint_cat = _hint_from_chunk_headers(docs)
    if hint_cat:
        log.debug("header-hint: item=%s cat=%s", representative, hint_cat)
        return _ClassificationOutcome(
            classification=DocumentClassification(
                category_id=hint_cat,
                category_name=CATEGORIES.get(hint_cat, UNCLASSIFIED),
                confidence=0.55, total_amount=total_amount, items=items,
                needs_human_review=True, review_reason="청크 헤더 키워드 기반 분류. 확인 권장.",
            ),
            signals=final_signals,
            signal_path="header-hint",
        )

    return _ClassificationOutcome(
        classification=DocumentClassification(
            category_id=UNCLASSIFIED,
            category_name=UNCLASSIFIED,
            confidence=0.0, total_amount=total_amount, items=items,
            needs_human_review=True,
            review_reason="RDB 규칙과 법령 청크 인용 모두 카테고리 후보를 찾지 못했습니다.",
        ),
        signals=final_signals,
        signal_path="unclassified",
    )


def _item_status(
    predicted: DocumentClassification,
    given_code: str | None,
    candidates: list | None = None,
    *,
    item_name: str = "",
    basic_info: dict | None = None,
) -> tuple[str, str, str, str]:
    """항목 판정 상태를 결정한다. 반환값: (판정상태, 최종카테고리코드, 사유, 판정경로)
    판정상태는 반드시 '유지' 또는 '카테고리변경' 중 하나. '검토필요' 없음.
    애매한 경우 LLM이 최종 결정한다.
    """
    candidates = candidates or []
    score_by_category = {c.category_code: c.score for c in candidates}
    predicted_score = score_by_category.get(predicted.category_id, 0.0)
    given_score = score_by_category.get(given_code or "", 0.0)
    top_score = candidates[0].score if candidates else 0.0

    def _fallback_to_llm(reason_tag: str) -> tuple[str, str, str, str]:
        """LLM에게 최종 분류를 위임한다. LLM 불가 시 '유지'로 안전하게 처리."""
        llm_code = _llm_classify_item(
            item_name=item_name,
            given_code=given_code or UNCLASSIFIED,
            basic_info=basic_info or {},
            candidates=candidates,
        )
        if llm_code and llm_code != given_code:
            cat_name = CATEGORIES.get(llm_code, llm_code)
            return "카테고리변경", llm_code, f"{cat_name} 카테고리로 변경이 필요함.", f"llm({reason_tag})"
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
    """classi 에이전트 결과를 agent_logs에 INSERT한다 (행당 1행)."""
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
    """
    사용내역서 row 목록을 받아 각 항목의 카테고리가 맞는지 검토하고
    필요하면 최종 카테고리 코드를 수정한다.

    project_id, usage_statement_id(int)가 모두 제공되면 agent_logs에 INSERT한다.
    """
    request = _coerce_usage_statement_input(
        payload=payload,
        usage_statement_id=usage_statement_id,
        rows=rows,
        basic_info=basic_info,
    )

    if not request.rows:
        results: list[RowReviewResult] = []
    else:
        max_workers = min(8, len(request.rows))

        def _run() -> list[RowReviewResult]:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                return list(
                    executor.map(
                        lambda row: copy_context().run(
                            _review_single_usage_statement_row,
                            row=row,
                            basic_info=request.basic_info,
                            collection=collection,
                        ),
                        request.rows,
                    )
                )

        results = _run()

    total_tokens: int | None = None

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


def review_usage_statement_items(
    payload: dict[str, Any] | None = None,
    *,
    usage_statement_id: int | str | None = None,
    rows: list[dict[str, Any]] | list[UsageStatementRow] | None = None,
    basic_info: dict[str, Any] | None = None,
    collection: str = DEFAULT_COLLECTION,
) -> UsageStatementItemsResponse:
    """
    OCR/사용내역서 입력 행과 classifier 판정 결과를 하나의 DTO로 병합해 반환한다.
    """
    request = _coerce_usage_statement_input(
        payload=payload,
        usage_statement_id=usage_statement_id,
        rows=rows,
        basic_info=basic_info,
    )
    reviewed = review_usage_statement(
        payload=request.model_dump(by_alias=True),
        collection=collection,
    )
    review_map = {result.row_id: result for result in reviewed.results}

    merged_rows: list[ClassifiedUsageStatementRow] = []
    for row in request.rows:
        review = review_map.get(row.row_id)
        if review is None:
            continue
        merged_rows.append(
            ClassifiedUsageStatementRow(
                usage_statement_id=request.usage_statement_id,
                row_id=row.row_id,
                given_category_code=row.given_category_code,
                used_on=row.used_on,
                item_name=row.item_name,
                unit=row.unit,
                quantity=row.quantity,
                unit_price=row.unit_price,
                total_amount=row.total_amount,
                final_category_code=review.final_category_code,
                decision_status=review.decision_status,
                needs_human_review=review.needs_human_review,
                reason=review.reason,
            )
        )

    return UsageStatementItemsResponse(
        usage_statement_id=request.usage_statement_id,
        rows=merged_rows,
    )


def verify_categories(
    categories: dict[str, dict[str, float]],
    basic_info: dict[str, Any] | None = None,
    collection: str = DEFAULT_COLLECTION,
) -> UsageStatementReviewResponse:
    """
    구형 카테고리맵 입력을 row 기반 입력으로 변환하는 호환용 래퍼.
    """
    rows: list[dict[str, Any]] = []
    row_id = 1
    for category_name, items in categories.items():
        category_code, _ = _rules_repo.resolve_category(category_name)
        for item_name, amount in items.items():
            rows.append(
                {
                    "행ID": row_id,
                    "기존카테고리코드": category_code or category_name,
                    "항목명": item_name,
                    "금액": amount,
                }
            )
            row_id += 1
    return review_usage_statement(
        usage_statement_id="legacy",
        rows=rows,
        basic_info=basic_info,
        collection=collection,
    )
