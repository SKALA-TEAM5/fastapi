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
import logging
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document

from src.core.rag import MAX_RETRY, build_retriever, rerank, retrieve, rewrite_query
from src.core.storage import load_vectorstore
from src.repositories import LegalRulesRepository
from src.schemas.classifier import (
    CATEGORIES,
    UNCLASSIFIED,
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
# RDB 점수가 이 값 이상이면 법령 규칙을 최우선으로 신뢰
_RDB_DOMINANT_SCORE = 9.0
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
    retriever = build_retriever(vectorstore)
    state = {
        "question": question,
        "documents": [],
        "judgment": None,
        "retry_count": 0,
    }
    state = retrieve(state, retriever)
    state = rerank(state)
    while not state["documents"] and state.get("retry_count", 0) < MAX_RETRY:
        state = rewrite_query(state)
        state = retrieve(state, retriever)
        state = rerank(state)
    return state["documents"]


def _build_classification_signals(
    *,
    items: dict[str, float],
    basic_info: dict[str, Any],
    collection: str,
) -> _ClassificationSignals:
    item_names = list(items.keys())
    total_amount = sum(items.values())
    representative = ", ".join(item_names[:3])
    basic_info_query = " ".join(f"{k} {v}" for k, v in basic_info.items())
    query = f"산안비에서 '{representative}' 항목이 해당하는 카테고리와 법령 조항 {basic_info_query}".strip()

    docs = _retrieve_docs(question=query, collection=collection)
    if not docs:
        return _ClassificationSignals(
            docs=[],
            candidates=[],
            vote_scores={},
            item_names=item_names,
            total_amount=total_amount,
        )

    context = "\n\n---\n\n".join(
        f"[출처: {d.metadata.get('source', '알 수 없음')}]\n{d.page_content}"
        for d in docs
    )
    candidates = _rules_repo.find_category_candidates(
        query_text=f"{representative} {basic_info_query}".strip(),
        retrieved_context=context,
        limit=5,
    )
    vote_scores = _vote_category_from_chunks(docs)
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
    """청크별 LEGAL_CITE 태그에서 제7조제1항제X호 인용을 추출해 역순위 가중 투표."""
    scores: dict[str, float] = {}
    for rank, doc in enumerate(docs):
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


def classify_document(
    items: dict[str, float] | None = None,
    basic_info: dict[str, Any] | None = None,
    document: dict[str, Any] | None = None,
    collection: str = "documents",
) -> DocumentClassification:
    """
    증빙자료 1건(항목:금액 딕셔너리)을 받아 산안비 카테고리 하나로 분류한다.

    1순위: 검색된 청크의 LEGAL_CITE 인용 투표
    2순위: RDB TF-IDF 규칙 매칭 (폴백)
    """
    items, basic_info = _coerce_input(items=items, basic_info=basic_info, document=document)
    signals = _build_classification_signals(
        items=items,
        basic_info=basic_info,
        collection=collection,
    )
    item_names = signals.item_names
    representative = ", ".join(item_names[:3])
    total_amount = signals.total_amount
    docs = signals.docs

    if not docs:
        return DocumentClassification(
            category_id=UNCLASSIFIED,
            category_name=UNCLASSIFIED,
            confidence=0.0,
            total_amount=total_amount,
            items=items,
            needs_human_review=True,
            review_reason="관련 법령 문맥을 검색하지 못했습니다.",
        )

    candidates = signals.candidates
    vote_scores = signals.vote_scores

    # RDB 최상위 후보 점수가 매우 높으면 법령 규칙을 우선 신뢰
    rdb_top_score = candidates[0].score if candidates else 0.0
    if rdb_top_score >= _RDB_DOMINANT_SCORE:
        top_candidate = candidates[0]
        category_id = top_candidate.category_code
        confidence = _confidence_from_candidates(candidates)
        needs_human_review, review_reason = _human_review_from_rdb(
            candidates=candidates, docs=docs, confidence=confidence,
        )
        needs_human_review, review_reason = _apply_generic_review_policy(
            item_names=item_names,
            category_id=category_id,
            needs_human_review=needs_human_review,
            review_reason=review_reason,
        )
        log.debug("rdb-dominant: item=%s cat=%s score=%.1f", representative, category_id, rdb_top_score)
        return DocumentClassification(
            category_id=category_id,
            category_name=CATEGORIES.get(category_id, UNCLASSIFIED),
            confidence=confidence,
            total_amount=total_amount,
            items=items,
            needs_human_review=needs_human_review,
            review_reason=review_reason,
        )

    # 인용 투표 신호가 명확하면 청크 기반 결과 사용
    if vote_scores:
        top_cat = max(vote_scores, key=lambda k: vote_scores[k])
        vote_total = sum(vote_scores.values())
        vote_ratio = vote_scores[top_cat] / vote_total
        if vote_ratio >= _VOTE_RATIO_MIN:
            confidence = _confidence_from_votes(vote_scores)
            needs_human_review, review_reason = _review_from_votes(vote_scores, top_cat, confidence)
            needs_human_review, review_reason = _apply_generic_review_policy(
                item_names=item_names,
                category_id=top_cat,
                needs_human_review=needs_human_review,
                review_reason=review_reason,
            )
            log.debug(
                "citation-vote: item=%s top=%s ratio=%.2f confidence=%.2f",
                representative, top_cat, vote_ratio, confidence,
            )
            return DocumentClassification(
                category_id=top_cat,
                category_name=CATEGORIES.get(top_cat, UNCLASSIFIED),
                confidence=confidence,
                total_amount=total_amount,
                items=items,
                needs_human_review=needs_human_review,
                review_reason=review_reason,
            )

    # RDB 폴백 (인용 투표 신호 약함)
    if candidates:
        top_candidate = candidates[0]
        category_id = top_candidate.category_code
        confidence = _confidence_from_candidates(candidates)
        needs_human_review, review_reason = _human_review_from_rdb(
            candidates=candidates, docs=docs, confidence=confidence,
        )
        needs_human_review, review_reason = _apply_generic_review_policy(
            item_names=item_names,
            category_id=category_id,
            needs_human_review=needs_human_review,
            review_reason=review_reason,
        )
        log.debug("rdb-fallback: item=%s cat=%s score=%.1f", representative, category_id, rdb_top_score)
        return DocumentClassification(
            category_id=category_id,
            category_name=CATEGORIES.get(category_id, UNCLASSIFIED),
            confidence=confidence,
            total_amount=total_amount,
            items=items,
            needs_human_review=needs_human_review,
            review_reason=review_reason,
        )

    # 헤더 힌트 폴백: 청크 섹션 제목에서 카테고리 추론
    hint_cat = _hint_from_chunk_headers(docs)
    if hint_cat:
        log.debug("header-hint: item=%s cat=%s", representative, hint_cat)
        return DocumentClassification(
            category_id=hint_cat,
            category_name=CATEGORIES.get(hint_cat, UNCLASSIFIED),
            confidence=0.72,
            total_amount=total_amount,
            items=items,
            needs_human_review=False,
            review_reason="",
        )

    return DocumentClassification(
        category_id=UNCLASSIFIED,
        category_name=UNCLASSIFIED,
        confidence=0.0,
        total_amount=total_amount,
        items=items,
        needs_human_review=True,
        review_reason="RDB 및 청크 인용 투표 모두 카테고리 후보를 찾지 못했습니다.",
    )


def _item_status(
    predicted: DocumentClassification,
    given_code: str | None,
    candidates: list | None = None,
) -> tuple[str, str, str]:
    candidates = candidates or []
    score_by_category = {candidate.category_code: candidate.score for candidate in candidates}
    predicted_score = score_by_category.get(predicted.category_id, 0.0)
    given_score = score_by_category.get(given_code or "", 0.0)
    top_score = candidates[0].score if candidates else 0.0

    if predicted.category_id == UNCLASSIFIED:
        return "검토필요", given_code or UNCLASSIFIED, "입력 항목만으로 카테고리 확정이 어려워 한번 더 확인해달라."

    # 분류 Agent는 적정/부적정을 판단하지 않으므로, 현재 카테고리에 분류 근거가 남아 있고
    # 대체 카테고리 우위가 강하지 않다면 기존 카테고리를 우선 유지한다.
    if given_code and predicted.category_id != given_code:
        if given_score > 0 and predicted_score <= 0:
            if given_score >= top_score:
                return "유지", given_code, ""
            return "검토필요", given_code, "입력 항목만으로 카테고리 확정이 어려워 한번 더 확인해달라."

        margin = predicted_score - given_score
        if given_score > 0 and margin < _RECLASSIFY_MARGIN_MIN:
            if predicted.needs_human_review or margin < 1.5:
                return "검토필요", given_code, "입력 항목만으로 카테고리 확정이 어려워 한번 더 확인해달라."
            return "유지", given_code, ""
        if predicted.confidence < _RECLASSIFY_CONFIDENCE_MIN:
            return "검토필요", given_code, f"재분류 신뢰도가 충분히 높지 않습니다. confidence={predicted.confidence:.2f}"

    if predicted.category_id == given_code:
        return "유지", predicted.category_id, ""

    if predicted.needs_human_review:
        return "검토필요", given_code or predicted.category_id, "입력 항목만으로 카테고리 확정이 어려워 한번 더 확인해달라."

    return "카테고리변경", predicted.category_id, f"{predicted.category_name} 카테고리로 변경이 필요함."


def review_usage_statement(
    payload: dict[str, Any] | None = None,
    *,
    usage_statement_id: int | str | None = None,
    rows: list[dict[str, Any]] | list[UsageStatementRow] | None = None,
    basic_info: dict[str, Any] | None = None,
    collection: str = "documents",
) -> UsageStatementReviewResponse:
    """
    사용내역서 row 목록을 받아 각 항목의 카테고리가 맞는지 검토하고
    필요하면 최종 카테고리 코드를 수정한다.
    """
    request = _coerce_usage_statement_input(
        payload=payload,
        usage_statement_id=usage_statement_id,
        rows=rows,
        basic_info=basic_info,
    )

    results: list[RowReviewResult] = []
    for row in request.rows:
        predicted = classify_document(
            items={row.item_name: row.total_amount},
            basic_info=request.basic_info,
            collection=collection,
        )
        signals = _build_classification_signals(
            items={row.item_name: row.total_amount},
            basic_info=request.basic_info,
            collection=collection,
        )
        decision_status, final_category_code, reason = _item_status(
            predicted=predicted,
            given_code=row.given_category_code,
            candidates=signals.candidates,
        )
        if decision_status == "검토필요" and not reason:
            reason = "한번 더 확인해달라."
        results.append(
            RowReviewResult(
                row_id=row.row_id,
                item_name=row.item_name,
                given_category_code=row.given_category_code,
                final_category_code=final_category_code,
                decision_status=decision_status,
                needs_human_review=decision_status == "검토필요",
                reason=reason if decision_status != "유지" else "",
            )
        )

    return UsageStatementReviewResponse(
        usage_statement_id=request.usage_statement_id,
        results=results,
    )


def verify_categories(
    categories: dict[str, dict[str, float]],
    basic_info: dict[str, Any] | None = None,
    collection: str = "documents",
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
