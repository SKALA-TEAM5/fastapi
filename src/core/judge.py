# --------------------------------------------------------------------------
# 작성자   : 송상민(ss19801)
# 작성일   : 2026-05-04
#
# [ 주요 클래스 및 함수 정의 ]
#
# 1. judge()              : LangGraph 노드 — 항목 판정 파이프라인 진입점
# 2. item_judge()         : 단일 항목 RAG 기반 적부 판정
# 3. extract_limit_rule() : 법령 컨텍스트에서 한도 규정 추출
# 4. _build_item_reasoning() : 항목별 판정 근거 텍스트 구성
# --------------------------------------------------------------------------
import re
from typing import List

from langchain_core.documents import Document

import src.core.llm_config as llm_config
from src.prompts import JUDGE_PROMPT
from src.repositories import LegalRulesRepository
from src.schemas.classifier import CATEGORIES
from src.schemas.shared import AgenticRAGState, AuditResult
from src.schemas.validator import ItemJudgment

_CITE_PATTERN = re.compile(r"\[LEGAL_CITE:\s*([^\]]+)\]")
_VALID_LAW_RE = re.compile(r"^(제\d+조|별표\d)")

# % 한도 추출 패턴 (총액 기준 규칙만 적용)
_PCT_PATTERNS = [
    (re.compile(r"100분의\s*(\d+)"), lambda m: int(m.group(1)) / 100),
    (re.compile(r"(\d+)분의\s*(\d+)"), lambda m: int(m.group(2)) / int(m.group(1))),
    (re.compile(r"(\d+(?:\.\d+)?)\s*%"), lambda m: float(m.group(1)) / 100),
]
_LIMIT_KEYWORDS = [
    "초과 불가", "초과할 수 없", "초과할수없",   # 공백 있는/없는 PDF 텍스트 모두 대응
    "이내", "를 넘을 수 없", "초과 금지", "초과금지",
]
_TOTAL_KEYWORDS = ["총액", "계상액"]


def extract_limit_rule(
    snippets: list[str],
    category_name: str | None = None,
) -> tuple[float | None, str]:
    """
    RAG snippets에서 '산안비 총액 대비 %' 한도 규정을 추출한다 (결정적 계산, LLM 미사용).

    category_name이 주어지면, 해당 카테고리명이 포함된 문장 또는 그 직후 문장만 대상으로 한다.
    이를 통해 한 chunk에 여러 카테고리 한도가 섞여 있을 때 엉뚱한 카테고리 규정을 선택하는 오류를 방지한다.
    """
    for raw_snippet in snippets:
        snippet = re.sub(r"\s*\n\s*", " ", raw_snippet).strip()

        if not any(kw in snippet for kw in _LIMIT_KEYWORDS):
            continue
        if not any(kw in snippet for kw in _TOTAL_KEYWORDS):
            continue

        sentences = re.split(r"[。]", snippet)

        # category_name 필터: 해당 카테고리명이 등장한 문장부터만 탐색
        start_idx = 0
        if category_name:
            for idx, sentence in enumerate(sentences):
                if category_name in sentence:
                    start_idx = idx
                    break

        for sentence in sentences[start_idx:]:
            has_limit = any(kw in sentence for kw in _LIMIT_KEYWORDS)
            has_total = any(kw in sentence for kw in _TOTAL_KEYWORDS)
            if not (has_limit and has_total):
                continue
            for pattern, extractor in _PCT_PATTERNS:
                m = pattern.search(sentence)
                if m:
                    pct = extractor(m)
                    if 0 < pct <= 1:
                        return pct, sentence.strip()
    return None, ""


def _extract_available_laws(context: str) -> str:
    seen: set = set()
    laws: list = []
    for raw in _CITE_PATTERN.findall(context):
        for part in raw.split("|"):
            law = part.strip()
            if law and law not in seen and _VALID_LAW_RE.match(law):
                seen.add(law)
                laws.append(law)
    if not laws:
        return "(없음 — 이 경우 referenced_laws는 빈 배열로 출력)"
    return "\n".join(f"- {law}" for law in laws)


def _extract_available_law_list(context: str) -> list[str]:
    seen: set[str] = set()
    laws: list[str] = []
    for raw in _CITE_PATTERN.findall(context):
        for part in raw.split("|"):
            law = part.strip()
            if law and law not in seen and _VALID_LAW_RE.match(law):
                seen.add(law)
                laws.append(law)
    return laws


# ── Item Judge ───────────────────────────────────────────────────────

_HUMAN_REVIEW_CONFIDENCE = 0.6
_rules_repo = LegalRulesRepository()


def item_judge(
    item: str,
    amount: float,
    category: str,
    documents: List[Document],
) -> ItemJudgment:
    """단일 항목의 산안비 집행 허용 여부를 판정한다 (금액 판단 없음)."""
    if not documents:
        return ItemJudgment(
            item=item,
            amount=amount,
            category=category,
            allowed=False,
            confidence=0.0,
            reasoning="관련 법령 조항을 검색할 수 없어 판정이 불가합니다.",
            needs_human_review=True,
            review_reason="관련 법령 문맥 검색 결과가 없습니다.",
        )

    context = "\n\n---\n\n".join(
        f"[출처: {d.metadata.get('source', '알 수 없음')}]\n{d.page_content}"
        for d in documents
    )
    category_code = next((code for code, name in CATEGORIES.items() if name == category), None)
    matches = _rules_repo.find_validator_matches(
        category=category,
        item_text=item,
        retrieved_context=context,
        limit=8,
    )
    limit_pct, limit_rule, limit_laws = _rules_repo.find_category_limit(category)

    if not matches:
        return ItemJudgment(
            item=item,
            amount=amount,
            category=category,
            allowed=False,
            confidence=0.0,
            reasoning=f"'{item}'에 대해 {category} 카테고리에서 직접 매칭되는 법령 규칙을 찾지 못했습니다.",
            referenced_laws=limit_laws,
            category_limit_pct=limit_pct,
            category_limit_rule=limit_rule,
            needs_human_review=True,
            review_reason="RDB에서 직접 매칭되는 항목 규칙이 없습니다.",
        )

    allowed_matches = [match for match in matches if match.allowed is True]
    disallowed_matches = [match for match in matches if match.allowed is False]
    top_allowed = allowed_matches[0] if allowed_matches else None
    top_disallowed = disallowed_matches[0] if disallowed_matches else None
    generic_policy = _get_generic_item_policy(item=item, category_code=category_code)

    allowed_score = top_allowed.score if top_allowed else 0.0
    disallowed_score = top_disallowed.score if top_disallowed else 0.0
    best_match = matches[0]
    allowed = allowed_score >= disallowed_score
    top_score = max(allowed_score, disallowed_score, best_match.score)
    rival_score = min(max(allowed_score, disallowed_score), sorted([allowed_score, disallowed_score], reverse=True)[1] if allowed_score and disallowed_score else 0.0)
    confidence = _validator_confidence(top_score=top_score, rival_score=rival_score, match_count=len(matches))

    needs_human_review = False
    review_reason = ""
    if confidence < _HUMAN_REVIEW_CONFIDENCE:
        needs_human_review = True
        review_reason = f"직접 규칙 매칭 근거가 약하거나 충돌합니다. confidence={confidence:.2f}"
    elif top_allowed and top_disallowed and abs(top_allowed.score - top_disallowed.score) < 2.0:
        needs_human_review = True
        review_reason = "허용 규칙과 불가 규칙 점수 차가 작아 사람 검토가 필요합니다."

    generic_item = _is_generic_item_name(item)
    exception_sensitive = bool(generic_policy) or any(_contains_exception_language(match.evidence) for match in matches[:3])
    caution = _generic_policy_warning(generic_policy)
    if not caution and generic_item and exception_sensitive:
        caution = (
            "입력 항목명이 일반적이어서 지급 대상, 사용 장소, 사용 목적 같은 조건 정보가 없으면 "
            "법령상 예외(단, 다만) 적용 여부를 확정할 수 없습니다. 실제 조건을 한번 더 확인해달라."
        )
    if caution:
        needs_human_review = True
        review_reason = f"{review_reason} / {caution}" if review_reason else caution

    reasoning = _build_item_reasoning(
        item=item,
        category=category,
        allowed=allowed,
        primary_match=top_allowed if allowed else top_disallowed or best_match,
        opposite_match=top_disallowed if allowed else top_allowed,
        generic_item=generic_item,
        exception_sensitive=exception_sensitive,
        caution=caution,
    )
    referenced_laws = _merge_laws(
        limit_laws,
        *(match.referenced_laws for match in matches[:3]),
        *([best_match.referenced_laws] if best_match else []),
        *([[_primary_category_law(category_code)]] if category_code else []),
    )
    evidence_snippets = _dedupe_snippets(
        [match.evidence for match in matches[:3]] +
        [_doc_snippet(doc) for doc in documents[:2]]
    )

    return ItemJudgment(
        item=item,
        amount=amount,
        category=category,
        allowed=allowed,
        confidence=confidence,
        reasoning=reasoning,
        evidence_snippets=evidence_snippets,
        referenced_laws=referenced_laws,
        category_limit_pct=limit_pct,
        category_limit_rule=limit_rule,
        needs_human_review=needs_human_review,
        review_reason=review_reason,
    )


def _validator_confidence(*, top_score: float, rival_score: float, match_count: int) -> float:
    if top_score <= 0:
        return 0.0
    ratio = top_score / (top_score + rival_score + 1.0)
    density = min(top_score / 18.0, 1.0)
    coverage = min(match_count / 3.0, 1.0)
    confidence = 0.3 + 0.45 * ratio + 0.2 * density + 0.05 * coverage
    return round(max(0.0, min(confidence, 0.95)), 2)


def _build_item_reasoning(
    *,
    item: str,
    category: str,
    allowed: bool,
    primary_match,
    opposite_match,
    generic_item: bool,
    exception_sensitive: bool,
    caution: str,
) -> str:
    if primary_match is None:
        return f"'{item}'에 대해 {category} 카테고리에서 직접 근거를 찾지 못했습니다."

    if allowed:
        base = f"'{item}'는 {category} 카테고리에서 허용되는 항목으로 판단됩니다. 근거: {primary_match.evidence}"
        if opposite_match is not None:
            base = f"{base} 다만, {opposite_match.evidence}"
        if caution:
            return f"{base} {caution}"
        if generic_item and exception_sensitive:
            return f"{base} 단, 입력값이 '{item}'처럼 일반 항목명만 있는 경우에는 지급 대상, 사용 장소, 사용 목적 정보가 없어 예외 적용 여부를 추가 확인해야 합니다."
        return base

    if opposite_match is not None:
        base = (
            f"'{item}'는 {category} 카테고리의 일반 허용 범위와 구분되어 불가로 판단됩니다. "
            f"허용 범위는 {opposite_match.evidence}이다. 다만, 본 항목에는 {primary_match.evidence}가 직접 적용됩니다."
        )
        if caution:
            return f"{base} {caution}"
        if generic_item and exception_sensitive:
            return f"{base} 단, 입력값이 일반 항목명만 남아 있는 경우에는 실제 지급 대상이나 사용 목적을 추가 확인하는 것이 안전합니다."
        return base

    base = f"'{item}'는 {category} 카테고리에서 불가 항목으로 판단됩니다. 근거: {primary_match.evidence}"
    if caution:
        return f"{base} {caution}"
    if generic_item and exception_sensitive:
        return f"{base} 단, 입력값이 일반 항목명만 남아 있는 경우에는 실제 지급 대상이나 사용 목적을 추가 확인하는 것이 안전합니다."
    return base


def _primary_category_law(category_code: str | None) -> str:
    mapping = {
        "CAT_01": "제7조제1항제1호",
        "CAT_02": "제7조제1항제2호",
        "CAT_03": "제7조제1항제3호",
        "CAT_04": "제7조제1항제4호",
        "CAT_05": "제7조제1항제5호",
        "CAT_06": "제7조제1항제6호",
        "CAT_07": "제7조제1항제7호",
        "CAT_08": "제7조제1항제8호",
        "CAT_09": "제7조제1항제9호",
    }
    return mapping.get(category_code or "", "")


def _merge_laws(*law_groups: list[str]) -> list[str]:
    merged: list[str] = []
    for group in law_groups:
        for law in group:
            if law and law not in merged:
                merged.append(law)
    return merged


def _doc_snippet(doc: Document) -> str:
    text = re.sub(r"\s+", " ", doc.page_content).strip()
    return text[:180]


def _dedupe_snippets(snippets: list[str]) -> list[str]:
    cleaned: list[str] = []
    for snippet in snippets:
        value = re.sub(r"\s+", " ", snippet or "").strip()
        if value and value not in cleaned:
            cleaned.append(value)
    return cleaned[:3]


def _is_generic_item_name(item: str) -> bool:
    normalized = re.sub(r"[\(\)\[\],./]", " ", item or "")
    tokens = [token for token in re.findall(r"[0-9A-Za-z가-힣]{2,}", normalized) if len(token) >= 2]
    if not tokens:
        return True
    if len(tokens) == 1:
        return True
    return False


def _get_generic_item_policy(*, item: str, category_code: str | None) -> dict | None:
    normalized_item = re.sub(r"\s+", " ", (item or "").strip().lower())
    policies = _rules_repo.generic_item_policies
    for policy in policies.values():
        categories = policy.get("conditional_categories") or []
        if categories and category_code and category_code not in categories:
            continue
        aliases = [str(alias).lower() for alias in policy.get("aliases") or []]
        for alias in aliases:
            if normalized_item == alias:
                return policy
    return None


def _generic_policy_warning(policy: dict | None) -> str:
    if not policy:
        return ""
    warning = str(policy.get("warning_template") or "").strip()
    return warning


def _contains_exception_language(text: str) -> bool:
    normalized = text or ""
    return any(keyword in normalized for keyword in ("다만", "단,", "단 ", "불가", "제외"))


def _fallback_audit_result(*, question: str, context: str, laws: list[str], top_source: str) -> AuditResult:
    negative_keywords = ("불가", "제외", "할 수 없다", "초과할 수 없", "초과 불가")
    positive_keywords = ("사용 가능", "사용이 가능", "할 수 있다", "해당한다")
    has_negative = any(keyword in context for keyword in negative_keywords)
    has_positive = any(keyword in context for keyword in positive_keywords)

    if has_negative and not has_positive:
        return AuditResult(
            is_compliant=False,
            confidence=0.68 if laws else 0.42,
            reasoning="LLM 미설정 상태라 규칙 기반으로 보수 판정했습니다. 검색 문맥에서 불가/제외 신호가 확인되어 부적합으로 봅니다.",
            referenced_laws=laws[:3],
            needs_human_review=True,
            top_source=top_source,
        )
    if has_positive and not has_negative:
        return AuditResult(
            is_compliant=True,
            confidence=0.62 if laws else 0.4,
            reasoning="LLM 미설정 상태라 규칙 기반으로 보수 판정했습니다. 검색 문맥에서 허용 신호가 확인되지만 추가 검토를 권장합니다.",
            referenced_laws=laws[:3],
            needs_human_review=True,
            top_source=top_source,
        )
    return AuditResult(
        is_compliant=False,
        confidence=0.35 if laws else 0.2,
        reasoning=f"LLM 미설정 상태라 규칙 기반으로 보수 판정했습니다. '{question}'에 대해 명확한 허용/불가 결론을 확정하기 어려워 추가 검토가 필요합니다.",
        referenced_laws=laws[:3],
        needs_human_review=True,
        top_source=top_source,
    )


def judge(state: AgenticRAGState) -> AgenticRAGState:
    """관련 문서를 바탕으로 산안비 적합성을 구조화된 형태로 판정."""
    if not state["retrieved_docs"]:
        result = AuditResult(
            is_compliant=False,
            confidence=0.0,
            reasoning="관련 법령 조항을 검색할 수 없어 판정이 불가합니다.",
            referenced_laws=[],
            needs_human_review=True,
        )
    else:
        context = "\n\n---\n\n".join(
            f"[출처: {d.metadata.get('source', '알 수 없음')}]\n{d.page_content}"
            for d in state["retrieved_docs"]
        )
        available_laws = _extract_available_laws(context)
        top_source = state["retrieved_docs"][0].metadata.get("source", "")
        law_list = _extract_available_law_list(context)
        try:
            llm = llm_config.get()
            result = (
                JUDGE_PROMPT | llm.with_structured_output(AuditResult)
            ).invoke(
                {
                    "context": context,
                    "available_laws": available_laws,
                    "question": state["question"],
                }
            )
            result.needs_human_review = (
                result.confidence < _HUMAN_REVIEW_CONFIDENCE or not result.referenced_laws
            )
            result.top_source = top_source
        except RuntimeError:
            result = _fallback_audit_result(
                question=state["question"],
                context=context,
                laws=law_list,
                top_source=top_source,
            )

    return {**state, "judgment": result.model_dump()}
