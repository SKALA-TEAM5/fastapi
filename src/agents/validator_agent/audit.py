# --------------------------------------------------------------------------
# 작성자   : 송상민(ss19801)
# 작성일   : 2026-05-04
#
# [ 주요 클래스 및 함수 정의 ]
#
# 1. decide_category() : 카테고리 최종 판정 및 근거 조립
# 2. _hard_status() : 하드룰 기반 판정 결정
# 3. _build_item_judgment() : 항목별 판정 결과 생성
# --------------------------------------------------------------------------
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from src.agents.validator_agent.calculator import CategoryComputation
from src.agents.validator_agent.context_retriever import CategoryRetrievedContext
from src.agents.validator_agent.parser import CategoryInputBlock
from src.agents.validator_agent.rule_matcher import CategoryRuleBundle, ItemRuleBundle
import src.core.llm_config as llm_config
from src.prompts import CATEGORY_DECISION_PROMPT
from src.schemas.validator import CategoryAuditResult, ItemJudgment

_EXCEPTION_PATTERNS = (
    re.compile(r"다만"),
    re.compile(r"단[, ]"),
    re.compile(r"제외"),
    re.compile(r"불가"),
    re.compile(r"초과할\s*수\s*없"),
    re.compile(r"이내"),
)


class CategoryDecisionOutput(BaseModel):
    status: Literal["적절", "부적절", "검토필요"] = Field(description="카테고리 최종 판정")
    legal_basis: str = Field(description="법령 근거")
    interpretation: str = Field(description="판정 해석")
    improvements: str = Field(description="보완점")
    referenced_laws: list[str] = Field(default_factory=list)


def decide_category(
    *,
    block: CategoryInputBlock,
    retrieved: CategoryRetrievedContext,
    rule_bundle: CategoryRuleBundle,
    computation: CategoryComputation,
) -> CategoryAuditResult:
    hard_status, hard_reason = _hard_status(rule_bundle=rule_bundle, computation=computation)
    item_judgments = [
        _build_item_judgment(bundle, category_name=block.category_name)
        for bundle in rule_bundle.items
    ]
    referenced_laws = _collect_laws(rule_bundle=rule_bundle, computation=computation)

    decision = _llm_decision(
        block=block,
        retrieved=retrieved,
        rule_bundle=rule_bundle,
        computation=computation,
        law_candidates=referenced_laws,
    )

    status = _resolve_final_status(
        hard_status=hard_status,
        decision=decision,
        rule_bundle=rule_bundle,
        computation=computation,
        retrieved=retrieved,
    )

    rejection_reason = _compose_category_reason(
        status=status,
        hard_reason=hard_reason,
        decision=decision,
    )

    final_laws = referenced_laws[:]
    if decision:
        for law in decision.referenced_laws:
            if law and law not in final_laws:
                final_laws.append(law)

    evidence_snippets = _build_evidence_snippets(retrieved)
    return CategoryAuditResult(
        status=status,
        total=computation.total,
        limit=computation.limit_amount,
        exceeded=computation.exceeded,
        limit_rule=rule_bundle.limit_rule,
        rejection_reason=rejection_reason,
        items=item_judgments,
        referenced_laws=final_laws,
        evidence_snippets=evidence_snippets,
        needs_human_review=(status == "검토필요"),
        progress_rate=computation.progress_rate,
        required_usage_rate=computation.required_usage_rate,
        required_used_amount=computation.required_used_amount,
        cumulative_used_amount=computation.cumulative_used_amount,
        usage_shortfall_amount=computation.usage_shortfall_amount,
    )


def _llm_decision(
    *,
    block: CategoryInputBlock,
    retrieved: CategoryRetrievedContext,
    rule_bundle: CategoryRuleBundle,
    computation: CategoryComputation,
    law_candidates: list[str],
) -> CategoryDecisionOutput | None:
    try:
        llm = llm_config.get()
    except RuntimeError:
        return None

    item_lines = "\n".join(
        f"- {item.item.item_name}: {item.item.amount:,.0f}원"
        for item in rule_bundle.items
    ) or "(없음)"
    rule_lines = "\n".join(_format_item_rule_bundle(bundle) for bundle in rule_bundle.items) or "(없음)"
    exception_lines = "\n".join(
        _clean_text(doc.page_content, limit=240)
        for doc in retrieved.exception_docs[:5]
    ) or "(없음)"
    metric_lines = _format_metric_lines(computation)
    law_lines = "\n".join(f"- {law}" for law in law_candidates) or "(없음)"

    try:
        return (
            CATEGORY_DECISION_PROMPT | llm.with_structured_output(CategoryDecisionOutput)
        ).invoke(
            {
                "category": block.category_name,
                "item_lines": item_lines,
                "rule_lines": rule_lines,
                "exception_lines": exception_lines,
                "metric_lines": metric_lines,
                "law_candidates": law_lines,
            }
        )
    except Exception:
        return None


def _hard_status(
    *,
    rule_bundle: CategoryRuleBundle,
    computation: CategoryComputation,
) -> tuple[Literal["적절", "부적절", "검토필요"], str]:
    if computation.exceeded and computation.limit_amount is not None:
        return (
            "부적절",
            f"카테고리 한도 초과: {computation.total:,.0f}원 > {computation.limit_amount:,.0f}원",
        )

    disallowed_items = []
    review_items = []
    for bundle in rule_bundle.items:
        if not bundle.matches:
            review_items.append(bundle.item.item_name)
            continue
        allowed = bundle.top_allowed
        disallowed = bundle.top_disallowed
        if disallowed and (
            not allowed
            or disallowed.score >= allowed.score + 0.5
            or any(keyword in (disallowed.evidence or "") for keyword in ("불가", "제외", "사무실", "감리원", "대지"))
        ):
            disallowed_items.append(bundle.item.item_name)
        elif disallowed and allowed and abs(disallowed.score - allowed.score) < 1.0:
            review_items.append(bundle.item.item_name)
    if disallowed_items:
        return ("부적절", f"직접적인 사용불가 근거가 확인된 항목: {', '.join(disallowed_items)}")
    if computation.has_progress_shortfall and computation.usage_shortfall_amount is not None:
        return (
            "검토필요",
            f"공정률 기준 부족: 누적 사용액이 {computation.usage_shortfall_amount:,.0f}원 부족합니다.",
        )
    if review_items:
        return ("검토필요", f"예외 조건 또는 근거 충돌로 검토가 필요한 항목: {', '.join(review_items)}")
    return ("적절", "")


def _build_item_judgment(bundle: ItemRuleBundle, *, category_name: str) -> ItemJudgment:
    allowed = bundle.top_allowed
    disallowed = bundle.top_disallowed
    best = allowed or disallowed or (bundle.matches[0] if bundle.matches else None)
    item_allowed = bool(allowed and (not disallowed or allowed.score >= disallowed.score))
    exception_source = _best_exception_source(bundle)
    exception_summary = _extract_exception_summary(exception_source)
    has_conflict = bool(allowed and disallowed and abs(allowed.score - disallowed.score) < 1.0)
    needs_review = bool(exception_summary or has_conflict)
    reasoning = "직접 매칭된 규칙이 없습니다."
    referenced_laws: list[str] = []
    if best is not None:
        reasoning = _compose_item_reasoning(
            bundle=bundle,
            best=best,
            item_allowed=item_allowed,
            category_name=category_name,
        )
        referenced_laws = best.referenced_laws[:]
    reasoning = _polish_item_reasoning(
        reasoning=reasoning,
        exception_summary=exception_summary,
        needs_review=needs_review,
    )

    return ItemJudgment(
        item=bundle.item.item_name,
        amount=bundle.item.amount,
        category=category_name,
        allowed=item_allowed,
        confidence=_bundle_confidence(bundle),
        reasoning=reasoning,
        evidence_snippets=_build_item_evidence_snippets(bundle=bundle, best=best),
        referenced_laws=referenced_laws,
        category_limit_pct=None,
        category_limit_rule="",
        needs_human_review=needs_review,
        review_reason=_build_item_review_reason(
            needs_review=needs_review,
            exception_summary=exception_summary,
            has_conflict=has_conflict,
        ),
        exception_summary=exception_summary,
    )


def _resolve_final_status(
    *,
    hard_status: Literal["적절", "부적절", "검토필요"],
    decision: CategoryDecisionOutput | None,
    rule_bundle: CategoryRuleBundle,
    computation: CategoryComputation,
    retrieved: CategoryRetrievedContext,
) -> Literal["적절", "부적절", "검토필요"]:
    if hard_status in {"부적절", "검토필요"}:
        return hard_status

    conservative_review = _needs_conservative_review(
        rule_bundle=rule_bundle,
        computation=computation,
        retrieved=retrieved,
    )
    if conservative_review:
        if decision and decision.status != "적절":
            return decision.status
        return "검토필요"

    return "적절"


def _bundle_confidence(bundle: ItemRuleBundle) -> float:
    allowed = bundle.top_allowed.score if bundle.top_allowed else 0.0
    disallowed = bundle.top_disallowed.score if bundle.top_disallowed else 0.0
    top = max(allowed, disallowed, bundle.matches[0].score if bundle.matches else 0.0)
    rival = min(max(allowed, disallowed), sorted([allowed, disallowed], reverse=True)[1] if allowed and disallowed else 0.0)
    if top <= 0:
        return 0.0
    ratio = top / (top + rival + 1.0)
    confidence = 0.4 + 0.45 * ratio
    return round(max(0.0, min(confidence, 0.95)), 2)


def _format_item_rule_bundle(bundle: ItemRuleBundle) -> str:
    lines = [f"- 항목: {bundle.item.item_name}"]
    for match in bundle.matches[:4]:
        lines.append(
            f"  * allowed={match.allowed} score={match.score:.2f} law={','.join(match.referenced_laws[:2])} evidence={_clean_text(match.evidence, limit=140)}"
        )
    if bundle.has_exception:
        summary = _extract_exception_summary(_best_exception_source(bundle))
        if summary:
            lines.append(f"  * 예외 문구: {summary}")
        else:
            lines.append("  * 예외 문구(단/다만/제외/불가) 포함")
    return "\n".join(lines)


def _build_item_evidence_snippets(*, bundle: ItemRuleBundle, best) -> list[str]:
    snippets: list[str] = []
    if best is not None and best.evidence:
        snippets.append(_clean_text(best.evidence, limit=220))
    if bundle.context_text:
        context_snippet = _clean_text(bundle.context_text, limit=180)
        if context_snippet and context_snippet not in snippets:
            snippets.append(context_snippet)
    return snippets[:3]


def _format_metric_lines(computation: CategoryComputation) -> str:
    lines = [f"- 카테고리 합계: {computation.total:,.0f}원"]
    if computation.limit_amount is not None:
        lines.append(f"- 카테고리 한도: {computation.limit_amount:,.0f}원")
        lines.append(f"- 한도 초과 여부: {computation.exceeded}")
    if computation.progress_rate is not None:
        lines.append(f"- 공정률: {computation.progress_rate:.1f}%")
    if computation.required_usage_rate is not None and computation.required_used_amount is not None:
        lines.append(f"- 요구 최소 사용률: {computation.required_usage_rate * 100:.0f}%")
        lines.append(f"- 요구 최소 사용액: {computation.required_used_amount:,.0f}원")
    if computation.cumulative_used_amount is not None:
        lines.append(f"- 실제 누적 사용액: {computation.cumulative_used_amount:,.0f}원")
    if computation.usage_shortfall_amount is not None:
        lines.append(f"- 부족액: {computation.usage_shortfall_amount:,.0f}원")
    return "\n".join(lines)


def _collect_laws(*, rule_bundle: CategoryRuleBundle, computation: CategoryComputation) -> list[str]:
    laws: list[str] = []
    for law in rule_bundle.primary_laws:
        if law and law not in laws:
            laws.append(law)
    for bundle in rule_bundle.items:
        for match in bundle.matches[:3]:
            for law in match.referenced_laws:
                if law and law not in laws:
                    laws.append(law)
    if computation.required_usage_rate is not None and rule_bundle.progress_law not in laws:
        laws.append(rule_bundle.progress_law)
    return laws


def _compose_category_reason(
    *,
    status: str,
    hard_reason: str,
    decision: CategoryDecisionOutput | None,
) -> str:
    if decision is None:
        return hard_reason
    if status == "적절" and not hard_reason:
        interpretation = decision.interpretation.strip()
        if any(keyword in interpretation for keyword in ("검토", "추가 확인", "불확실")):
            return decision.legal_basis.strip()
    parts = [decision.legal_basis.strip(), decision.interpretation.strip()]
    if status == "검토필요" and decision.improvements.strip():
        parts.append(decision.improvements.strip())
    merged = " ".join(part for part in parts if part)
    if hard_reason:
        if merged:
            if hard_reason not in merged:
                return f"{hard_reason} {merged}".strip()
        return hard_reason
    return merged


def _needs_conservative_review(
    *,
    rule_bundle: CategoryRuleBundle,
    computation: CategoryComputation,
    retrieved: CategoryRetrievedContext,
) -> bool:
    if computation.has_progress_shortfall or computation.exceeded:
        return True
    for bundle in rule_bundle.items:
        if not bundle.matches:
            return True
        allowed = bundle.top_allowed
        disallowed = bundle.top_disallowed
        if allowed and disallowed and abs(allowed.score - disallowed.score) < 1.0:
            return True
    return False


def _build_evidence_snippets(retrieved: CategoryRetrievedContext) -> list[str]:
    snippets: list[str] = []
    for doc in (retrieved.category_docs + retrieved.exception_docs)[:5]:
        cleaned = _clean_text(doc.page_content, limit=180)
        if cleaned and cleaned not in snippets:
            snippets.append(cleaned)
    return snippets


def _clean_text(text: str, *, limit: int) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    return cleaned[:limit]


def _extract_exception_summary(text: str) -> str:
    cleaned = re.sub(r"\[LEGAL_CITE:[^\]]+\]", " ", text or "")
    cleaned = re.sub(r"[#<>]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return ""
    candidates = re.split(r"(?<=[.!?])\s+|\s+-\s+|\s+\*\s+", cleaned)
    keyword_hits = []
    for part in candidates:
        snippet = part.strip(" -")
        if _contains_exception_phrase(snippet):
            keyword_hits.append(snippet)
    if keyword_hits:
        best = max(keyword_hits, key=_exception_snippet_score)
        best = re.sub(r"\s+", " ", best).strip(" -")
        return best[:160]
    return ""


def _best_exception_source(bundle: ItemRuleBundle) -> str:
    prioritized = []
    if bundle.top_disallowed is not None:
        prioritized.append(bundle.top_disallowed)
    if bundle.top_allowed is not None and bundle.top_allowed is not bundle.top_disallowed:
        prioritized.append(bundle.top_allowed)
    for match in bundle.matches[:4]:
        if match not in prioritized:
            prioritized.append(match)
    for match in prioritized:
        if _contains_exception_phrase(match.evidence or ""):
            return match.evidence
    return ""


def _compose_item_reasoning(*, bundle: ItemRuleBundle, best, item_allowed: bool, category_name: str) -> str:
    item_name = bundle.item.item_name
    detail = _naturalize_evidence_statement(
        _summarize_match_evidence(best.evidence, item_name=item_name),
        item_allowed=item_allowed,
    )
    law = best.referenced_laws[0] if best.referenced_laws else ""

    if item_allowed:
        opening = f"{item_name} 항목은 {category_name} 카테고리에서 허용되는 집행 항목에 해당합니다."
    else:
        opening = f"{item_name} 항목은 {category_name} 카테고리의 허용 범위를 벗어납니다."

    if law:
        opening = f"{law} 기준상 {opening}"

    if detail:
        return f"{opening} {detail}".strip()
    return opening


def _summarize_match_evidence(text: str, *, item_name: str) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return ""

    segments = re.split(r"\s*[-\u2022\u25cf\u25a3]\s*|\s*\s*", cleaned)
    candidates = []
    for segment in segments:
        seg = segment.strip(" -")
        if not seg:
            continue
        if len(seg) < 8:
            continue
        candidates.append(seg)

    item_tokens = [token for token in re.findall(r"[0-9A-Za-z가-힣]{2,}", item_name) if len(token) >= 2]
    preferred = max(
        candidates or [cleaned],
        key=lambda segment: _evidence_segment_score(segment, item_tokens=item_tokens),
    )

    preferred = _normalize_evidence_segment(preferred)
    if "가능한지" in preferred:
        tail = preferred.split("가능한지", 1)[1].strip(" :.-")
        if tail:
            preferred = tail
    if "사용 가능함." in preferred:
        preferred = preferred.split("사용 가능함.", 1)[-1].strip(" :.-") or preferred
    if "사용이 가능함." in preferred:
        tail = preferred.split("사용이 가능함.", 1)[-1].strip(" :.-")
        if tail:
            preferred = tail

    preferred = _normalize_evidence_segment(preferred)
    if len(preferred) > 140:
        preferred = preferred[:140].rstrip(" ,.") + "..."
    return preferred


def _polish_item_reasoning(*, reasoning: str, exception_summary: str, needs_review: bool) -> str:
    text = re.sub(r"\s+", " ", reasoning or "").strip()
    text = re.sub(r"\s*다만 예외 또는 단서 문구가 함께 확인되었습니다\.?\s*$", "", text)
    text = re.sub(r"\s*다만 예외 또는 단서 문구가 함께 확인되었습니다", "", text)
    if exception_summary:
        if needs_review:
            return f"{text} 다만 \"{exception_summary}\"라는 단서가 함께 확인되어 적용 대상을 한 번 더 확인할 필요가 있습니다.".strip()
        return f"{text} 참고로 \"{exception_summary}\"라는 단서 문구도 함께 확인되었습니다.".strip()
    return text or "직접 매칭된 규칙이 없습니다."


def _build_item_review_reason(*, needs_review: bool, exception_summary: str, has_conflict: bool) -> str:
    if not needs_review:
        return ""
    if exception_summary and has_conflict:
        return f"예외 문구와 상반된 근거가 함께 있어 확인이 필요합니다: {exception_summary}"
    if exception_summary:
        return f"예외 문구 적용 여부 확인 필요: {exception_summary}"
    if has_conflict:
        return "허용 근거와 불가 근거가 함께 확인되어 추가 검토가 필요합니다."
    return "예외 문구 또는 근거 충돌 확인이 필요합니다."


def _exception_snippet_score(text: str) -> tuple[int, int]:
    snippet = re.sub(r"\s+", " ", text).strip()
    keywords = sum(1 for pattern in _EXCEPTION_PATTERNS if pattern.search(snippet))
    has_paren = 1 if "(" in snippet or ")" in snippet else 0
    length_score = min(len(snippet), 160)
    return (keywords + has_paren, length_score)


def _contains_exception_phrase(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    return any(pattern.search(normalized) for pattern in _EXCEPTION_PATTERNS)


def _evidence_segment_score(text: str, *, item_tokens: list[str]) -> tuple[int, int, int, int]:
    segment = _normalize_evidence_segment(text)
    token_hits = sum(1 for token in item_tokens if token in segment)
    answer_hits = sum(1 for keyword in ("가능", "불가", "제외", "해당", "사용", "구입", "지급", "아닌", "비용") if keyword in segment)
    question_penalty = 0
    if "가능한지" in segment:
        question_penalty = 3
        if any(keyword in segment for keyword in ("불가", "제외", "아닌")):
            question_penalty = 1
    law_boilerplate_penalty = 1 if "제7조제1항" in segment and token_hits == 0 and len(segment) > 80 else 0
    length_score = min(len(segment), 160)
    return (
        token_hits + answer_hits - question_penalty - law_boilerplate_penalty,
        token_hits,
        answer_hits,
        length_score,
    )


def _normalize_evidence_segment(text: str) -> str:
    segment = re.sub(r"\s+", " ", text or "").strip()
    segment = re.sub(r"^[0-9]+[.)]?\s*", "", segment)
    segment = segment.replace("으로 사용 가능한지 으로 사용 가능한지", "으로 사용 가능한지")
    segment = segment.strip(" -:.")
    return segment


def _naturalize_evidence_statement(text: str, *, item_allowed: bool) -> str:
    segment = _normalize_evidence_segment(text)
    if not segment:
        return ""

    if item_allowed and any(keyword in segment for keyword in ("화재 위험작업", "용접", "인화성물질")):
        return "화재 위험작업 구간에서 근로자 보호를 위해 사용하는 항목은 해당 카테고리 범위에 포함됩니다."
    if (not item_allowed) and any(keyword in segment for keyword in ("사무실", "분전반", "근로자 보호 목적이 아닌")):
        return "근로자 보호 목적이 아닌 장소나 용도로 사용하는 항목은 해당 카테고리로 집행할 수 없습니다."
    if (not item_allowed) and any(keyword in segment for keyword in ("불가", "제외")):
        return "해당 항목은 예외 또는 제외 대상으로 분류되어 이 카테고리로 집행할 수 없습니다."

    if "보호구 항목으로 사용이 가능함" in segment:
        return "법정 보호구의 구입·수리·관리 비용에 해당하므로 이 카테고리에서 집행할 수 있습니다."

    if "일반 허용 범위와" in segment and "신호가 일치합니다" in segment:
        return segment.rstrip(".") + "."

    return segment.rstrip(".") + "."
