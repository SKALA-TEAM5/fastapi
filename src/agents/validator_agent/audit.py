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

# fallback evidence: RDB 미매칭 시 _build_fallback_validator_match()가 생성하는 진단 문자열 패턴
_FALLBACK_EVIDENCE_PATTERNS = (
    "카테고리의 일반 허용 범위와",
    "신호가 일치합니다",
    "예외 또는 제한 조건으로 다뤄질 수 있습니다",
)

_DUPLICATE_COST_KEYWORDS = (
    "중복",
    "이중",
    "이중계상",
    "타 비용",
    "환경관리비",
    "공사비",
    "기포함",
    "별도 계상",
    "동일 목적",
    "타 법령",
)


class CategoryDecisionOutput(BaseModel):
    status: Literal["적절", "부적절", "검토필요"] = Field(description="카테고리 최종 판정")
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
        llm_interpretation="",
        llm_improvements="",
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
    duplicate_risk_items = []
    no_match_items = []
    conflict_items = []
    exception_review_items = []
    for bundle in rule_bundle.items:
        if not bundle.matches:
            no_match_items.append(bundle.item.item_name)
            continue
        if _has_duplicate_cost_risk(bundle):
            duplicate_risk_items.append(bundle.item.item_name)
            continue
        allowed = bundle.top_allowed
        disallowed = bundle.top_disallowed
        exception_summary = _extract_exception_summary(_best_exception_source(bundle))
        if disallowed and (
            not allowed
            or disallowed.score >= allowed.score + 0.5
            or any(keyword in (disallowed.evidence or "") for keyword in ("불가", "제외", "사무실", "감리원", "대지"))
        ):
            disallowed_items.append(bundle.item.item_name)
        elif disallowed and allowed and abs(disallowed.score - allowed.score) < 1.0:
            conflict_items.append(bundle.item.item_name)
        elif exception_summary:
            exception_review_items.append(bundle.item.item_name)
    if disallowed_items:
        return ("부적절", f"직접적인 사용불가 근거가 확인된 항목: {', '.join(disallowed_items)}")
    if duplicate_risk_items:
        return (
            "검토필요",
            f"타 비용과의 중복 계상 가능성이 있는 항목: {', '.join(duplicate_risk_items)}",
        )
    if computation.has_progress_shortfall and computation.usage_shortfall_amount is not None:
        return (
            "검토필요",
            f"공정률 기준 부족: 누적 사용액이 {computation.usage_shortfall_amount:,.0f}원 부족합니다.",
        )
    if no_match_items and not (conflict_items or exception_review_items):
        return (
            "검토필요",
            f"직접적인 허용 근거가 충분히 확인되지 않은 항목: {', '.join(no_match_items)}",
        )
    if conflict_items or exception_review_items or no_match_items:
        segments: list[str] = []
        if conflict_items or exception_review_items:
            review_targets = conflict_items + [item for item in exception_review_items if item not in conflict_items]
            if review_targets:
                segments.append(
                    f"예외 조건 또는 근거 충돌로 검토가 필요한 항목: {', '.join(review_targets)}"
                )
        if no_match_items:
            segments.append(
                f"직접적인 허용 근거가 충분히 확인되지 않은 항목: {', '.join(no_match_items)}"
            )
        return ("검토필요", " ".join(segments).strip())
    return ("적절", "")


def _is_fallback_evidence(text: str) -> bool:
    """RDB 미매칭 시 _build_fallback_validator_match()가 생성한 진단 문자열인지 판별."""
    return any(pattern in text for pattern in _FALLBACK_EVIDENCE_PATTERNS)


def _verbalize_from_match(
    *,
    item_name: str,
    category_name: str,
    best,
    item_allowed: bool,
) -> str:
    """
    fallback evidence가 감지된 경우, DB 필드(referenced_laws, rule_type)만을 사용해
    최소한의 법령 기반 근거 문장을 생성한다.
    하드코딩된 도메인 지식 없이 DB에서 가져온 조항 번호와 허용 여부만 활용한다.
    """
    law = best.referenced_laws[0] if best.referenced_laws else ""
    rule_type = getattr(best, "rule_type", "") or ""

    if not law:
        return ""

    if item_allowed:
        return f"{law}에 따른 허용 항목으로 확인됩니다."

    # disallowed rule_type이 명시된 경우 제외 근거 표현
    if any(tag in rule_type for tag in ("disallowed", "profile_disallowed")):
        return f"{law} 기준 집행 제외 대상으로 확인됩니다."

    return f"{law} 기준 허용 범위를 벗어난 것으로 확인됩니다."


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
        raw_evidence = _clean_text(best.evidence or "", limit=220)
        if raw_evidence and not _is_fallback_evidence(raw_evidence):
            # 실제 법령 원문 → Zero Verbalization (그대로 사용)
            reasoning = raw_evidence
        else:
            # fallback 진단 문자열 → DB 필드 기반 최소 verbalization
            reasoning = _verbalize_from_match(
                item_name=bundle.item.item_name,
                category_name=category_name,
                best=best,
                item_allowed=item_allowed,
            ) or "직접 매칭된 규칙이 없습니다."
        referenced_laws = best.referenced_laws[:]

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
    return hard_reason


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


def _has_duplicate_cost_risk(bundle: ItemRuleBundle) -> bool:
    texts = [bundle.context_text, bundle.item_exception_text]
    for match in bundle.matches[:4]:
        texts.append(match.evidence or "")
    normalized = " ".join(_clean_text(text, limit=600) for text in texts if text)
    return any(keyword in normalized for keyword in _DUPLICATE_COST_KEYWORDS)


def _exception_snippet_score(text: str) -> tuple[int, int]:
    snippet = re.sub(r"\s+", " ", text).strip()
    keywords = sum(1 for pattern in _EXCEPTION_PATTERNS if pattern.search(snippet))
    has_paren = 1 if "(" in snippet or ")" in snippet else 0
    length_score = min(len(snippet), 160)
    return (keywords + has_paren, length_score)


def _contains_exception_phrase(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    return any(pattern.search(normalized) for pattern in _EXCEPTION_PATTERNS)


