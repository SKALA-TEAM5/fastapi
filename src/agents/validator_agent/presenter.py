# --------------------------------------------------------------------------
# 작성자   : 송상민(ss19801)
# 작성일   : 2026-05-04
#
# [ 주요 클래스 및 함수 정의 ]
#
# 1. summarize_audit_response()      : 카테고리별 감사 결과 요약 생성
# 2. to_validator_response()         : 전체 감사 응답 DTO 변환
# 3. _build_sources()                : 출처 목록(법령·규정) 구성
# 4. _build_reason()                 : 항목별 LLM 사유 텍스트 생성
# 5. _derive_reasoning_summary_for_law() : 법령별 출처 요지 추출
# --------------------------------------------------------------------------
from __future__ import annotations

import logging
import re

from pydantic import BaseModel, Field

import src.core.llm_config as llm_config
from src.prompts import (
    AUDIT_REASON_SYNTHESIS_PROMPT,
    ITEM_REASON_SYNTHESIS_PROMPT,
)
from src.schemas.classifier import CATEGORIES
from src.schemas.validator import (
    AuditResponse,
    AuditSourceSummary,
    CategoryAuditSummary,
    UsageStatementAuditSummaryResponse,
    ValidatorAuditResponse,
    ValidatorCategoryMetrics,
)

logger = logging.getLogger(__name__)

_PROGRESS_RULE_LAW = "별표 3 공사진척에 따른 산업안전보건관리비 사용기준"
_REGULATION_NAME = "「건설업 산업안전보건관리비 계상 및 사용기준」"
_SCHEDULE_2_REF = f"{_REGULATION_NAME} 별표 2"
_SCHEDULE_3_REF = f"{_REGULATION_NAME} 별표 3"
_DUPLICATE_REVIEW_KEYWORDS = ("중복", "이중", "이중계상", "타 비용", "환경관리비", "공사비", "기포함", "동일 목적")

# ── Reason-code templates ─────────────────────────────────────────────────────
# {law_ref}        : e.g. "「건설업 산업안전보건관리비 계상 및 사용기준」 제7조제1항제2호에 따르면"  (may be empty)
# {items}          : comma-joined relevant item names
# {allowed_items}  : comma-joined allowed item names
# {disallowed_items}: comma-joined disallowed item names
# {total}          : formatted total amount (원)
# {limit}          : formatted limit amount (원)
# {exceeded}       : formatted exceeded amount (원)
# {progress_rate}  : progress rate (%)
# {shortfall}      : formatted shortfall amount (원)
# {exception_texts}: joined exception clause texts
# LLM 텍스트가 실질적 내용을 담고 있는지 판별할 때 사용하는 순환/범용 표현 목록
def summarize_audit_response(
    *,
    response: AuditResponse,
    usage_statement_id: int | str | None = None,
) -> UsageStatementAuditSummaryResponse:
    summaries: list[CategoryAuditSummary] = []
    for category_name, result in response.categories.items():
        category_code = next((code for code, name in CATEGORIES.items() if name == category_name), category_name)
        detailed_sources = _build_sources(
            category_name=category_name,
            category_code=category_code,
            result=result,
            base_amount=response.base_amount,
        )
        display_sources = _compact_display_sources(detailed_sources)
        summaries.append(
            CategoryAuditSummary(
                category_code=category_code,
                status=result.status,
                reason=_build_reason(
                    category_name=category_name,
                    category_code=category_code,
                    result=result,
                    base_amount=response.base_amount,
                    sources=detailed_sources,
                ),
                sources=display_sources,
            )
        )
    return UsageStatementAuditSummaryResponse(
        usage_statement_id=usage_statement_id,
        results=summaries,
    )


def to_validator_response(
    *,
    response: AuditResponse,
    usage_statement_id: int | str | None = None,
) -> ValidatorAuditResponse:
    summary = summarize_audit_response(response=response, usage_statement_id=usage_statement_id)
    metrics: dict[str, ValidatorCategoryMetrics] = {}
    for category_name, cat_result in response.categories.items():
        category_code = next((code for code, name in CATEGORIES.items() if name == category_name), category_name)
        avg_confidence = sum(j.confidence for j in cat_result.items) / len(cat_result.items) if cat_result.items else 0.0
        metrics[category_code] = ValidatorCategoryMetrics(
            confidence=round(avg_confidence, 2),
            total=cat_result.total,
            limit=cat_result.limit,
            exceeded=cat_result.exceeded,
            needs_human_review=cat_result.needs_human_review,
            progress_rate=cat_result.progress_rate,
            required_usage_rate=cat_result.required_usage_rate,
            usage_shortfall_amount=cat_result.usage_shortfall_amount,
        )
    return ValidatorAuditResponse(result=summary, metrics=metrics)


def _build_sources(*, category_name: str, category_code: str, result, base_amount: float) -> list[AuditSourceSummary]:
    sources: list[AuditSourceSummary] = []
    seen_law_keys: set[str] = set()
    limit_pct = (result.limit / base_amount) if (result.limit is not None and base_amount) else None
    primary_item_law = _best_item_law(result=result)

    ordered_laws = _arrange_source_laws(
        result=result,
        primary_item_law=primary_item_law,
        ordered_laws=_prioritize_referenced_laws(
            _collect_candidate_laws(result=result),
            result=result,
        ),
    )

    mandatory_laws: list[str] = []
    if result.required_usage_rate is not None:
        mandatory_laws.append(_SCHEDULE_3_REF)
    if result.exceeded:
        mandatory_laws.append("제4조")
    if result.status == "부적절" and any(not item.allowed for item in result.items) and not any(item.allowed for item in result.items):
        mandatory_laws.append(_SCHEDULE_2_REF)
    for law in mandatory_laws + ordered_laws:
        law_key = _law_key(law)
        if not law or law_key in seen_law_keys:
            continue
        if not result.exceeded and law.startswith("제4조"):
            continue
        summary = _build_additional_law_summary(
            law=law,
            category_name=category_name,
            result=result,
        )
        if law.startswith("별표 ") and law not in {"별표 2", "별표 3", _SCHEDULE_2_REF, _SCHEDULE_3_REF} and _is_generic_source_summary(summary):
            continue
        if limit_pct is not None and law.startswith("제4조") and summary:
            limit_pct_str = f"{limit_pct * 100:.0f}%"
            if limit_pct_str not in summary:
                summary = f"{category_name}의 집행액은 산업안전보건관리비 총액의 {limit_pct_str} 이내로 제한됩니다."
        if summary:
            sources.append(AuditSourceSummary(law=_qualify_law_for_display(law), summary=summary))
            seen_law_keys.add(law_key)

    if not sources and ordered_laws:
        sources.append(
            AuditSourceSummary(
                law=_qualify_law_for_display(ordered_laws[0]),
                summary=f"{category_name} 판정의 근거 조항입니다.",
            )
        )
    return sources


class _AuditSynthesisOutput(BaseModel):
    reason: str = Field(description="전문 감사 의견 사유 (합니다체, 3단락 이상)")


class _ItemReasonSynthesisOutput(BaseModel):
    reason: str = Field(description="항목별 검토 사유 2~3문장")


def _synthesize_reason_with_llm(
    *,
    category_name: str,
    result,
    base_amount: float,
    sources: list[AuditSourceSummary],
) -> str:
    """
    AUDIT_REASON_SYNTHESIS_PROMPT를 사용해 전문 감사 의견 사유를 한 번에 생성한다.
    audit.py가 pack한 수치·원본 텍스트·법령 후보만을 사용하며
    새로운 수치나 법령을 창작하지 않도록 프롬프트가 제한한다.
    """
    try:
        llm = llm_config.get()
    except RuntimeError as exc:
        logger.info("[validator_reason] LLM unavailable for category synthesis: %s", exc)
        return ""

    # ── 판정 유형 코드 (조치 문장 선택 기준) ─────────────────────────────────
    # ★ _classify_reason_code와 동일한 로직으로 산출해야 PROMPT 지침과 일치한다.
    disallowed_items_for_code = [item.item for item in getattr(result, "items", []) if not item.allowed]
    allowed_items_for_code = [item.item for item in getattr(result, "items", []) if item.allowed]
    exception_texts_for_code = _collect_exception_summaries(result, allowed=None)
    _reason_code = _classify_reason_code(
        result=result,
        disallowed_items=disallowed_items_for_code,
        allowed_items=allowed_items_for_code,
        exception_texts=exception_texts_for_code,
    )

    # ── 집행 수치 ────────────────────────────────────────────────────────────
    metric_parts: list[str] = [
        f"- 판정 유형 코드: {_reason_code}",
        f"- 카테고리 집행 합계: {result.total:,.0f}원",
    ]
    if result.limit is not None:
        metric_parts.append(f"- 법정 한도: {result.limit:,.0f}원")
        if base_amount:
            pct = result.limit / base_amount * 100
            metric_parts.append(f"- 한도 비율: 산안비 총액의 {pct:.0f}%")
        exceeded_amount = max(0.0, result.total - result.limit)
        metric_parts.append(f"- 초과 여부: {'초과 ' + f'{exceeded_amount:,.0f}원' if result.exceeded else '한도 이내'}")
    if result.progress_rate is not None:
        metric_parts.append(f"- 공정률: {result.progress_rate:.1f}%")
    if result.required_usage_rate is not None:
        metric_parts.append(f"- 요구 최소 사용률: {result.required_usage_rate * 100:.0f}%")
        # ★ 요구 최소 사용액을 명시해야 LLM이 직접 계산을 시도하지 않는다.
        #   이 값이 없으면 LLM이 금회사용금액 × 사용률로 잘못 계산해 틀린 수치가 사유에 나온다.
        if getattr(result, "required_used_amount", None) is not None:
            metric_parts.append(f"- 요구 최소 사용액: {result.required_used_amount:,.0f}원")
    if getattr(result, "cumulative_used_amount", None) is not None:
        metric_parts.append(f"- 실제 누적 사용액: {result.cumulative_used_amount:,.0f}원")
    if result.usage_shortfall_amount is not None and result.usage_shortfall_amount > 0:
        metric_parts.append(f"- 공정률 기준 부족액: {result.usage_shortfall_amount:,.0f}원")
    metric_lines = "\n".join(metric_parts)

    # ── 항목별 판정 데이터 ────────────────────────────────────────────────────
    item_parts: list[str] = []
    for item in getattr(result, "items", []) or []:
        verdict = "허용" if item.allowed else "불허"
        laws = ", ".join(getattr(item, "referenced_laws", []) or []) or "(법령 미확인)"
        reasoning_raw = " ".join((getattr(item, "reasoning", "") or "").split())[:180]
        item_parts.append(
            f"- {item.item} ({item.amount:,.0f}원): {verdict} | 근거조항={laws}\n"
            f"  원본근거={reasoning_raw}"
        )
    item_lines = "\n".join(item_parts) or "(항목 없음)"

    # ── 예외·단서 문구 ────────────────────────────────────────────────────────
    exception_parts: list[str] = []
    for item in getattr(result, "items", []) or []:
        exc = " ".join((getattr(item, "exception_summary", "") or "").split())
        if exc and exc not in exception_parts:
            exception_parts.append(exc)
    exception_lines = "\n".join(f"- {e}" for e in exception_parts) or "(단서 없음)"

    # ── 법령 후보 ─────────────────────────────────────────────────────────────
    law_parts: list[str] = []
    seen_laws: set[str] = set()
    for source in sources:
        law = (source.law or "").strip()
        if law and law not in seen_laws:
            seen_laws.add(law)
            summary = (source.summary or "").strip()
            law_parts.append(f"- {law}: {summary}" if summary else f"- {law}")
    for law in getattr(result, "referenced_laws", []) or []:
        if law and law not in seen_laws:
            seen_laws.add(law)
            law_parts.append(f"- {law}")
    law_candidates = "\n".join(law_parts) or "(법령 정보 없음)"

    try:
        output: _AuditSynthesisOutput = (
            AUDIT_REASON_SYNTHESIS_PROMPT
            | llm.with_structured_output(_AuditSynthesisOutput)
        ).invoke(
            {
                "category": category_name,
                "status": result.status,
                "hard_reason": (result.rejection_reason or "").strip() or "(하드룰 판정 없음)",
                "metric_lines": metric_lines,
                "item_lines": item_lines,
                "exception_lines": exception_lines,
                "law_candidates": law_candidates,
            }
        )
    except Exception as exc:
        logger.warning("[validator_reason] category synthesis failed: %s", exc)
        return ""

    text = " ".join((output.reason or "").split()).strip()
    if len(text) < 60:
        return ""
    return text


def _synthesize_item_reason_with_llm(
    *,
    category_name: str,
    item,
    result=None,
    legal_basis_context: str = "",
) -> str:
    """
    확정된 항목 판정값을 바탕으로 item-level reason 한 문장만 LLM으로 재작성한다.
    LLM은 판정을 바꾸지 않고, 원본 reasoning/법령을 자연스럽게 설명하는 역할만 맡는다.
    """
    try:
        llm = llm_config.get()
    except RuntimeError as exc:
        logger.info("[validator_reason] LLM unavailable for item reason: %s", exc)
        return ""

    is_conditional_review = bool(item.allowed and getattr(item, "conditional_review", False))
    verdict = "조건부 허용(확인 필요)" if is_conditional_review else ("허용" if item.allowed else "불허")
    conditional_note = (
        "법령상 조건(전담·선임·신고·자격 등)을 실제로 충족했는지 아직 확인되지 않아 검토가 필요합니다."
        if is_conditional_review
        else "(없음)"
    )
    item_laws = [
        str(law)
        for law in (getattr(item, "referenced_laws", []) or [])
        if law and not _is_progress_law_reference(str(law))
    ]
    laws = ", ".join(item_laws) or "(법령 미확인)"
    reasoning = _item_reasoning_context_for_llm(item)
    category_issue = _item_category_issue_context(result)
    try:
        output: _ItemReasonSynthesisOutput = (
            ITEM_REASON_SYNTHESIS_PROMPT
            | llm.with_structured_output(_ItemReasonSynthesisOutput)
        ).invoke(
            {
                "category": category_name,
                "item_name": getattr(item, "item", ""),
                "amount": f"{getattr(item, 'amount', 0):,.0f}원",
                "verdict": verdict,
                "laws": laws,
                "category_issue": category_issue,
                "conditional_note": conditional_note,
                "legal_basis_context": legal_basis_context or "(각주/법령 원문 없음)",
                "reasoning": reasoning or "(원본 근거 없음)",
            }
        )
    except Exception as exc:
        logger.warning("[validator_reason] item reason generation failed: %s", exc)
        return ""

    text = _normalize_reason_output(output.reason or "")
    if len(text) < 15:
        return ""
    text = _ensure_reason_has_footnoted_law_reference(
        text=text,
        laws=laws,
        legal_basis_context=legal_basis_context,
    )
    has_category_issue = category_issue != "(없음)"
    if item.allowed and not has_category_issue and any(keyword in text for keyword in ("불가", "부적절", "제외", "인정하기 어렵")):
        logger.info("[validator_reason] rejected conflicting item reason for allowed item: %s", item.item)
        return ""
    return text


def _ensure_reason_has_footnoted_law_reference(*, text: str, laws: str, legal_basis_context: str) -> str:
    reason = " ".join((text or "").split()).strip()
    if not reason:
        return ""
    reason = _wrap_first_law_reference(reason)
    if _reason_has_footnoted_law_reference(reason):
        return reason
    if _reason_has_law_reference(reason):
        return reason
    law_reference = _primary_law_reference(laws=laws, legal_basis_context=legal_basis_context)
    if not law_reference:
        return reason
    return f"[{law_reference}]에 따르면, {reason}"


def _reason_has_footnoted_law_reference(text: str) -> bool:
    return bool(
        re.search(r"\[[^\]]*(?:산업안전보건관리비|산업안전보건법|근로기준법|제\d+조)[^\]]*\]", text or "")
    )


def _wrap_first_law_reference(text: str) -> str:
    if _reason_has_footnoted_law_reference(text):
        return text

    patterns = (
        r"(「[^」]+」\s*제\d+조(?:제\d+항)?(?:제\d+호)?)",
        r"((?:건설업\s*)?산업안전보건관리비\s*계상\s*및\s*사용기준\s*제\d+조(?:제\d+항)?(?:제\d+호)?)",
        r"((?:산업안전보건법|근로기준법)\s*제\d+조(?:제\d+항)?(?:제\d+호)?)",
        r"(제\d+조(?:제\d+항)?(?:제\d+호)?)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        start, end = match.span(1)
        return f"{text[:start]}[{match.group(1)}]{text[end:]}"
    return text


def _reason_has_law_reference(text: str) -> bool:
    return bool(
        re.search(r"(건설업\s*)?산업안전보건관리비|산업안전보건법|근로기준법|제\d+조", text or "")
    )


def _primary_law_reference(*, laws: str, legal_basis_context: str) -> str:
    context = " ".join((legal_basis_context or "").split())
    law_match = re.search(
        r"((?:「[^」]+」|건설업\s*산업안전보건관리비\s*계상\s*및\s*사용기준|산업안전보건법|근로기준법)[^\\n]{0,80}?제\d+조(?:제\d+항)?(?:제\d+호)?)",
        context,
    )
    if law_match:
        return law_match.group(1).strip(" ,.")

    law_text = " ".join((laws or "").split()).strip()
    if not law_text or law_text == "(법령 미확인)":
        return ""
    first_law = law_text.split(",")[0].split("|")[0].strip()
    if _is_progress_law_reference(first_law):
        return ""
    if re.fullmatch(r"제\d+조(?:제\d+항)?(?:제\d+호)?", first_law):
        return f"「건설업 산업안전보건관리비 계상 및 사용기준」 {first_law}"
    return first_law


def _is_progress_law_reference(law: str) -> bool:
    normalized = " ".join((law or "").split())
    return "별표 3" in normalized or "공사진척" in normalized or "공정률" in normalized


def _item_category_issue_context(result) -> str:
    if result is None:
        return "(없음)"

    issue_parts: list[str] = []
    if getattr(result, "exceeded", False):
        total = getattr(result, "total", None)
        limit = getattr(result, "limit", None)
        if total is not None and limit is not None:
            issue_parts.append(
                f"카테고리 집행액 {total:,.0f}원이 법정 한도 {limit:,.0f}원을 초과합니다."
            )
        else:
            issue_parts.append("카테고리 법정 한도를 초과합니다.")

    if getattr(result, "needs_human_review", False) and not issue_parts:
        rejection = " ".join((getattr(result, "rejection_reason", "") or "").split()).strip()
        if "공정률" not in rejection and "공사진척" not in rejection:
            issue_parts.append(rejection or "카테고리 차원의 추가 검토가 필요합니다.")

    return " ".join(issue_parts) if issue_parts else "(없음)"


def _item_reasoning_context_for_llm(item) -> str:
    """
    항목별 reason LLM 입력용 근거 요약.

    RDB 법령 원문이나 깨진 PDF 텍스트를 길게 넣으면 LLM이 이를 그대로 복사할 수
    있으므로, 문장 경계 기준으로 짧게 줄이고 원문 복사 금지 힌트를 함께 제공한다.
    """
    reasoning = _clean_db_raw_text(
        " ".join((getattr(item, "reasoning", "") or "").split()).strip()
    )
    if not reasoning:
        return "(원본 근거 없음)"

    sentences = _split_reason_sentences(reasoning)
    if sentences:
        condensed = " ".join(sentences[:2])
    else:
        condensed = reasoning
    condensed = condensed[:260].strip()
    return (
        f"{condensed}\n"
        "주의: 위 원본 근거는 검색/RDB 원문 조각일 수 있으므로 그대로 복사하지 말고, "
        "항목명·판정·참조 법령에 맞는 자연스러운 검토 사유로 재작성합니다."
    )


def _build_reason(
    *,
    category_name: str,
    category_code: str,
    result,
    base_amount: float,
    sources: list[AuditSourceSummary],
) -> str:
    """
    항목별 검토 사유를 생성한다.

    rule_matcher에서 LLM이 생성한 reason_text를 우선 사용하고,
    없으면 reasoning을 그대로 사용한다. (별도 LLM 호출 없음)
    """
    lines: list[str] = []
    for item in getattr(result, "items", []) or []:
        verdict = "집행 가능" if item.allowed else "집행 불가"
        # rule_matcher에서 생성된 reason_text 우선 사용, 없으면 reasoning 사용
        item_reason = getattr(item, "reason_text", "") or item.reasoning or ""
        item_reason = _normalize_reason_output(item_reason)
        law_raw = item.referenced_laws[0] if item.referenced_laws else ""
        law = _qualify_law_for_display(law_raw) if law_raw else ""

        if law:
            lines.append(f"{item.item} ({item.amount:,.0f}원): {verdict} — {item_reason} ({law})")
        else:
            lines.append(f"{item.item} ({item.amount:,.0f}원): {verdict} — {item_reason}")

    body = "\n".join(lines)

    notes: list[str] = []
    if result.exceeded and result.limit is not None:
        exceeded_amount = max(0.0, result.total - result.limit)
        notes.append(f"한도 {result.limit:,.0f}원 초과 {exceeded_amount:,.0f}원")
    if result.usage_shortfall_amount and result.usage_shortfall_amount > 0:
        req = getattr(result, "required_used_amount", None)
        if req is not None:
            notes.append(
                f"공정률 {result.progress_rate:.1f}% 기준 최소 {req:,.0f}원 대비 "
                f"{result.usage_shortfall_amount:,.0f}원 부족"
            )
        else:
            notes.append(f"공정률 기준 {result.usage_shortfall_amount:,.0f}원 부족")

    if notes:
        body = body + "\n" + " | ".join(notes)

    return _qualify_law_refs_in_reason(body)


def _classify_reason_code(*, result, disallowed_items: list[str], allowed_items: list[str], exception_texts: list[str]) -> str:
    if result.status == "부적절" and result.exceeded and result.limit is not None:
        return "improper_limit_exceeded"
    if result.status == "부적절" and result.required_usage_rate is not None and result.usage_shortfall_amount is not None:
        return "improper_progress_shortfall"
    if result.status == "부적절" and disallowed_items:
        return "improper_mixed_items" if allowed_items else "improper_scope_exclusion"
    if result.status == "부적절":
        return "improper_scope_exclusion"
    if result.status == "적절" and result.required_usage_rate is not None:
        return "appropriate_progress_compliant"
    return "appropriate_compliant"


def _compact_display_sources(sources: list[AuditSourceSummary]) -> list[AuditSourceSummary]:
    grouped = _group_sources_by_summary(sources)
    compacted: list[AuditSourceSummary] = []
    for laws, summary in grouped:
        deduped_laws: list[str] = []
        seen: set[str] = set()
        for law in laws:
            normalized = (law or "").strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                deduped_laws.append(normalized)
        compacted.append(
            AuditSourceSummary(
                law=", ".join(deduped_laws),
                summary=summary,
            )
        )
    return compacted


def _group_sources_by_summary(sources: list[AuditSourceSummary]) -> list[tuple[list[str], str]]:
    groups: list[tuple[list[str], str]] = []
    for source in sources:
        summary = (source.summary or "").strip()
        law = (source.law or "").strip()
        if not law and not summary:
            continue
        matched = False
        for idx, (existing_laws, existing_summary) in enumerate(groups):
            if existing_summary == summary:
                if law:
                    existing_laws.append(law)
                groups[idx] = (existing_laws, existing_summary)
                matched = True
                break
        if not matched:
            groups.append(([law] if law else [], summary))
    return groups


def _build_additional_law_summary(*, law: str, category_name: str, result) -> str:
    raw = _extract_snippet_for_law(result=result, law=law)
    if law in {_SCHEDULE_2_REF, "별표 2"} or law.startswith("별표 2"):
        if raw:
            return raw
        return "산업안전보건관리비로 사용할 수 없는 제외 항목 기준을 정합니다."
    if law in {_PROGRESS_RULE_LAW, _SCHEDULE_3_REF} or law.startswith("별표 3"):
        if result.required_usage_rate is not None and result.progress_rate is not None:
            return f"공정률 {result.progress_rate:.1f}% 구간에서는 산업안전보건관리비를 {result.required_usage_rate * 100:.0f}% 이상 사용해야 합니다."
        return "공정률 구간별 최소 사용률 기준을 정합니다."
    if law.startswith("제4조"):
        if raw and any(kw in raw for kw in ("한도", "이내", "초과")):
            return raw
        if result.limit is not None:
            return "산업안전보건관리비 총액 기준 한도를 함께 적용합니다."
        return "산업안전보건관리비 총액 기준을 함께 적용합니다."
    if law.startswith("제2조") or law.startswith("제62조"):
        return raw
    if raw:
        return raw
    reasoning_summary = _derive_reasoning_summary_for_law(result=result, law=law)
    if reasoning_summary:
        return reasoning_summary
    if law.startswith("별표"):
        return "세부 기준과 적용 구간을 함께 확인합니다."
    return f"{category_name} 판단의 근거 조항입니다."


def _is_generic_source_summary(summary: str) -> bool:
    generic_phrases = (
        "참고합니다",
        "함께 확인합니다",
        "함께 참고합니다",
        "함께 적용합니다",
        "판단의 근거 조항입니다",
        "허용 범위에 부합합니다",
        "허용 범위에 해당합니다",
        "허용 범위를 벗어납니다",
        "허용 범위에 부합하지만",
    )
    return any(p in summary for p in generic_phrases)


def _prioritize_referenced_laws(laws: list[str], *, result) -> list[str]:
    item_level_laws = {
        normalized
        for item in getattr(result, "items", []) or []
        for law in getattr(item, "referenced_laws", []) or []
        for normalized in _normalize_law_refs(law)
    }
    context_text = " ".join(
        part for part in (
            getattr(result, "limit_rule", "") or "",
            getattr(result, "rejection_reason", "") or "",
            " ".join(getattr(result, "referenced_laws", []) or []),
        )
        if part
    )

    def sort_key(law: str) -> tuple[int, int]:
        if result.required_usage_rate is not None and ("별표 3" in law or "공사진척" in law):
            return (0, 0)
        if result.exceeded and law.startswith("제4조"):
            return (1, 0)
        if result.exceeded and law and law in context_text:
            return (2, 0)
        if law in item_level_laws and ("제" in law and ("항" in law or "호" in law)):
            return (3, 0)
        if law and law in context_text:
            return (4, 0)
        if law.startswith("제2조"):
            return (5, 0)
        if law.startswith("제62조"):
            return (6, 0)
        if law.startswith("별표"):
            return (7, 0)
        return (8, laws.index(law))

    seen: set[str] = set()
    ordered: list[str] = []
    for law in sorted(laws, key=sort_key):
        if law and law not in seen:
            seen.add(law)
            ordered.append(law)
    return ordered


def _law_key(law: str) -> str:
    cleaned = (law or "").strip()
    if cleaned in {_PROGRESS_RULE_LAW, _SCHEDULE_3_REF} or cleaned.startswith("별표 3"):
        return "schedule_3"
    if cleaned.startswith("별표 2"):
        return "schedule_2"
    return cleaned


def _qualify_law_for_display(law: str) -> str:
    """
    출처(조항) 표시용 법령명 완전 표기.

    - 이미 법령명이 포함된 경우: 그대로 반환
    - 별표 2 / 별표 3: 고시명 + 별표 번호로 반환
    - bare 제X조 (고시 조항): 고시명 prefix 추가
      * 제1조~제15조 → 「건설업 산안비 계상 및 사용기준」 조항
      * 제29조 이상 큰 번호 → 산업안전보건법 조항일 가능성이 높으므로 그대로
    - ','로 묶인 복수 조항: 각각 처리
    """
    import re

    cleaned = (law or "").strip()
    if not cleaned:
        return cleaned

    # 여러 조항이 콤마 또는 파이프로 묶인 경우
    if re.search(r"(?:,|\|)\s*제\d+조", cleaned):
        parts = [p.strip() for p in re.split(r"\s*(?:,|\|)\s*", cleaned)]
        return " 및 ".join(_qualify_law_for_display(p) for p in parts if p)

    # 이미 법령명 포함
    if cleaned.startswith("「") or cleaned.startswith(_REGULATION_NAME):
        return cleaned
    if any(kw in cleaned for kw in ("산업안전보건법", "건설기술진흥법", "건설업 산업안전")):
        return cleaned

    # 별표
    if cleaned in {_PROGRESS_RULE_LAW, _SCHEDULE_3_REF} or cleaned.startswith("별표 3"):
        return _SCHEDULE_3_REF
    if cleaned.startswith("별표 2"):
        return _SCHEDULE_2_REF

    # bare 조항 → 고시 조항 번호(제1조~제15조)는 고시명 prefix 추가
    m = re.match(r"제(\d+)조", cleaned)
    if m:
        article_num = int(m.group(1))
        if article_num <= 15:
            return f"{_REGULATION_NAME} {cleaned}"
        # 제16조 이상은 다른 법령 조항일 수 있으므로 그대로
        return cleaned

    return cleaned


def _collect_exception_summaries(result, *, allowed: bool | None) -> list[str]:
    seen: list[str] = []
    for item in result.items:
        if allowed is not None and item.allowed is not allowed:
            continue
        text = (item.exception_summary or "").strip()
        if text and text not in seen:
            seen.append(text)
    return seen


def _collect_candidate_laws(*, result) -> list[str]:
    candidate_laws: list[str] = []
    seen: set[str] = set()

    def add(law: str) -> None:
        for cleaned in _normalize_law_refs(law):
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                candidate_laws.append(cleaned)

    for law in getattr(result, "referenced_laws", []) or []:
        add(law)
    for item in getattr(result, "items", []) or []:
        for law in getattr(item, "referenced_laws", []) or []:
            add(law)
    for text in (getattr(result, "limit_rule", "") or "", getattr(result, "rejection_reason", "") or ""):
        for law in _extract_law_mentions(text):
            add(law)
    if result.required_usage_rate is not None:
        add(_SCHEDULE_3_REF)
    if result.exceeded:
        add("제4조")
    return _collapse_parent_laws(candidate_laws)


def _extract_law_mentions(text: str) -> list[str]:
    import re

    if not text:
        return []
    pattern = re.compile(r"(제\d+조(?:제\d+항(?:제\d+호)?)?|별표\s*\d+)")
    refs: list[str] = []
    for match in pattern.finditer(text):
        group = match.group(1)
        if group.startswith("제"):
            refs.append(group.replace(" ", ""))
            continue
        digits = "".join(ch for ch in group if ch.isdigit())
        if digits:
            refs.append(f"별표 {digits}")
    return refs


def _normalize_law_refs(law: str) -> list[str]:
    import re

    cleaned = " ".join((law or "").split()).strip()
    if not cleaned:
        return []

    _ws = re.compile(r"\s+")

    # ── 1순위: 「법령명」 제X조 → 법령명 포함 그대로 보존 ─────────────────────
    bracket_pat = re.compile(r"「([^」]+)」\s*(제\d+조(?:제\d+항(?:제\d+호)?)?)")
    bracket_hits = [
        f"「{m.group(1).strip()}」 {_ws.sub('', m.group(2))}"
        for m in bracket_pat.finditer(cleaned)
    ]
    if bracket_hits:
        return bracket_hits

    # ── 2순위: 법령명(한글) + 제X조 → 법령명 포함 그대로 보존 ─────────────────
    #   예: "산업안전보건법 시행령 제74조", "건설기술진흥법 제62조의3"
    named_pat = re.compile(
        r"([가-힣ㆍ·\s]+(?:법|령|규칙|지침)(?:\s+시행령|\s+시행규칙)?)"
        r"\s+(제\d+조(?:의\d+)?(?:제\d+항(?:제\d+호)?)?)"
    )
    named_hits = [
        f"{m.group(1).strip()} {_ws.sub('', m.group(2))}"
        for m in named_pat.finditer(cleaned)
    ]
    if named_hits:
        return named_hits

    refs = _extract_law_mentions(cleaned)
    if refs:
        return refs
    if cleaned in {_PROGRESS_RULE_LAW, _SCHEDULE_3_REF} or "별표 3" in cleaned:
        return [_SCHEDULE_3_REF]
    if "별표 2" in cleaned:
        return [_SCHEDULE_2_REF]
    split_parts = re.split(r"[|/,]", cleaned)
    normalized = [part.strip() for part in split_parts if part.strip()]
    return normalized or [cleaned]


def _collapse_parent_laws(laws: list[str]) -> list[str]:
    collapsed: list[str] = []
    for law in laws:
        if law.startswith("제") and "항" not in law and "호" not in law:
            if any(other != law and other.startswith(law) for other in laws):
                continue
        collapsed.append(law)
    return collapsed


def _derive_reasoning_summary_for_law(*, result, law: str) -> str:
    category_name = getattr(result, "category", "") or ""
    evidence_for_law: list[str] = []
    reasoning_for_law: list[str] = []
    for item in getattr(result, "items", []) or []:
        normalized_laws = {
            normalized
            for item_law in getattr(item, "referenced_laws", []) or []
            for normalized in _normalize_law_refs(item_law)
        }
        if law not in normalized_laws:
            continue
        # 1순위: evidence_snippets (법령 원문 발췌)
        for snippet in getattr(item, "evidence_snippets", []) or []:
            cleaned = _clean_legal_snippet(snippet)
            if cleaned and cleaned not in evidence_for_law:
                evidence_for_law.append(cleaned)
        # 2순위: reasoning 필드 (LLM verbalization 또는 RDB 법령 텍스트)
        reasoning_raw = " ".join((getattr(item, "reasoning", "") or "").split()).strip()
        if reasoning_raw and len(reasoning_raw) >= 20 and reasoning_raw not in reasoning_for_law:
            reasoning_for_law.append(reasoning_raw)

    # 1순위: evidence_snippets — 실제 법령 텍스트
    if evidence_for_law:
        return evidence_for_law[0]

    # 2순위: reasoning 필드
    # - 순환·공허 표현 제외
    # - 현재 카테고리가 아닌 다른 카테고리명을 언급하는 reasoning은 오염된 것으로 판단 → 제외
    for reasoning in reasoning_for_law:
        if _reasoning_mentions_other_category(reasoning, category_name):
            continue
        # DB raw 포맷 prefix 제거 후 정제
        cleaned = _clean_db_raw_text(reasoning)
        if not cleaned or _is_generic_source_summary(cleaned):
            continue
        if len(cleaned) >= 15:
            return cleaned[:140].rstrip(" .,")

    # 순환 표현 대신 빈 문자열 반환 → 상위 함수에서 처리
    return ""


def _reasoning_mentions_other_category(reasoning: str, current_category: str) -> bool:
    """
    reasoning 텍스트에 현재 카테고리가 아닌 다른 카테고리명이 언급되면
    오염된(cross-category) reasoning으로 판단해 출처 요지로 사용하지 않는다.
    예: 보호구 항목인데 reasoning이 '건강장해예방비 항목으로 사용이 가능함'을 포함
    """
    from src.schemas.classifier import CATEGORIES
    for cat_name in CATEGORIES.values():
        if cat_name == current_category:
            continue
        if cat_name in reasoning:
            return True
    return False


def _clean_db_raw_text(text: str) -> str:
    """
    DB에서 가져온 raw 법령 텍스트의 불필요한 prefix·포맷 문자를 제거한다.
    예: '사 역 가. 안전ㆍ보건관리자의...' → '안전ㆍ보건관리자의...'
        '가. 법 제29조부터...' → '법 제29조부터...'
    """
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    # '사 역 가.' 같은 분류 기호 prefix 제거
    cleaned = re.sub(r"^(사\s+역\s+)?[가나다라마바사아자차카타파하]\.\s*", "", cleaned)
    # 번호 목록 시작부('1)','2)','①' 등) 제거
    cleaned = re.sub(r"^[\d①②③④⑤]+[).]\s*", "", cleaned)
    # 남은 leading 특수문자 제거
    cleaned = cleaned.strip("- *·•")
    return cleaned.strip()


def _best_item_law(*, result) -> str:
    counts: dict[str, tuple[int, int]] = {}
    for item in getattr(result, "items", []) or []:
        for raw_law in getattr(item, "referenced_laws", []) or []:
            for law in _normalize_law_refs(raw_law):
                specificity = 2 if "호" in law else 1 if "항" in law else 0
                current_count, current_specificity = counts.get(law, (0, -1))
                counts[law] = (current_count + 1, max(current_specificity, specificity))
    if not counts:
        return ""
    return sorted(counts.items(), key=lambda kv: (-kv[1][1], -kv[1][0], kv[0]))[0][0]


def _arrange_source_laws(*, result, primary_item_law: str, ordered_laws: list[str]) -> list[str]:
    arranged: list[str] = []
    seen: set[str] = set()

    def add(law: str) -> None:
        if law and law not in seen:
            seen.add(law)
            arranged.append(law)

    if result.required_usage_rate is not None:
        add(_SCHEDULE_3_REF)
    if result.exceeded:
        add("제4조")
    add(primary_item_law)
    for law in ordered_laws:
        add(law)
    return arranged


def _extract_snippet_for_law(*, result, law: str) -> str:
    for item in result.items:
        if law not in (item.referenced_laws or []):
            continue
        for snippet in item.evidence_snippets or []:
            cleaned = _clean_legal_snippet(snippet)
            if cleaned:
                return cleaned
    if law.startswith("제4조") and result.limit_rule:
        cleaned = _clean_legal_snippet(result.limit_rule)
        if cleaned:
            return cleaned
    return ""


def _clean_legal_snippet(text: str) -> str:
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return ""
    if "LEGAL_CITE" in cleaned:
        return ""
    if cleaned.startswith("※"):
        return ""
    cleaned = cleaned.replace("으로 사용 가능한지 으로 사용 가능한지", "으로 사용 가능한지")
    for marker in ("- ", "• ", " "):
        if marker in cleaned:
            parts = [part.strip() for part in cleaned.split(marker) if part.strip()]
            preferred = _prefer_explanatory_part(parts)
            if preferred:
                cleaned = preferred
                break
    cleaned = cleaned.strip(" -")
    if not _is_usable_legal_snippet(cleaned):
        return ""
    if len(cleaned) > 140:
        cleaned = cleaned[:140].rstrip(" ,.") + "..."
    return cleaned


def _prefer_explanatory_part(parts: list[str]) -> str:
    for part in parts:
        if any(keyword in part for keyword in ("해당", "사용", "불가", "가능", "초과", "이상", "이내")):
            return part
    return parts[-1] if parts else ""


def _is_usable_legal_snippet(text: str) -> bool:
    cleaned = " ".join(text.split())
    if len(cleaned) < 18:
        return False
    if cleaned.endswith("..."):
        return False
    if "가능한지" in cleaned:
        return False
    if "|" in cleaned:
        return False
    if cleaned.count("/") >= 2:
        return False
    if cleaned.count("「") >= 2 and len(cleaned) > 100:
        return False
    return True


def _normalize_reason_output(text: str) -> str:
    cleaned = " ".join((text or "").split()).strip()
    if not cleaned:
        return ""
    replacements = (
        ("판단된다.", "판단됩니다."),
        ("판단된다", "판단됩니다"),
        ("확인된다.", "확인됩니다."),
        ("확인된다", "확인됩니다"),
        ("가능하다.", "가능합니다."),
        ("가능하다", "가능합니다"),
        ("가능함.", "가능합니다."),
        ("가능함", "가능합니다"),
        ("불가함.", "불가합니다."),
        ("불가함", "불가합니다"),
        ("불가하다.", "불가합니다."),
        ("불가하다", "불가합니다"),
        ("필요하다.", "필요합니다."),
        ("필요하다", "필요합니다"),
        ("어렵다.", "어렵습니다."),
        ("어렵다", "어렵습니다"),
        ("해당한다.", "해당합니다."),
        ("해당한다", "해당합니다"),
        ("인정된다.", "인정됩니다."),
        ("인정된다", "인정됩니다"),
        ("보인다.", "보입니다."),
        ("보인다", "보입니다"),
        ("있다.", "있습니다."),
        ("없다.", "없습니다."),
    )
    for before, after in replacements:
        cleaned = cleaned.replace(before, after)
    cleaned = cleaned.replace("합니다..", "합니다.")
    cleaned = cleaned.replace("됩니다..", "됩니다.")
    cleaned = cleaned.replace("바랍니다..", "바랍니다.")
    cleaned = _dedupe_reason_sentences(cleaned)
    cleaned = _trim_redundant_review_closer(cleaned)
    cleaned = _qualify_law_refs_in_reason(cleaned)
    return cleaned.strip()


def _dedupe_reason_sentences(text: str) -> str:
    sentences = _split_reason_sentences(text)
    if len(sentences) <= 1:
        return text

    deduped: list[str] = []
    normalized_seen: list[str] = []
    for sentence in sentences:
        normalized = _normalize_reason_sentence(sentence)
        if not normalized:
            continue
        if any(_is_redundant_reason_sentence(normalized, existing) for existing in normalized_seen):
            continue
        normalized_seen.append(normalized)
        deduped.append(sentence.strip())

    result = " ".join(deduped).strip()
    return result or text


def _split_reason_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _normalize_reason_sentence(sentence: str) -> str:
    text = sentence.strip()
    text = re.sub(r"「[^」]+」\s*", "", text)
    text = re.sub(r"\[[^\]]+\]\s*", "", text)
    text = re.sub(r"제\d+조(?:제\d+항(?:제\d+호)?)?", "", text)
    text = re.sub(r"별표\s*\d+", "", text)
    text = re.sub(r"[0-9,]+원", "", text)
    text = re.sub(r"[0-9.]+%", "", text)
    text = text.replace("에 따르면", "")
    text = text.replace("항목은", "")
    text = text.replace("항목", "")
    text = text.replace("현재", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .")


def _is_redundant_reason_sentence(current: str, existing: str) -> bool:
    if current == existing:
        return True
    current_tokens = {token for token in current.split() if len(token) >= 2}
    existing_tokens = {token for token in existing.split() if len(token) >= 2}
    if not current_tokens or not existing_tokens:
        return False
    overlap = len(current_tokens & existing_tokens)
    ratio = overlap / min(len(current_tokens), len(existing_tokens))
    return ratio >= 0.8


def _qualify_law_refs_in_reason(text: str) -> str:
    """
    사유(reason) 텍스트 내 bare 고시 조항(제1조~제15조)에 고시명 prefix를 추가한다.

    이미 법령명이 붙어 있는 패턴(「…」 제X조 / 한글법령명 제X조)은 건드리지 않는다.
    산업안전보건법(제29조~), 건설기술진흥법(제62조~) 등 큰 번호 조항도 건드리지 않는다.

    예:
      "제7조제1항제2호에 따라" → "「건설업 산업안전보건관리비 계상 및 사용기준」 제7조제1항제2호에 따라"
      "산업안전보건법 시행령 제74조" → 그대로 (이미 법령명 포함)
    """
    if not text:
        return text

    # ── 이미 법령명이 있는 패턴 위치 마킹 (이 범위 내 bare 조항은 건드리지 않는다) ──
    _qualified_pat = re.compile(
        r"(?:「[^」]+」|[가-힣ㆍ·]+(?:법|령|규칙|기준|지침)(?:\s+시행령|\s+시행규칙)?)"
        r"\s*제\d+조(?:의\d+)?(?:제\d+항(?:제\d+호)?)?"
    )
    covered: list[tuple[int, int]] = [
        (m.start(), m.end()) for m in _qualified_pat.finditer(text)
    ]

    def _is_covered(start: int) -> bool:
        return any(s <= start < e for s, e in covered)

    # ── bare 조항 치환 ──────────────────────────────────────────────────────────
    _bare_pat = re.compile(r"제(\d+)조(?:의\d+)?(?:제\d+항(?:제\d+호)?)?")
    result: list[str] = []
    prev = 0
    for m in _bare_pat.finditer(text):
        if _is_covered(m.start()):
            continue  # 이미 법령명 포함 — 건드리지 않음
        article_num = int(m.group(1))
        if article_num > 15:
            continue  # 고시 조항이 아닌 것으로 간주 (산안보건법 등)
        result.append(text[prev:m.start()])
        result.append(f"{_REGULATION_NAME} {m.group(0)}")
        prev = m.end()
    result.append(text[prev:])
    return "".join(result)


def _trim_redundant_review_closer(text: str) -> str:
    sentences = _split_reason_sentences(text)
    if len(sentences) < 2:
        return text
    last = sentences[-1]
    previous = " ".join(sentences[:-1])
    if last == "검토가 필요합니다." and any(
        phrase in previous
        for phrase in ("추가 확인이 필요합니다", "소명 자료가 필요합니다", "재검토가 필요합니다", "추가로 확인해야 합니다", "확인해야 합니다")
    ):
        return " ".join(sentences[:-1]).strip()
    return text
