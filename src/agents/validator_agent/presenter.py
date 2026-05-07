from __future__ import annotations

import re

from pydantic import BaseModel, Field

import src.core.llm_config as llm_config
from src.prompts import (
    AUDIT_REASON_SYNTHESIS_PROMPT,
    CATEGORY_REASON_APPROPRIATE_PROMPT,
    CATEGORY_REASON_IMPROPER_PROMPT,
    CATEGORY_REASON_REVIEW_PROMPT,
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

_PROGRESS_RULE_LAW = "별표 3 공사진척에 따른 산업안전보건관리비 사용기준"
_REGULATION_NAME = "「건설업 산업안전보건관리비 계상 및 사용기준」"
_SCHEDULE_2_REF = f"{_REGULATION_NAME} [별표 2]"
_SCHEDULE_3_REF = f"{_REGULATION_NAME} [별표 3]"
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
_CIRCULAR_PHRASES = (
    "허용 범위에 부합합니다",
    "허용 범위에 해당합니다",
    "허용 범위를 벗어납니다",
    "법령 기준에 따라 판단",
    "관련 조항에 따라 판단",
    "추가 확인이 필요합니다",
    "추가 검토가 필요합니다",
)


def _is_meaningful_llm_text(text: str) -> bool:
    """
    LLM이 생성한 interpretation / improvements 텍스트가
    실질적인 내용을 담고 있으면 True, 너무 짧거나 순환 표현이면 False.
    """
    cleaned = " ".join((text or "").split()).strip()
    if len(cleaned) < 25:
        return False
    if any(phrase in cleaned for phrase in _CIRCULAR_PHRASES) and len(cleaned) < 60:
        return False
    return True


_REASON_TEMPLATES: dict[str, str] = {
    "improper_scope_exclusion": (
        "{law_basis} "
        "이번에 검토한 {items} 항목은 관련 기준상 {category_name}에서 허용하는 사용 범위를 벗어나 집행 가능한 대상으로 보기 어렵고, "
        "{schedule_2_ref}의 제외 기준에도 저촉될 수 있어 산안비 집행 대상으로 인정하기 어렵습니다. "
        "따라서 본 건은 부적절하며, 해당 항목을 삭제하거나 현장 안전 확보와 직접 관련된 적정 항목으로 조정한 후 재제출하시기 바랍니다."
    ),
    "improper_limit_exceeded": (
        "{law_basis} "
        "이번에 검토한 {items} 항목은 {category_name} 범위에서 집행이 가능한 항목이나, "
        "현재 누적 집행액 {total}원이 법정 허용 한도 {limit_detail}을 {exceeded}원 초과하였습니다. "
        "집행액을 한도 이내로 조정하거나 초과분을 타 예산으로 전환 처리하시기 바랍니다."
    ),
    "review_progress_shortfall": (
        "{law_basis} "
        "이번에 검토한 {items} 항목은 집행 항목 자체의 적정성에는 특별한 문제가 없는 것으로 보입니다. "
        "현재 공정률 {progress_rate}% 구간에서 요구되는 최소 집행 기준에 비해 누적 사용 실적이 {shortfall}원 부족합니다. "
        "정산 단계에서 집행 적정성 문제가 발생할 수 있으므로, 집행 계획을 점검하고 법정 하한선을 충족하도록 조정하시기 바랍니다."
    ),
    "review_exception_or_conflict": (
        "{law_basis} "
        "이번에 검토한 {items} 항목은 원칙적으로 {category_name}의 허용 범위에 포함될 여지가 있습니다. "
        "단서 조항('{exception_texts}') 또는 상반된 근거의 적용 여부에 따라 판단이 달라질 수 있습니다. "
        "현 단계에서는 일률적으로 적정 또는 부적정으로 단정하기 어렵기 때문에, 예외 요건 충족 여부를 확인할 수 있는 상세 증빙과 사실관계 소명 자료를 추가로 제출해 주시기 바랍니다."
    ),
    "review_insufficient_basis": (
        "{law_basis} "
        "다만 이번에 검토한 {items} 항목은 현행 법령, 질의회시 및 확보된 문맥만으로는 {category_name} 집행 대상으로 직접 인정할 수 있는 허용 근거가 충분히 확인되지 않습니다. "
        "따라서 현재 단계에서는 보수적으로 추가 검토가 필요하며, 실제 사용 목적, 투입 장소, 현장 안전관리와의 직접 관련성을 설명하는 소명 자료를 보완한 후 재검토를 요청하시기 바랍니다."
    ),
    "review_duplicate_cost_risk": (
        "{law_basis} "
        "이번에 검토한 {items} 항목은 {category_name}에서 집행 가능한 항목에 해당할 수 있으나, "
        "공사비 또는 타 비용 항목에 이미 반영되거나 기포함된 항목과 중복/이중 계상되었을 가능성을 배제하기 어렵습니다. "
        "회계 처리의 적정성을 확인하기 위해 해당 비용이 산안비 전용 목적으로 별도 집행되었음을 입증하는 추가 증빙과 비용 구분 근거를 제출하시기 바랍니다."
    ),
    "improper_mixed_items": (
        "{law_basis} "
        "이번에 함께 검토한 항목 중 {allowed_items}는 해당 카테고리에서 집행 가능한 항목으로 볼 수 있습니다. "
        "반면 {disallowed_items}는 현장 안전 확보와의 직접 관련성이 약하거나 제외 대상으로 해석될 여지가 있어 집행 대상으로 보기 어렵습니다. "
        "동일 카테고리 내에 적정 항목과 부적정 항목이 혼재된 경우 현재 신청 형태 그대로는 적정성 인정이 어렵습니다. "
        "부적격 항목을 분리하거나 삭제한 후, 적정 항목만으로 내역을 재구성하여 다시 제출하시기 바랍니다."
    ),
    "appropriate_compliant": (
        "{law_basis} "
        "이번에 검토한 {items} 항목은 {category_name}의 사용 범위에 부합하는 것으로 확인됩니다. "
        "사용 범위, 집행 한도, 공정률 대비 집행 적정성 및 현재 확인 가능한 중복 계상 위험 여부를 함께 검토한 결과, 본 건은 산안비 집행 기준에 부합하며 모든 법적 기준을 충족하는 것으로 판단됩니다. "
        "향후 정산 단계에서도 동일한 결론이 유지될 수 있도록 실제 투입 내역과 증빙 자료의 일치성을 계속 관리하고, 관련 근거를 투명하게 관리해 주시기 바랍니다."
    ),
    "appropriate_progress_compliant": (
        "{law_basis} "
        "이번에 검토한 {items} 항목은 {category_name}의 사용 범위와 직접 관련된 집행으로 확인됩니다. "
        "현재 공정률 {progress_rate}% 구간에서는 누적 사용액이 총액의 {required_usage_rate_pct}% 이상이어야 하며, "
        "실제 누적 집행액 {cumulative_used_amount}원이 이를 충족하므로 공정률 기준상 적정합니다. "
        "따라서 항목 적정성과 공정률 기준을 함께 고려하더라도 본 건은 적정하게 집행된 것으로 판단됩니다."
    ),
}


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
            sources.append(AuditSourceSummary(law=law, summary=summary))
            seen_law_keys.add(law_key)

    if not sources and ordered_laws:
        sources.append(
            AuditSourceSummary(
                law=ordered_laws[0],
                summary=f"{category_name} 판정의 근거 조항입니다.",
            )
        )
    return sources


class _AuditSynthesisOutput(BaseModel):
    reason: str = Field(description="전문 감사 의견 사유 (합니다체, 3단락 이상)")


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
    except RuntimeError:
        return ""

    # ── 판정 유형 코드 (조치 문장 선택 기준) ─────────────────────────────────
    if result.status == "부적절" and getattr(result, "exceeded", False):
        _reason_code = "improper_limit_exceeded"
    elif result.status == "부적절":
        _reason_code = "improper_item_validity"
    elif result.status == "적절" and result.required_usage_rate is not None:
        _reason_code = "appropriate_progress_compliant"
    elif result.status == "적절":
        _reason_code = "appropriate_compliant"
    elif result.status == "검토필요" and result.required_usage_rate is not None:
        _reason_code = "review_progress_shortfall"
    else:
        _reason_code = "review_other"

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
    except Exception:
        return ""

    text = " ".join((output.reason or "").split()).strip()
    if len(text) < 60:
        return ""
    return text


def _build_reason(*, category_name: str, category_code: str, result, base_amount: float, sources: list[AuditSourceSummary]) -> str:
    # ── 1차: 전문 감사 의견 합성 (AUDIT_REASON_SYNTHESIS_PROMPT) ──────────────
    synthesis = _synthesize_reason_with_llm(
        category_name=category_name,
        result=result,
        base_amount=base_amount,
        sources=sources,
    )
    if synthesis:
        return _normalize_reason_output(synthesis)

    # ── 폴백: 기존 경로 (reason_code 기반 LLM → 템플릿) ─────────────────────
    disallowed_items = [item.item for item in result.items if not item.allowed]
    allowed_items = [item.item for item in result.items if item.allowed]
    exception_texts = _collect_exception_summaries(result, allowed=None)

    reason_code = _classify_reason_code(
        result=result,
        disallowed_items=disallowed_items,
        allowed_items=allowed_items,
        exception_texts=exception_texts,
    )
    primary_law = sources[0].law if sources else ""
    law_ref = _format_law_reference(primary_law)
    law_basis = _compose_reason_law_basis(
        reason_code=reason_code,
        sources=sources,
        primary_law=primary_law,
        law_ref=law_ref,
        category_name=category_name,
        result=result,
    )
    if reason_code == "improper_mixed_items":
        return _normalize_reason_output(_build_emergency_reason_fallback(
            reason_code=reason_code,
            result=result,
            category_name=category_name,
            law_basis=law_basis,
            disallowed_items=disallowed_items,
            allowed_items=allowed_items,
        ))
    llm_reason = _generate_reason_with_llm(
        reason_code=reason_code,
        result=result,
        base_amount=base_amount,
        category_name=category_name,
        law_basis=law_basis,
        sources=sources,
        disallowed_items=disallowed_items,
        allowed_items=allowed_items,
    )
    if llm_reason:
        return _normalize_reason_output(llm_reason)

    return _normalize_reason_output(_build_emergency_reason_fallback(
        reason_code=reason_code,
        result=result,
        category_name=category_name,
        law_basis=law_basis,
        disallowed_items=disallowed_items,
        allowed_items=allowed_items,
    ))


def _classify_reason_code(*, result, disallowed_items: list[str], allowed_items: list[str], exception_texts: list[str]) -> str:
    rejection = (result.rejection_reason or "").strip()
    if result.status == "부적절" and result.exceeded and result.limit is not None:
        return "improper_limit_exceeded"
    if result.status == "부적절" and disallowed_items:
        return "improper_mixed_items" if allowed_items else "improper_scope_exclusion"
    if result.status == "검토필요" and result.required_usage_rate is not None and result.usage_shortfall_amount is not None:
        return "review_progress_shortfall"
    if result.status == "검토필요" and any(keyword in rejection for keyword in _DUPLICATE_REVIEW_KEYWORDS):
        return "review_duplicate_cost_risk"
    if result.status == "검토필요" and (
        exception_texts
        or "충돌" in rejection
        or "예외 조건" in rejection
    ):
        return "review_exception_or_conflict"
    if result.status == "검토필요" and (
        "직접적인 허용 근거가 충분히 확인되지 않은 항목" in rejection
        or "직접 근거 미확인" in rejection
        or "허용 근거가 충분히 확인되지" in rejection
    ):
        return "review_insufficient_basis"
    if result.status == "검토필요":
        return "review_insufficient_basis"
    if result.status == "적절" and result.required_usage_rate is not None:
        return "appropriate_progress_compliant"
    return "appropriate_compliant"


def _apply_reason_template(
    *,
    reason_code: str,
    result,
    base_amount: float = 0.0,
    category_name: str = "해당 카테고리",
    law_ref: str,
    law_basis: str,
    disallowed_items: list[str],
    allowed_items: list[str],
    exception_texts: list[str],
) -> str:
    template = _REASON_TEMPLATES.get(reason_code)
    if not template:
        return result.rejection_reason or "추가 검토가 필요합니다."

    exceeded_amount = max(0.0, result.total - result.limit) if result.limit is not None else 0.0
    items_text = _join_items(disallowed_items if disallowed_items else allowed_items) or "해당 항목"
    exception_text = _join_exception_texts(exception_texts)
    allowed_items_text = _join_items(allowed_items)
    disallowed_items_text = _join_items(disallowed_items)

    # 한도 초과 케이스: "7,500,000원(산안비 총액 50,000,000원의 15%)" 형태로 조합
    if result.limit is not None and base_amount:
        limit_pct_calc = result.limit / base_amount * 100
        limit_detail = f"{result.limit:,.0f}원(산안비 총액 {base_amount:,.0f}원의 {limit_pct_calc:.0f}%)"
    elif result.limit is not None:
        limit_detail = f"{result.limit:,.0f}원"
    else:
        limit_detail = ""

    template_vars = {
        "law_ref": law_ref,
        "law_basis": law_basis,
        "law_basis_prefix": f"{law_basis} " if law_basis else "",
        "law_basis_with_and": f"{law_basis} 및 " if law_basis else "",
        "category_name": category_name,
        "items": items_text,
        "items_subject": f"이번에 검토한 {items_text} 항목은" if items_text and items_text != "해당 항목" else "이번에 검토한 항목은",
        "allowed_items": allowed_items_text,
        "disallowed_items": disallowed_items_text,
        "total": f"{result.total:,.0f}",
        "limit": f"{result.limit:,.0f}" if result.limit is not None else "",
        "exceeded": f"{exceeded_amount:,.0f}",
        "progress_rate": f"{result.progress_rate:.1f}" if result.progress_rate is not None else "",
        "required_usage_rate_pct": f"{result.required_usage_rate * 100:.0f}" if result.required_usage_rate is not None else "",
        "cumulative_used_amount": f"{result.cumulative_used_amount:,.0f}" if getattr(result, 'cumulative_used_amount', None) is not None else "",
        "shortfall": f"{result.usage_shortfall_amount:,.0f}" if result.usage_shortfall_amount is not None else "",
        "exception_texts": exception_text,
        "items_clause": f"{items_text} 등은 " if items_text and items_text != "해당 항목" else "",
        "schedule_2_ref": _SCHEDULE_2_REF,
        "schedule_3_ref": _SCHEDULE_3_REF,
        "limit_detail": limit_detail,
    }

    try:
        filled = template.format(**template_vars)
    except KeyError:
        filled = template

    filled = " ".join(filled.strip().split())
    supplement = _build_secondary_issue_note(
        reason_code=reason_code,
        result=result,
        disallowed_items=disallowed_items,
        allowed_items=allowed_items,
    )
    if supplement:
        filled = f"{filled} {supplement}".strip()
    return filled


def _humanize_category_name(result, *, allowed_items: list[str], disallowed_items: list[str]) -> str:
    if allowed_items or disallowed_items:
        return "해당 카테고리"
    return "해당 카테고리"


def _compose_law_basis(*, sources: list[AuditSourceSummary], primary_law: str, law_ref: str) -> str:
    if not sources:
        return law_ref

    groups = _group_sources_by_summary(sources)
    pieces: list[str] = []
    for idx, (laws, summary) in enumerate(groups):
        source_ref = _format_law_reference_list(laws)
        piece = f"{source_ref} {summary}".strip() if source_ref and summary else source_ref or summary
        if not piece:
            continue
        if idx > 0:
            piece = f"또한 {piece}"
        pieces.append(piece)
        if len(pieces) >= 2:
            break

    combined = " ".join(piece.strip() for piece in pieces if piece.strip()).strip()
    if combined and not combined.endswith("."):
        combined += "."
    return combined


def _compose_reason_law_basis(
    *,
    reason_code: str,
    sources: list[AuditSourceSummary],
    primary_law: str,
    law_ref: str,
    category_name: str,
    result,
) -> str:
    if reason_code == "improper_limit_exceeded":
        limit_basis = _compose_selected_law_basis(
            sources=sources,
            preferred_laws=("제4조", "별표 3"),
            fallback_law_ref=law_ref,
            only_preferred=True,
        )
        if limit_basis and "제4조" in limit_basis:
            return limit_basis
        if result.limit is not None:
            manual_limit_basis = f"{_format_law_reference('제4조')} {_build_limit_rule_summary(result=result)}".strip()
            if manual_limit_basis and not manual_limit_basis.endswith("."):
                manual_limit_basis += "."
            return manual_limit_basis
        if limit_basis:
            return limit_basis
    if reason_code in ("review_progress_shortfall", "appropriate_progress_compliant"):
        progress_basis = _compose_selected_law_basis(
            sources=sources,
            preferred_laws=(_SCHEDULE_3_REF, "별표 3"),
            fallback_law_ref=law_ref,
            only_preferred=True,
        )
        if progress_basis:
            return progress_basis
    if reason_code in {
        "improper_mixed_items",
        "review_duplicate_cost_risk",
        "appropriate_compliant",
    }:
        filtered_sources = sources
        if not result.exceeded:
            filtered_sources = [source for source in sources if not (source.law or "").startswith("제4조")] or sources
        primary_basis = _compose_selected_law_basis(
            sources=filtered_sources[:1],
            preferred_laws=(),
            fallback_law_ref=law_ref,
            only_preferred=False,
        )
        if primary_basis:
            return primary_basis
    if reason_code == "improper_scope_exclusion":
        filtered_sources = [
            source
            for source in sources
            if not (
                _law_key(source.law) == "schedule_2"
                and _is_generic_source_summary((source.summary or "").strip())
            )
        ] or sources
        primary_basis = _compose_selected_law_basis(
            sources=filtered_sources[:1],
            preferred_laws=(),
            fallback_law_ref=law_ref,
            only_preferred=False,
        )
        if primary_basis:
            return primary_basis
    return _compose_law_basis(
        sources=sources,
        primary_law=primary_law,
        law_ref=law_ref,
    )


def _build_secondary_issue_note(
    *,
    reason_code: str,
    result,
    disallowed_items: list[str],
    allowed_items: list[str],
) -> str:
    if result.status == "적절":
        return ""
    notes: list[str] = []
    if (
        reason_code != "improper_limit_exceeded"
        and result.exceeded
        and result.limit is not None
    ):
        exceeded_amount = max(0.0, result.total - result.limit)
        if exceeded_amount > 0:
            notes.append(
                f"또한 {_format_law_reference('제4조')} 누적 집행액 {result.total:,.0f}원은 "
                f"법정 한도 {result.limit:,.0f}원을 {exceeded_amount:,.0f}원 초과하고 있어, 한도 기준에서도 추가 조정이 필요합니다."
            )
    if (
        reason_code != "review_progress_shortfall"
        and result.required_usage_rate is not None
        and result.usage_shortfall_amount is not None
    ):
        if result.usage_shortfall_amount > 0:
            notes.append(
                f"또한 {_format_law_reference('별표 3')} 현재 공정률 {result.progress_rate:.1f}% 구간의 최소 집행 기준에 비해 "
                f"{result.usage_shortfall_amount:,.0f}원이 부족하므로, 공정률 기준도 함께 보완해야 합니다."
            )
    if (
        reason_code not in {"improper_scope_exclusion", "improper_mixed_items"}
        and disallowed_items
    ):
        disallowed_text = _join_items(disallowed_items)
        if disallowed_text:
            notes.append(
                f"또한 {_format_law_reference('제7조')} {disallowed_text} 항목은 현장 안전 확보와의 직접 관련성이 약하거나 제외 대상으로 해석될 여지가 있어 별도 정리가 필요합니다."
            )
    if (
        reason_code not in {"review_insufficient_basis", "review_exception_or_conflict"}
        and result.status == "검토필요"
        and "직접적인 허용 근거가 충분히 확인되지 않은 항목" in (result.rejection_reason or "")
    ):
        unresolved = _extract_reason_items(result.rejection_reason or "")
        if unresolved:
            notes.append(
                f"아울러 {unresolved} 항목은 직접적인 허용 근거가 충분히 확인되지 않아, 사용 목적과 투입 장소를 뒷받침하는 추가 소명 자료가 필요합니다."
            )
    return " ".join(note.strip() for note in notes if note.strip()).strip()


def _compose_selected_law_basis(
    *,
    sources: list[AuditSourceSummary],
    preferred_laws: tuple[str, ...],
    fallback_law_ref: str,
    only_preferred: bool = False,
) -> str:
    selected_sources: list[AuditSourceSummary] = []
    used_law_keys: set[str] = set()
    for preferred in preferred_laws:
        for source in sources:
            law = (source.law or "").strip()
            if not law:
                continue
            if preferred in law:
                law_key = _law_key(law)
                if law_key not in used_law_keys:
                    selected_sources.append(source)
                    used_law_keys.add(law_key)
                break
    if not only_preferred:
        for source in sources:
            law_key = _law_key(source.law)
            if law_key in used_law_keys:
                continue
            selected_sources.append(source)
            if len(selected_sources) >= 2:
                break
    groups = _group_sources_by_summary(selected_sources)
    pieces: list[str] = []
    for idx, (laws, summary) in enumerate(groups):
        source_ref = _format_law_reference_list(laws)
        piece = f"{source_ref} {summary}".strip() if source_ref and summary else source_ref or summary
        if not piece:
            continue
        if idx > 0:
            piece = f"또한 {piece}"
        pieces.append(piece)
    combined = " ".join(piece.strip() for piece in pieces if piece.strip()).strip()
    if not combined:
        combined = fallback_law_ref
    if combined and not combined.endswith("."):
        combined += "."
    return combined


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


def _compose_source_piece(source: AuditSourceSummary, *, lead: str = "") -> str:
    source_ref = _format_law_reference(source.law)
    summary = (source.summary or "").strip()
    if source_ref and summary:
        return f"{lead}{source_ref} {summary}".strip()
    if source_ref:
        return f"{lead}{source_ref}".strip()
    if summary:
        return f"{lead}{summary}".strip()
    return ""


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


def _format_law_reference_list(laws: list[str]) -> str:
    filtered = [law for law in laws if law]
    if not filtered:
        return ""
    deduped: list[str] = []
    seen: set[str] = set()
    for law in filtered:
        if law not in seen:
            seen.add(law)
            deduped.append(law)
    formatted = [_format_law_label(law) for law in deduped]
    if len(formatted) == 1:
        return f"{_REGULATION_NAME} {formatted[0]}에 따르면"
    joined = ", ".join(formatted)
    return f"{_REGULATION_NAME} {joined}에 따르면"


def _format_law_label(law: str) -> str:
    cleaned = (law or "").strip()
    if not cleaned:
        return ""
    if cleaned.startswith(_REGULATION_NAME):
        cleaned = cleaned[len(_REGULATION_NAME):].strip()
    if cleaned in {_PROGRESS_RULE_LAW, _SCHEDULE_3_REF} or cleaned.startswith("별표 3"):
        return "[별표 3]"
    if cleaned.startswith("별표 2"):
        return "[별표 2]"
    return cleaned


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


def _build_limit_rule_summary(*, result) -> str:
    text = str(result.limit_rule or "")
    if not text:
        return ""
    cleaned = _clean_legal_snippet(text)
    if cleaned:
        return cleaned
    return ""


def _is_generic_source_summary(summary: str) -> bool:
    generic_phrases = (
        "참고합니다",
        "함께 확인합니다",
        "함께 참고합니다",
        "함께 적용합니다",
        "판단의 근거 조항입니다",
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


def _join_items(items: list[str], limit: int | None = None) -> str:
    filtered = [item for item in items if item]
    if not filtered:
        return ""
    selected = filtered if limit is None else filtered[:limit]
    suffix = ""
    remaining = len(filtered) - len(selected)
    if remaining > 0:
        suffix = f" 외 {remaining}건"
    return ", ".join(selected) + suffix


def _label_items(items: list[str], label: str, limit: int | None = None) -> str:
    filtered = [item for item in items if item]
    if not filtered:
        return ""
    selected = filtered if limit is None else filtered[:limit]
    suffix = ""
    remaining = len(filtered) - len(selected)
    if remaining > 0:
        suffix = f" 외 {remaining}건"
    return ", ".join(f"{item}({label})" for item in selected) + suffix


def _join_exception_texts(exception_texts: list[str]) -> str:
    selected = [text for text in exception_texts[:2] if text]
    if not selected:
        return "적용 범위가 상충하는 단서 조항"
    return " / ".join(selected)


def _extract_reason_items(text: str) -> str:
    if ":" not in text:
        return ""
    return text.split(":", 1)[1].strip()


def _format_law_reference(law: str) -> str:
    cleaned = (law or "").strip()
    if not cleaned:
        return ""
    if cleaned.startswith(_REGULATION_NAME):
        if cleaned.endswith("에 따르면"):
            return cleaned
        return f"{cleaned}에 따르면"
    if cleaned in {_PROGRESS_RULE_LAW, _SCHEDULE_3_REF} or cleaned.startswith("별표 3"):
        return f"{_SCHEDULE_3_REF}에 따르면"
    if cleaned.startswith("별표 2"):
        return f"{_SCHEDULE_2_REF}에 따르면"
    return f"{_REGULATION_NAME} {cleaned}에 따르면"


def _law_key(law: str) -> str:
    cleaned = (law or "").strip()
    if cleaned in {_PROGRESS_RULE_LAW, _SCHEDULE_3_REF} or cleaned.startswith("별표 3"):
        return "schedule_3"
    if cleaned.startswith("별표 2"):
        return "schedule_2"
    return cleaned


def _has_final_consonant(char: str) -> bool:
    if not char or not ("가" <= char <= "힣"):
        return False
    code = ord(char) - ord("가")
    return (code % 28) != 0


def _collect_exception_summaries(result, *, allowed: bool | None) -> list[str]:
    seen: list[str] = []
    for item in result.items:
        if allowed is not None and item.allowed is not allowed:
            continue
        text = (item.exception_summary or "").strip()
        if text and text not in seen:
            seen.append(text)
    return seen


def _derive_primary_summary(*, primary_law: str, category_name: str, result) -> str:
    raw = _extract_snippet_for_law(result=result, law=primary_law)
    if raw:
        return raw
    return f"{category_name}의 집행 가능 범위와 직접 관련된 조항입니다."


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
    allowed_items: list[str] = []
    disallowed_items: list[str] = []
    evidence_for_law: list[str] = []
    for item in getattr(result, "items", []) or []:
        normalized_laws = {
            normalized
            for item_law in getattr(item, "referenced_laws", []) or []
            for normalized in _normalize_law_refs(item_law)
        }
        if law not in normalized_laws:
            continue
        if item.allowed:
            allowed_items.append(item.item)
        else:
            disallowed_items.append(item.item)
        # 해당 조항과 연결된 항목의 evidence_snippets 수집
        for snippet in getattr(item, "evidence_snippets", []) or []:
            cleaned = _clean_legal_snippet(snippet)
            if cleaned and cleaned not in evidence_for_law:
                evidence_for_law.append(cleaned)

    # 실제 법령 텍스트가 있으면 우선 사용 — 순환 표현 방지
    if evidence_for_law:
        return evidence_for_law[0]

    # fallback: 허용/불허 항목 기반 요약
    allowed_text = _join_items(allowed_items)
    disallowed_text = _join_items(disallowed_items)
    if allowed_text and disallowed_text:
        return f"{allowed_text} 항목은 해당 조항상 허용 범위에 부합하지만, {disallowed_text} 항목은 허용 범위를 벗어납니다."
    if allowed_text:
        return f"{allowed_text} 항목은 해당 조항상 허용 범위에 부합합니다."
    if disallowed_text:
        return f"{disallowed_text} 항목은 해당 조항상 허용 범위를 벗어납니다."
    return ""


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


def _build_hard_number_sentence(
    *,
    reason_code: str,
    result,
    base_amount: float,
    category_name: str,
) -> str:
    """
    수치가 판정의 핵심인 케이스에서 금액·비율·공정률 팩트를 한 문장으로 확정.
    LLM interpretation 앞에 삽입하여 맥락을 고정한다.
    """
    if reason_code == "improper_limit_exceeded" and result.limit is not None:
        exceeded_amount = max(0.0, result.total - result.limit)
        if base_amount:
            limit_pct_calc = result.limit / base_amount * 100
            limit_detail = f"{result.limit:,.0f}원(산안비 총액의 {limit_pct_calc:.0f}%)"
        else:
            limit_detail = f"{result.limit:,.0f}원"
        return (
            f"현재 누적 집행액 {result.total:,.0f}원이 법정 허용 한도 {limit_detail}을 "
            f"{exceeded_amount:,.0f}원 초과하였습니다."
        )
    if (
        reason_code == "review_progress_shortfall"
        and result.progress_rate is not None
        and result.usage_shortfall_amount is not None
    ):
        return (
            f"현재 공정률 {result.progress_rate:.1f}% 구간에서 요구되는 최소 집행 기준에 비해 "
            f"누적 사용 실적이 {result.usage_shortfall_amount:,.0f}원 부족합니다."
        )
    if (
        reason_code == "appropriate_progress_compliant"
        and getattr(result, "cumulative_used_amount", None) is not None
        and result.progress_rate is not None
    ):
        return (
            f"실제 누적 집행액 {result.cumulative_used_amount:,.0f}원이 "
            f"공정률 {result.progress_rate:.1f}% 구간의 법정 하한선을 충족합니다."
        )
    return ""


def _get_template_action_suffix(*, reason_code: str, disallowed_items: list[str]) -> str:
    """
    LLM improvements가 없거나 부실할 때 폴백으로 사용할
    reason_code별 마무리 행동 지침 문장.
    """
    _ACTIONS: dict[str, str] = {
        "improper_scope_exclusion": (
            "해당 항목을 삭제하거나 현장 안전 확보와 직접 관련된 적정 항목으로 교체한 후 재제출하시기 바랍니다."
        ),
        "improper_limit_exceeded": (
            "집행액을 한도 이내로 조정하거나 초과분을 타 예산으로 전환 처리하시기 바랍니다."
        ),
        "review_progress_shortfall": (
            "집행 계획을 점검하고 법정 하한선을 충족하도록 조정하시기 바랍니다."
        ),
        "review_exception_or_conflict": (
            "예외 요건 충족 여부를 확인할 수 있는 상세 증빙과 사실관계 소명 자료를 추가로 제출해 주시기 바랍니다."
        ),
        "review_insufficient_basis": (
            "실제 사용 목적, 투입 장소, 현장 안전관리와의 직접 관련성을 설명하는 소명 자료를 보완한 후 재검토를 요청하시기 바랍니다."
        ),
        "review_duplicate_cost_risk": (
            "해당 비용이 산안비 전용 목적으로 별도 집행되었음을 입증하는 추가 증빙과 비용 구분 근거를 제출하시기 바랍니다."
        ),
        "improper_mixed_items": (
            "해당 항목을 삭제한 후 적정 항목만으로 내역을 재구성하여 다시 제출하시기 바랍니다."
        ),
    }
    return _ACTIONS.get(reason_code, "")


def _generate_reason_with_llm(
    *,
    reason_code: str,
    result,
    base_amount: float,
    category_name: str,
    law_basis: str,
    sources: list[AuditSourceSummary],
    disallowed_items: list[str],
    allowed_items: list[str],
) -> str:
    interpretation = " ".join((result.llm_interpretation or "").split()).strip()
    improvements = " ".join((result.llm_improvements or "").split()).strip()
    if reason_code == "improper_mixed_items":
        return ""
    try:
        llm = llm_config.get()
    except RuntimeError:
        return ""

    if result.status == "적절":
        action = "향후 정산 단계에서도 관련 증빙과 집행 근거를 일관되게 관리해 주시기 바랍니다."
    elif result.status == "부적절":
        action = _get_template_action_suffix(
            reason_code=reason_code,
            disallowed_items=disallowed_items,
        )
    elif reason_code == "review_progress_shortfall":
        action = _get_template_action_suffix(
            reason_code=reason_code,
            disallowed_items=disallowed_items,
        )
    else:
        action = improvements if _is_meaningful_llm_text(improvements) else _get_template_action_suffix(
            reason_code=reason_code,
            disallowed_items=disallowed_items,
        )
    facts = _build_reason_facts(
        reason_code=reason_code,
        result=result,
        base_amount=base_amount,
    )
    law_basis_for_writer = _build_llm_reason_law_basis(
        reason_code=reason_code,
        law_basis=law_basis,
        sources=sources,
    )
    items_text = _join_items(_all_item_names(result)) or "해당 항목"
    prompt = _select_reason_prompt(result.status)
    try:
        rendered = (prompt | llm).invoke(
            {
                "status": result.status,
                "reason_code": reason_code,
                "category": category_name,
                "items": items_text,
                "allowed_items": _join_items(allowed_items) or "(없음)",
                "disallowed_items": _join_items(disallowed_items) or "(없음)",
                "law_basis": law_basis_for_writer,
                "facts": facts,
                "action_hint": action,
            }
        )
    except Exception:
        return ""

    text = getattr(rendered, "content", rendered)
    text = " ".join(str(text or "").split()).strip()
    text = re.sub(r"^출력:\s*", "", text)
    if _llm_reason_conflicts_with_result(
        reason_code=reason_code,
        result=result,
        interpretation=text,
        improvements="",
    ):
        return ""
    return text


def _select_reason_prompt(status: str):
    if status == "부적절":
        return CATEGORY_REASON_IMPROPER_PROMPT
    if status == "검토필요":
        return CATEGORY_REASON_REVIEW_PROMPT
    return CATEGORY_REASON_APPROPRIATE_PROMPT


def _all_item_names(result) -> list[str]:
    names: list[str] = []
    for item in getattr(result, "items", []) or []:
        name = getattr(item, "item", "")
        if name and name not in names:
            names.append(name)
    return names


def _build_llm_reason_law_basis(
    *,
    reason_code: str,
    law_basis: str,
    sources: list[AuditSourceSummary],
) -> str:
    selected_laws: list[str] = []

    def add_law(law: str) -> None:
        cleaned = (law or "").strip()
        if cleaned and cleaned not in selected_laws:
            selected_laws.append(cleaned)

    if reason_code == "improper_mixed_items":
        for source in sources[:1]:
            add_law(source.law)
        return _format_law_reference_list(selected_laws) or law_basis
    if reason_code == "improper_limit_exceeded":
        for source in sources:
            law = (source.law or "").strip()
            if law.startswith("제4조"):
                add_law(law)
                break
        for source in sources:
            law = (source.law or "").strip()
            if law and not law.startswith("제4조"):
                add_law(law)
                break
        return _format_law_reference_list(selected_laws) or law_basis
    if reason_code in {"review_progress_shortfall", "appropriate_progress_compliant"}:
        for source in sources:
            law = (source.law or "").strip()
            if "별표 3" in law:
                add_law(law)
                break
        for source in sources:
            law = (source.law or "").strip()
            if law and "별표 3" not in law:
                add_law(law)
                break
        return _format_law_reference_list(selected_laws) or law_basis
    for source in sources[:2]:
        law = (source.law or "").strip()
        if law:
            add_law(law)
    return _format_law_reference_list(selected_laws) or law_basis


def _llm_reason_conflicts_with_result(
    *,
    reason_code: str,
    result,
    interpretation: str,
    improvements: str,
) -> bool:
    text = f"{interpretation} {improvements}".strip()
    if not text:
        return False
    if reason_code in {"appropriate_compliant", "appropriate_progress_compliant"}:
        if any(keyword in text for keyword in ("초과", "부족", "부적절", "불가", "제외", "검토", "추가 검토", "확인이 필요", "예외")):
            return True
    if reason_code.startswith("improper_"):
        if any(
            keyword in text
            for keyword in (
                "검토가 필요",
                "추가 검토",
                "추가 확인이 필요",
                "추가로 확인",
                "확인해야",
                "확인할 필요",
                "판단이 달라질 수",
                "단정하기 어렵",
            )
        ):
            return True
        if "부적절" not in text and "인정하기 어렵" not in text and "집행 대상으로 보기 어렵" not in text:
            return True
    if reason_code.startswith("review_"):
        if any(keyword in text for keyword in ("부적절합니다", "승인이 불가", "인정하기 어렵습니다")):
            return True
        if "검토" not in text and "확인" not in text and "소명" not in text:
            return True
    if reason_code == "improper_limit_exceeded" and "초과" not in text and "한도" not in text:
        return True
    if reason_code == "review_progress_shortfall" and "부족" not in text and "공정률" not in text:
        return True
    if reason_code == "improper_mixed_items":
        sentences = _split_reason_sentences(text)
        if len(sentences) > 3:
            return True
        if "반면" not in text and "혼재" not in text and "함께 포함" not in text:
            return True
        if len(sentences) >= 2:
            later_text = " ".join(sentences[1:])
            item_names = [item.item for item in getattr(result, "items", []) or [] if getattr(item, "item", "")]
            repeated_names = sum(1 for name in item_names if name and name in later_text)
            if repeated_names >= 2:
                return True
    return False


def _build_reason_facts(*, reason_code: str, result, base_amount: float) -> str:
    parts: list[str] = []
    number_sentence = _build_hard_number_sentence(
        reason_code=reason_code,
        result=result,
        base_amount=base_amount,
        category_name="",
    )
    if number_sentence:
        parts.append(number_sentence)
    if reason_code == "review_duplicate_cost_risk":
        parts.append("해당 항목 자체의 허용 근거는 확인되었으나 공사비 또는 타 비용과의 중복 계상 가능성이 남아 있습니다.")
    return " ".join(parts).strip() or "특이 수치 사실 없음"


def _build_reason_decision_basis(*, reason_code: str, result, allowed_items: list[str], disallowed_items: list[str]) -> str:
    if reason_code == "improper_mixed_items":
        allowed_text = _join_items(allowed_items)
        disallowed_text = _join_items(disallowed_items)
        return (
            f"{allowed_text}는 집행 가능한 항목으로 볼 수 있으나, "
            f"{disallowed_text}는 같은 카테고리에서 인정하기 어려워 동일 신청 건으로는 승인하기 어렵습니다."
        ).strip()
    if reason_code == "improper_scope_exclusion":
        return "대상 항목은 현장 안전 확보와 직접 관련된 비용으로 보기 어려워 해당 카테고리 사용 범위를 벗어납니다."
    if reason_code == "improper_limit_exceeded":
        return "항목 자체의 집행 가능성과 별개로, 법정 한도 위반이 확인되어 초과 범위는 인정하기 어렵습니다."
    if reason_code == "review_progress_shortfall":
        return "항목 자체의 적정성과 별개로 공정률 구간별 최소 집행 기준 미달 여부가 핵심 쟁점입니다."
    if reason_code == "review_exception_or_conflict":
        return "원칙적 허용 가능성은 있으나 예외 조항 또는 상반된 근거의 적용 여부를 추가로 확인해야 합니다."
    if reason_code == "review_insufficient_basis":
        return "직접적인 허용 근거가 충분히 확인되지 않아 보수적으로 추가 검토가 필요합니다."
    if reason_code == "review_duplicate_cost_risk":
        return "허용 가능성은 확인되지만 공사비 또는 타 예산과의 중복 계상 여부가 정리되지 않았습니다."
    if reason_code == "appropriate_progress_compliant":
        return "항목 적정성과 공정률 기준 충족 여부가 모두 확인되었습니다."
    if reason_code == "appropriate_compliant":
        return "허용 범위와 현재 확인 가능한 제한 요소를 함께 검토한 결과 특이사항이 없습니다."
    return ""


def _build_reason_issue_summary(*, reason_code: str, result, allowed_items: list[str], disallowed_items: list[str]) -> str:
    if reason_code == "improper_mixed_items":
        return "허용 항목과 부적정 항목이 혼재되어 있습니다."
    if reason_code == "improper_scope_exclusion":
        return "사용 범위를 벗어난 항목입니다."
    if reason_code == "improper_limit_exceeded" and result.limit is not None:
        return f"법정 허용 한도 {result.limit:,.0f}원을 초과했습니다."
    if reason_code == "review_progress_shortfall" and result.usage_shortfall_amount is not None:
        return f"공정률 기준 대비 {result.usage_shortfall_amount:,.0f}원이 부족합니다."
    if reason_code == "review_exception_or_conflict":
        return "예외 조항 또는 근거 충돌이 있습니다."
    if reason_code == "review_insufficient_basis":
        return "직접적인 허용 근거가 부족합니다."
    if reason_code == "review_duplicate_cost_risk":
        return "중복 계상 가능성이 남아 있습니다."
    if reason_code == "appropriate_progress_compliant":
        return "공정률 기준을 충족합니다."
    if reason_code == "appropriate_compliant":
        return "특이한 제한 요소가 확인되지 않았습니다."
    return ""


def _build_reason_writer_guidance(*, reason_code: str, result, allowed_items: list[str], disallowed_items: list[str]) -> str:
    guidance_map = {
        "improper_scope_exclusion": "항목명으로 바로 시작하고, 대상 항목이 왜 허용 범위를 벗어나는지 명확히 설명한 뒤 부적절 결론과 재제출 조치를 적으십시오.",
        "improper_limit_exceeded": "항목명으로 시작하고, 집행 가능성은 짧게 언급하되 한도 위반이 핵심이라는 점을 중심으로 부적절 결론을 적으십시오.",
        "review_progress_shortfall": "항목명으로 시작하고, 항목 자체의 적정성과 공정률 기준 미달을 분리해서 설명한 뒤 검토필요로 마무리하십시오.",
        "review_exception_or_conflict": "항목명으로 시작하고, 원칙적 허용 가능성과 예외/충돌 가능성을 함께 설명하되 부적절로 단정하지 말고 검토필요로 마무리하십시오.",
        "review_insufficient_basis": "항목명으로 시작하고, 직접 근거가 부족하다는 점을 핵심으로 설명한 뒤 필요한 소명 자료를 적으십시오.",
        "review_duplicate_cost_risk": "항목명으로 시작하고, 허용 가능성보다 중복 계상 확인이 필요하다는 점을 중심으로 검토필요 흐름을 유지하십시오.",
        "improper_mixed_items": "3~4문장으로 작성하고, 첫 문장은 법령 근거, 둘째 문장은 허용 항목과 부적절 항목의 대비, 셋째 문장은 혼재 상태 때문에 같은 신청 건으로는 인정하기 어렵다는 결론, 필요 시 마지막 문장은 조치만 적으십시오. 같은 의미를 반복하지 마십시오.",
        "appropriate_compliant": "항목명으로 시작하고, 허용 범위와 확인된 기준 충족 사실을 차분히 설명한 뒤 적절하다고 마무리하십시오.",
        "appropriate_progress_compliant": "항목명으로 시작하고, 항목 적정성과 공정률 기준 충족을 함께 설명하되 부족/초과 같은 부정 표현은 넣지 마십시오.",
    }
    guidance = guidance_map.get(reason_code, "주어진 사실과 법령을 바탕으로 자연스럽고 전문적으로 작성하십시오.")
    if reason_code == "improper_mixed_items" and allowed_items and disallowed_items:
        guidance += f" 허용 항목은 {', '.join(allowed_items)}이고, 부적절 항목은 {', '.join(disallowed_items)}입니다."
    return guidance


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
    return cleaned.strip()


def _default_reason(category_name: str, status: str) -> str:
    if status == "적절":
        return f"{category_name} 항목은 현재 기준상 적정하게 집행되었습니다."
    if status == "부적절":
        return f"{category_name} 항목은 법령 기준에 맞지 않아 부적절합니다."
    return f"{category_name} 항목은 추가 검토가 필요합니다."


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


def _build_emergency_reason_fallback(
    *,
    reason_code: str,
    result,
    category_name: str,
    law_basis: str,
    disallowed_items: list[str],
    allowed_items: list[str],
) -> str:
    items_text = _join_items(disallowed_items if disallowed_items else allowed_items) or "해당 항목"
    action = _get_template_action_suffix(reason_code=reason_code, disallowed_items=disallowed_items)
    facts = _build_reason_facts(reason_code=reason_code, result=result, base_amount=0.0)
    subject = _build_reason_subject(
        reason_code=reason_code,
        items_text=items_text,
        allowed_items=allowed_items,
        disallowed_items=disallowed_items,
    )
    if result.status == "적절":
        return (
            f"{law_basis} {subject} {category_name} 기준에 부합하는 것으로 판단됩니다. "
            f"현재 확인된 범위에서는 적절한 집행으로 볼 수 있으며, 관련 증빙과 집행 근거를 계속 관리해 주시기 바랍니다."
        ).strip()
    if result.status == "부적절":
        if reason_code == "improper_mixed_items" and allowed_items and disallowed_items:
            allowed_text = _join_items(allowed_items)
            disallowed_text = _join_items(disallowed_items)
            return (
                f"{law_basis} {allowed_text}는 집행 가능한 항목으로 볼 수 있으나, {disallowed_text}는 같은 기준에서 인정하기 어렵습니다. "
                f"따라서 두 항목이 함께 포함된 현재 신청 형태는 그대로 인정하기 어렵습니다. "
                f"{action}"
            ).strip()
        base = (
            f"{law_basis} {subject} 현재 기준에 비추어 부적절합니다."
        ).strip()
        if facts and facts != "특이 수치 사실 없음":
            base = f"{base} {facts}"
        if action:
            base = f"{base} {action}"
        return base
    base = (
        f"{law_basis} {subject} 현재 자료만으로 단정하기 어려워 검토가 필요합니다."
    ).strip()
    if facts and facts != "특이 수치 사실 없음":
        base = f"{base} {facts}"
    if action:
        base = f"{base} {action}"
    return base


def _build_reason_subject(*, reason_code: str, items_text: str, allowed_items: list[str], disallowed_items: list[str]) -> str:
    if reason_code == "improper_mixed_items" and allowed_items and disallowed_items:
        allowed_text = _join_items(allowed_items)
        disallowed_text = _join_items(disallowed_items)
        return f"{allowed_text}는 집행 가능 항목으로 볼 수 있으나, {disallowed_text}는"
    return f"{items_text} 항목은"


def _build_items_basis_text(*, result, item_names: list[str], allowed: bool) -> str:
    snippets: list[str] = []
    for item in getattr(result, "items", []) or []:
        if item.item not in item_names:
            continue
        reasoning = " ".join((getattr(item, "reasoning", "") or "").split()).strip()
        if not reasoning:
            continue
        cleaned = reasoning
        cleaned = re.sub(r"^제\d+조(?:제\d+항(?:제\d+호)?)?에 비추어\s*", "", cleaned)
        cleaned = re.sub(rf"^{re.escape(item.item)} 항목은\s*", "", cleaned)
        if allowed:
            cleaned = cleaned.replace("집행 가능한 항목으로 판단됩니다.", "").strip()
        else:
            cleaned = cleaned.replace("집행 대상으로 보기 어렵습니다.", "").strip()
        cleaned = cleaned.strip()
        if cleaned and cleaned not in snippets:
            snippets.append(cleaned)
    return " / ".join(snippets[:2])
