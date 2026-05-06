from __future__ import annotations

from src.schemas.classifier import CATEGORIES
from src.schemas.validator import (
    AuditResponse,
    AuditSourceSummary,
    CategoryAuditSummary,
    UsageStatementAuditSummaryResponse,
    ValidatorAuditResponse,
    ValidatorCategoryMetrics,
)

_PRIMARY_CATEGORY_LAW_TO_SUMMARY = {
    "제7조제1항제1호": "안전관리자 등의 인건비 및 업무수당 등은 해당 카테고리에서 집행할 수 있습니다.",
    "제7조제1항제2호": "산업재해 예방을 위한 안전시설, 안전장비, 화재위험작업용 소화기 등은 안전시설비 항목에 해당합니다.",
    "제7조제1항제3호": "근로자에게 지급하는 법정 보호구의 구입 및 관리 비용은 보호구 항목에 해당합니다.",
    "제7조제1항제4호": "작업환경 측정, 안전보건진단, 유해위험 진단 목적의 장비와 비용은 안전보건진단비 항목에 해당합니다.",
    "제7조제1항제5호": "안전보건교육을 위한 강사비, 자료비, 교육 실시 비용은 안전보건교육비 항목에 해당합니다.",
    "제7조제1항제6호": "근로자 건강장해 예방을 위한 장비, 임시 휴게시설, 감염병 예방 비용 등은 건강장해예방비 항목에 해당합니다.",
    "제7조제1항제7호": "건설재해예방 전문지도기관의 기술지도 대가는 기술지도비 항목에 해당합니다.",
    "제7조제1항제8호": "본사 안전전담부서 운영과 관련된 인건비 및 출장비는 본사 운영비 항목에 해당합니다.",
    "제7조제1항제9호": "위험성평가와 노사협의체 결정에 따른 유해위험요인 개선 비용은 위험성평가 비용 항목에 해당합니다.",
}
_CATEGORY_CODE_TO_PRIMARY_LAW = {
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
_REASON_TEMPLATES: dict[str, str] = {
    "improper_scope_exclusion": (
        "{items_subject} {category_name}에서 허용하는 사용 범위에 포함된다고 보기 어렵습니다. {law_basis} "
        "관련 기준상 사용 범위를 벗어나 집행 대상으로 보기 어렵고, "
        f"{_SCHEDULE_2_REF}의 제외 기준에도 저촉될 수 있어 산안비 집행 대상으로 인정하기 어렵습니다. "
        "따라서 본 건은 부적절하며, 해당 항목을 삭제하거나 현장 안전 확보와 직접 관련된 적정 항목으로 조정한 후 재제출하시기 바랍니다."
    ),
    "improper_limit_exceeded": (
        "{items_subject} {category_name} 범위에서 집행 가능한 항목으로 볼 수 있으나, {law_basis} "
        "현재 누적 집행액 {total}원은 법정 허용 한도 {limit}원을 {exceeded}원 초과하고 있습니다. "
        "한도 규정은 항목의 적정성과 별개로 반드시 준수되어야 하므로, 초과분은 산안비로 인정되기 어렵습니다. "
        "집행 금액을 한도 내로 조정하거나 초과분을 타 예산으로 전환 처리하시기 바랍니다."
    ),
    "review_progress_shortfall": (
        "{items_subject} 집행 항목 자체의 적정성에는 특별한 문제가 없으나, {law_basis} 현재 공정률 {progress_rate}% 구간에서 요구되는 "
        f"{_SCHEDULE_3_REF}상 최소 집행 기준에 비해 누적 사용 실적이 {{shortfall}}원 부족합니다. "
        "이 경우 정산 단계에서 집행 적정성 이슈가 발생할 수 있으므로, 누적 집행 계획을 점검하고 집행 시점을 재조정하여 법정 하한선을 충족하도록 관리할 필요가 있습니다."
    ),
    "review_exception_or_conflict": (
        "{items_subject} 원칙적으로 {category_name}의 허용 범위에 포함될 여지가 있으나, {law_basis} "
        "단서 조항('{exception_texts}')의 적용 여부 또는 상반된 근거의 존재에 따라 최종 판단이 달라질 수 있습니다. "
        "현 단계에서는 일률적으로 적정 또는 부적정으로 단정하기 어렵기 때문에, 예외 요건 충족 여부를 확인할 수 있는 상세 증빙과 사실관계 소명 자료를 추가로 제출해 주시기 바랍니다."
    ),
    "review_insufficient_basis": (
        "{items_subject} 현행 법령, 질의회시 및 확보된 문맥만으로는 {category_name} 집행 대상으로 직접 인정할 수 있는 허용 근거가 충분히 확인되지 않습니다. {law_basis} "
        "따라서 현재 단계에서는 보수적으로 추가 검토가 필요하며, 실제 사용 목적, 투입 장소, 현장 안전관리와의 직접 관련성을 설명하는 소명 자료를 보완한 후 재검토를 요청하시기 바랍니다."
    ),
    "review_duplicate_cost_risk": (
        "{items_subject} 형식상 {category_name}에서 집행 가능한 항목으로 볼 여지가 있으나, {law_basis} "
        "공사비 또는 타 비용 항목에 이미 반영되거나 기포함된 항목과 중복/이중 계상되었을 가능성을 배제하기 어렵습니다. "
        "회계 처리의 적정성을 확인하기 위해 해당 비용이 산안비 전용 목적으로 별도 집행되었음을 입증하는 추가 증빙과 비용 구분 근거를 제출하시기 바랍니다."
    ),
    "improper_mixed_items": (
        "이번에 함께 검토한 항목 중에는 {allowed_items_labeled}처럼 집행 가능한 항목과 "
        "{disallowed_items_labeled}처럼 집행 대상으로 보기 어려운 항목이 함께 포함되어 있습니다. {law_basis} "
        "동일 카테고리 내에 적정 항목과 부적정 항목이 혼재된 경우 현재 신청 형태 그대로는 적정성 인정이 어렵습니다. "
        "부적격 항목을 분리하거나 삭제한 후, 적정 항목만으로 내역을 재구성하여 다시 제출하시기 바랍니다."
    ),
    "appropriate_compliant": (
        "{items_subject} {category_name}의 사용 범위에 부합하는 것으로 확인됩니다. {law_basis} "
        "사용 범위, 집행 한도, 공정률 대비 집행 적정성 및 현재 확인 가능한 중복 계상 위험 여부를 함께 검토한 결과, 본 건은 산안비 집행 기준에 부합하며 모든 법적 기준을 충족하는 것으로 판단됩니다. "
        "향후 정산 단계에서도 동일한 결론이 유지될 수 있도록 실제 투입 내역과 증빙 자료의 일치성을 계속 관리하고, 관련 근거를 투명하게 관리해 주시기 바랍니다."
    ),
    "appropriate_progress_compliant": (
        "{law_basis} "
        "현재 공정률 {progress_rate}% 구간에서는 누적 사용액이 총액의 {required_usage_rate_pct}% 이상이어야 하며, "
        "실제 누적 집행액 {cumulative_used_amount}원이 이를 충족하므로 공정률 기준상 적정합니다. "
        "{items_subject} {category_name}의 집행 항목에도 특별한 문제가 없는 것으로 확인됩니다."
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
        sources = _build_sources(
            category_name=category_name,
            category_code=category_code,
            result=result,
            base_amount=response.base_amount,
        )
        summaries.append(
            CategoryAuditSummary(
                category_code=category_code,
                status=result.status,
                reason=_build_reason(
                    category_name=category_name,
                    category_code=category_code,
                    result=result,
                    base_amount=response.base_amount,
                    sources=sources,
                ),
                sources=sources,
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
    primary_law = _CATEGORY_CODE_TO_PRIMARY_LAW.get(category_code, "")
    limit_pct = (result.limit / base_amount) if (result.limit is not None and base_amount) else None

    if primary_law:
        summary = _derive_primary_summary(
            primary_law=primary_law,
            category_name=category_name,
            result=result,
        )
        if limit_pct is not None and category_code in {"CAT_02", "CAT_09"} and "한도" not in summary:
            if summary.endswith("합니다."):
                summary = summary[:-len("합니다.")] + f"하며, 총액의 {limit_pct * 100:.0f}% 한도도 함께 적용됩니다."
            else:
                summary = f"{summary} 총액의 {limit_pct * 100:.0f}% 한도도 함께 적용됩니다."
        sources.append(AuditSourceSummary(law=primary_law, summary=summary))
        seen_law_keys.add(_law_key(primary_law))

    if result.required_usage_rate is not None:
        sources.append(
            AuditSourceSummary(
                law=_SCHEDULE_3_REF,
                summary=f"공정률 {result.progress_rate:.1f}% 구간에서는 산업안전보건관리비를 {result.required_usage_rate * 100:.0f}% 이상 사용해야 합니다.",
            )
        )
        seen_law_keys.add(_law_key(_SCHEDULE_3_REF))

    for law in _prioritize_referenced_laws(result.referenced_laws, result=result):
        law_key = _law_key(law)
        if not law or law_key in seen_law_keys:
            continue
        summary = _build_additional_law_summary(
            law=law,
            category_name=category_name,
            result=result,
        )
        if summary and not _is_generic_source_summary(summary):
            sources.append(AuditSourceSummary(law=law, summary=summary))
            seen_law_keys.add(law_key)
        if len(sources) >= 3:
            break

    if not sources and result.referenced_laws:
        sources.append(
            AuditSourceSummary(
                law=result.referenced_laws[0],
                summary=f"{category_name} 판정의 근거 조항입니다.",
            )
        )
    return sources


def _build_reason(*, category_name: str, category_code: str, result, base_amount: float, sources: list[AuditSourceSummary]) -> str:
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
    return _apply_reason_template(
        reason_code=reason_code,
        result=result,
        law_ref=law_ref,
        law_basis=law_basis,
        disallowed_items=disallowed_items,
        allowed_items=allowed_items,
        exception_texts=exception_texts,
    )


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
    if result.status == "검토필요" and (exception_texts or "충돌" in rejection):
        return "review_exception_or_conflict"
    if result.status == "검토필요":
        return "review_insufficient_basis"
    if result.status == "적절" and result.required_usage_rate is not None:
        return "appropriate_progress_compliant"
    return "appropriate_compliant"


def _apply_reason_template(
    *,
    reason_code: str,
    result,
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

    template_vars = {
        "law_ref": law_ref,
        "law_basis": law_basis,
        "category_name": _humanize_category_name(result, allowed_items=allowed_items, disallowed_items=disallowed_items),
        "items": items_text,
        "items_subject": f"이번에 검토한 {items_text} 항목은" if items_text and items_text != "해당 항목" else "이번에 검토한 항목은",
        "allowed_items": allowed_items_text,
        "disallowed_items": disallowed_items_text,
        "allowed_items_labeled": _label_items(allowed_items, "적정"),
        "disallowed_items_labeled": _label_items(disallowed_items, "부적정"),
        "total": f"{result.total:,.0f}",
        "limit": f"{result.limit:,.0f}" if result.limit is not None else "",
        "exceeded": f"{exceeded_amount:,.0f}",
        "progress_rate": f"{result.progress_rate:.1f}" if result.progress_rate is not None else "",
        "required_usage_rate_pct": f"{result.required_usage_rate * 100:.0f}" if result.required_usage_rate is not None else "",
        "cumulative_used_amount": f"{result.cumulative_used_amount:,.0f}" if getattr(result, 'cumulative_used_amount', None) is not None else "",
        "shortfall": f"{result.usage_shortfall_amount:,.0f}" if result.usage_shortfall_amount is not None else "",
        "exception_texts": exception_text,
        "items_clause": f"{items_text} 등은 " if items_text and items_text != "해당 항목" else "",
    }

    try:
        filled = template.format(**template_vars)
    except KeyError:
        filled = template

    return " ".join(filled.strip().split())


def _humanize_category_name(result, *, allowed_items: list[str], disallowed_items: list[str]) -> str:
    first_law = ""
    if getattr(result, "referenced_laws", None):
        first_law = result.referenced_laws[0]
    for code, law in _CATEGORY_CODE_TO_PRIMARY_LAW.items():
        if law == first_law:
            return CATEGORIES.get(code, "")
    if allowed_items or disallowed_items:
        return "해당 카테고리"
    return "해당 카테고리"


def _compose_law_basis(*, sources: list[AuditSourceSummary], primary_law: str, law_ref: str) -> str:
    if not sources:
        return law_ref

    pieces: list[str] = []
    seen_basis: set[str] = set()
    primary_summary = sources[0].summary.strip() if sources and sources[0].summary else ""
    if law_ref and primary_summary:
        primary_piece = f"{law_ref} {primary_summary}".strip()
        pieces.append(primary_piece)
        seen_basis.add(primary_piece)
    elif law_ref:
        pieces.append(law_ref)
        seen_basis.add(law_ref)
    elif primary_summary:
        pieces.append(primary_summary)
        seen_basis.add(primary_summary)

    for source in sources[1:3]:
        summary = (source.summary or "").strip()
        if not summary:
            continue
        source_ref = _format_law_reference(source.law)
        if source_ref:
            piece = f"또한 {source_ref} {summary}".strip()
        else:
            piece = summary
        if piece in seen_basis:
            continue
        seen_basis.add(piece)
        pieces.append(piece)

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
    return _compose_law_basis(
        sources=sources,
        primary_law=primary_law,
        law_ref=law_ref,
    )


def _compose_selected_law_basis(
    *,
    sources: list[AuditSourceSummary],
    preferred_laws: tuple[str, ...],
    fallback_law_ref: str,
    only_preferred: bool = False,
) -> str:
    pieces: list[str] = []
    used_law_keys: set[str] = set()
    for preferred in preferred_laws:
        for source in sources:
            law = (source.law or "").strip()
            if not law:
                continue
            if preferred in law:
                piece = _compose_source_piece(source)
                law_key = _law_key(law)
                if piece and law_key not in used_law_keys:
                    pieces.append(piece)
                    used_law_keys.add(law_key)
                break
    if not only_preferred:
        for source in sources:
            law_key = _law_key(source.law)
            if law_key in used_law_keys:
                continue
            piece = _compose_source_piece(source, lead="또한 ")
            if piece and piece not in pieces:
                pieces.append(piece)
            if len(pieces) >= 2:
                break
    combined = " ".join(piece.strip() for piece in pieces if piece.strip()).strip()
    if not combined:
        combined = fallback_law_ref
    if combined and not combined.endswith("."):
        combined += "."
    return combined


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


def _build_additional_law_summary(*, law: str, category_name: str, result) -> str:
    if law in {_PROGRESS_RULE_LAW, _SCHEDULE_3_REF}:
        if result.required_usage_rate is not None and result.progress_rate is not None:
            return f"공정률 {result.progress_rate:.1f}% 구간에서는 산업안전보건관리비를 {result.required_usage_rate * 100:.0f}% 이상 사용해야 합니다."
        return "공정률 구간별 최소 사용률 기준을 정합니다."
    if law.startswith("제4조"):
        raw = _extract_snippet_for_law(result=result, law=law)
        if raw and any(kw in raw for kw in ("한도", "이내", "초과")):
            return raw
        if result.limit is not None:
            return "산업안전보건관리비 총액 기준 한도를 함께 적용합니다."
        return "산업안전보건관리비 총액 기준을 함께 적용합니다."
    if law.startswith("제2조"):
        raw = _extract_snippet_for_law(result=result, law=law)
        if raw and any(kw in raw for kw in ("제외", "해당", "적용", "불가", "가능")):
            return raw
        return "세부 적용 대상과 제외 기준을 함께 확인합니다."
    if law.startswith("제62조"):
        raw = _extract_snippet_for_law(result=result, law=law)
        if raw and any(kw in raw for kw in ("스마트", "안전장비", "장비", "기기")):
            return raw
        return "스마트 안전장비 관련 근거를 함께 확인합니다."
    if law.startswith("별표"):
        return "세부 기준과 적용 구간을 함께 확인합니다."
    return f"{category_name} 판단에 함께 참고합니다."


def _build_limit_rule_summary(*, result) -> str:
    text = str(result.limit_rule or "")
    if not text:
        return "해당 카테고리 집행액은 산업안전보건관리비 총액 기준 한도 내에서 관리해야 합니다."
    if "스마트안전장비" in text or "스마트 안전장비" in text:
        return "스마트안전장비 구입·임대 비용에는 산업안전보건관리비 총액의 20% 한도가 적용됩니다."
    cleaned = _clean_legal_snippet(text)
    if cleaned:
        return cleaned
    return "해당 카테고리 집행액은 산업안전보건관리비 총액 기준 한도 내에서 관리해야 합니다."


def _is_generic_source_summary(summary: str) -> bool:
    generic_phrases = ("참고합니다", "함께 확인합니다", "함께 참고합니다", "함께 적용합니다", "판정의 근거 조항입니다")
    return any(p in summary for p in generic_phrases)


def _prioritize_referenced_laws(laws: list[str], *, result) -> list[str]:
    def sort_key(law: str) -> tuple[int, int]:
        if result.limit is not None or result.exceeded:
            if law.startswith("제4조"):
                return (0, 0)
            if law.startswith("제2조"):
                return (1, 0)
        if result.required_usage_rate is not None and law in {_PROGRESS_RULE_LAW, _SCHEDULE_3_REF}:
            return (0, 0)
        if law.startswith("제2조"):
            return (2, 0)
        if law.startswith("제62조"):
            return (3, 0)
        if law.startswith("별표"):
            return (4, 0)
        return (5, laws.index(law))

    seen: set[str] = set()
    ordered: list[str] = []
    for law in sorted(laws, key=sort_key):
        if law and law not in seen:
            seen.add(law)
            ordered.append(law)
    return ordered


def _join_items(items: list[str], limit: int = 3) -> str:
    selected = [item for item in items[:limit] if item]
    return ", ".join(selected)


def _label_items(items: list[str], label: str, limit: int = 3) -> str:
    selected = [item for item in items[:limit] if item]
    return ", ".join(f"{item}({label})" for item in selected)


def _join_exception_texts(exception_texts: list[str]) -> str:
    selected = [text for text in exception_texts[:2] if text]
    if not selected:
        return "적용 범위가 상충하는 단서 조항"
    return " / ".join(selected)


def _format_law_reference(law: str) -> str:
    cleaned = (law or "").strip()
    if not cleaned:
        return ""
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
    return _PRIMARY_CATEGORY_LAW_TO_SUMMARY.get(primary_law, f"{category_name}의 기본 법령 조항입니다.")


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
    if cleaned.count("「") >= 2 and len(cleaned) > 100:
        return False
    return True


def _default_reason(category_name: str, status: str) -> str:
    if status == "적절":
        return f"{category_name} 항목은 현재 기준상 적정하게 집행되었습니다."
    if status == "부적절":
        return f"{category_name} 항목은 법령 기준에 맞지 않아 부적절합니다."
    return f"{category_name} 항목은 추가 검토가 필요합니다."
