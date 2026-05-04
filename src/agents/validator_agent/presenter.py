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
    seen_laws: set[str] = set()
    primary_law = _CATEGORY_CODE_TO_PRIMARY_LAW.get(category_code, "")
    limit_pct = (result.limit / base_amount) if (result.limit is not None and base_amount) else None

    if primary_law:
        summary = _derive_primary_summary(
            primary_law=primary_law,
            category_name=category_name,
            result=result,
        )
        if limit_pct is not None and category_code in {"CAT_02", "CAT_09"} and "한도" not in summary:
            summary = f"{summary} 이 카테고리는 총액의 {limit_pct * 100:.0f}% 한도가 함께 적용됩니다."
        sources.append(AuditSourceSummary(law=primary_law, summary=summary))
        seen_laws.add(primary_law)

    if result.required_usage_rate is not None:
        sources.append(
            AuditSourceSummary(
                law=_PROGRESS_RULE_LAW,
                summary=f"공정률 {result.progress_rate:.1f}% 구간에서는 산업안전보건관리비를 {result.required_usage_rate * 100:.0f}% 이상 사용해야 합니다.",
            )
        )
        seen_laws.add(_PROGRESS_RULE_LAW)

    for law in _prioritize_referenced_laws(result.referenced_laws, result=result):
        if not law or law in seen_laws:
            continue
        summary = _build_additional_law_summary(
            law=law,
            category_name=category_name,
            result=result,
        )
        if summary:
            sources.append(AuditSourceSummary(law=law, summary=summary))
            seen_laws.add(law)
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
    law_basis = _build_law_basis(category_name=category_name, sources=sources, result=result)
    primary_law = sources[0].law if sources else ""
    disallowed_items = [item.item for item in result.items if not item.allowed]
    allowed_items = [item.item for item in result.items if item.allowed]

    if result.status == "부적절":
        exception_texts = _collect_exception_summaries(result, allowed=False)
        interpretation = _build_improper_interpretation(
            result=result,
            primary_law=primary_law,
            disallowed_items=disallowed_items,
            allowed_items=allowed_items,
            exception_texts=exception_texts,
        )
        supplement = _build_supplement(
            status=result.status,
            result=result,
            disallowed_items=disallowed_items,
            exception_texts=exception_texts,
        )
        return " ".join(part for part in [law_basis, interpretation, supplement] if part).strip()

    if result.status == "검토필요":
        exception_texts = _collect_exception_summaries(result, allowed=None)
        interpretation = _build_review_interpretation(
            result=result,
            allowed_items=allowed_items,
            exception_texts=exception_texts,
        )
        supplement = _build_supplement(
            status=result.status,
            result=result,
            disallowed_items=disallowed_items,
            exception_texts=exception_texts,
        )
        return " ".join(part for part in [law_basis, interpretation, supplement] if part).strip()

    exception_texts = _collect_exception_summaries(result, allowed=True)
    interpretation = _build_appropriate_interpretation(
        result=result,
        allowed_items=allowed_items,
        exception_texts=exception_texts,
    )
    supplement = _build_supplement(
        status=result.status,
        result=result,
        disallowed_items=disallowed_items,
        exception_texts=exception_texts,
    )
    return " ".join(part for part in [law_basis, interpretation, supplement] if part).strip()


def _build_law_basis(*, category_name: str, sources: list[AuditSourceSummary], result) -> str:
    if not sources:
        return f"{category_name} 관련 근거를 종합해 설명합니다."
    primary_source = sources[0]
    parts = [f"{_short_regulation_name()} {primary_source.law}에 따르면 {primary_source.summary}"]
    if _should_use_primary_only(result):
        return " ".join(parts)
    support = _select_supporting_source(sources=sources, result=result)
    if support is not None:
        connector = "를 보면"
        if support.law.endswith("기준"):
            connector = "을 보면"
        parts.append(f"{support.law}{connector} {support.summary}")
    return " ".join(parts)


def _build_additional_law_summary(*, law: str, category_name: str, result) -> str:
    if law == _PROGRESS_RULE_LAW:
        if result.required_usage_rate is not None and result.progress_rate is not None:
            return f"공정률 {result.progress_rate:.1f}% 구간에서는 산업안전보건관리비를 {result.required_usage_rate * 100:.0f}% 이상 사용해야 합니다."
        return "공정률 구간별 최소 사용률 기준을 정합니다."
    if law.startswith("제4조"):
        raw = _extract_snippet_for_law(result=result, law=law)
        if raw:
            return raw
        if result.limit is not None:
            return "산업안전보건관리비 총액 기준 한도를 함께 적용합니다."
        return "산업안전보건관리비 총액 기준을 함께 적용합니다."
    if law.startswith("제2조"):
        raw = _extract_snippet_for_law(result=result, law=law)
        if raw:
            return raw
        return "세부 적용 대상과 제외 기준을 함께 확인합니다."
    if law.startswith("제62조"):
        raw = _extract_snippet_for_law(result=result, law=law)
        if raw:
            return raw
        return "스마트 안전장비 관련 근거를 함께 확인합니다."
    if law.startswith("별표"):
        return "세부 기준과 적용 구간을 함께 확인합니다."
    return f"{category_name} 판단에 함께 참고합니다."


def _prioritize_referenced_laws(laws: list[str], *, result) -> list[str]:
    def sort_key(law: str) -> tuple[int, int]:
        if result.limit is not None or result.exceeded:
            if law.startswith("제4조"):
                return (0, 0)
            if law.startswith("제2조"):
                return (1, 0)
        if result.required_usage_rate is not None and law == _PROGRESS_RULE_LAW:
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


def _build_improper_interpretation(*, result, primary_law: str, disallowed_items: list[str], allowed_items: list[str], exception_texts: list[str]) -> str:
    if result.exceeded and result.limit is not None:
        return f"이번 산안비 집행액 {result.total:,.0f}원은 제4조에 따른 허용 한도 {result.limit:,.0f}원을 초과하므로 부적절합니다."
    if allowed_items and disallowed_items:
        allowed_text = _join_items(allowed_items)
        disallowed_text = _join_items(disallowed_items)
        law_text = f"{primary_law}에 따른 허용 범위를 벗어나" if primary_law else "관련 법령 기준상"
        base = f"{_with_topic_particle(allowed_text)} 허용되지만, {_with_topic_particle(disallowed_text)} {law_text} 산안비 집행이 부적절합니다."
        if exception_texts:
            return f"{base} {_format_exception_reference(exception_texts)}"
        return base
    if disallowed_items:
        items = _join_items(disallowed_items)
        law_text = f"{primary_law}에 따라" if primary_law else "관련 법령 기준상"
        base = f"{_with_topic_particle(items)} {law_text} 산안비 집행이 부적절합니다."
        if exception_texts:
            return f"{base} {_format_exception_reference(exception_texts)}"
        return base
    return result.rejection_reason or "해당 카테고리의 집행 기준을 충족하지 못해 부적절합니다."


def _build_review_interpretation(*, result, allowed_items: list[str], exception_texts: list[str]) -> str:
    if result.required_usage_rate is not None and result.required_used_amount is not None and result.usage_shortfall_amount is not None:
        return (
            f"요구 최소 사용액은 {result.required_used_amount:,.0f}원인데 실제 누적 사용액은 "
            f"{(result.cumulative_used_amount or 0):,.0f}원으로 {result.usage_shortfall_amount:,.0f}원 부족해 추가 검토가 필요합니다."
        )
    if exception_texts:
        items = _join_items(allowed_items[:2]) if allowed_items else "해당 항목"
        return (
            f"{_with_topic_particle(items)} 기본적으로 해당 카테고리에서 집행 가능한 항목이지만, "
            f"{_format_exception_reference(exception_texts)} 추가 검토가 필요합니다."
        )
    return result.rejection_reason or "관련 근거가 충돌하거나 부족해 추가 검토가 필요합니다."


def _build_appropriate_interpretation(*, result, allowed_items: list[str], exception_texts: list[str]) -> str:
    if allowed_items:
        items = _join_items(allowed_items)
        base = f"{_with_topic_particle(items)} 해당 카테고리의 허용 범위에 해당하고 수치 기준도 충족하므로 적절합니다."
        if exception_texts:
            return f"{base} 다만 {_format_exception_reference(exception_texts)}"
        return base
    return "해당 카테고리의 항목과 수치 기준이 모두 충족되므로 적절합니다."


def _build_supplement(*, status: str, result, disallowed_items: list[str], exception_texts: list[str]) -> str:
    if status == "검토필요" and result.required_usage_rate is not None:
        return "보완점으로는 누적 집행 계획과 집행 시점을 다시 확인하시기 바랍니다."
    if status == "검토필요" and exception_texts:
        joined = " / ".join(exception_texts[:2])
        return f"보완점으로는 \"{joined}\" 문구가 실제 집행 대상과 사용 상황에 어떻게 적용되는지 관련 증빙과 함께 확인하시기 바랍니다."
    if status == "부적절" and result.exceeded:
        return "보완점으로는 집행 금액을 카테고리 한도 내로 조정하거나, 관련 집행 시점을 다시 나눠 검토하시기 바랍니다."
    if status == "부적절" and disallowed_items:
        return "보완점으로는 해당 항목을 제외하거나, 허용되는 대체 집행 항목으로 조정하시기 바랍니다."
    if status == "적절":
        if exception_texts:
            return "보완점으로는 예외 문구가 문제되지 않는 집행 대상과 사용 목적을 메모나 증빙으로 함께 남겨 두시기 바랍니다."
        return "보완점으로는 현재 집행 목적과 관련 증빙을 함께 정리해 두시기 바랍니다."
    return "보완점으로는 관련 증빙과 적용 대상을 한 번 더 점검하시기 바랍니다."


def _join_items(items: list[str], limit: int = 3) -> str:
    selected = [item for item in items[:limit] if item]
    return ", ".join(selected)


def _with_topic_particle(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    if "," in cleaned:
        return f"{cleaned}는"
    last = cleaned[-1]
    return f"{cleaned}{'은' if _has_final_consonant(last) else '는'}"


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


def _format_exception_reference(exception_texts: list[str]) -> str:
    if not exception_texts:
        return ""
    primary = exception_texts[0]
    return f"\"{primary}\"라는 단서 문구를 보면 예외 적용 여부를 더 확인해야 합니다."


def _short_regulation_name() -> str:
    return "「산안비 사용기준」"


def _select_supporting_source(*, sources: list[AuditSourceSummary], result) -> AuditSourceSummary | None:
    if len(sources) <= 1:
        return None
    candidates = sources[1:]
    if result.required_usage_rate is not None:
        for source in candidates:
            if source.law == _PROGRESS_RULE_LAW:
                return source
    if any(not item.allowed for item in result.items):
        for source in candidates:
            if source.law.startswith("제2조"):
                return source
    if result.exceeded:
        for source in candidates:
            if source.law.startswith("제4조"):
                return source
    for source in candidates:
        if source.law.startswith("제2조"):
            return source
    for source in candidates:
        if result.limit is not None and source.law.startswith("제4조"):
            return source
    return sources[1]


def _should_use_primary_only(result) -> bool:
    if any(not item.allowed for item in result.items):
        return True
    if result.exceeded:
        return True
    return False


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
    for marker in ("- ", "• ", " "):
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
