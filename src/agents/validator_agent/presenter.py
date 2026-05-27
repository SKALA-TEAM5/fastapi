# --------------------------------------------------------------------------
# 작성자   : 송상민(ss19801)
# 작성일   : 2026-05-04
#
# [ 주요 클래스 및 함수 정의 ]
#
# 1. summarize_audit_response()      : 카테고리별 감사 결과 요약 생성
# 2. to_validator_response()         : 전체 감사 응답 DTO 변환
# 3. _build_sources()                : 출처 목록(법령·규정) 구성
# 4. _build_reason()                 : 항목별 구조화 사유 텍스트 조립 (LLM 없음)
# 5. _item_reason_snippet()          : reasoning 첫 문장 추출 (대괄호 제거)
# 6. _derive_reasoning_summary_for_law() : 법령별 출처 요지 추출
# --------------------------------------------------------------------------
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
    except Exception:
        return ""

    text = " ".join((output.reason or "").split()).strip()
    if len(text) < 60:
        return ""
    return text


def _item_reason_snippet(reasoning: str) -> str:
    """
    reasoning에서 의미 있는 첫 문장을 최대 100자로 반환한다.
    - DB raw 포맷 prefix 제거 ('사 역 가.', '1)' 등)
    - 대괄호 내용 제거
    - 마침표 기준 첫 문장, 없으면 100자 자름
    """
    text = " ".join((reasoning or "").split()).strip()
    if not text:
        return ""
    text = _clean_db_raw_text(text)
    if not text:
        return ""
    # 대괄호 제거 (예: [LEGAL_CITE:...], [허용] 등)
    text = re.sub(r"\[[^\]]*\]", "", text).strip()
    if not text:
        return ""
    # 첫 문장 (마침표/느낌표/물음표 기준)
    m = re.search(r"[.!?]", text)
    if m and m.start() <= 100:
        return text[: m.start()].strip()
    return text[:100].rstrip(" ,.")


def _build_reason(
    *,
    category_name: str,
    category_code: str,
    result,
    base_amount: float,
    sources: list[AuditSourceSummary],
) -> str:
    """
    LLM으로 전문 감사 의견 사유를 생성한다.
    LLM 실패 시 구조화된 텍스트를 폴백으로 반환한다.
    """
    # ── LLM 합성 우선 시도 ──────────────────────────────────────────────────
    llm_reason = _synthesize_reason_with_llm(
        category_name=category_name,
        result=result,
        base_amount=base_amount,
        sources=sources,
    )
    if llm_reason:
        return llm_reason

    # ── 폴백: 구조화 텍스트 직접 생성 ──────────────────────────────────────
    lines: list[str] = []
    for item in result.items:
        verdict  = "집행 가능" if item.allowed else "집행 불가"
        # RDB 직접 매칭(law_rule / qa_rule)은 reasoning이 raw 법령 텍스트 —
        # snippet으로 잘라내면 단어 중간 절단 or 의미없는 조각이 되므로 스킵.
        # LLM이 생성한 자연어 reasoning(llm_fallback 등)만 snippet으로 표시한다.
        is_rdb_source = getattr(item, "judgment_source", "") in ("law_rule", "qa_rule")
        snippet  = "" if is_rdb_source else _item_reason_snippet(item.reasoning)
        law_raw  = item.referenced_laws[0] if item.referenced_laws else ""
        law      = _qualify_law_for_display(law_raw) if law_raw else ""

        if snippet and law:
            lines.append(f"{item.item} ({item.amount:,.0f}원): {verdict} — {snippet} ({law})")
        elif snippet:
            lines.append(f"{item.item} ({item.amount:,.0f}원): {verdict} — {snippet}")
        elif law:
            lines.append(f"{item.item} ({item.amount:,.0f}원): {verdict} ({law})")
        else:
            lines.append(f"{item.item} ({item.amount:,.0f}원): {verdict}")

    body = "\n".join(lines)

    # ── 카테고리 수준 수치 부가 ────────────────────────────────────────────────
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

    # bare 조항에 법령명 prefix 추가
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
    if reason_code in ("improper_progress_shortfall", "appropriate_progress_compliant"):
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
        return "별표 3"
    if cleaned.startswith("별표 2"):
        return "별표 2"
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

    # 여러 조항이 콤마로 묶인 경우
    if re.search(r",\s*제\d+조", cleaned):
        parts = [p.strip() for p in re.split(r",\s*", cleaned)]
        return ", ".join(_qualify_law_for_display(p) for p in parts if p)

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

    # fallback: 순환 표현 대신 빈 문자열 반환 → 상위 함수에서 처리
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
            f"{law_basis} {subject} {category_name} 항목의 집행 기준에 부합하는 것으로 판단됩니다. "
            f"관련 증빙과 집행 근거를 지속적으로 관리해 주시기 바랍니다."
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


