# --------------------------------------------------------------------------
# 작성자   : 송상민(ss19801)
# 작성일   : 2026-05-04
#
# [ 주요 클래스 및 함수 정의 ]
#
# 1. ItemRuleBundle       : 항목별 허용/불허 규칙 매칭 결과 집계
# 2. CategoryRuleBundle   : 카테고리 전체 규칙 매칭 결과 집계
# 3. match_category_rules() : RDB + LLM fallback 기반 규칙 매칭 수행
# 4. _llm_item_fallback() : RDB 매칭 실패 시 LLM 기반 보조 판정
# --------------------------------------------------------------------------
from __future__ import annotations

import hashlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

# Qdrant 문서에 삽입된 내부 마커 — LLM 입력 및 응답에 유출되지 않도록 제거
_LEGAL_CITE_RE = re.compile(r"\[LEGAL_CITE:[^\]]*\]\s*")

from pydantic import BaseModel, Field

import src.core.llm_config as llm_config
from src.agents.validator_agent.context_retriever import CategoryRetrievedContext
from src.agents.validator_agent.parser import CategoryInputBlock, CategoryItemRow
from src.prompts.validator_prompt import ITEM_JUDGMENT_PROMPT, ITEM_REASON_ONLY_PROMPT
from src.repositories import LegalRulesRepository, ValidatorRuleMatch

_PROGRESS_RULE_LAW = "별표 3 공사진척에 따른 산업안전보건관리비 사용기준"

# RDB 매칭이 있다고 볼 최소 confidence score 기준
_RDB_MATCH_SCORE_THRESHOLD = 1.5


class _ItemJudgmentLLMOutput(BaseModel):
    """LLM 항목 판단 결과 (판단 + 사유 동시 생성)"""
    allowed: bool | None = Field(description="허용 여부 (불확실하면 null)")
    confidence: float = Field(description="판정 확신도 0.0~1.0")
    reasoning: str = Field(description="내부 판단 근거 (Chain of Thought)")
    reason_text: str = Field(default="", description="사용자 표시용 사유 (2~3문장 합니다체)")
    referenced_laws: list[str] = Field(default=[], description="참조 법령 조항")


class _ItemReasonOnlyLLMOutput(BaseModel):
    """LLM 사유 전용 출력 (RDB 확정 항목용)"""
    reason_text: str = Field(description="사용자 표시용 사유 (2~3문장 합니다체)")


@dataclass
class ItemRuleBundle:
    item: CategoryItemRow
    matches: list[ValidatorRuleMatch]
    context_text: str
    item_exception_text: str = ""
    judgment_tier: str = "rdb"  # "rdb" | "llm" — 항목 판정에 사용된 계층
    reason_text: str = ""       # LLM이 생성한 사용자 표시용 사유
    qdrant_citations: list[dict] = field(default_factory=list)

    @property
    def top_allowed(self) -> ValidatorRuleMatch | None:
        for match in self.matches:
            if match.allowed is True:
                return match
        return None

    @property
    def top_disallowed(self) -> ValidatorRuleMatch | None:
        for match in self.matches:
            if match.allowed is False:
                return match
        return None

    @property
    def has_exception(self) -> bool:
        return any(keyword in self.item_exception_text for keyword in ("단,", "다만", "제외", "불가"))


@dataclass
class CategoryRuleBundle:
    category_code: str
    category_name: str
    limit_pct: float | None
    limit_rule: str
    primary_laws: list[str] = field(default_factory=list)
    progress_law: str = _PROGRESS_RULE_LAW
    progress_required_rate: float | None = None
    progress_rule_text: str = ""
    items: list[ItemRuleBundle] = field(default_factory=list)


_AUTHORITATIVE_SOURCES = {"law_rule", "qa_rule"}  # [9] DB 기반 근거


def _has_rdb_match(matches: list[ValidatorRuleMatch], category_code: str) -> bool:
    """
    같은 카테고리 내에서 신뢰할 수 있는 DB 규칙이 있는지 확인.
    카테고리 코드가 정확히 일치하는 규칙만 인정 — 카테고리 경계 침범 방지.
    (예: CAT_02 규칙이 CAT_03 항목에 토큰 겹침으로 잘못 매칭되는 케이스 차단)
    DB 규칙 있음 → DB가 주 판단 / DB 규칙 없음 → LLM이 법령 맥락 읽고 판단

    [9] match_source "rdb" → "law_rule" | "qa_rule" 으로 세분화됨.
    """
    return any(
        m.match_source in _AUTHORITATIVE_SOURCES
        and m.score >= _RDB_MATCH_SCORE_THRESHOLD
        and m.category_code == category_code
        for m in matches
    )


def _has_rdb_disallowed_match(matches: list[ValidatorRuleMatch], category_code: str) -> bool:
    """
    같은 카테고리 내에서 신뢰할 수 있는 DB 불허 규칙이 있는지 확인.

    허용 방향 DB 규칙은 키워드 겹침으로 과매칭될 수 있어 LLM 재검증이 필요.
    (예: '소화기 허용' 규칙이 '사무실 소화기'에도 매칭되는 케이스)
    불허 방향 DB 규칙은 명시적 제외 근거이므로 LLM 없이도 신뢰 가능.
    """
    return any(
        m.match_source in _AUTHORITATIVE_SOURCES
        and m.score >= _RDB_MATCH_SCORE_THRESHOLD
        and m.category_code == category_code
        and m.allowed is False
        for m in matches
    )


def _has_strong_allowed_match(matches: list[ValidatorRuleMatch], category_code: str) -> bool:
    """강한 허용 RDB 매칭 여부 확인 — 이 경우 LLM 판단 스킵 가능."""
    return any(
        m.match_source in _AUTHORITATIVE_SOURCES
        and m.score >= 3.5
        and m.category_code == category_code
        and m.allowed is True
        for m in matches
    )


def _select_exception_context(context_text: str, limit: int = 1500) -> str:
    """Qdrant 청크에서 예외/단서 조항이 있는 부분만 선별해 LLM에 넘긴다.
    예외 문구가 없으면 앞부분만 반환."""
    _EXCEPTION_KEYWORDS = ("단,", "다만", "제외", "불가", "아니한", "이외")
    lines = context_text.split("\n")
    selected: list[str] = []
    total = 0
    for line in lines:
        if any(kw in line for kw in _EXCEPTION_KEYWORDS):
            selected.append(line)
            total += len(line)
        if total >= limit:
            break
    if not selected:
        return context_text[:limit]
    return "\n".join(selected)[:limit]


def _llm_generate_reason_only(
    *,
    item_text: str,
    category_name: str,
    allowed: bool,
    rdb_evidence: str,
    referenced_laws: list[str],
    retrieved_context: str = "",
) -> str:
    """RDB 확정 항목에 대해 사유 텍스트만 LLM으로 생성한다.
    RDB evidence + Qdrant 원문 맥락을 함께 제공해 풍부한 사유를 생성한다.
    """
    try:
        llm = llm_config.get()
    except RuntimeError:
        return rdb_evidence[:200] if rdb_evidence else ""

    verdict = "허용" if allowed else "불허"
    laws_str = ", ".join(referenced_laws[:3]) if referenced_laws else "(법령 미확인)"
    # Qdrant 원문: 예외/단서 조항 위주로 선별
    law_context = _select_exception_context(retrieved_context)[:1200] if retrieved_context else ""
    try:
        result: _ItemReasonOnlyLLMOutput = (
            ITEM_REASON_ONLY_PROMPT
            | llm.with_structured_output(_ItemReasonOnlyLLMOutput)
        ).invoke(
            {
                "category_name": category_name,
                "item_text": item_text,
                "verdict": verdict,
                "rdb_evidence": rdb_evidence[:400] if rdb_evidence else "(근거 없음)",
                "referenced_laws": laws_str,
                "law_context": law_context if law_context else "(원문 없음)",
            }
        )
        return result.reason_text or ""
    except Exception:
        return rdb_evidence[:200] if rdb_evidence else ""


def _llm_item_fallback(
    *,
    item_text: str,
    category_name: str,
    category_code: str,
    retrieved_context: str,
) -> tuple[ValidatorRuleMatch | None, str]:
    """
    RDB 규칙 미매칭 시 LLM이 법령 맥락을 읽고 허용 여부 판단 + 사유 동시 생성.

    반환: (ValidatorRuleMatch | None, reason_text)
    판단 불가 → (None, "") 반환
    """
    try:
        llm = llm_config.get()
    except RuntimeError:
        return None, ""

    # 예외/단서 조항 위주로 선별 (lost in the middle 방지)
    law_context = _select_exception_context(retrieved_context) if retrieved_context else "(관련 법령 맥락 없음)"

    try:
        result: _ItemJudgmentLLMOutput = (
            ITEM_JUDGMENT_PROMPT
            | llm.with_structured_output(_ItemJudgmentLLMOutput)
        ).invoke(
            {
                "category_name": category_name,
                "item_text": item_text,
                "law_context": law_context,
            }
        )
    except Exception:
        return None, ""

    if result.allowed is None:
        # 판단 불가(애매) — reason_text는 살려서 "현장 확인 필요" 멘트 유지
        return None, result.reason_text or ""

    if result.confidence < 0.4:
        return None, result.reason_text or ""

    score = 5.0 + result.confidence * 2.0

    match = ValidatorRuleMatch(
        category_code=category_code,
        category_name=category_name,
        rule_type="llm_judgment",
        allowed=result.allowed,
        score=score,
        evidence="",  # llm_fallback은 법령 원문 근거 없음 — reasoning은 ItemJudgment.reasoning에만 저장
        referenced_laws=result.referenced_laws,
        limit_pct=None,
        source_id="llm_fallback",
        match_source="llm_fallback",
    )
    return match, result.reason_text or ""


def _qdrant_citations_from_docs(*, docs, referenced_laws: list[str], judgment_source: str) -> list[dict]:
    for doc in docs or []:
        original_text = _clean_qdrant_original_text(str(getattr(doc, "page_content", "") or ""))
        if not original_text:
            continue
        metadata = dict(getattr(doc, "metadata", {}) or {})
        source = str(metadata.get("source") or metadata.get("source_name") or metadata.get("title") or "").strip()
        source_key = source or hashlib.sha1(original_text[:300].encode("utf-8")).hexdigest()[:12]
        legal_basis = _primary_qdrant_legal_basis(referenced_laws=referenced_laws, metadata=metadata, source=source)
        return [
            {
                "source_id": f"qdrant:{source_key}",
                "legal_basis": legal_basis,
                "summary": "Qdrant 검색 원문 보조 근거" if judgment_source == "qdrant_support" else None,
                "judgment_source": judgment_source,
                "original_text": original_text,
            }
        ]
    return []


def _clean_qdrant_original_text(value: str) -> str:
    text = _LEGAL_CITE_RE.sub("", value or "")
    cleaned_lines: list[str] = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        line = " ".join(raw_line.split())
        if not line:
            if cleaned_lines and cleaned_lines[-1]:
                cleaned_lines.append("")
            continue
        if " > " in line:
            line = line.split(" > ")[-1].strip()
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def _primary_qdrant_legal_basis(*, referenced_laws: list[str], metadata: dict, source: str) -> str:
    for law in referenced_laws or []:
        text = str(law or "").strip()
        if text:
            return text
    for key in ("legal_basis", "law_name", "source_name", "title"):
        text = str(metadata.get(key) or "").strip()
        if text:
            return text
    return source


def _process_single_item(
    *,
    item: "CategoryItemRow",
    block: "CategoryInputBlock",
    retrieved: "CategoryRetrievedContext",
    rules_repo: "LegalRulesRepository",
) -> ItemRuleBundle:
    """
    단일 항목에 대해 컨텍스트 구성 → RDB 매칭 → LLM fallback 을 수행한다.
    ThreadPoolExecutor에서 병렬 호출된다.
    """
    docs = retrieved.item_docs.get(item.item_name) or []
    context_parts = [doc.page_content for doc in (docs or retrieved.category_docs)]
    raw_context = "\n\n---\n\n".join(context_parts)
    # [LEGAL_CITE: ...] 내부 마커 제거 — LLM 입력 및 evidence_snippets 오염 방지
    context_text = _LEGAL_CITE_RE.sub("", raw_context)
    item_exception_text = _LEGAL_CITE_RE.sub(
        "",
        "\n\n---\n\n".join(
            doc.page_content for doc in docs
            if any(keyword in (doc.page_content or "") for keyword in ("단,", "다만", "제외", "불가"))
        ),
    )
    # 비고(remark)가 있으면 항목명에 붙여서 LLM 판단 정확도 향상
    # 예: "안전표지판 설치 (현장 진입로 표지판)" → LLM이 용도를 명확히 파악
    item_text_with_remark = (
        f"{item.item_name} ({item.remark})" if item.remark else item.item_name
    )

    matches = rules_repo.find_validator_matches(
        category=block.category_name,
        item_text=item.item_name,
        retrieved_context=context_text,
        limit=8,
    )

    # 계층 판단:
    # 1. 강한 불허 RDB → LLM 판단 스킵, 사유만 LLM 생성
    # 2. 강한 허용 RDB → LLM 판단 스킵, 사유만 LLM 생성
    # 3. 애매함 → LLM이 판단 + 사유 동시 생성
    reason_text = ""
    qdrant_citations: list[dict] = []

    if _has_rdb_disallowed_match(matches, block.category_code):
        # 강한 불허 → 판단은 RDB, 사유만 LLM
        judgment_tier = "rdb"
        best = next((m for m in matches if m.allowed is False and m.match_source in _AUTHORITATIVE_SOURCES), None)
        qdrant_citations = _qdrant_citations_from_docs(
            docs=(docs or retrieved.category_docs),
            referenced_laws=best.referenced_laws if best else [],
            judgment_source="qdrant_support",
        )
        reason_text = _llm_generate_reason_only(
            item_text=item_text_with_remark,
            category_name=block.category_name,
            allowed=False,
            rdb_evidence=best.evidence if best else "",
            referenced_laws=best.referenced_laws if best else [],
            retrieved_context=context_text,
        )
    elif _has_strong_allowed_match(matches, block.category_code):
        # 강한 허용 → 판단은 RDB, 사유만 LLM
        judgment_tier = "rdb"
        best = next((m for m in matches if m.allowed is True and m.match_source in _AUTHORITATIVE_SOURCES), None)
        qdrant_citations = _qdrant_citations_from_docs(
            docs=(docs or retrieved.category_docs),
            referenced_laws=best.referenced_laws if best else [],
            judgment_source="qdrant_support",
        )
        reason_text = _llm_generate_reason_only(
            item_text=item_text_with_remark,
            category_name=block.category_name,
            allowed=True,
            rdb_evidence=best.evidence if best else "",
            referenced_laws=best.referenced_laws if best else [],
            retrieved_context=context_text,
        )
    else:
        # 애매함 → LLM 판단 + 사유 동시
        llm_match, reason_text = _llm_item_fallback(
            item_text=item_text_with_remark,
            category_name=block.category_name,
            category_code=block.category_code,
            retrieved_context=context_text,
        )
        if llm_match is not None:
            qdrant_citations = _qdrant_citations_from_docs(
                docs=(docs or retrieved.category_docs),
                referenced_laws=llm_match.referenced_laws,
                judgment_source="llm_fallback",
            )
            if _has_rdb_match(matches, block.category_code):
                matches = matches + [llm_match]
            else:
                matches = [llm_match] + matches
        judgment_tier = "llm"

    return ItemRuleBundle(
        item=item,
        matches=matches,
        context_text=context_text,
        item_exception_text=item_exception_text,
        judgment_tier=judgment_tier,
        reason_text=reason_text,
        qdrant_citations=qdrant_citations,
    )


def match_category_rules(
    *,
    block: CategoryInputBlock,
    retrieved: CategoryRetrievedContext,
    repo: LegalRulesRepository | None = None,
) -> CategoryRuleBundle:
    rules_repo = repo or LegalRulesRepository()
    limit_pct, limit_rule, limit_laws = rules_repo.find_category_limit(block.category_name)
    progress_required_rate, progress_rule_text, progress_laws = rules_repo.find_progress_requirement(block.progress_rate)

    # 카테고리 검증도 병렬 실행되므로 항목별 동시 실행 수를 낮게 유지한다.
    n_workers = min(max(len(block.items), 1), 2)
    item_bundles: list[ItemRuleBundle] = [None] * len(block.items)  # type: ignore[list-item]

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {
            executor.submit(
                _process_single_item,
                item=item,
                block=block,
                retrieved=retrieved,
                rules_repo=rules_repo,
            ): idx
            for idx, item in enumerate(block.items)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            item_bundles[idx] = future.result()

    return CategoryRuleBundle(
        category_code=block.category_code,
        category_name=block.category_name,
        limit_pct=limit_pct,
        limit_rule=limit_rule,
        primary_laws=limit_laws,
        progress_law=(progress_laws[0] if progress_laws else _PROGRESS_RULE_LAW),
        progress_required_rate=progress_required_rate,
        progress_rule_text=progress_rule_text,
        items=item_bundles,
    )
