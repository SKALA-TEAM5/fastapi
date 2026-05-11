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

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

import src.core.llm_config as llm_config
from src.agents.validator_agent.context_retriever import CategoryRetrievedContext
from src.agents.validator_agent.parser import CategoryInputBlock, CategoryItemRow
from src.prompts.validator_prompt import ITEM_JUDGMENT_PROMPT
from src.repositories import LegalRulesRepository, ValidatorRuleMatch

_PROGRESS_RULE_LAW = "별표 3 공사진척에 따른 산업안전보건관리비 사용기준"

# RDB 매칭이 있다고 볼 최소 confidence score 기준
_RDB_MATCH_SCORE_THRESHOLD = 1.5


class _ItemJudgmentLLMOutput(BaseModel):
    """LLM 항목 판단 결과"""
    allowed: bool | None = Field(description="허용 여부 (불확실하면 null)")
    confidence: float = Field(description="판정 확신도 0.0~1.0")
    reasoning: str = Field(description="판정 근거 (법령 맥락 기반)")
    referenced_laws: list[str] = Field(default=[], description="참조 법령 조항")


@dataclass
class ItemRuleBundle:
    item: CategoryItemRow
    matches: list[ValidatorRuleMatch]
    context_text: str
    item_exception_text: str = ""

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


def _has_rdb_match(matches: list[ValidatorRuleMatch], category_code: str) -> bool:
    """
    같은 카테고리 내에서 신뢰할 수 있는 RDB 규칙이 있는지 확인.
    카테고리 코드가 정확히 일치하는 규칙만 인정 — 카테고리 경계 침범 방지.
    (예: CAT_02 규칙이 CAT_03 항목에 토큰 겹침으로 잘못 매칭되는 케이스 차단)
    RDB 있음 → RDB가 주 판단 / RDB 없음 → LLM이 법령 맥락 읽고 판단
    """
    return any(
        m.match_source == "rdb"
        and m.score >= _RDB_MATCH_SCORE_THRESHOLD
        and m.category_code == category_code
        for m in matches
    )


def _llm_item_fallback(
    *,
    item_text: str,
    category_name: str,
    category_code: str,
    retrieved_context: str,
) -> ValidatorRuleMatch | None:
    """
    RDB 규칙 미매칭 시 LLM이 법령 맥락을 읽고 허용 여부 판단.

    Tier 2: 법령 조항을 읽고 맥락상 허용 가능 → allowed=True
    Tier 3: 법령 취지와 무관하거나 명시 불가 → allowed=False
    판단 불가 → None 반환 (기존 profile fallback으로 위임)
    """
    try:
        llm = llm_config.get()
    except RuntimeError:
        return None

    # 컨텍스트가 너무 길면 앞부분만 사용
    law_context = retrieved_context[:2000] if retrieved_context else "(관련 법령 맥락 없음)"

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
        return None

    if result.allowed is None:
        return None

    # confidence가 너무 낮으면 불확실하다고 보고 None 반환
    if result.confidence < 0.4:
        return None

    # score: LLM confidence를 scale해서 부여 (profile fallback 4.0~5.5 보다 높게)
    score = 5.0 + result.confidence * 2.0

    return ValidatorRuleMatch(
        category_code=category_code,
        category_name=category_name,
        rule_type="llm_judgment",
        allowed=result.allowed,
        score=score,
        evidence=result.reasoning[:220] if result.reasoning else "",
        referenced_laws=result.referenced_laws,
        limit_pct=None,
        source_id="llm_fallback",
        match_source="llm_fallback",
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

    item_bundles: list[ItemRuleBundle] = []
    for item in block.items:
        docs = retrieved.item_docs.get(item.item_name) or []
        context_parts = [doc.page_content for doc in (docs or retrieved.category_docs)]
        context_text = "\n\n---\n\n".join(context_parts)
        item_exception_text = "\n\n---\n\n".join(
            doc.page_content for doc in docs
            if any(keyword in (doc.page_content or "") for keyword in ("단,", "다만", "제외", "불가"))
        )
        matches = rules_repo.find_validator_matches(
            category=block.category_name,
            item_text=item.item_name,
            retrieved_context=context_text,
            limit=8,
        )

        # LLM이 법령 맥락을 읽고 항상 판단
        # RDB 강한 매칭 존재 → LLM은 보조로 뒤에 추가
        # RDB 매칭 약함  → LLM이 주 판단, 최상단에 삽입
        llm_match = _llm_item_fallback(
            item_text=item.item_name,
            category_name=block.category_name,
            category_code=block.category_code,
            retrieved_context=context_text,
        )
        if llm_match is not None:
            if _has_rdb_match(matches, block.category_code):
                matches = matches + [llm_match]
            else:
                matches = [llm_match] + matches

        item_bundles.append(
            ItemRuleBundle(
                item=item,
                matches=matches,
                context_text=context_text,
                item_exception_text=item_exception_text,
            )
        )

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
