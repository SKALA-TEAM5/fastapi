# --------------------------------------------------------------------------
# 작성자   : 송상민(ss19801)
# 작성일   : 2026-05-04
# 수정일   : 2026-06-18
#
# [ 주요 클래스 및 함수 정의 ]
#
# 1. CategoryComputation     : 카테고리별 집행 지표 집계 데이터 클래스
# 2. calculate_category_metrics() : 카테고리 항목 리스트로부터 지표 계산
# 3. _to_float()             : 다양한 타입의 금액 값을 float로 변환
# --------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass

from src.agents.validator_agent.parser import CategoryInputBlock
from src.agents.validator_agent.rule_matcher import CategoryRuleBundle


@dataclass
class CategoryComputation:
    """Computed category-level legal metrics used by audit decisions."""

    total: float
    limit_checked_total: float
    limit_pct: float | None
    limit_amount: float | None
    exceeded: bool
    progress_rate: float | None
    required_usage_rate: float | None
    required_used_amount: float | None
    cumulative_used_amount: float | None
    usage_shortfall_amount: float | None

    @property
    def has_progress_shortfall(self) -> bool:
        """Return whether cumulative usage is below the progress-based minimum."""
        return bool(self.usage_shortfall_amount and self.usage_shortfall_amount > 0)


def calculate_category_metrics(
    *,
    block: CategoryInputBlock,
    rule_bundle: CategoryRuleBundle,
    total_cumulative_used_amount: float | None = None,
) -> CategoryComputation:
    """Calculate amount limits and progress shortfall for one category block."""
    total = sum(item.amount for item in block.items)
    limit_pct = rule_bundle.limit_pct
    limit_checked_total = total
    limit_amount = (
        block.base_amount * limit_pct
        if limit_pct is not None else None
    )
    exceeded = bool(limit_amount is not None and limit_checked_total > limit_amount)

    cumulative_used_amount = _to_float(
        block.summary.get("누적사용금액") or block.summary.get("cumulative_amount")
    )
    required_usage_rate = rule_bundle.progress_required_rate
    required_used_amount = (
        block.base_amount * required_usage_rate
        if required_usage_rate is not None else None
    )
    # 공정률 shortfall은 전체 카테고리 누적합 기준으로 판단한다.
    # total_cumulative_used_amount가 없으면 이 카테고리의 누적금액으로 대체(하위 호환).
    progress_cumulative = total_cumulative_used_amount if total_cumulative_used_amount is not None else cumulative_used_amount
    usage_shortfall_amount = None
    if required_used_amount is not None and progress_cumulative is not None:
        usage_shortfall_amount = max(required_used_amount - progress_cumulative, 0.0)

    return CategoryComputation(
        total=total,
        limit_checked_total=limit_checked_total,
        limit_pct=limit_pct,
        limit_amount=limit_amount,
        exceeded=exceeded,
        progress_rate=block.progress_rate,
        required_usage_rate=required_usage_rate,
        required_used_amount=required_used_amount,
        cumulative_used_amount=cumulative_used_amount,
        usage_shortfall_amount=usage_shortfall_amount,
    )


def _to_float(value) -> float | None:
    """Coerce numeric-ish values into float, returning ``None`` on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
