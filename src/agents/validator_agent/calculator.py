from __future__ import annotations

from dataclasses import dataclass

from src.agents.validator_agent.parser import CategoryInputBlock
from src.agents.validator_agent.rule_matcher import CategoryRuleBundle


@dataclass
class CategoryComputation:
    total: float
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
        return bool(self.usage_shortfall_amount and self.usage_shortfall_amount > 0)


def calculate_category_metrics(
    *,
    block: CategoryInputBlock,
    rule_bundle: CategoryRuleBundle,
) -> CategoryComputation:
    total = sum(item.amount for item in block.items)
    limit_amount = (
        block.base_amount * rule_bundle.limit_pct
        if rule_bundle.limit_pct is not None else None
    )
    exceeded = bool(limit_amount is not None and total > limit_amount)

    cumulative_used_amount = _to_float(
        block.summary.get("누적사용금액") or block.summary.get("cumulative_amount")
    )
    required_usage_rate = required_usage_rate_for_progress(block.progress_rate)
    required_used_amount = (
        block.base_amount * required_usage_rate
        if required_usage_rate is not None else None
    )
    usage_shortfall_amount = None
    if required_used_amount is not None and cumulative_used_amount is not None:
        usage_shortfall_amount = max(required_used_amount - cumulative_used_amount, 0.0)

    return CategoryComputation(
        total=total,
        limit_pct=rule_bundle.limit_pct,
        limit_amount=limit_amount,
        exceeded=exceeded,
        progress_rate=block.progress_rate,
        required_usage_rate=required_usage_rate,
        required_used_amount=required_used_amount,
        cumulative_used_amount=cumulative_used_amount,
        usage_shortfall_amount=usage_shortfall_amount,
    )


def required_usage_rate_for_progress(progress_rate: float | None) -> float | None:
    if progress_rate is None:
        return None
    if 50 <= progress_rate < 70:
        return 0.5
    if 70 <= progress_rate < 90:
        return 0.7
    if progress_rate >= 90:
        return 0.9
    return None


def _to_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
