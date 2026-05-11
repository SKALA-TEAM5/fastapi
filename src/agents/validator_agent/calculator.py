# --------------------------------------------------------------------------
# 작성자   : 송상민(ss19801)
# 작성일   : 2026-05-04
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
    required_usage_rate = rule_bundle.progress_required_rate
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


def _to_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
