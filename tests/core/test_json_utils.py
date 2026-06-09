from datetime import date, datetime
from decimal import Decimal

from src.core.json_utils import json_dumps, to_json_compatible


def test_json_dumps_serializes_decimal_as_json_number() -> None:
    payload = {
        "whole": Decimal("1850000.00"),
        "fraction": Decimal("0.95"),
    }

    assert json_dumps(payload) == '{"whole": 1850000, "fraction": 0.95}'


def test_to_json_compatible_handles_nested_agent_payload() -> None:
    payload = {
        "usage_item": {"amount": Decimal("850000.00")},
        "scores": [Decimal("0.91")],
        "used_on": date(2026, 6, 9),
        "created_at": datetime(2026, 6, 9, 12, 0, 0),
    }

    assert to_json_compatible(payload) == {
        "usage_item": {"amount": 850000},
        "scores": [0.91],
        "used_on": "2026-06-09",
        "created_at": "2026-06-09T12:00:00",
    }
