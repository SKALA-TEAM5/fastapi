"""JSON serialization helpers for database and agent payloads."""

from __future__ import annotations

import json
from datetime import date, datetime
from decimal import Decimal
from typing import Any


def _json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()

    item = getattr(value, "item", None)
    if callable(item):
        return item()

    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def json_dumps(value: Any, **kwargs: Any) -> str:
    """Serialize agent payloads while preserving numeric JSON values."""
    return json.dumps(value, default=_json_default, **kwargs)


def to_json_compatible(value: Any) -> Any:
    """Return a recursively JSON-compatible copy of a value."""
    return json.loads(json_dumps(value, ensure_ascii=False))
