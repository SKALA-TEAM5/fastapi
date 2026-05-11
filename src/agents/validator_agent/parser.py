# --------------------------------------------------------------------------
# 작성자   : 송상민(ss19801)
# 작성일   : 2026-05-04
#
# [ 주요 클래스 및 함수 정의 ]
#
# 1. CategoryItemRow / CategoryInputBlock : Validator 입력 표준화 구조
# 2. parse_usage_statement() : 사용내역서 입력 파싱
# 3. parse_single_category_request() : 단일 카테고리 요청 파싱
# 4. build_legacy_blocks() : 레거시 형식 블록 변환
# --------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.schemas.classifier import CATEGORIES


@dataclass
class CategoryItemRow:
    row_id: int | None
    item_name: str
    amount: float
    remark: str = ""
    used_on: str | None = None
    unit: str | None = None
    quantity: float | None = None
    unit_price: float | None = None


@dataclass
class CategoryInputBlock:
    usage_statement_id: int | str | None
    category_code: str
    category_name: str
    base_amount: float
    progress_rate: float | None
    summary: dict[str, Any]
    basic_info: dict[str, Any] = field(default_factory=dict)
    items: list[CategoryItemRow] = field(default_factory=list)


@dataclass
class ParsedUsageStatement:
    usage_statement_id: int | str | None
    base_amount: float
    progress_rate: float | None
    blocks: list[CategoryInputBlock]


def parse_usage_statement(document: dict[str, Any]) -> ParsedUsageStatement:
    basic_info = document.get("기본정보") or {}
    usage_statement_id = document.get("사용내역서ID")
    base_amount = _to_float(
        basic_info.get("산안비총액")
        or basic_info.get("base_amount")
        or basic_info.get("safety_budget_total")
    )
    if base_amount is None:
        raise ValueError("사용내역서 validator 입력에는 기본정보.산안비총액이 필요합니다.")

    progress_rate = _to_float(
        basic_info.get("누계공정률")
        or basic_info.get("공정률")
        or basic_info.get("progress_rate")
        or basic_info.get("construction_progress_rate")
    )

    blocks: list[CategoryInputBlock] = []
    for raw_block in document.get("카테고리별데이터") or []:
        block = _parse_category_block(
            usage_statement_id=usage_statement_id,
            base_amount=base_amount,
            progress_rate=progress_rate,
            basic_info=basic_info,
            raw_block=raw_block,
        )
        if block.items:
            blocks.append(block)

    if not blocks:
        raise ValueError("사용내역서 validator 입력에는 카테고리별데이터.항목목록이 필요합니다.")

    return ParsedUsageStatement(
        usage_statement_id=usage_statement_id,
        base_amount=base_amount,
        progress_rate=progress_rate,
        blocks=blocks,
    )


def parse_single_category_request(
    *,
    category: str,
    items: dict[str, float] | None = None,
    basic_info: dict[str, Any] | None = None,
    base_amount: float | None = None,
    document: dict[str, Any] | None = None,
) -> CategoryInputBlock:
    if document:
        if items is None:
            items = document.get("items") or document.get("data") or {}
        if basic_info is None:
            basic_info = document.get("basic_info") or document.get("meta") or {}
        if base_amount is None:
            base_amount = (
                document.get("base_amount")
                or document.get("safety_budget_total")
                or document.get("산안비총액")
            )

    basic_info = basic_info or {}
    if base_amount is None:
        base_amount = (
            basic_info.get("base_amount")
            or basic_info.get("safety_budget_total")
            or basic_info.get("산안비총액")
            or basic_info.get("총계상액")
        )
    normalized_base_amount = _to_float(base_amount)
    if normalized_base_amount is None:
        raise ValueError("validator에 필요한 base_amount가 없습니다.")

    progress_rate = _to_float(
        basic_info.get("누계공정률")
        or basic_info.get("공정률")
        or basic_info.get("progress_rate")
        or basic_info.get("construction_progress_rate")
    )

    normalized_items: list[CategoryItemRow] = []
    for key, value in (items or {}).items():
        amount = _to_float(value)
        if amount is None:
            continue
        normalized_items.append(
            CategoryItemRow(
                row_id=None,
                item_name=str(key),
                amount=amount,
            )
        )
    if not normalized_items:
        raise ValueError("validator에 전달할 items 데이터가 비어 있습니다.")

    category_code, category_name = resolve_category(category)
    return CategoryInputBlock(
        usage_statement_id=document.get("사용내역서ID") if document else None,
        category_code=category_code or category,
        category_name=category_name,
        base_amount=normalized_base_amount,
        progress_rate=progress_rate,
        summary=basic_info.get("집계정보") or {},
        basic_info=basic_info,
        items=normalized_items,
    )


def build_legacy_blocks(
    *,
    base_amount: float,
    categories: dict[str, dict[str, float]],
    basic_info_by_category: dict[str, dict[str, Any]] | None = None,
    summaries_by_category: dict[str, dict[str, Any]] | None = None,
    progress_rate: float | None = None,
) -> ParsedUsageStatement:
    blocks: list[CategoryInputBlock] = []
    for category, item_map in categories.items():
        category_code, category_name = resolve_category(category)
        items: list[CategoryItemRow] = []
        for item_name, amount in item_map.items():
            normalized_amount = _to_float(amount)
            if normalized_amount is None:
                continue
            items.append(
                CategoryItemRow(
                    row_id=None,
                    item_name=item_name,
                    amount=normalized_amount,
                )
            )
        if not items:
            continue
        blocks.append(
            CategoryInputBlock(
                usage_statement_id=None,
                category_code=category_code or category,
                category_name=category_name,
                base_amount=float(base_amount),
                progress_rate=progress_rate,
                summary=(summaries_by_category or {}).get(category_name) or (summaries_by_category or {}).get(category) or {},
                basic_info=(basic_info_by_category or {}).get(category_name) or (basic_info_by_category or {}).get(category) or {},
                items=items,
            )
        )
    return ParsedUsageStatement(
        usage_statement_id=None,
        base_amount=float(base_amount),
        progress_rate=progress_rate,
        blocks=blocks,
    )


def resolve_category(category: str | None) -> tuple[str | None, str]:
    if not category:
        return None, ""
    if category in CATEGORIES:
        return category, CATEGORIES[category]
    for code, name in CATEGORIES.items():
        if name == category:
            return code, name
    return None, category


def _parse_category_block(
    *,
    usage_statement_id: int | str | None,
    base_amount: float,
    progress_rate: float | None,
    basic_info: dict[str, Any],
    raw_block: dict[str, Any],
) -> CategoryInputBlock:
    category_code, category_name = resolve_category(
        raw_block.get("카테고리코드") or raw_block.get("카테고리명")
    )
    rows: list[CategoryItemRow] = []
    for raw_row in raw_block.get("항목목록") or []:
        item_name = raw_row.get("항목명")
        amount = _to_float(raw_row.get("금액"))
        if not item_name or amount is None:
            continue
        rows.append(
            CategoryItemRow(
                row_id=raw_row.get("행ID"),
                item_name=str(item_name),
                amount=amount,
                remark=str(raw_row.get("비고") or ""),
                used_on=raw_row.get("사용일자"),
                unit=raw_row.get("단위"),
                quantity=_to_float(raw_row.get("수량")),
                unit_price=_to_float(raw_row.get("단가")),
            )
        )

    return CategoryInputBlock(
        usage_statement_id=usage_statement_id,
        category_code=category_code or "",
        category_name=category_name,
        base_amount=base_amount,
        progress_rate=progress_rate,
        summary=raw_block.get("집계정보") or {},
        basic_info={**basic_info, "집계정보": raw_block.get("집계정보") or {}},
        items=rows,
    )


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
