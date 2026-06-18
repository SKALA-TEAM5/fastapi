# --------------------------------------------------------------------------
# 작성자   : 송상민(ss19801)
# 작성일   : 2026-05-29
# 수정일   : 2026-06-18
#
# [ 사용 현황 ]
#
# - 이 모듈의 라우터 엔드포인트는 현재 등록되어 있지 않다.
# - services.orchestrator_service가 insert_usage_statement_item()을 import해
#   저장된 사용내역서 세부항목 classi 재분류 결과를 usage_statement_items에 적재한다.
#
# [ 주요 함수 정의 ]
#
# 1. insert_usage_statement_item() : usage_statement_items INSERT
# --------------------------------------------------------------------------
from __future__ import annotations

_PAGE_NO_MANUAL = 999  # 수동 추가 항목 고정값


_INSERT_ITEM_SQL = """
    INSERT INTO usage_statement_items (
        usage_statement_id, category_code,
        used_on, item_name, unit,
        quantity, unit_price, total_amount,
        page_no
    )
    VALUES (
        %(usage_statement_id)s, %(category_code)s,
        %(used_on)s, %(item_name)s, %(unit)s,
        %(quantity)s, %(unit_price)s, %(total_amount)s,
        %(page_no)s
    )
    RETURNING id
"""


def insert_usage_statement_item(
    conn,
    *,
    usage_statement_id: int,
    category_code: str,
    used_on: str,
    item_name: str,
    unit: str | None,
    quantity: float,
    unit_price: float,
    total_amount: float,
) -> int:
    """Insert one usage-statement item row and return its database id.

    Args:
        conn: Open database connection with cursor support.
        usage_statement_id: Usage statement identifier.
        category_code: Final category code to store.
        used_on: Usage date string.
        item_name: Item name.
        unit: Unit label, if present.
        quantity: Item quantity.
        unit_price: Unit price.
        total_amount: Total amount for the row.

    Returns:
        Inserted ``usage_statement_items.id``.
    """
    with conn.cursor() as cur:
        cur.execute(
            _INSERT_ITEM_SQL,
            {
                "usage_statement_id": usage_statement_id,
                "category_code": category_code,
                "used_on": used_on,
                "item_name": item_name,
                "unit": unit,
                "quantity": quantity,
                "unit_price": unit_price,
                "total_amount": int(total_amount),
                "page_no": _PAGE_NO_MANUAL,
            },
        )
        return cur.fetchone()[0]
