# --------------------------------------------------------------------------
# 작성자   : 송상민(ss19801)
# 작성일   : 2026-05-29
#
# [ 엔드포인트 ]
#
# /classi/item
#   - 설명 : 단일 항목 카테고리 분류 + DB 적재
#   - Body : {
#               "project_id": int,
#               "usage_statement_id": int,
#               "item_name": str,
#               "used_on": str,          # YYYY-MM-DD
#               "unit": str | null,
#               "quantity": float,
#               "unit_price": float,
#               "total_amount": float,
#               "basic_info": dict | null
#             }
#   - 호출 : classify_item_service()
#   - 적재 : usage_statement_items INSERT → agent_logs INSERT (classi)
#
# [ 주요 함수 정의 ]
#
# 1. classify_item_service()        : 단일 항목 카테고리 분류 + DB 적재
# 2. _insert_usage_statement_item() : usage_statement_items INSERT
# 3. _write_classi_log()            : agent_logs INSERT (classi 전용)
# --------------------------------------------------------------------------
from __future__ import annotations

import json
import logging
from typing import Any

from src.agents.classifier_agent.agent import classify_document

try:
    from langchain_community.callbacks import get_openai_callback
except ImportError:  # pragma: no cover
    get_openai_callback = None  # type: ignore
from src.core.storage import DEFAULT_COLLECTION
from src.schemas.classifier import DocumentClassification

logger = logging.getLogger(__name__)

_MODEL_NAME = "classifier_agent"
_PAGE_NO_MANUAL = 999  # 수동 추가 항목 고정값


# ── 단일 항목 분류 + DB 적재 ──────────────────────────────────────────────────

def classify_item_service(
    *,
    project_id: int,
    usage_statement_id: int,
    item_name: str,
    used_on: str,
    unit: str | None = None,
    quantity: float = 0,
    unit_price: float = 0,
    total_amount: float,
    basic_info: dict[str, Any] | None = None,
    collection: str = DEFAULT_COLLECTION,
) -> DocumentClassification:
    """
    단일 항목을 카테고리 분류 후 usage_statement_items와 agent_logs에 INSERT한다.

    Args:
        project_id          : projects.id
        usage_statement_id  : usage_statements.id
        item_name           : 품목명
        used_on             : 사용일자 (YYYY-MM-DD)
        unit                : 단위 (선택)
        quantity            : 수량 (기본 0)
        unit_price          : 단가 (기본 0)
        total_amount        : 합계금액
        basic_info          : 분류 참고 기본정보 (선택)
        collection          : Qdrant 컬렉션명

    Returns:
        DocumentClassification (category_id, category_name, confidence 등)
    """
    # 1. 카테고리 분류 (토큰 집계 포함)
    total_tokens: int | None = None
    if get_openai_callback is not None:
        with get_openai_callback() as cb:
            result = classify_document(
                items={item_name: total_amount},
                basic_info=basic_info or {},
                collection=collection,
            )
        total_tokens = cb.total_tokens or None
    else:
        result = classify_document(
            items={item_name: total_amount},
            basic_info=basic_info or {},
            collection=collection,
        )

    from src.repositories.db import get_connection

    with get_connection() as conn:
        # 2. usage_statement_items INSERT
        item_id = _insert_usage_statement_item(
            conn,
            usage_statement_id=usage_statement_id,
            category_code=result.category_id,
            used_on=used_on,
            item_name=item_name,
            unit=unit,
            quantity=quantity,
            unit_price=unit_price,
            total_amount=total_amount,
        )

        # 3. agent_logs INSERT
        _write_classi_log(
            conn,
            project_id=project_id,
            usage_statement_id=usage_statement_id,
            usage_statement_item_id=item_id,
            item_name=item_name,
            result=result,
            token=total_tokens,
        )

    return result


# ── usage_statement_items INSERT ──────────────────────────────────────────────

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


def _insert_usage_statement_item(
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
    with conn.cursor() as cur:
        cur.execute(_INSERT_ITEM_SQL, {
            "usage_statement_id": usage_statement_id,
            "category_code":      category_code,
            "used_on":            used_on,
            "item_name":          item_name,
            "unit":               unit,
            "quantity":           quantity,
            "unit_price":         unit_price,
            "total_amount":       int(total_amount),
            "page_no":            _PAGE_NO_MANUAL,
        })
        return cur.fetchone()[0]


# ── agent_logs INSERT ─────────────────────────────────────────────────────────

_INSERT_LOG_SQL = """
    INSERT INTO agent_logs (
        project_id, usage_statement_id, usage_statement_item_id,
        agent_type_code, status_code, result_code,
        reason, details, model_name, token
    )
    VALUES (
        %(project_id)s, %(usage_statement_id)s, %(usage_statement_item_id)s,
        'classi', 'success', 'success',
        '', %(details)s::jsonb, %(model_name)s, %(token)s
    )
    RETURNING id
"""


def _write_classi_log(
    conn,
    *,
    project_id: int,
    usage_statement_id: int,
    usage_statement_item_id: int,
    item_name: str,
    result: DocumentClassification,
    token: int | None = None,
) -> None:
    details = {
        "item": {
            "item_name":     item_name,
            "category_id":   result.category_id,
            "category_name": result.category_name,
            "confidence":    result.confidence,
        }
    }
    with conn.cursor() as cur:
        cur.execute(_INSERT_LOG_SQL, {
            "project_id":               project_id,
            "usage_statement_id":       usage_statement_id,
            "usage_statement_item_id":  usage_statement_item_id,
            "details":                  json.dumps(details, ensure_ascii=False),
            "model_name":               _MODEL_NAME,
            "token":                    token,
        })
    logger.info("[agent_log] classi INSERT 완료 (item=%s, category=%s)", item_name, result.category_id)
