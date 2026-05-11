"""
validation_logs 적재 저장소
━━━━━━━━━━━━━━━━━━━━━━━━━━
OCR 매칭 결과(match_all_usage_to_receipts 반환값)를
service.validation_logs 테이블에 INSERT한다.

주요 함수:
    insert_match_batch(conn, project_id, usage_statement_id, batch)
        → 배치 매칭 결과 전체를 validation_logs에 적재

    insert_single_match(conn, project_id, usage_statement_id,
                        usage_statement_item_id, result)
        → 단건 매칭 결과를 validation_logs에 적재

DB 스키마 (V1__init.sql):
    validation_logs (
        id                      BIGSERIAL PK,
        project_id              BIGINT NOT NULL,
        usage_statement_id      BIGINT,
        usage_statement_item_id BIGINT,
        validation_type_code    VARCHAR(50) NOT NULL,
        result_code             VARCHAR(30) NOT NULL,
        details                 JSONB,
        model_name              VARCHAR(100),
        created_at              TIMESTAMPTZ DEFAULT now()
    )
"""

from __future__ import annotations

import json
from typing import Optional

import psycopg2.extras
from psycopg2.extensions import connection as PgConnection


# validation_type_code 고정값
_VALIDATION_TYPE = "ocr_receipt_match"
_MODEL_NAME      = "clova_ocr_v2 + matching_service"


# ══════════════════════════════════════════════════════════════
# 공개 함수
# ══════════════════════════════════════════════════════════════

def insert_match_batch(
    conn: PgConnection,
    project_id: int,
    usage_statement_id: int,
    batch: dict,
    item_id_map: Optional[dict[str, int]] = None,
) -> int:
    """
    배치 매칭 결과 전체를 validation_logs에 INSERT한다.

    Args:
        conn                : psycopg2 커넥션 (get_connection() 사용 권장)
        project_id          : service.projects.id
        usage_statement_id  : service.usage_statements.id
        batch               : match_all_usage_to_receipts() 반환 dict
        item_id_map         : {usage_item seq → usage_statement_items.id}
                              None이면 usage_statement_item_id = NULL로 저장

    Returns:
        삽입된 행 수
    """
    results = batch.get("results", [])
    if not results:
        return 0

    item_id_map = item_id_map or {}
    rows_inserted = 0

    for result in results:
        seq = result.get("usage_item", {}).get("seq")
        item_db_id = item_id_map.get(seq)  # None이면 NULL

        rows_inserted += insert_single_match(
            conn=conn,
            project_id=project_id,
            usage_statement_id=usage_statement_id,
            usage_statement_item_id=item_db_id,
            result=result,
        )

    return rows_inserted


def insert_single_match(
    conn: PgConnection,
    project_id: int,
    usage_statement_id: int,
    usage_statement_item_id: Optional[int],
    result: dict,
) -> int:
    """
    단건 매칭 결과를 validation_logs에 INSERT한다.

    Args:
        conn                    : psycopg2 커넥션
        project_id              : service.projects.id
        usage_statement_id      : service.usage_statements.id
        usage_statement_item_id : service.usage_statement_items.id (없으면 None)
        result                  : match_threeway() 또는 match_best() 반환 dict

    Returns:
        삽입된 행 수 (항상 1)
    """
    match_status = result.get("match_status", "unmatched")

    # result_code: matched / review_needed / unmatched / rejected
    result_code = match_status

    # details JSONB: 영수증·사용항목 정보 포함
    details = _build_details(result)

    sql = """
        INSERT INTO validation_logs (
            project_id,
            usage_statement_id,
            usage_statement_item_id,
            validation_type_code,
            result_code,
            details,
            model_name
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id
    """
    params = (
        project_id,
        usage_statement_id,
        usage_statement_item_id,
        _VALIDATION_TYPE,
        result_code,
        json.dumps(details, ensure_ascii=False),
        _MODEL_NAME,
    )

    with conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
        log_id = row[0] if row else None

    item_desc = result.get("usage_item", {}).get("description", "?")
    print(f"  [DB] validation_logs INSERT: id={log_id}  item={item_desc}  result={result_code}")

    return 1


# ══════════════════════════════════════════════════════════════
# 내부 헬퍼
# ══════════════════════════════════════════════════════════════

def _build_details(result: dict) -> dict:
    """match_threeway 결과에서 validation_logs.details JSONB를 구성한다."""
    usage_item = result.get("usage_item", {})
    receipt    = result.get("receipt", {})
    comp       = result.get("component_scores", {})

    usage_entry = {
        "seq":              usage_item.get("seq"),
        "date":             usage_item.get("date"),
        "name":             usage_item.get("description") or usage_item.get("item_name"),
        "claimed_amount":   usage_item.get("amount") or usage_item.get("total_amount"),
        "similarity_score": result.get("similarity_score", 0),
        "item_match_status": result.get("match_status"),
        "gate_failed":      result.get("gate_failed", []),
    }

    receipt_entry = None
    if receipt and receipt.get("receipt_id") != "NO_RECEIPT":
        receipt_entry = {
            "source_file":   receipt.get("source_file") or receipt.get("receipt_id"),
            "total_amount":  receipt.get("total_amount"),
            "vendor":        receipt.get("vendor"),
            "date":          receipt.get("date"),
        }

    details = {
        "match_status":      result.get("match_status"),
        "similarity_score":  result.get("similarity_score", 0),
        "component_scores":  comp,
        "usage_item":        usage_entry,
        "receipt":           receipt_entry,
        "reject_reason":     result.get("reject_reason"),
    }

    return {k: v for k, v in details.items() if v is not None}
