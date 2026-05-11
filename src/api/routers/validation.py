"""
검증 로그 & HITL 라우터
━━━━━━━━━━━━━━━━━━━━━━━
GET  /api/v1/validation-logs              — 검증 로그 목록 조회
POST /api/v1/validation-logs/{id}/review  — HITL 검토 결과 제출
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Body, HTTPException, Path, Query, status

from src.schemas.ocr import ReviewRequest, ReviewResponse, ValidationLogItem
from src.repositories.db import get_connection

router = APIRouter(prefix="/validation-logs", tags=["검증 로그 & HITL"])


# ══════════════════════════════════════════════
# GET /validation-logs
# ══════════════════════════════════════════════

@router.get(
    "",
    response_model=list[ValidationLogItem],
    summary="검증 로그 목록 조회",
    description="""
`service.validation_logs` 테이블에서 매칭 결과를 조회합니다.

**result_code 필터**

| 값 | 설명 |
|----|------|
| `matched` | 자동 매칭 성공 |
| `review_needed` | AI 불확실 → **HITL 검토 필요** |
| `unmatched` | 매칭 실패 |
| `rejected` | 영수증 무효 |
| `approved` | 담당자 승인 완료 |
| `rejected_by_human` | 담당자 반려 |
    """,
)
async def list_validation_logs(
    project_id: int = Query(..., description="프로젝트 ID", examples=[1]),
    usage_statement_id: Optional[int] = Query(None, description="사용내역서 ID (미입력 시 전체)"),
    result_code: Optional[str] = Query(None, description="결과 코드 필터 (예: review_needed)"),
    limit: int = Query(50, ge=1, le=200, description="최대 반환 건수"),
) -> list[ValidationLogItem]:
    sql = """
        SELECT
            vl.id,
            vl.usage_statement_item_id,
            usi.item_name,
            usi.total_amount,
            vl.validation_type_code,
            vl.result_code,
            vl.details->'gate_failed'          AS gate_failed,
            (vl.details->>'similarity_score')::float AS similarity_score,
            vl.model_name,
            vl.created_at
        FROM service.validation_logs vl
        LEFT JOIN service.usage_statement_items usi
            ON vl.usage_statement_item_id = usi.id
        WHERE vl.project_id = %s
    """
    params: list = [project_id]

    if usage_statement_id is not None:
        sql += " AND vl.usage_statement_id = %s"
        params.append(usage_statement_id)
    if result_code:
        sql += " AND vl.result_code = %s"
        params.append(result_code)

    sql += " ORDER BY vl.id DESC LIMIT %s"
    params.append(limit)

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"DB 조회 실패: {e}",
        )

    results = []
    for row in rows:
        gate_failed_raw = row[6]
        if isinstance(gate_failed_raw, str):
            gate_failed = json.loads(gate_failed_raw)
        elif isinstance(gate_failed_raw, list):
            gate_failed = gate_failed_raw
        else:
            gate_failed = None

        results.append(
            ValidationLogItem(
                id=row[0],
                usage_statement_item_id=row[1],
                item_name=row[2],
                total_amount=row[3],
                validation_type_code=row[4],
                result_code=row[5],
                gate_failed=gate_failed,
                similarity_score=row[7],
                model_name=row[8],
                created_at=row[9],
            )
        )
    return results


# ══════════════════════════════════════════════
# POST /validation-logs/{log_id}/review
# ══════════════════════════════════════════════

@router.post(
    "/{log_id}/review",
    response_model=ReviewResponse,
    status_code=status.HTTP_201_CREATED,
    summary="HITL 검토 결과 제출",
    description="""
`review_needed` 상태인 항목에 대한 담당자 최종 결정을 기록합니다.

- **approved**: 매칭 내용을 그대로 승인
- **rejected**: 영수증 재제출 또는 반려 처리

결정 내용은 `validation_logs`에 `validation_type_code = 'human_review'`로 별도 저장되어
감사 추적(audit trail)이 가능합니다.
    """,
    responses={
        201: {"description": "검토 결과 저장 완료"},
        404: {"description": "대상 로그를 찾을 수 없음"},
        409: {"description": "이미 검토 완료된 항목"},
        503: {"description": "DB 오류"},
    },
)
async def submit_review(
    log_id: int = Path(..., description="검토 대상 validation_log ID", examples=[2]),
    body: ReviewRequest = Body(...),
) -> ReviewResponse:
    # ── 대상 로그 존재 여부 확인 ──────────────────────
    check_sql = """
        SELECT id, project_id, usage_statement_id, usage_statement_item_id, result_code
        FROM service.validation_logs
        WHERE id = %s
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(check_sql, [log_id])
                row = cur.fetchone()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"DB 조회 실패: {e}")

    if not row:
        raise HTTPException(status_code=404, detail=f"validation_log id={log_id} 를 찾을 수 없습니다.")

    _, project_id, usage_statement_id, item_id, current_status = row

    if current_status not in ("review_needed",):
        raise HTTPException(
            status_code=409,
            detail=f"현재 상태({current_status})는 HITL 검토 대상이 아닙니다. review_needed 상태만 검토 가능합니다.",
        )

    # ── human_review 로그 INSERT ──────────────────────
    insert_sql = """
        INSERT INTO service.validation_logs (
            project_id,
            usage_statement_id,
            usage_statement_item_id,
            validation_type_code,
            result_code,
            details,
            model_name
        ) VALUES (%s, %s, %s, 'human_review', %s, %s::jsonb, 'human')
        RETURNING id, created_at
    """
    result_code = "approved" if body.decision == "approved" else "rejected"
    details = json.dumps({
        "original_log_id": log_id,
        "reviewer_id": body.reviewer_id,
        "decision": body.decision,
        "reason": body.reason,
    }, ensure_ascii=False)

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(insert_sql, [
                    project_id,
                    usage_statement_id,
                    item_id,
                    result_code,
                    details,
                ])
                new_id, created_at = cur.fetchone()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"DB 저장 실패: {e}")

    return ReviewResponse(
        log_id=new_id,
        original_log_id=log_id,
        decision=body.decision,
        reviewer_id=body.reviewer_id,
        created_at=created_at,
    )
