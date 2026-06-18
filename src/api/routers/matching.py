"""
OCR 매칭 라우터
━━━━━━━━━━━━━━
POST /api/v1/matching/run  — 사용내역서 ↔ 영수증 2-way 매칭 실행
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from src.schemas.ocr import MatchRequest, MatchResponse
from src.services.matching_service import match_all_usage_to_receipts

router = APIRouter(prefix="/matching", tags=["OCR 매칭"])


@router.post(
    "/run",
    response_model=MatchResponse,
    status_code=status.HTTP_200_OK,
    summary="사용내역서 ↔ 영수증 매칭 실행",
    description="""
사용내역서 항목 목록과 영수증 OCR 결과를 2-way 매칭합니다.

**매칭 전략**

1. **Hard Gate** (모두 통과해야 후보 인정)
   - 날짜 Gate: 같은 연월이면 통과 (월 경계 ±2일 허용)
   - 금액 Gate: |사용금액 − 영수증금액| / max ≤ 1%
   - 업체명 Gate: 정규화 후 완전일치 (미기재 시 면제)

2. **유사도 점수** → Gate 통과 영수증 중 최고 점수 선택
   - `matched`       : 유사도 ≥ 0.85
   - `review_needed` : 유사도 0.75 ~ 0.84  ← **HITL 대상**
   - `unmatched`     : 유사도 < 0.75 또는 Gate 미통과

**gate_failed 코드**

| 코드 | 설명 |
|------|------|
| `no_receipt` | 매칭 가능한 영수증 없음 |
| `amount_gate` | 금액 불일치 |
| `date_gate` | 날짜 불일치 |
| `vendor_gate` | 업체명 불일치 |
    """,
    responses={
        200: {"description": "매칭 성공"},
        422: {"description": "입력 데이터 형식 오류"},
    },
)
async def run_matching(request: MatchRequest) -> MatchResponse:
    usage_statement = {
        "source_file": f"usage_statement_{request.usage_statement_id}.pdf",
        "parse_status": "SUCCESS",
        "meta": {"project_id": request.project_id},
        "items": [item.model_dump() for item in request.usage_items],
    }

    try:
        batch = match_all_usage_to_receipts(
            usage_statement=usage_statement,
            receipts=request.receipt_ocr_results,
            photo_texts=request.photo_texts or {},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"매칭 처리 중 오류 발생: {e}",
        )

    return MatchResponse(**batch)
