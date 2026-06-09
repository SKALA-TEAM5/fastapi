"""
Link Agent 라우터
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
POST /link/run

오케스트레이터가 호출하는 엔드포인트.
사용내역서 ID와 증빙 파일 ID 목록(혼합)을 받아
OCR + 2-way 매칭을 실행하고 결과를 DB에 저장한다.

- 영수증·거래명세표·세금계산서 구분은 files.uploaded_evidence_type_code로 내부 처리
- 응답 형태: 다른 Agent와 동일한 success/fail 구조 (ParseResponse)
"""

from __future__ import annotations

from fastapi import APIRouter, status
from pydantic import BaseModel, Field

from src.schemas.ocr import ParseError, ParseResponse
from src.services.usage_statement_pipeline_service import run_link_pipeline

router = APIRouter(prefix="/link", tags=["Link Agent"])


@router.post(
    "/run",
    response_model=ParseResponse,
    status_code=status.HTTP_200_OK,
    summary="영수증·세금계산서 OCR + 사용내역서 매칭",
    description="""
분류 Agent + safety-doc 완료 후 오케스트레이터가 호출합니다.

**처리 흐름**

1. `receipt_file_ids`(영수증·거래명세표) + `tax_invoice_file_ids`(세금계산서) 수신
2. 영수증·거래명세표 → OCR 엔진 실행
3. 세금계산서 → pdfplumber / CLOVA 파싱 후 영수증 사전 검증
4. 사용내역서 항목(DB 조회) ↔ 영수증 2-way 매칭
5. 매칭 결과 → `evidence_file_links` INSERT, `files.status_code` 업데이트

**파일 유형 구분**

`files.uploaded_evidence_type_code` 기준으로 내부에서 자동 분리합니다.

| 코드 | 처리 |
|---|---|
| `receipt` | 영수증 OCR → 매칭 |
| `transaction_statement` | 거래명세표 OCR → 매칭 |
| `wage_statement` | 임금명세서 OCR → 매칭 (Gate 3 면제) |
| `tax_invoice` | 세금계산서 파싱 → 영수증 사전 검증 |
    """,
    responses={
        200: {"description": "성공(success=true) 또는 비즈니스 실패(success=false)"},
        422: {"description": "입력 데이터 형식 오류"},
        503: {"description": "OCR 엔진 또는 DB 연결 오류"},
    },
)
async def link_run(body: LinkRunRequest) -> ParseResponse:
    try:
        result = run_link_pipeline(
            usage_statement_id=body.usage_statement_id,
            receipt_file_ids=body.receipt_file_ids,
            tax_invoice_file_ids=body.tax_invoice_file_ids,
        )
        return ParseResponse(
            success=True,
            data=result,
            error=None,
            message="매칭 완료",
        )
    except ValueError as e:
        return ParseResponse(
            success=False,
            data=None,
            error=ParseError(code="not_found", message=str(e)),
            message="입력값 오류",
        )
    except Exception as e:
        return ParseResponse(
            success=False,
            data=None,
            error=ParseError(code="link_error", message=str(e)),
            message="매칭 실패",
        )


class LinkRunRequest(BaseModel):
    """POST /link/run 요청"""
    usage_statement_id: int = Field(
        ...,
        description="사용내역서 DB ID",
        examples=[3],
    )
    receipt_file_ids: list[int] = Field(
        default_factory=list,
        description="영수증·거래명세표 파일 ID 목록",
        examples=[[10, 11, 12]],
    )
    tax_invoice_file_ids: list[int] = Field(
        default_factory=list,
        description="세금계산서 파일 ID 목록",
        examples=[[20]],
    )
