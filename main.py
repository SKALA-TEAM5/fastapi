"""
산업안전관리비 AI 검증 시스템 — OCR 파이프라인 API
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
엔드포인트:
  POST /ocr/parse   — 사용내역서 PDF 파싱 → DB 저장 → usage_statement_id 반환
                      (Spring이 파일 업로드 직후 호출)

  POST /link/run    — 영수증·세금계산서 OCR + 2-way 매칭 → DB 저장
                      (분류 Agent + safety-doc 완료 후 Spring이 호출)

실행:
  uvicorn main:app --host 0.0.0.0 --port 8001 --reload
"""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.services.pipeline_runner import parse_usage_statement, run_link_pipeline

app = FastAPI(
    title="OCR 파이프라인 API",
    description="사용내역서 PDF 파싱과 영수증 OCR·매칭을 수행하고 결과를 DB에 저장합니다.",
    version="2.0.0",
)


# ─────────────────────────────────────────────────────────────
# 요청 / 응답 스키마
# ─────────────────────────────────────────────────────────────

class OcrParseRequest(BaseModel):
    usage_file_id: int = Field(
        ...,
        description="사용내역서 PDF 파일의 files.id (DB PK)",
        example=1,
    )


class OcrParseResponse(BaseModel):
    success: bool
    usage_statement_id: Optional[int] = None
    parse_status: Optional[str] = None   # SUCCESS / PARTIAL / FAILED
    item_count: Optional[int] = None
    elapsed_sec: Optional[float] = None
    message: str


class LinkRunRequest(BaseModel):
    usage_statement_id: int = Field(
        ...,
        description="이미 저장된 usage_statements.id (/ocr/parse 응답값)",
        example=10,
    )
    receipt_file_ids: list[int] = Field(
        ...,
        description="영수증·거래명세표 파일들의 files.id 목록",
        example=[2, 3, 4],
    )
    tax_invoice_file_ids: Optional[list[int]] = Field(
        default=None,
        description="세금계산서 파일들의 files.id 목록 (선택)",
        example=[5],
    )


class LinkRunResponse(BaseModel):
    success: bool
    usage_statement_id: Optional[int] = None
    summary: Optional[dict] = None
    elapsed_sec: Optional[float] = None
    message: str


# ─────────────────────────────────────────────────────────────
# 엔드포인트 1: 사용내역서 파싱
# ─────────────────────────────────────────────────────────────

@app.post(
    "/ocr/parse",
    response_model=OcrParseResponse,
    status_code=status.HTTP_200_OK,
    summary="사용내역서 PDF 파싱",
    description="""
사용내역서 PDF 파일의 DB file_id를 받아 파싱하고 DB에 저장합니다.
Spring이 파일 업로드 직후 즉시 호출합니다.

**내부 처리 순서**

1. DB `files` 테이블에서 `storage_key` 조회
2. S3에서 파일 fetch
3. 사용내역서 PDF 파싱 (pdfplumber)
4. DB 저장: `usage_statements`, `usage_statement_items`, `usage_statement_summaries`

**parse_status 기준**

| 값 | 조건 |
|---|---|
| `SUCCESS` | 파싱 완료 + line_items 1건 이상 |
| `PARTIAL` | 파싱 완료 + line_items 0건 (빈 양식 등) |
| `FAILED` | 파싱 중 예외 발생 |
    """,
    responses={
        200: {"description": "파싱 성공"},
        404: {"description": "파일을 찾을 수 없음"},
        503: {"description": "S3 접근 오류 또는 파싱 실패"},
    },
)
async def ocr_parse(request: OcrParseRequest) -> OcrParseResponse:
    try:
        result = parse_usage_statement(usage_file_id=request.usage_file_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"파싱 중 오류 발생: {e}",
        )

    return OcrParseResponse(
        success=True,
        usage_statement_id=result.get("usage_statement_id"),
        parse_status=result.get("parse_status"),
        item_count=result.get("item_count"),
        elapsed_sec=result.get("elapsed_sec"),
        message=f"사용내역서 파싱 완료 — {result.get('item_count', 0)}건 항목",
    )


# ─────────────────────────────────────────────────────────────
# 엔드포인트 2: 영수증 OCR + 매칭
# ─────────────────────────────────────────────────────────────

@app.post(
    "/link/run",
    response_model=LinkRunResponse,
    status_code=status.HTTP_200_OK,
    summary="영수증 OCR + 2-way 매칭",
    description="""
usage_statement_id와 영수증·세금계산서 파일 목록을 받아
OCR 후 사용내역서와 매칭하고 결과를 DB에 저장합니다.

분류 Agent + safety-doc 완료 후 Spring이 호출합니다.

**내부 처리 순서**

1. DB에서 usage_statement_items 조회 (매칭 기준)
2. S3에서 영수증·세금계산서 파일 fetch
3. 영수증/거래명세표 OCR (CLOVA)
4. 세금계산서 파싱 (선택)
5. 2-way 매칭 (날짜 연월 기준 / 금액 ±1% / 업체명 완전일치)
6. DB 저장: `files.status_code` 업데이트, `evidence_file_links` INSERT

**match_status 기준**

| 값 | 조건 |
|---|---|
| `matched` | 유사도 ≥ 0.85 |
| `review_needed` | 0.75 ≤ 유사도 < 0.85 (담당자 검토 필요) |
| `unmatched` | 유사도 < 0.75 또는 Gate 미통과 |
| `rejected` | OCR 실패 / 품목명 없음 |
    """,
    responses={
        200: {"description": "OCR + 매칭 성공"},
        404: {"description": "usage_statement 또는 파일을 찾을 수 없음"},
        503: {"description": "S3 또는 CLOVA OCR 연결 오류"},
    },
)
async def link_run(request: LinkRunRequest) -> LinkRunResponse:
    try:
        result = run_link_pipeline(
            usage_statement_id=request.usage_statement_id,
            receipt_file_ids=request.receipt_file_ids,
            tax_invoice_file_ids=request.tax_invoice_file_ids,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR·매칭 중 오류 발생: {e}",
        )

    summary = result.get("summary") or {}
    return LinkRunResponse(
        success=True,
        usage_statement_id=result.get("usage_statement_id"),
        summary=summary,
        elapsed_sec=result.get("elapsed_sec"),
        message=(
            f"매칭 완료 — "
            f"matched {summary.get('matched', 0)}건 / "
            f"review {summary.get('review_needed', 0)}건 / "
            f"unmatched {summary.get('unmatched', 0)}건"
        ),
    )


# ─────────────────────────────────────────────────────────────
# 헬스체크
# ─────────────────────────────────────────────────────────────

@app.get("/health", summary="헬스체크")
async def health() -> dict:
    return {"status": "ok"}
