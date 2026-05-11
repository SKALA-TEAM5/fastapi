"""
세금계산서 OCR 라우터
━━━━━━━━━━━━━━━━━━━
POST /api/v1/tax-invoices/ocr  — 세금계산서 PDF/이미지 업로드 → 파싱
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from src.schemas.ocr import TaxInvoiceOCRResponse, TaxInvoiceParty, TaxInvoiceValidation
from src.core.config import CLOVA_OCR_SECRET, CLOVA_OCR_URL
from src.ocr.parse_tax_invoice import ALL_EXTS, PDF_EXTS, parse_tax_invoice

router = APIRouter(prefix="/tax-invoices", tags=["세금계산서 OCR"])


@router.post(
    "/ocr",
    response_model=TaxInvoiceOCRResponse,
    status_code=status.HTTP_200_OK,
    summary="세금계산서 파싱",
    description="""
세금계산서 파일(PDF 또는 이미지)을 업로드하면 공급자·공급받는자·금액·품목을 추출합니다.

**처리 흐름**
1. 파일 확장자 자동 감지
2. PDF → `pdfplumber` 텍스트 직접 추출
3. 이미지 → CLOVA OCR → 텍스트 추출 → 정규식 파싱
4. 공급가액 + 세액 = 합계금액 검증

**지원 형식**: pdf, jpg, jpeg, png, tif, tiff

**참고**: 이미지 파일 처리 시 `CLOVA_OCR_URL`, `CLOVA_OCR_SECRET` 환경변수가 필요합니다.
    """,
    responses={
        200: {"description": "파싱 성공"},
        400: {"description": "지원하지 않는 파일 형식"},
        503: {"description": "이미지 처리 시 CLOVA 설정 누락"},
    },
)
async def ocr_tax_invoice(
    file: UploadFile = File(..., description="세금계산서 파일 (pdf, jpg, png 등)"),
) -> TaxInvoiceOCRResponse:
    # ── 확장자 검증 ──────────────────────────────────
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALL_EXTS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"지원하지 않는 파일 형식: {suffix}. 지원 형식: {sorted(ALL_EXTS)}",
        )

    # ── 이미지 파일 → CLOVA 설정 확인 ────────────────
    if suffix not in PDF_EXTS and (not CLOVA_OCR_URL or not CLOVA_OCR_SECRET):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="이미지 처리를 위한 CLOVA_OCR_URL 또는 CLOVA_OCR_SECRET 환경변수가 설정되지 않았습니다.",
        )

    # ── 임시 파일 저장 후 파싱 ───────────────────────
    file_bytes = await file.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        parsed = parse_tax_invoice(tmp_path, secret=CLOVA_OCR_SECRET, url=CLOVA_OCR_URL)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"세금계산서 파싱 실패: {e}",
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # ── 스키마 매핑 ───────────────────────────────────
    # parse_tax_invoice 출력 → TaxInvoiceOCRResponse 변환
    sup_raw  = parsed.get("supplier") or {}
    buy_raw  = parsed.get("buyer") or {}
    val_raw  = parsed.get("validation") or {}

    mapped = {
        "invoice_id":          f"inv_{uuid.uuid4().hex[:8]}",
        "source_file":         file.filename or "unknown",
        "parse_method":        parsed.get("parse_method", "unknown"),
        "supplier": TaxInvoiceParty(
            company_name=sup_raw.get("company_name"),
            business_number=sup_raw.get("business_number"),
            representative=sup_raw.get("representative"),
        ),
        "buyer": TaxInvoiceParty(
            company_name=buy_raw.get("company_name"),
            business_number=buy_raw.get("business_number"),
            representative=buy_raw.get("representative"),
        ),
        "issue_date":          parsed.get("issue_date"),
        "items":               parsed.get("items") or [],
        "total_supply_amount": parsed.get("total_supply_amount"),
        "total_tax_amount":    parsed.get("total_tax_amount"),
        "total_amount":        parsed.get("total_amount"),
        "validation": TaxInvoiceValidation(
            is_valid=val_raw.get("is_valid", False),
            warnings=val_raw.get("warnings") or [],
        ),
    }

    try:
        return TaxInvoiceOCRResponse(**mapped)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"응답 스키마 변환 실패: {exc}",
        )
