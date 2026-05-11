"""
영수증 / 거래명세표 / 임금명세서 OCR 라우터
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
POST /api/v1/receipts/ocr

프론트엔드의 '영수증' 업로드란에는 영수증·거래명세표·임금명세서가
함께 업로드된다. 파일을 받은 뒤 문서 유형을 자동 판별하고,
유형에 맞는 파서로 분기한 다음 동일한 응답 구조로 반환한다.

─── 문서 유형(doc_type) 3종 ────────────────────────────────────────
  receipt              — 영수증
    · CLOVA OCR 영수증 특화 모델 호출
    · 카드 승인일시가 날짜 기준 (결제 시점)

  transaction_statement — 거래명세표
    · PDF → pdfplumber 텍스트 직접 추출
    · 이미지 → CLOVA OCR 일반 텍스트 추출 후 정규식 파싱
    · 작성일자가 날짜 기준 (납품 시점)
    · 매칭 엔진은 doc_type 값을 보고 날짜 허용 범위를 다르게 적용

  wage_statement        — 임금명세서
    · 거래명세표와 동일한 파서(parse_tax_invoice.py) 사용
    · 업체명(vendor) 개념 없음 → 매칭 Gate 3 자동 면제
    · 세금계산서 대조 단계 건너뜀 (tax_invoice_status = "exempt")

─── 문서 유형 판별 순서 ──────────────────────────────────────────────
  1) 파일명에 "임금명세서" 키워드 포함 → wage_statement
  2) 파일명에 "거래명세표" / "명세표" 키워드 포함 → transaction_statement
  3) PDF 파일이면 pdfplumber로 앞 3페이지 텍스트 추출 후 키워드 검색
  4) 1·2·3에서 판별 불가 → 기본값 영수증으로 처리
     (CLOVA 영수증 모델이 거래명세표를 인식 못하면 infer_result=FAILURE
      로 반환되므로 호출부에서 사후 보정 가능)
"""

from __future__ import annotations

import io
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from src.schemas.ocr import ReceiptOCRResponse, ReceiptValidation
from src.core.config import CLOVA_OCR_SECRET, CLOVA_OCR_URL
from src.ocr.clova_ocr_receipt import (
    SUPPORTED_EXTS,
    call_clova_receipt,
    parse_clova_response,
    validate_result,
)
from src.ocr.parse_tax_invoice import parse_from_pdf, parse_from_image

router = APIRouter(prefix="/receipts", tags=["영수증 OCR"])

# ──────────────────────────────────────────────────────────────────────
# 상수 정의
# ──────────────────────────────────────────────────────────────────────

# 문서 유형 판별 키워드 (파일명·본문 텍스트 모두에서 검색, 우선순위 순)
_WAGE_KEYWORDS        = ["임금명세서", "임금 명세서"]
_TRANSACTION_KEYWORDS = ["거래명세표", "명세표"]

# 파일 확장자별 허용 여부
_PDF_EXT   = {".pdf"}
_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
_ALL_EXT   = _PDF_EXT | _IMAGE_EXT


# ──────────────────────────────────────────────────────────────────────
# 헬퍼 1 — 문서 유형 판별
# ──────────────────────────────────────────────────────────────────────

def _classify_document(file_bytes: bytes, suffix: str, filename: str) -> str:
    """
    파일의 문서 유형을 판별한다.

    Returns:
        "wage_statement"        — 임금명세서 (근로자 임금 지급 내역)
        "transaction_statement" — 거래명세표 (납품 확인 + 대금 청구)
        "receipt"               — 영수증 (기본값)

    판별 흐름 (우선순위 순):
      ① 파일명 키워드 확인 (임금명세서 > 거래명세표 순)
      ② PDF 텍스트 키워드 확인 (pdfplumber, 앞 3페이지)
      ③ 기본값 receipt
    """
    filename_lower = (filename or "").lower()

    # ① 파일명 키워드 — 임금명세서 먼저 확인 (우선순위 높음)
    if any(kw in filename_lower for kw in _WAGE_KEYWORDS):
        return "wage_statement"
    if any(kw in filename_lower for kw in _TRANSACTION_KEYWORDS):
        return "transaction_statement"

    # ② PDF이면 텍스트를 직접 뽑아서 키워드 검색
    if suffix in _PDF_EXT:
        try:
            import pdfplumber  # 설치 여부 런타임 확인

            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                # 앞 3페이지만 확인 (문서 헤더에 유형명 표기되는 경우)
                text = "\n".join(
                    page.extract_text() or "" for page in pdf.pages[:3]
                )
            if any(kw in text for kw in _WAGE_KEYWORDS):
                return "wage_statement"
            if any(kw in text for kw in _TRANSACTION_KEYWORDS):
                return "transaction_statement"
        except Exception:
            pass  # pdfplumber 실패 시 기본값으로 계속 진행

    # ③ 이미지이거나 판별 실패 → 기본값 영수증
    #    (CLOVA 영수증 모델이 거래명세표를 인식 못하면 infer_result=FAILURE
    #     로 내려오므로, 필요하다면 이후 단계에서 재분류 가능)
    return "receipt"


# ──────────────────────────────────────────────────────────────────────
# 헬퍼 2 — 거래명세표 / 임금명세서 파싱 결과 → 공통 응답 스키마 매핑
# ──────────────────────────────────────────────────────────────────────

def _map_transaction_statement(parsed: dict, filename: str, doc_type: str = "transaction_statement") -> dict:
    """
    parse_tax_invoice.py 의 출력(거래명세표·임금명세서 공통 구조)을
    ReceiptOCRResponse 스키마로 변환한다.

    거래명세표 / 임금명세서 ↔ 영수증 필드 대응:
      공급자 상호               → vendor  (임금명세서는 사업주명)
      작성일자                  → date    (납품·지급 시점 기준)
      품목별 합계(공급가액+세액) → items[].amount
      합계금액                  → total_amount

    Args:
        doc_type: "transaction_statement" 또는 "wage_statement"
    """
    sup  = parsed.get("supplier") or {}
    val  = parsed.get("validation") or {}

    items = []
    for item in (parsed.get("items") or []):
        # 거래명세표 품목의 금액 = 공급가액 + 세액
        # 둘 다 없으면 None 유지 (OCR 미인식)
        supply = item.get("supply_amount")
        tax    = item.get("tax_amount")
        if supply is not None and tax is not None:
            amount = supply + tax
        elif supply is not None:
            amount = supply
        else:
            amount = None

        items.append({
            "item_name":  item.get("item_name") or item.get("name") or "",
            "count":      item.get("count") or item.get("quantity"),
            "unit_price": item.get("unit_price"),
            "amount":     amount,
            "roi_box":    None,  # 거래명세표·임금명세서는 ROI 미지원
        })

    return {
        "receipt_id":   f"rec_{uuid.uuid4().hex[:8]}",
        "source_file":  filename or "unknown",
        "doc_type":     doc_type,
        "infer_result": "SUCCESS" if parsed.get("validation", {}).get("is_valid") else "PARTIAL",
        "vendor":       sup.get("company_name"),
        "date":         parsed.get("issue_date"),
        "total_amount": parsed.get("total_amount"),
        "items":        items,
        "validation": {
            "is_valid":        val.get("is_valid", False),
            "items_sum_match": None,  # 거래명세표·임금명세서는 품목 합산 검증 별도 로직
            "warnings":        val.get("warnings") or [],
        },
    }


# ──────────────────────────────────────────────────────────────────────
# 헬퍼 3 — 영수증 파싱 결과 → 공통 응답 스키마 매핑
# ──────────────────────────────────────────────────────────────────────

def _map_receipt(parsed: dict, filename: str) -> dict:
    """
    clova_ocr_receipt.py 의 출력을 ReceiptOCRResponse 스키마로 변환한다.

    영수증 특이사항:
      · vendor   = store.name   (카드 영수증의 매장명)
      · date     = payment.date (카드 승인일시)
      · items[].name → item_name (필드명 변환 필요)
    """
    val_raw      = parsed.get("validation") or {}
    all_messages = (val_raw.get("warnings") or []) + (val_raw.get("errors") or [])

    items = []
    for item in (parsed.get("items") or []):
        items.append({
            "item_name":  item.get("name") or "",
            "count":      item.get("count"),
            "unit_price": item.get("unit_price"),
            "amount":     item.get("amount"),
            "roi_box":    item.get("roi_box"),
        })

    return {
        "receipt_id":   parsed.get("receipt_id") or f"rec_{uuid.uuid4().hex[:8]}",
        "source_file":  filename or "unknown",
        "doc_type":     "receipt",
        "infer_result": parsed.get("infer_result", "ERROR"),
        "vendor":       (parsed.get("store") or {}).get("name"),
        "date":         (parsed.get("payment") or {}).get("date"),
        "total_amount": parsed.get("total_amount"),
        "items":        items,
        "validation": {
            "is_valid":        val_raw.get("has_required_fields", False),
            "items_sum_match": val_raw.get("items_sum_match"),
            "warnings":        all_messages,
        },
    }


# ──────────────────────────────────────────────────────────────────────
# 엔드포인트
# ──────────────────────────────────────────────────────────────────────

@router.post(
    "/ocr",
    response_model=ReceiptOCRResponse,
    status_code=status.HTTP_200_OK,
    summary="영수증 / 거래명세표 OCR",
    description="""
영수증 또는 거래명세표 파일을 업로드하면 문서 유형을 자동 판별한 뒤 파싱합니다.

**문서 유형 자동 판별 기준**
| 판별 순서 | 방법 | 대상 |
|-----------|------|------|
| 1순위 | 파일명에 "거래명세표" / "명세표" 포함 | 모든 형식 |
| 2순위 | PDF 텍스트에서 "거래명세표" 키워드 검색 | PDF 한정 |
| 기본값 | 위 조건 미해당 시 영수증으로 처리 | 이미지 등 |

**문서 유형별 파싱 전략**
- `receipt` (영수증): CLOVA OCR 영수증 특화 모델 → 카드 승인일시 기준
- `transaction_statement` (거래명세표): pdfplumber / 정규식 → 작성일자 기준
- `wage_statement` (임금명세서): 거래명세표와 동일 파서 → Gate 3·세금계산서 검증 면제

**지원 형식**: pdf, jpg, jpeg, png, tif, tiff
    """,
    responses={
        200: {"description": "OCR / 파싱 성공"},
        400: {"description": "지원하지 않는 파일 형식"},
        503: {"description": "CLOVA 설정 누락 또는 API 오류"},
    },
)
async def ocr_receipt(
    file: UploadFile = File(..., description="영수증 또는 거래명세표 파일"),
) -> ReceiptOCRResponse:

    # ── 확장자 검증 ───────────────────────────────────────────
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _ALL_EXT:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"지원하지 않는 파일 형식: {suffix}. 지원 형식: {sorted(_ALL_EXT)}",
        )

    # ── 파일 읽기 ─────────────────────────────────────────────
    file_bytes = await file.read()

    # ══════════════════════════════════════════════════════════
    # STEP 1. 문서 유형 자동 판별
    #   파일명 → PDF 텍스트 순서로 확인.
    #   이미지이거나 판별 불가하면 기본값 'receipt' 사용.
    # ══════════════════════════════════════════════════════════
    doc_type = _classify_document(file_bytes, suffix, file.filename or "")

    # ── 임시 파일 저장 (파서가 파일 경로를 필요로 함) ─────────
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:

        # ══════════════════════════════════════════════════════
        # STEP 2-A. 거래명세표 / 임금명세서 처리
        #   세금계산서 파서(parse_tax_invoice.py)와 동일한 로직 사용.
        #   PDF → pdfplumber 직접 추출
        #   이미지 → CLOVA 일반 OCR + 정규식
        # ══════════════════════════════════════════════════════
        if doc_type in ("transaction_statement", "wage_statement"):

            # 이미지 문서는 CLOVA 설정이 필요
            if suffix in _IMAGE_EXT and (not CLOVA_OCR_URL or not CLOVA_OCR_SECRET):
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="이미지 문서 처리를 위한 CLOVA 환경변수가 설정되지 않았습니다.",
                )

            try:
                if suffix in _PDF_EXT:
                    parsed = parse_from_pdf(tmp_path)
                else:
                    parsed = parse_from_image(tmp_path, CLOVA_OCR_SECRET, CLOVA_OCR_URL)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"문서 파싱 실패 ({doc_type}): {e}",
                )

            mapped = _map_transaction_statement(parsed, file.filename or "unknown", doc_type=doc_type)

        # ══════════════════════════════════════════════════════
        # STEP 2-B. 영수증 처리
        #   CLOVA OCR 영수증 특화 모델 호출.
        #   카드 승인일시를 날짜 기준으로 사용.
        # ══════════════════════════════════════════════════════
        else:
            # 영수증은 이미지만 지원 (PDF 영수증은 거의 없음)
            if not CLOVA_OCR_URL or not CLOVA_OCR_SECRET:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="CLOVA_OCR_URL 또는 CLOVA_OCR_SECRET 환경변수가 설정되지 않았습니다.",
                )

            try:
                raw = call_clova_receipt(tmp_path, CLOVA_OCR_SECRET, CLOVA_OCR_URL)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"CLOVA OCR API 호출 실패: {e}",
                )

            parsed = parse_clova_response(raw)
            parsed["source_file"] = file.filename or "unknown"
            parsed = validate_result(parsed)

            mapped = _map_receipt(parsed, file.filename or "unknown")

    finally:
        # 임시 파일은 항상 정리
        Path(tmp_path).unlink(missing_ok=True)

    # ══════════════════════════════════════════════════════════
    # STEP 3. Pydantic 스키마로 변환 후 반환
    #   변환 실패 시 500 대신 422로 상세 오류 노출
    # ══════════════════════════════════════════════════════════
    try:
        return ReceiptOCRResponse(**mapped)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"응답 스키마 변환 실패: {exc}",
        )
