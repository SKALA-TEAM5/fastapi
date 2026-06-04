"""
단일 OCR 파싱 엔드포인트
━━━━━━━━━━━━━━━━━━━━━━━━
POST /api/v1/ocr/parse

백엔드 서버가 DB files 테이블 레코드를 JSON으로 전달하면,
OCR 엔진이 storage_key 로 S3에서 파일을 직접 가져와 파싱한다.

문서 유형(uploaded_evidence_type_code)에 따라 파서를 분기한다.

  usage_statement      → pdfplumber 사용내역서 파서 (파싱만, 매칭 없음)
  receipt              → OCR 엔진 (OCR_ENGINE=vlm: Gemini/OpenAI, clova: CLOVA OCR)
  transaction_statement→ pdfplumber / OCR 엔진
  wage_statement       → 거래명세표 파서 공유 (매칭 없음, Gate 3 면제)
  tax_invoice          → pdfplumber / CLOVA 세금계산서 파서

매칭은 POST /api/v1/matching/run 별도 엔드포인트에서 처리한다.
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, status

from src.schemas.ocr import (
    FileRecord,
    ParseError,
    ParseResponse,
    TaxInvoiceParty,
    TaxInvoiceValidation,
)
from src.core.config import CLOVA_OCR_SECRET, CLOVA_OCR_URL
from src.ocr.ocr_engine import parse_receipt, parse_document_image, get_engine_name
from src.ocr.parse_tax_invoice import (
    ALL_EXTS as TAX_EXTS,
    PDF_EXTS,
    parse_from_image,
    parse_from_pdf,
    parse_tax_invoice,
)
from src.ocr.parse_usage_statement import parse_pdf as parse_usage_statement, is_usage_statement
from src.services.minio_client import create_presigned_file_url, fetch_file

router = APIRouter(prefix="/ocr", tags=["OCR 파싱"])

# ── 지원 확장자 ───────────────────────────────────────────────────────
_PDF_EXT   = {".pdf"}
_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
_ALL_EXT   = _PDF_EXT | _IMAGE_EXT

# ── 문서 유형 판별 키워드 ──────────────────────────────────────────────
_WAGE_KEYWORDS        = ["임금명세서", "임금 명세서"]
_TRANSACTION_KEYWORDS = ["거래명세표", "명세표"]


# ─────────────────────────────────────────────────────────────────────
# 헬퍼 — 파일명 기반 doc_type 보정
# receipt/transaction_statement/wage_statement 모두 동일 엔드포인트이므로
# uploaded_evidence_type_code 가 'receipt' 이더라도 파일명으로 재판별한다.
# ─────────────────────────────────────────────────────────────────────
def _refine_doc_type(declared: str, filename: str, file_bytes: bytes, suffix: str) -> str:
    """
    백엔드가 전달한 uploaded_evidence_type_code 를 파일명으로 보정한다.
    wage_statement / transaction_statement 는 파일명 키워드로 확정하고,
    나머지는 declared 값을 그대로 신뢰한다.
    """
    name_lower = filename.lower()
    if any(kw in filename for kw in _WAGE_KEYWORDS):
        return "wage_statement"
    if any(kw in name_lower for kw in [k.lower() for k in _TRANSACTION_KEYWORDS]):
        return "transaction_statement"

    # PDF 본문 텍스트로 재확인 (이미지면 스킵)
    if suffix == ".pdf" and declared in ("receipt", "transaction_statement"):
        try:
            import pdfplumber
            import io
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                text = " ".join(
                    page.extract_text() or ""
                    for page in pdf.pages[:3]
                )
            if any(kw in text for kw in _WAGE_KEYWORDS):
                return "wage_statement"
            if any(kw in text for kw in _TRANSACTION_KEYWORDS):
                return "transaction_statement"
        except Exception:
            pass

    return declared


# ─────────────────────────────────────────────────────────────────────
# 파서 함수
# ─────────────────────────────────────────────────────────────────────

def _parse_usage_statement(file_bytes: bytes, suffix: str, filename: str) -> dict:
    """사용내역서 파싱 — PDF 전용"""
    if suffix != ".pdf":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="사용내역서는 PDF 형식만 지원합니다.",
        )
    # ── 사용내역서 형식 사전 판별 ─────────────────────────
    if not is_usage_statement(file_bytes):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "업로드된 파일이 사용내역서 형식이 아닙니다. "
                "산업안전보건관리비 사용내역서 PDF를 업로드해 주세요."
            ),
        )
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        result = parse_usage_statement(tmp_path)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"사용내역서 파싱 실패: {e}",
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return result


def _parse_receipt_or_statement(
    file_bytes: bytes, suffix: str, filename: str, doc_type: str
) -> dict:
    """영수증 / 거래명세표 / 임금명세서 파싱"""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        if doc_type in ("transaction_statement", "wage_statement"):
            if suffix in _PDF_EXT:
                parsed = parse_from_pdf(tmp_path)
            else:
                parsed = parse_document_image(tmp_path, type_hint=doc_type)

            return {
                "receipt_id":   f"rec_{uuid.uuid4().hex[:8]}",
                "source_file":  filename,
                "doc_type":     doc_type,
                "infer_result": parsed.get("parse_status", "SUCCESS").upper(),
                "vendor":       parsed.get("supplier", {}).get("company_name"),
                "date":         parsed.get("issue_date"),
                "total_amount": parsed.get("total_amount"),
                "items": [
                    {
                        "item_name":  item.get("name", ""),
                        "count":      item.get("count"),
                        "unit_price": item.get("unit_price"),
                        "amount":     item.get("supply_amount"),
                    }
                    for item in (parsed.get("items") or [])
                ],
                "validation": {
                    "is_valid":       parsed.get("validation", {}).get("is_valid", False),
                    "items_sum_match": None,
                    "warnings":       parsed.get("validation", {}).get("warnings", []),
                },
            }

        else:  # receipt
            parsed = parse_receipt(tmp_path)

            return {
                "receipt_id":   f"rec_{uuid.uuid4().hex[:8]}",
                "source_file":  filename,
                "doc_type":     "receipt",
                "ocr_engine":   get_engine_name(),
                "infer_result": parsed.get("infer_result", "SUCCESS"),
                "vendor":       parsed.get("store", {}).get("name") or parsed.get("store_name"),
                "date":         parsed.get("payment", {}).get("date") or parsed.get("payment_date"),
                "total_amount": parsed.get("total_amount") or parsed.get("total_price"),
                "items": [
                    {
                        "item_name":  item.get("name", "") or item.get("item_name", ""),
                        "count":      item.get("count"),
                        "unit_price": item.get("unit_price"),
                        "amount":     item.get("amount"),
                    }
                    for item in (parsed.get("items") or parsed.get("sub_results") or [])
                ],
                "validation": {
                    "is_valid":        parsed.get("validation", {}).get("is_valid", False),
                    "items_sum_match": parsed.get("validation", {}).get("items_sum_match"),
                    "warnings":        parsed.get("validation", {}).get("warnings", []),
                },
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"파싱 실패 ({doc_type}): {e}",
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _parse_tax_invoice(file_bytes: bytes, suffix: str, filename: str) -> dict:
    """세금계산서 파싱"""
    if suffix not in TAX_EXTS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"지원하지 않는 파일 형식: {suffix}",
        )
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

    sup = parsed.get("supplier") or {}
    buy = parsed.get("buyer") or {}
    val = parsed.get("validation") or {}

    return {
        "invoice_id":          f"inv_{uuid.uuid4().hex[:8]}",
        "source_file":         filename,
        "parse_method":        parsed.get("parse_method", "unknown"),
        "supplier":            sup,
        "buyer":               buy,
        "issue_date":          parsed.get("issue_date"),
        "items":               parsed.get("items") or [],
        "total_supply_amount": parsed.get("total_supply_amount"),
        "total_tax_amount":    parsed.get("total_tax_amount"),
        "total_amount":        parsed.get("total_amount"),
        "validation":          val,
    }


# ─────────────────────────────────────────────────────────────────────
# 엔드포인트
# ─────────────────────────────────────────────────────────────────────

@router.post(
    "/parse",
    response_model=ParseResponse,
    status_code=status.HTTP_200_OK,
    summary="단일 OCR 파싱 엔드포인트",
    description="""
DB `files` 테이블 레코드를 JSON으로 전달하면 `storage_key`로 S3에서 파일을 가져와 파싱합니다.

**문서 유형별 처리**

| uploaded_evidence_type_code | 파서 | 매칭 |
|---|---|---|
| `usage_statement` | pdfplumber | ❌ |
| `receipt` | OCR 엔진 (VLM/CLOVA, OCR_ENGINE 환경변수로 전환) | ❌ (별도 /matching/run 호출) |
| `transaction_statement` | pdfplumber / OCR 엔진 | ❌ |
| `wage_statement` | 거래명세표 파서 공유 | ❌ |
| `tax_invoice` | pdfplumber / CLOVA | ❌ |

매칭은 `POST /api/v1/matching/run`에서 별도로 실행합니다.
    """,
    responses={
        200: {"description": "파싱 성공 (success: true) 또는 비즈니스 실패 (success: false)"},
        422: {"description": "지원하지 않는 파일 형식 또는 MinIO 파일 접근 불가"},
        503: {"description": "OCR 엔진(VLM/CLOVA) 또는 MinIO 연결 오류"},
    },
)
async def parse(body: FileRecord) -> ParseResponse:
    # ── 확장자 추출 ──────────────────────────────────────────────────
    suffix = Path(body.original_filename).suffix.lower()
    if suffix not in _ALL_EXT:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"지원하지 않는 파일 형식: {suffix}. 지원 형식: {sorted(_ALL_EXT)}",
        )

    # ── S3에서 파일 fetch ────────────────────────────────────────────
    file_bytes = fetch_file(body.storage_key)
    file_input = {
        "file_id": body.id,
        "original_filename": body.original_filename,
        "storage_key": body.storage_key,
        "presigned_url": create_presigned_file_url(body.storage_key),
    }

    evidence_code = body.uploaded_evidence_type_code

    try:
        # ── 사용내역서 ───────────────────────────────────────────────
        if evidence_code == "usage_statement":
            result = _parse_usage_statement(file_bytes, suffix, body.original_filename)
            return ParseResponse(
                success=True,
                data={
                    "file":              file_input,
                    "usage_statement": result.get("usage_statement"),
                    "summaries":       result.get("summaries", []),
                    "items":           result.get("items", []),
                },
                error=None,
                message="사용내역서 파싱 성공",
            )

        # ── 세금계산서 ───────────────────────────────────────────────
        elif evidence_code == "tax_invoice":
            result = _parse_tax_invoice(file_bytes, suffix, body.original_filename)
            return ParseResponse(
                success=True,
                data={"file": file_input, "ocr_result": result},
                error=None,
                message="세금계산서 파싱 성공",
            )

        # ── 영수증 / 거래명세표 / 임금명세서 ─────────────────────────
        elif evidence_code in ("receipt", "transaction_statement", "wage_statement"):
            # 파일명으로 doc_type 보정 (백엔드가 'receipt'로 보내도 실제 파일명 기준)
            doc_type = _refine_doc_type(
                evidence_code, body.original_filename, file_bytes, suffix
            )
            result = _parse_receipt_or_statement(
                file_bytes, suffix, body.original_filename, doc_type
            )
            return ParseResponse(
                success=True,
                data={"file": file_input, "ocr_result": result},
                error=None,
                message="파싱 성공",
            )

        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"지원하지 않는 uploaded_evidence_type_code: {evidence_code}",
            )

    except HTTPException:
        raise
    except Exception as e:
        return ParseResponse(
            success=False,
            data=None,
            error=ParseError(code="parse_error", message=str(e)),
            message="파싱 실패",
        )
