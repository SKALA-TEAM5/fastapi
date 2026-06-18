"""
OCR — 세금계산서 VLM 전환 검증 테스트
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
배포 전 직접 테스트 항목 1·2·3·5를 자동화한 테스트.

VLM 실제 호출(call_vision)과 MinIO(fetch_file)를 모킹하여 API 키·네트워크 없이
실제 매핑·라우팅·검증 로직을 검증한다.

검증 항목
  1. 세금계산서 이미지 → VLM 파싱 결과가 표준 스키마로 정확히 매핑되는가
     (+ /tax-invoices/ocr, /ocr/parse 엔드포인트 응답)
  2. /ocr/parse (evidence_code=tax_invoice) 라우팅
  3. 세금계산서 PDF도 VLM 경로로 가는가 (전면 VLM 전환)
  5. 검증 경고(필수 필드 누락 / 품목 합산 불일치 / VLM 오류) 동작

모킹 방식
  call_vision 등은 문자열 경로가 아니라 import한 모듈 객체를 patch.object로 패치한다.
  (editable 설치 환경에서 지연 import된 서브모듈을 문자열로 못 찾는 문제 회피)

실행:
    python -m pytest tests/services/test_ocr_tax_invoice_vlm.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ocr import parse_tax_invoice as pti
from src.ocr import vlm_ocr  # patch.object 대상으로 직접 import (문자열 경로 회피)


# ──────────────────────────────────────────────────────────────────────
# 공통 픽스처 — VLM 원시 응답 샘플 (vlm_ocr._TAX_INVOICE_PROMPT 스키마)
# ──────────────────────────────────────────────────────────────────────

def _vlm_ok() -> dict:
    """정상 세금계산서 VLM 응답 (내부 정합: 공급가액 450,000 + 세액 45,000 = 495,000).

    품목 공급가액 합(200,000+250,000=450,000) == 공급가액합계(495,000−45,000) 가 되도록 구성.
    """
    return {
        "doc_type": "tax_invoice",
        "infer_result": "SUCCESS",
        "supplier": {"name": "(주)안전상사", "biz_num": "1234567890"},
        "buyer": {"name": "(주)건설현장", "biz_num": "987-65-43210"},
        "date": "2026-04-15",
        "items": [
            {"name": "안전모", "count": 10, "unit_price": 20000,
             "amount": 200000, "tax_amount": 20000},
            {"name": "안전화", "count": 5, "unit_price": 50000,
             "amount": 250000, "tax_amount": 25000},
        ],
        "total_amount": 495000,
        "tax_amount": 45000,
        "confidence": 0.95,
        "fail_reason": None,
    }


# ══════════════════════════════════════════════════════════════════════
# [항목 1] 세금계산서 이미지 → VLM 매핑 정확성
# ══════════════════════════════════════════════════════════════════════

@patch.object(vlm_ocr, "call_vision")
def test_image_maps_to_standard_schema(mock_call_vision):
    """VLM 출력이 parse_tax_invoice 표준 스키마로 정확히 매핑된다."""
    mock_call_vision.return_value = _vlm_ok()

    result = pti.parse_with_vlm("세금계산서.jpg")

    # 공급자 / 공급받는자
    assert result["supplier"]["company_name"] == "(주)안전상사"
    assert result["supplier"]["business_number"] == "123-45-67890"  # 정규화됨
    assert result["buyer"]["company_name"] == "(주)건설현장"
    assert result["buyer"]["business_number"] == "987-65-43210"
    # 작성일자(누락 위험으로 별도 보존한 필드)
    assert result["issue_date"] == "2026-04-15"
    # 합계 / 세액 / 공급가액(= 합계 − 세액)
    assert result["total_amount"] == 495000
    assert result["total_tax_amount"] == 45000
    assert result["total_supply_amount"] == 495000 - 45000


@patch.object(vlm_ocr, "call_vision")
def test_image_items_mapping(mock_call_vision):
    """품목 VLM 키(name/count/unit_price/amount/tax_amount) → 표준 키 매핑."""
    mock_call_vision.return_value = _vlm_ok()

    items = pti.parse_with_vlm("세금계산서.png")["items"]

    assert len(items) == 2
    first = items[0]
    assert first["item_name"] == "안전모"
    assert first["quantity"] == 10
    assert first["unit_price"] == 20000
    assert first["supply_amount"] == 200000
    assert first["tax_amount"] == 20000


@patch.object(vlm_ocr, "call_vision")
def test_image_calls_vlm_with_tax_invoice_hint(mock_call_vision):
    """이미지는 type_hint='tax_invoice'로 VLM을 호출한다."""
    mock_call_vision.return_value = _vlm_ok()

    pti.parse_with_vlm("세금계산서.jpg")

    mock_call_vision.assert_called_once()
    _, kwargs = mock_call_vision.call_args
    assert kwargs.get("type_hint") == "tax_invoice"


# ══════════════════════════════════════════════════════════════════════
# [항목 3] 세금계산서 PDF도 VLM 경로 — 전면 VLM 전환
# ══════════════════════════════════════════════════════════════════════

@patch.object(pti, "parse_from_pdf")
@patch.object(vlm_ocr, "call_vision")
def test_pdf_goes_through_vlm_not_pdfplumber(mock_call_vision, mock_parse_from_pdf):
    """PDF 입력도 VLM(call_vision)으로 처리하고 pdfplumber(parse_from_pdf)는 쓰지 않는다."""
    mock_call_vision.return_value = _vlm_ok()

    result = pti.parse_tax_invoice("세금계산서.pdf")

    mock_call_vision.assert_called_once()          # VLM 사용됨
    mock_parse_from_pdf.assert_not_called()        # pdfplumber 미사용
    assert result["parse_method"] == "vlm"
    assert result["file_type"] == "pdf"
    assert result["total_amount"] == 495000


@patch.object(vlm_ocr, "call_vision")
def test_image_dispatch_uses_vlm(mock_call_vision):
    """이미지 입력도 디스패처를 통해 VLM으로 간다."""
    mock_call_vision.return_value = _vlm_ok()

    result = pti.parse_tax_invoice("세금계산서.jpeg")

    mock_call_vision.assert_called_once()
    assert result["parse_method"] == "vlm"
    assert result["file_type"] == "image"


def test_unsupported_extension_returns_warning():
    """지원하지 않는 확장자는 경고와 함께 빈 결과를 반환(기능 보존)."""
    result = pti.parse_tax_invoice("문서.hwp")
    assert result["parse_method"] == "none"
    assert any("지원하지 않는" in w for w in result["validation"]["warnings"])


# ══════════════════════════════════════════════════════════════════════
# [항목 5] 검증 경고 동작
# ══════════════════════════════════════════════════════════════════════

@patch.object(vlm_ocr, "call_vision")
def test_valid_invoice_has_no_warnings(mock_call_vision):
    """정상 세금계산서는 is_valid=True, 경고 없음."""
    mock_call_vision.return_value = _vlm_ok()

    val = pti.parse_with_vlm("세금계산서.jpg")["validation"]

    assert val["is_valid"] is True
    assert val["warnings"] == []


@patch.object(vlm_ocr, "call_vision")
def test_missing_required_fields_warns(mock_call_vision):
    """필수 필드(사업자번호/작성일자/합계)가 없으면 경고 + is_valid=False."""
    raw = _vlm_ok()
    raw["supplier"]["biz_num"] = None
    raw["date"] = None
    raw["total_amount"] = None
    mock_call_vision.return_value = raw

    val = pti.parse_with_vlm("세금계산서.jpg")["validation"]

    assert val["is_valid"] is False
    assert any("필수 필드 누락" in w for w in val["warnings"])


@patch.object(vlm_ocr, "call_vision")
def test_item_supply_sum_mismatch_warns(mock_call_vision):
    """품목 공급가액 합산이 공급가액합계와 다르면 경고."""
    raw = _vlm_ok()
    raw["total_amount"] = 1200000
    raw["tax_amount"] = 200000          # → total_supply = 1,000,000
    raw["items"] = [{"name": "안전모", "count": 1, "unit_price": 500000,
                     "amount": 500000, "tax_amount": 50000}]  # 합산 500,000
    mock_call_vision.return_value = raw

    val = pti.parse_with_vlm("세금계산서.jpg")["validation"]

    assert any("품목 공급가액 합산" in w for w in val["warnings"])


@patch.object(vlm_ocr, "call_vision")
def test_vlm_error_is_handled_gracefully(mock_call_vision):
    """VLM이 오류를 반환하면 예외 없이 경고로 처리한다."""
    mock_call_vision.return_value = {"error": "Gemini API 호출 실패: timeout"}

    result = pti.parse_with_vlm("세금계산서.jpg")

    assert result["validation"]["is_valid"] is False
    assert any("VLM 오류" in w for w in result["validation"]["warnings"])


@patch.object(vlm_ocr, "call_vision")
def test_vlm_failed_infer_warns(mock_call_vision):
    """infer_result=FAILED 이면 판독 실패 경고를 남긴다."""
    raw = _vlm_ok()
    raw["infer_result"] = "FAILED"
    raw["fail_reason"] = "이미지 판독 불가"
    mock_call_vision.return_value = raw

    warnings = pti.parse_with_vlm("세금계산서.jpg")["validation"]["warnings"]
    assert any("VLM 판독 실패" in w for w in warnings)


# ══════════════════════════════════════════════════════════════════════
# [항목 1·2·3] 엔드포인트 테스트 (TestClient, VLM·MinIO 모킹)
#   fastapi 미설치 환경에서는 자동 skip.
#   라우터 모듈은 테스트 내부에서 import해 patch.object로 모킹(문자열 경로 회피).
# ══════════════════════════════════════════════════════════════════════

def _make_client(router_module_name: str):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import importlib
    router_mod = importlib.import_module(router_module_name)
    app = fastapi.FastAPI()
    app.include_router(router_mod.router)
    return TestClient(app), router_mod


def test_endpoint_tax_invoices_ocr_image():
    """[항목 1] POST /tax-invoices/ocr (이미지) → 200 + 매핑된 응답."""
    client, _ = _make_client("src.api.routers.tax_invoices")

    with patch.object(vlm_ocr, "call_vision", return_value=_vlm_ok()) as m:
        resp = client.post(
            "/tax-invoices/ocr",
            files={"file": ("세금계산서.jpg", b"fakeimagebytes", "image/jpeg")},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["supplier"]["company_name"] == "(주)안전상사"
    assert data["supplier"]["business_number"] == "123-45-67890"
    assert data["total_amount"] == 495000
    m.assert_called_once()


def test_endpoint_tax_invoices_ocr_pdf():
    """[항목 3] POST /tax-invoices/ocr (PDF) → 200, PDF도 VLM 경로."""
    client, _ = _make_client("src.api.routers.tax_invoices")

    with patch.object(vlm_ocr, "call_vision", return_value=_vlm_ok()) as m:
        resp = client.post(
            "/tax-invoices/ocr",
            files={"file": ("세금계산서.pdf", b"%PDF-1.4 fake", "application/pdf")},
        )

    assert resp.status_code == 200
    assert resp.json()["total_amount"] == 495000
    m.assert_called_once()  # PDF가 VLM으로 처리됨


def test_endpoint_parse_tax_invoice():
    """[항목 2] POST /ocr/parse (evidence_code=tax_invoice) → 성공 + ocr_result."""
    client, parse_mod = _make_client("src.api.routers.parse")

    body = {
        "id": 1,
        "uploaded_evidence_type_code": "tax_invoice",
        "original_filename": "세금계산서_20260415.jpg",
        "storage_key": "projects/10/tax/세금계산서_20260415.jpg",
        "mime_type": "image/jpeg",
    }

    with patch.object(parse_mod, "fetch_file", return_value=b"fakebytes"), \
         patch.object(parse_mod, "create_presigned_file_url", return_value="http://x/presigned"), \
         patch.object(vlm_ocr, "call_vision", return_value=_vlm_ok()) as m:
        resp = client.post("/ocr/parse", json=body)

    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    ocr_result = data["data"]["ocr_result"]
    assert ocr_result["supplier"]["company_name"] == "(주)안전상사"
    assert ocr_result["total_amount"] == 495000
    m.assert_called_once()
