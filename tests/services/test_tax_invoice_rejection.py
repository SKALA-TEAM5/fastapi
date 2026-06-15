"""
세금계산서 매칭 반려 회귀 테스트
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
버그: 세금계산서를 레거시 CLOVA 경로로 파싱하면 items가 빈 배열로 나와,
매칭에서 "영수증 품목명 없음 — 반려 처리"로 오반려되었다.
수정: 세금계산서도 VLM(OCR 엔진) 경로로 파싱해 items를 정상 추출.

이 테스트는 _check_rejection의 계약을 잠근다:
  - items에 품목명이 있으면 반려하지 않는다(수정 후 기대 동작).
  - items가 비면 "영수증 품목명 없음"으로 반려한다(버그 재현 조건).
  - wage_statement는 items가 비어도 면제한다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.services.matching_service_monthly import _check_rejection  # noqa: E402

USAGE_ITEM = {
    "name": "낙하물 방지망 구입",
    "description": "낙하물 방지망 구입",
    "category": "",
}


def test_tax_invoice_with_items_not_rejected():
    receipt = {
        "doc_type": "tax_invoice",
        "infer_result": "SUCCESS",
        "items": [{"name": "낙하물 방지망", "amount": 1200000}],
    }
    assert _check_rejection(USAGE_ITEM, receipt) is None


def test_tax_invoice_empty_items_rejected_with_context_reason():
    # 세금계산서 빈 items → 세금계산서 전용 안내 문구
    receipt = {"doc_type": "tax_invoice", "infer_result": "SUCCESS", "items": []}
    reason = _check_rejection(USAGE_ITEM, receipt)
    assert reason == "세금계산서에 품목 내역이 없어 매칭 불가 — 품목이 기재된 거래명세표/영수증 증빙 필요"


def test_receipt_empty_items_reason():
    # 영수증 빈 items → 영수증 품목 인식 실패 문구
    receipt = {"doc_type": "receipt", "infer_result": "SUCCESS", "items": []}
    assert _check_rejection(USAGE_ITEM, receipt) == "영수증 품목 인식 실패 — 증빙 재제출 필요"


def test_delivery_statement_empty_items_reason():
    # 거래명세표 빈 items → 거래명세표 품목 인식 실패 문구
    receipt = {"doc_type": "delivery_statement", "infer_result": "SUCCESS", "items": []}
    assert _check_rejection(USAGE_ITEM, receipt) == "거래명세표 품목 인식 실패 — 증빙 재제출 필요"


def test_item_name_key_also_accepted():
    receipt = {
        "doc_type": "tax_invoice",
        "infer_result": "SUCCESS",
        "items": [{"item_name": "안전모 (ABS 산업용)"}],
    }
    assert _check_rejection(USAGE_ITEM, receipt) is None


def test_wage_statement_empty_items_exempt():
    receipt = {"doc_type": "wage_statement", "infer_result": "SUCCESS", "items": []}
    assert _check_rejection(USAGE_ITEM, receipt) is None


def test_ocr_failure_rejected():
    receipt = {"doc_type": "tax_invoice", "infer_result": "FAILURE", "items": []}
    reason = _check_rejection(USAGE_ITEM, receipt)
    assert reason and "OCR 인식 실패" in reason
