"""
세금계산서 금액 게이트 VAT 보정 회귀 테스트
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
버그: VAT 보정(_resolve_gate2_amount)이 delivery_statement(거래명세표)에만 적용되어,
세금계산서(tax_invoice)는 공급가액(VAT 미포함)으로 비교돼 사용내역서 금액(VAT 포함)과
~9.1% 차이가 나 금액 게이트에서 탈락했다. 그 결과 품목명이 동일(유사도 1.00)해도
"매칭 실패"가 되었다.
수정: 세금계산서도 거래명세표와 동일하게 VAT 보정.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.services.matching_service import (  # noqa: E402
    _check_hard_gates,
    _resolve_gate2_amount,
)


# ── _resolve_gate2_amount ─────────────────────────────────────────────

def test_tax_invoice_vat_correction_with_tax_amount():
    # 공급가액 1,090,908 + 세액 109,092 = 1,200,000
    ti = {"doc_type": "tax_invoice", "total_amount": 1090908, "tax_amount": 109092}
    assert _resolve_gate2_amount(ti, 1200000) == 1200000


def test_tax_invoice_vat_correction_x11_fallback():
    # 세액 필드 없음 → ×1.1 추정으로 게이트 통과
    ti = {"doc_type": "tax_invoice", "total_amount": 1090909}
    corrected = _resolve_gate2_amount(ti, 1200000)
    assert abs(corrected - 1200000) / 1200000 <= 0.01


def test_delivery_statement_still_corrected():
    ds = {"doc_type": "delivery_statement", "total_amount": 1090908, "tax_amount": 109092}
    assert _resolve_gate2_amount(ds, 1200000) == 1200000


def test_receipt_doctype_not_corrected():
    # 일반 영수증(receipt)은 보정하지 않고 raw 그대로
    r = {"doc_type": "receipt", "total_amount": 200000}
    assert _resolve_gate2_amount(r, 200000) == 200000


# ── _check_hard_gates (세금계산서 금액 게이트 통과) ───────────────────

def test_tax_invoice_amount_gate_now_passes():
    usage = {"date": "2026-05-08", "amount": 1200000, "vendor": "",
             "name": "안전모 (ABS 산업용)"}
    receipt = {
        "doc_type": "tax_invoice",
        "total_amount": 1090908,
        "tax_amount": 109092,
        "date": None,  # 세금계산서 날짜 미인식 → 날짜 게이트 스킵
        "items": [{"name": "안전모 (ABS 산업용)"}],
    }
    passed, failed = _check_hard_gates(usage, receipt)
    assert passed, f"게이트 실패: {failed}"
