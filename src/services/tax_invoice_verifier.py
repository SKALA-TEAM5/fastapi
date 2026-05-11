"""
세금계산서 ↔ 영수증/거래명세표 사전 유효성 검증
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
산업안전관리비 AI 검증 시스템 — tax_invoice_verifier.py

[역할]
  메인 매칭(사용내역서 ↔ 영수증) 이전에 실행되는 사전 검증 단계.
  세금계산서를 기준으로 영수증·거래명세표의 효력을 확인한다.

[처리 결과]
  "verified"   — 세금계산서와 월·금액·업체명이 모두 일치
  "unverified" — 매칭되는 세금계산서 없음 (없거나 불일치)
               → Step 2(사용내역서 ↔ 영수증 매칭)에 그대로 포함
               → RAG 단계에서 추가 검토 권장

[unverified 허용 이유]
  건설현장 소액 거래 등 세금계산서가 발행되지 않는 경우가 있으므로
  unverified를 즉시 탈락시키지 않고 표시만 하고 통과시킨다.
  실제 건설현장 데이터에서는 대다수 거래에 세금계산서가 발행되므로
  unverified 비율 자체가 품질 지표로 활용 가능하다.

[매칭 기준 — Hard Gate 3가지]
  Gate 1 — 날짜  : 같은 연월 (또는 월 경계 ±2일)
  Gate 2 — 금액  : |영수증금액 − 세금계산서금액| / max ≤ 1%
  Gate 3 — 업체명: 정규화 후 완전일치 (영수증에 업체명 미기재 시 면제)
"""

from __future__ import annotations

import re
import logging
from typing import Optional

from services.matching_service_monthly import (
    _date_gate_cycle,
    _get_settlement_cycle,
    _which_cycle,
    _extract_receipt_date,
    _extract_receipt_vendor,
    _normalize_vendor,
    GATE_AMOUNT_PCT,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# 내부 유틸
# ══════════════════════════════════════════════════════════════

def _clean_vendor_for_gate(text: str) -> str:
    """업체명 Gate용 정규화 (matching_service_monthly._check_hard_gates와 동일 로직)"""
    if not text:
        return ""
    t = _normalize_vendor(text)
    t = re.sub(r"^[주사유]\s*", "", t)
    t = re.sub(r"\s*[주사유]$", "", t)
    t = re.sub(r"[^가-힣a-zA-Z0-9]", "", t)
    return t.lower()


def _ti_summary(tax_invoice: dict) -> dict:
    """세금계산서 요약본 (결과 포함용)"""
    return {
        "vendor":       _extract_receipt_vendor(tax_invoice),
        "date":         _extract_receipt_date(tax_invoice),
        "total_amount": tax_invoice.get("total_amount"),
        "source_file":  tax_invoice.get("source_file", ""),
    }


# ══════════════════════════════════════════════════════════════
# 핵심 함수 — 단일 영수증 검증
# ══════════════════════════════════════════════════════════════

def verify_one_receipt(
    receipt: dict,
    tax_invoices: list[dict],
) -> dict:
    """
    영수증/거래명세표 1건을 세금계산서 목록과 비교하여 유효성 판정.

    Args:
        receipt      : doc_type이 "receipt" 또는 "transaction_statement"인 딕셔너리
        tax_invoices : doc_type이 "tax_invoice"인 딕셔너리 목록

    Returns:
        {
            "tax_invoice_status":   "verified" | "unverified",
            "matched_tax_invoice":  {요약} | None,
            "failed_gates":         [실패 사유, ...]   # unverified 시 최근접 후보 기준
        }
    """
    if not tax_invoices:
        return {
            "tax_invoice_status":  "unverified",
            "matched_tax_invoice": None,
            "failed_gates":        ["세금계산서 없음"],
        }

    receipt_date   = _extract_receipt_date(receipt)
    receipt_vendor = _extract_receipt_vendor(receipt)
    receipt_amount = receipt.get("total_amount")

    r_vendor_clean = _clean_vendor_for_gate(receipt_vendor)

    best_failed: list[str] = []

    for ti in tax_invoices:
        ti_date   = _extract_receipt_date(ti)
        ti_vendor = _extract_receipt_vendor(ti)
        ti_amount = ti.get("total_amount")
        ti_vendor_clean = _clean_vendor_for_gate(ti_vendor)

        failed: list[str] = []

        # ── Gate 1: 날짜 (정산 사이클 기반) ─────────────────────
        # 세금계산서 검증에서는 영수증 날짜를 기준 사이클 ref_date로 사용.
        # 세금계산서 발행일(목요일)은 결제일(수요일) 다음날이므로
        # 같은 사이클 내에 포함되어 자연스럽게 통과.
        if receipt_date and ti_date:
            if not _date_gate_cycle(receipt_date, ti_date):
                from services.matching_service_monthly import _parse_date_safe
                d_receipt = _parse_date_safe(receipt_date)
                if d_receipt:
                    cs, ce = _which_cycle(d_receipt)   # _get_settlement_cycle → _which_cycle
                    cycle_str = f"{cs.strftime('%Y-%m-%d')} ~ {ce.strftime('%Y-%m-%d')}"
                else:
                    cycle_str = "계산 불가"
                failed.append(
                    f"날짜 정산 사이클 불일치 "
                    f"(영수증: {receipt_date} / 세금계산서: {ti_date}, "
                    f"허용 사이클: {cycle_str})"
                )

        # ── Gate 2: 금액 ────────────────────────────────────────
        if receipt_amount is not None and ti_amount is not None:
            try:
                a1, a2 = int(receipt_amount), int(ti_amount)
                if max(a1, a2) > 0:
                    diff_pct = abs(a1 - a2) / max(a1, a2)
                    if diff_pct > GATE_AMOUNT_PCT:
                        failed.append(
                            f"금액 {diff_pct * 100:.1f}% 차이 "
                            f"(영수증: {a1:,}원 / 세금계산서: {a2:,}원)"
                        )
            except (TypeError, ValueError):
                failed.append("금액 파싱 오류")

        # ── Gate 3: 업체명 ──────────────────────────────────────
        if r_vendor_clean:                          # 영수증에 업체명 있을 때만 검사
            if not ti_vendor_clean or r_vendor_clean != ti_vendor_clean:
                failed.append(
                    f"업체명 불일치 "
                    f"(영수증: '{receipt_vendor}' / 세금계산서: '{ti_vendor}')"
                )

        if not failed:
            # 모든 Gate 통과 → verified
            logger.debug(
                "세금계산서 검증 통과: %s ↔ %s",
                receipt.get("source_file", "-"),
                ti.get("source_file", "-"),
            )
            return {
                "tax_invoice_status":  "verified",
                "matched_tax_invoice": _ti_summary(ti),
                "failed_gates":        [],
            }

        # 가장 적게 실패한 후보를 기록 (진단 목적)
        if not best_failed or len(failed) < len(best_failed):
            best_failed = failed

    logger.debug(
        "세금계산서 검증 실패: %s — %s",
        receipt.get("source_file", "-"),
        best_failed,
    )
    return {
        "tax_invoice_status":  "unverified",
        "matched_tax_invoice": None,
        "failed_gates":        best_failed,
    }


# ══════════════════════════════════════════════════════════════
# 배치 함수 — 영수증 전체 검증
# ══════════════════════════════════════════════════════════════

def verify_receipts_against_tax_invoices(
    receipts: list[dict],
    tax_invoices: list[dict],
) -> list[dict]:
    """
    영수증/거래명세표 목록 전체에 세금계산서 검증 결과를 추가하여 반환.

    각 영수증 딕셔너리에 다음 필드가 추가된다:
        "tax_invoice_status"   : "verified" | "unverified"
        "matched_tax_invoice"  : {요약} | None
        "ti_failed_gates"      : [실패 사유]  (unverified 시)

    Args:
        receipts      : doc_type이 "receipt" 또는 "transaction_statement"인 목록
        tax_invoices  : doc_type이 "tax_invoice"인 목록

    Returns:
        검증 결과가 추가된 영수증 목록 (원본 딕셔너리는 변경하지 않음)
    """
    verified_count   = 0
    unverified_count = 0
    result: list[dict] = []

    for receipt in receipts:
        verification = verify_one_receipt(receipt, tax_invoices)
        enriched = {
            **receipt,
            "tax_invoice_status":  verification["tax_invoice_status"],
            "matched_tax_invoice": verification["matched_tax_invoice"],
            "ti_failed_gates":     verification["failed_gates"],
        }
        result.append(enriched)

        if verification["tax_invoice_status"] == "verified":
            verified_count += 1
        else:
            unverified_count += 1

    logger.info(
        "세금계산서 사전 검증 완료: 총 %d건 — verified %d / unverified %d",
        len(receipts), verified_count, unverified_count,
    )
    return result
