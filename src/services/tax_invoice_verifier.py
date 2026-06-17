"""
세금계산서 ↔ 영수증/거래명세표 사전 유효성 검증
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
산업안전관리비 AI 검증 시스템 — tax_invoice_verifier.py

[역할]
  메인 매칭(사용내역서 ↔ 영수증) 이전에 실행되는 사전 검증 단계.
  세금계산서를 기준으로 영수증·거래명세표의 효력을 확인한다.

[처리 결과]
  "verified"   — 세금계산서와 월·금액·업체명(공급자)이 모두 일치
  "unverified" — 매칭되는 세금계산서 없음 (없거나 불일치)

[MVP 검증 정책 — unverified 반려]
  세금계산서 없음 / 금액 불일치(±1% 초과) / 업체명(공급자) 불일치 → 반려(rejected).
  날짜는 월 ±2일 유예로 판정하며, 벗어나면 unverified → 반려.
  ※ 이 모듈은 verified/unverified 판정만 한다. 실제 반려(match_status="rejected") 전환은
    matching_service_monthly.match_all_usage_to_receipts 에서 수행한다.

[매칭 기준 — Hard Gate 3가지]
  Gate 1 — 날짜  : 같은 연월 (또는 월 경계 ±2일)
  Gate 2 — 금액  : |영수증금액 − 세금계산서금액| / max ≤ 1%
  Gate 3 — 업체명: 정규화 후 완전일치 (공급자 기준, 영수증에 업체명 미기재 시 면제)
                  ※ 세금계산서는 공급자/공급받는자 2개 업체명이 있으므로 OCR(vlm)에서
                    vendor 를 '공급자'로 매핑한 값으로 비교한다.
"""

from __future__ import annotations

import re
import logging
from typing import Optional

from src.services.matching_service_monthly import (
    _date_gate_monthly,
    _extract_receipt_date,
    _extract_receipt_vendor,
    _normalize_vendor,
    _find_best_receipt_item_amount,
    _get_item_name,
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
        receipt      : doc_type이 "receipt" 또는 "delivery_statement"인 딕셔너리
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

    # 다품목 대응: 영수증/거래명세표는 품목별 가상 영수증으로 분리되어 total_amount가
    # '해당 품목 공급가액+세액'이다. 세금계산서도 합계(total)가 아니라 같은 품목의
    # 공급가액+세액으로 비교해야 다품목에서 어긋나지 않는다. (매칭용 품목명)
    receipt_item_name = (
        receipt.get("_item_name")
        or _get_item_name((receipt.get("items") or [{}])[0])
        or ""
    )

    best_failed: list[str] = []

    for ti in tax_invoices:
        ti_date   = _extract_receipt_date(ti)
        ti_vendor = _extract_receipt_vendor(ti)
        ti_vendor_clean = _clean_vendor_for_gate(ti_vendor)

        # 세금계산서 금액: 합계(total)가 아니라 영수증 품목과 매칭되는 세금계산서 품목의
        # 공급가액+세액으로 비교한다. 매칭 품목을 못 찾으면 합계금액으로 폴백.
        ti_amount = ti.get("total_amount")
        _ti_items = ti.get("items") or []
        if receipt_item_name and _ti_items:
            _sim, _item_amt = _find_best_receipt_item_amount(
                receipt_item_name, _ti_items, doc_type="tax_invoice"
            )
            if _item_amt is not None:
                ti_amount = _item_amt

        failed: list[str] = []

        # ── Gate 1: 날짜 (연월 및 월 경계 기준) ──────────────────
        if receipt_date and ti_date:
            if not _date_gate_monthly(receipt_date, ti_date):
                failed.append(
                    f"날짜 불일치 (같은 연월 또는 월 경계 ±2일 이내 아님) "
                    f"(영수증: {receipt_date} / 세금계산서: {ti_date})"
                )

        # ── Gate 2: 금액 ────────────────────────────────────────
        if receipt_amount is not None and ti_amount is not None:
            try:
                a1, a2 = int(receipt_amount), int(ti_amount)
                if max(a1, a2) > 0:
                    diff_pct = abs(a1 - a2) / max(a1, a2)
                    if diff_pct > GATE_AMOUNT_PCT:
                        failed.append(
                            f"금액 불일치 (영수증: {a1:,}원 / 세금계산서: {a2:,}원)"
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
        receipts      : doc_type이 "receipt" 또는 "delivery_statement"인 목록
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
