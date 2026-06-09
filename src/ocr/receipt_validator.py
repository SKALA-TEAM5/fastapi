"""
영수증 파싱 결과 후처리 검증 모듈
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
산업안전관리비 AI 검증 시스템 — receipt_validator.py

[역할]
  VLM / CLOVA 등 OCR 엔진에 무관하게 파싱 결과를 검증한다.
  clova_ocr_receipt.py 의존성을 제거하고 독립 모듈로 분리.

[검증 항목]
  1. 필수 필드 존재 여부  (store.biz_num / payment.date / total_amount)
  2. 품목 금액 합산 vs 총액 일치 여부
  3. 부가세 10% VAT 정합성
  4. 단가 × 수량 = 품목 합계 검증
  5. 날짜 미래값 / 이상값 탐지
  6. 사업자등록번호 형식 검증
"""

from __future__ import annotations

import re
from datetime import date as _date


def validate_result(parsed: dict) -> dict:
    """
    파싱 결과 후처리 검증.

    Returns:
        validation 키가 추가/갱신된 parsed dict
    """
    flags = {
        "items_sum_match":     None,
        "tax_calc_match":      None,
        "has_required_fields": False,
        "missing_fields":      [],
        "warnings":            [],
        "errors":              [],
    }

    # ── 필수 필드 점검 ──────────────────────────────
    # 날짜(payment.date)는 date_status에 따라 처리 방식이 다르므로
    # has_required_fields 판단에서 제외하고 별도 경고로 처리한다.
    # - recognized : 날짜 정상 추출 → 필수 필드 충족
    # - not_written : 기재 없음      → 경고만 (매칭 엔진에서 플래그)
    # - unreadable  : 판독 불가      → 경고만 (매칭 엔진에서 플래그)
    hard_required = ["store.biz_num", "total_amount"]   # 날짜 제외
    missing = []
    for field in hard_required:
        keys = field.split(".")
        val = parsed
        for k in keys:
            val = val.get(k) if isinstance(val, dict) else None
        if val is None:
            missing.append(field)

    # 날짜 상태 별도 점검
    date_status = (parsed.get("payment") or {}).get("date_status", "recognized")
    pay_date    = (parsed.get("payment") or {}).get("date")
    if date_status == "not_written":
        flags["warnings"].append("날짜 미기재 — 담당자 확인 필요")
    elif date_status == "unreadable":
        flags["warnings"].append("날짜 판독 불가 — 담당자 확인 필요")
    elif pay_date is None:
        # date_status 없이 날짜만 null인 경우 (구버전 호환)
        missing.append("payment.date")

    flags["missing_fields"]      = missing
    flags["has_required_fields"] = (len(missing) == 0)
    if missing:
        flags["warnings"].append(f"필수 필드 누락: {', '.join(missing)}")

    # ── 사업자등록번호 형식 검증 ─────────────────────
    biz_num = (parsed.get("store") or {}).get("biz_num") or ""
    if biz_num:
        if not re.match(r"^\d{3}-\d{2}-\d{5}$", biz_num):
            flags["warnings"].append(
                f"사업자등록번호 형식 이상: '{biz_num}' — 000-00-00000 형식 확인 필요"
            )

    # ── 품목 합산 검증 ──────────────────────────────
    #   ±1원  → 정상
    #   ±2~10원 → 반올림 의심 경고 (통과)
    #   10원 초과 → 금액 불일치 오류 (실패)
    total = parsed.get("total_amount")
    tax   = parsed.get("tax_amount") or 0
    items = parsed.get("items", [])

    if total and items:
        items_sum     = sum(item.get("amount") or 0 for item in items)
        if items_sum > 0:
            diff_direct   = abs(total - items_sum)
            diff_with_tax = abs(total - items_sum - tax)
            best_diff     = min(diff_direct, diff_with_tax)

            if best_diff <= 1:
                flags["items_sum_match"] = True
            elif best_diff <= 10:
                flags["items_sum_match"] = True
                flags["warnings"].append(
                    f"품목 합산 반올림 의심 ({best_diff}원 차이): "
                    f"품목합산 {items_sum:,}원, 총액 {total:,}원 — 검토 권장"
                )
            else:
                flags["items_sum_match"] = False
                flags["errors"].append(
                    f"품목 합산 불일치: 품목합산({items_sum:,}원) ≠ 총액({total:,}원) "
                    f"— 차이 {diff_direct:,}원, 반드시 확인 필요"
                )

    # ── 부가세 검증 (10% VAT) ───────────────────────
    if total and tax:
        expected_tax = round(total / 11)
        diff = abs(tax - expected_tax)
        flags["tax_calc_match"] = (diff <= 10)
        if diff > 10:
            flags["warnings"].append(
                f"부가세 불일치: 인식값={tax:,}원, 역산값={expected_tax:,}원"
            )

    # ── 단가 × 수량 = 품목 합계 검증 ────────────────
    # 영수증: 부가세 포함 금액 표기  → unit_price × count ≈ amount
    # 거래명세표: 부가세 별도 공급가액 표기 → unit_price × count / 1.1 ≈ amount
    # 둘 중 하나라도 ±1원 이내로 일치하면 통과
    for item in parsed.get("items", []):
        iname      = item.get("name") or "미상"
        unit_price = item.get("unit_price")
        amount     = item.get("amount")
        count      = item.get("count")
        if unit_price and amount and count:
            expected_incl = unit_price * count               # 부가세 포함
            expected_excl = round(unit_price * count / 1.1)  # 부가세 제외 공급가액

            match_incl = abs(expected_incl - amount) <= 1
            match_excl = abs(expected_excl - amount) <= 1

            if not match_incl and not match_excl:
                flags["warnings"].append(
                    f"단가×수량≠합계: '{iname}' "
                    f"단가({unit_price:,}원) × 수량({count}개) = {expected_incl:,}원 ≠ 합계({amount:,}원)"
                )

    # ── 날짜 이상값 탐지 ─────────────────────────────
    pay_date = (parsed.get("payment") or {}).get("date")
    if pay_date:
        try:
            parsed_date = _date.fromisoformat(pay_date)
            if parsed_date > _date.today():
                flags["errors"].append(
                    f"날짜 미래값: {pay_date} — 환각 의심, 반드시 확인 필요"
                )
            elif parsed_date.year < 2000:
                flags["warnings"].append(
                    f"날짜 이상값: {pay_date} — 2000년 이전, 확인 권장"
                )
        except ValueError:
            flags["warnings"].append(f"날짜 형식 오류: '{pay_date}' — YYYY-MM-DD 형식 확인 필요")

    # ── 간이영수증 / 불완전 인식 감지 ────────────────
    if not parsed.get("items") and not (parsed.get("payment") or {}).get("date"):
        flags["warnings"].append(
            "간이영수증 또는 불완전 인식 의심 — 품목 목록 없음, 결제일 없음"
        )

    parsed["validation"] = flags
    return parsed
