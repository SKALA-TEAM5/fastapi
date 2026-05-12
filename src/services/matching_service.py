"""
2-way 매칭 엔진: 사용내역서 항목 ↔ 영수증 OCR 결과
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
산업안전관리비 AI 검증 시스템  —  matching_service.py  v4

[역할]
  src/ocr/parse_usage_statement.py  → 사용내역서 항목 JSON
  src/ocr/clova_ocr_receipt.py      → 영수증 OCR JSON
  이 두 가지를 연결해 매칭 결과를 출력한다.

  ※ 현장사진 텍스트 분석은 비전 모델 파트에서 별도 담당.

[매칭 전략 — Hard Gate + 점수 보조]
  1단계 Hard Gate (3가지 조건, 모두 통과해야 후보 인정)
    · 날짜 Gate  : |사용일자 − 영수증일자| ≤ 1일
    · 금액 Gate  : |사용금액 − 영수증금액| / max ≤ 1%
    · 업체명 Gate: 정규화 후 완전일치
                   (사용내역서에 업체명 미기재 시 면제)
  → Gate 미통과 시 즉시 unmatched (점수 계산 없이)

  2단계 보조 점수 (Gate 통과 영수증에 대해 선별 기준)
    · 동일 Gate 통과 영수증이 여러 개일 때 최고 점수 선택
    · matched / review_needed 구분 (0.85 / 0.75 임계값)
  - 품목명 없는 영수증 → rejected(반려) 처리

[출력 형태]
  {
    "match_id": "...",
    "usage_item": {...},       // 사용내역서 항목
    "receipt": {...},          // 영수증 OCR 결과 요약
    "similarity_score": 0.92,
    "component_scores": {...}, // 세부 점수
    "match_status": "matched" | "review_needed" | "unmatched" | "rejected",
    "reject_reason": "..."
  }

사용법:
    # 단일 매칭
    from matching_engine import match_twoway
    result = match_twoway(usage_item, receipt_ocr)

    # 배치 매칭 (사용내역서 전체 ↔ 영수증 목록)
    from matching_engine import match_all_usage_to_receipts
    results = match_all_usage_to_receipts(usage_statement, receipts)
"""

from __future__ import annotations

import re
import json
import uuid
import logging
import argparse
import calendar
from pathlib import Path
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from typing import Optional

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# 0. 설정 상수
# ══════════════════════════════════════════════════════════════

# ── 3단계 매칭 임계값 ───────────────────────────────────────────
# 0.85 이상         → "matched"       (자동 통과)
# 0.75 이상 ~ 0.85 미만 → "review_needed" (담당자 검토 필요)
# 0.75 미만         → "unmatched"     (자동 반려)
THRESHOLD_MATCHED: float       = 0.85
THRESHOLD_REVIEW:  float       = 0.75
# 하위 호환성 유지용 (CLI --threshold 기본값에 사용)
MATCH_THRESHOLD:   float       = THRESHOLD_REVIEW

# ── Hard Gate 허용 오차 ─────────────────────────────────────────────
# 날짜: 정산 사이클 범위 내 (전달 마지막 목요일 다음날 ~ 이번달 마지막 수요일)
# 금액: ±1%  /  업체명: 정규화 후 완전일치
#
# [정산 사이클 정의]
#   결제: 마지막 수요일 / 세금계산서: 목요일 / 사용내역서: 금요일
#   해당 주에 수·목·금 없으면 전주 실행
#   → 한 달 정산 범위: 전달 마지막 목요일 다음날 ~ 이번달 마지막 수요일
GATE_AMOUNT_PCT:  float = 0.01  # 금액 허용 오차 (1%)


# 점수 가중치 (합산 = 1.0)
# ※ 현장사진 텍스트 항목 제거 (비전 모델 파트 담당으로 분리)
WEIGHTS: dict[str, float] = {
    "date":      0.30,   # 날짜 근접도
    "amount":    0.35,   # 금액 일치
    "vendor":    0.20,   # 거래처/점포명 유사도
    "item_desc": 0.15,   # 품목명·내역 키워드 유사도
}


# ══════════════════════════════════════════════════════════════
# 1. 텍스트 유사도 유틸리티
# ══════════════════════════════════════════════════════════════

def _normalize_vendor(text: str) -> str:
    """
    거래처명 정규화:
    - `(주)`, `(株)`, `㈜` → 모두 `주` 로 통일
    - `(유)`, `㈔`, `(사)` 등 법인 표기도 통일
    - 앞뒤 공백 제거
    """
    if not text:
        return ""
    # ㈜ (U+338E) / (주) / (株) → 주
    text = text.replace("㈜", "주").replace("(주)", "주").replace("(株)", "주")
    # ㈔ (U+3214) / (사) → 사
    text = text.replace("㈔", "사").replace("(사)", "사")
    # (유) → 유
    text = text.replace("(유)", "유")
    return text.strip()


def _normalize(text: str) -> str:
    """
    텍스트 정규화:
    - 소문자 변환
    - 특수문자·괄호·조사 제거 (단, 한글·숫자·영어 유지)
    - 연속 공백 단일화
    """
    if not text:
        return ""
    # 괄호 및 특수문자 제거
    text = re.sub(r"[（）()\[\]{}<>【】「」『』""''·•※…]", " ", text)
    # 조사/단위 패턴 제거 (구입, 형, 개, 켤레 등 단독 형태소 제거는 하지 않음 — 오히려 매칭에 방해)
    text = re.sub(r"[^가-힣a-zA-Z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text.strip()).lower()


def _bigrams(text: str) -> set[str]:
    """한글 텍스트용 2-gram (공백 제거 후 문자 단위)"""
    clean = re.sub(r"\s", "", text)
    if len(clean) < 2:
        return set()
    return {clean[i:i+2] for i in range(len(clean) - 1)}


def _tokens(text: str) -> set[str]:
    """공백 기준 토큰화 (길이 1 토큰 제외)"""
    return {t for t in text.split() if len(t) > 1}


def _jaccard(set_a: set, set_b: set) -> float:
    """Jaccard 유사도 (교집합 / 합집합)"""
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def text_similarity(a: str, b: str) -> float:
    """
    한글 텍스트 유사도 (0.0 ~ 1.0).

    세 가지 지표를 혼합:
    1. bigram Jaccard  (한글 n-gram, 어순 무관)
    2. token  Jaccard  (단어 수준 overlap)
    3. SequenceMatcher (순서 반영 문자열 유사도)

    가중 평균: bigram 40% + token 30% + sequence 30%
    """
    if not a or not b:
        return 0.0

    a_n = _normalize(a)
    b_n = _normalize(b)

    if not a_n or not b_n:
        return 0.0

    bigram_sim  = _jaccard(_bigrams(a_n), _bigrams(b_n))
    token_sim   = _jaccard(_tokens(a_n),  _tokens(b_n))
    seq_sim     = SequenceMatcher(None, a_n, b_n).ratio()

    return round(0.40 * bigram_sim + 0.30 * token_sim + 0.30 * seq_sim, 4)


# ══════════════════════════════════════════════════════════════
# 2. 날짜 / 금액 점수 계산
# ══════════════════════════════════════════════════════════════

def _parse_date_safe(date_str: Optional[str]) -> Optional[datetime]:
    """YYYY-MM-DD 파싱, 실패 시 None"""
    if not date_str:
        return None
    try:
        return datetime.strptime(str(date_str).strip()[:10], "%Y-%m-%d")
    except ValueError:
        return None


# ══════════════════════════════════════════════════════════════
# 2-a. 정산 사이클 기반 날짜 Gate (Gate 1)
# ══════════════════════════════════════════════════════════════

def _last_weekday_of_month(year: int, month: int, weekday: int) -> datetime:
    """
    해당 연월의 마지막 특정 요일 반환.
    weekday: 0=월, 1=화, 2=수, 3=목, 4=금, 5=토, 6=일
    """
    last_day  = calendar.monthrange(year, month)[1]
    last_date = datetime(year, month, last_day)
    days_back = (last_date.weekday() - weekday) % 7
    return last_date - timedelta(days=days_back)


def _get_settlement_cycle(ref_date: datetime) -> tuple[datetime, datetime]:
    """
    ref_date 달 기준 정산 사이클의 시작·종료일 반환.

    cycle_end   = ref_date 달의 마지막 수요일 (결제일)
    cycle_start = 전달 마지막 목요일 + 1일

    예) ref_date = 2026-04-15
      4월 마지막 수요일 → 4/29  (cycle_end)
      3월 마지막 목요일 → 3/26
      cycle_start       → 3/27
    """
    last_wed = _last_weekday_of_month(ref_date.year, ref_date.month, weekday=2)

    first_of_month  = ref_date.replace(day=1)
    prev_month_last = first_of_month - timedelta(days=1)
    last_thu_prev   = _last_weekday_of_month(
        prev_month_last.year, prev_month_last.month, weekday=3
    )

    cycle_start = last_thu_prev + timedelta(days=1)
    cycle_end   = last_wed
    return cycle_start, cycle_end


def _which_cycle(date: datetime) -> tuple[datetime, datetime]:
    """
    특정 날짜가 실제로 귀속되는 정산 사이클 반환.

    탐색 순서: 현재달 사이클 → 다음달 사이클 (최대 2단계)

    예) 2026-03-27
      · 3월 사이클: 2026-02-27 ~ 2026-03-25  → 03-27 범위 밖
      · 4월 사이클: 2026-03-27 ~ 2026-04-29  → 03-27 범위 안 ✅
    """
    cs, ce = _get_settlement_cycle(date)
    if cs <= date <= ce:
        return cs, ce
    # 현재달 사이클 밖 → 다음달 사이클에 귀속
    next_month_first = (date.replace(day=1) + timedelta(days=32)).replace(day=1)
    return _get_settlement_cycle(next_month_first)


def _date_gate_cycle(usage_date: Optional[str], doc_date: Optional[str]) -> bool:
    """
    정산 사이클 기반 날짜 Gate.

    통과 조건:
      ① 날짜 중 하나라도 없거나 파싱 실패 → 통과 (면제)
      ② usage_date 가 속한 정산 사이클 내에 doc_date 포함 → 통과
      ③ 그 외 → 실패
    """
    if not usage_date or not doc_date:
        return True

    d_usage = _parse_date_safe(usage_date)
    d_doc   = _parse_date_safe(doc_date)

    if d_usage is None or d_doc is None:
        return True

    cycle_start, cycle_end = _which_cycle(d_usage)
    return cycle_start <= d_doc <= cycle_end


# ══════════════════════════════════════════════════════════════
# 2-b. 영수증 필드 추출 헬퍼 — 신·구 포맷 공통 지원
# ══════════════════════════════════════════════════════════════

def _extract_receipt_date(receipt: dict) -> Optional[str]:
    """
    영수증 딕셔너리에서 날짜를 추출한다.

    포맷 우선순위:
      ① 신버전 통합 포맷: receipt["date"]
         (영수증: 카드 승인일시 / 거래명세표: 작성일자)
      ② 구버전 CLOVA 포맷: receipt["payment"]["date"]
      ③ match_format 변환 후: receipt["payment"]["date"]  (동일)
    """
    date_top = receipt.get("date")
    if date_top:
        return date_top
    return (receipt.get("payment") or {}).get("date")


def _extract_receipt_vendor(receipt: dict) -> str:
    """
    영수증 딕셔너리에서 업체명을 추출한다.

    포맷 우선순위:
      ① 신버전 통합 포맷: receipt["vendor"]
      ② match_format / 구버전: receipt["store"]["name"]
    """
    vendor_top = receipt.get("vendor") or ""
    if vendor_top:
        return vendor_top
    return (receipt.get("store") or {}).get("name") or ""


def date_score(date1: Optional[str], date2: Optional[str]) -> Optional[float]:
    """
    날짜 근접도 점수.
    - 두 날짜 모두 없으면 None (가중치 재분배)
    - 한쪽만 없으면 0.4 (부분 불확실)
    - 당일: 1.0 / 3일 이내: 0.85 / 7일 이내: 0.60 / 14일 이내: 0.30 / 초과: 0.0
    """
    d1 = _parse_date_safe(date1)
    d2 = _parse_date_safe(date2)

    if d1 is None and d2 is None:
        return None
    if d1 is None or d2 is None:
        return 0.4

    diff = abs((d1 - d2).days)
    if diff == 0:
        return 1.0
    elif diff <= 3:
        return 0.85
    elif diff <= 7:
        return 0.60
    elif diff <= 14:
        return 0.30
    else:
        return 0.0


def amount_score(amount1: Optional[int], amount2: Optional[int]) -> Optional[float]:
    """
    금액 근접도 점수.
    - 둘 다 없으면 None
    - 한쪽만 없으면 0.3
    - 1% 이내: 1.0 / 5% 이내: 0.85 / 10% 이내: 0.65 / 20% 이내: 0.30 / 초과: 0.0
    """
    if amount1 is None and amount2 is None:
        return None
    if amount1 is None or amount2 is None:
        return 0.3

    try:
        a1, a2 = int(amount1), int(amount2)
    except (TypeError, ValueError):
        return 0.0

    if max(a1, a2) == 0:
        return 1.0

    diff_ratio = abs(a1 - a2) / max(a1, a2)
    if diff_ratio <= 0.01:
        return 1.0
    elif diff_ratio <= 0.05:
        return 0.85
    elif diff_ratio <= 0.10:
        return 0.65
    elif diff_ratio <= 0.20:
        return 0.30
    else:
        return 0.0


# ══════════════════════════════════════════════════════════════
# 2-b. 사용내역서 항목 키 정규화 (한글 ↔ 영문 통합 지원)
# ══════════════════════════════════════════════════════════════

def _normalize_usage_item(item: dict) -> dict:
    """
    parse_usage_statement.py 버전별 키를 내부 표준 키(영문)로 정규화한다.

    지원 포맷 (우선순위 순):
      v3 영문 키 (현재 버전): used_on, total_amount, item_name, category_code, remark
      구버전 영문 키         : date, amount, description
      구버전 한글 키         : 사용일자, 금액, 사용내역, 항목명, 추가정보

    vendor 추출:
      v3: remark 필드에서 "institution:값 | org:값" 형식의 첫 번째 값을 사용
      구버전 한글: 추가정보 내 업체명 / 진단기관 / 지도기관 / 교육주관 / 진단병원
    """
    # ── v3 영문 키 (parse_usage_statement v3 현재 포맷) ──────────
    if "used_on" in item:
        vendor = ""
        remark = item.get("remark") or ""
        # remark 형식: "institution:한국안전원 | org:소속" — 첫 값을 vendor로 사용
        for part in remark.split("|"):
            part = part.strip()
            if ":" in part:
                vendor = part.split(":", 1)[1].strip()
                break
        return {
            **item,                                      # 원본 키도 유지 (필요 시 접근 가능)
            "date":        item.get("used_on", ""),
            "amount":      item.get("total_amount"),
            "description": item.get("item_name", ""),
            "name":        item.get("item_name", ""),
            "category":    item.get("category_code", ""),
            "vendor":      vendor,
        }

    # ── 구버전 영문 키 (이미 정규화된 경우) ─────────────────────
    if "date" in item or "amount" in item or "description" in item:
        return item

    # ── 구버전 한글 키 (하위 호환) ──────────────────────────────
    extra = item.get("추가정보") or {}

    # vendor 후보: 추가정보 내 여러 필드 중 첫 번째 유효값 사용
    vendor_candidates = [
        extra.get("업체명"), extra.get("진단기관"), extra.get("지도기관"),
        extra.get("교육주관"), extra.get("진단병원"),
    ]
    vendor = next((v for v in vendor_candidates if v), "")

    return {
        **item,                                          # 원본 키도 유지 (필요 시 접근 가능)
        "date":        item.get("사용일자", ""),
        "amount":      item.get("금액"),
        "description": item.get("사용내역", ""),
        "name":        item.get("사용내역", ""),         # name 별칭 추가
        "category":    item.get("항목명", ""),
        "vendor":      vendor,
    }


# ══════════════════════════════════════════════════════════════
# 3. 반려 조건 검사
# ══════════════════════════════════════════════════════════════

def _check_rejection(usage_item: dict, receipt: dict) -> Optional[str]:
    """
    반려(rejected) 조건을 검사하고 사유 문자열 반환.
    문제 없으면 None 반환.

    반려 조건:
    1. 영수증 OCR 인식 실패
    2. 영수증에 품목명이 하나도 없음 (items 빈 배열 포함)
    3. 사용내역서 항목에 설명(description)이 없음
    """
    infer = receipt.get("infer_result", "")
    if infer not in ("SUCCESS", ""):
        # infer_result가 없는 경우(테스트용 수동 딕셔너리)는 패스
        if infer:
            return f"영수증 OCR 인식 실패 (상태: {infer})"

    # 임금명세서(wage_statement)는 items가 없는 것이 구조적으로 정상 → 품목명 검사 면제
    if receipt.get("doc_type") != "wage_statement":
        items = receipt.get("items", [])
        has_named_item = any(
            (item.get("name") and str(item["name"]).strip())
            or (item.get("item_name") and str(item.get("item_name", "")).strip())
            for item in items
        )
        if not has_named_item:
            return "영수증 품목명 없음 — 반려 처리"

    if not usage_item.get("name") and not usage_item.get("description") and not usage_item.get("category"):
        return "사용내역서 항목에 내용 설명 누락"

    return None


# ══════════════════════════════════════════════════════════════
# 3-b. Hard Gate 검사 (날짜·금액·업체명 필수 조건)
# ══════════════════════════════════════════════════════════════

def _check_hard_gates(
    usage_item: dict,
    receipt: dict,
) -> tuple[bool, list[str]]:
    """
    Hard Gate 3가지를 검사한다.
    모두 통과해야 후보 영수증으로 인정.
    하나라도 실패하면 즉시 unmatched.

    Gate 1 — 날짜  : 정산 사이클 기반 (전달 마지막 목요일+1 ~ 이번달 마지막 수요일)
    Gate 2 — 금액  : |사용금액 − 영수증금액| / max ≤ GATE_AMOUNT_PCT (1%)
    Gate 3 — 업체명: 정규화 후 완전일치
              · wage_statement(임금명세서)는 업체명 개념이 없으므로 자동 면제
              · 사용내역서에 업체명이 기재되지 않은 경우에도 면제

    Returns:
        (passed: bool, failed_gates: list[str])
    """
    failed: list[str] = []

    # ── Gate 1: 날짜 (정산 사이클 기반) ──────────────────────
    usage_date   = usage_item.get("date")
    receipt_date = _extract_receipt_date(receipt)
    if usage_date and receipt_date:
        if not _date_gate_cycle(usage_date, receipt_date):
            d_usage = _parse_date_safe(usage_date)
            if d_usage:
                cs, ce    = _which_cycle(d_usage)
                cycle_str = f"{cs.strftime('%Y-%m-%d')} ~ {ce.strftime('%Y-%m-%d')}"
            else:
                cycle_str = "계산 불가"
            failed.append(
                f"날짜 정산 사이클 불일치 "
                f"(내역서: {usage_date} / 영수증: {receipt_date}, "
                f"허용 사이클: {cycle_str})"
            )

    # ── Gate 2: 금액 ─────────────────────────────────────────
    # [다품목 영수증 대응]
    # 영수증 한 장에 여러 품목이 있을 경우 total_amount는 항목 합산이므로
    # 사용내역서 단일 항목 금액과 직접 비교하면 항상 실패한다.
    # 우선순위:
    #   1순위 — 품목명 유사도 기반 라인 소계 (item.amount: count × unit_price)
    #   2순위 — 영수증 총액 (단품 영수증 또는 1순위 미탐지 시)
    #   3순위 — 공급가액 (VAT 포함 total 대응: total / 1.1 추정)
    # 후보 중 하나라도 ±1% 이내면 Gate 통과.
    usage_amount = usage_item.get("amount")

    if usage_amount is not None:
        receipt_items = receipt.get("items", [])
        usage_desc    = usage_item.get("name") or usage_item.get("description", "")

        # 1순위: 품목명 유사도로 찾은 라인 소계
        _, best_item_amt = _find_best_receipt_item_amount(usage_desc, receipt_items)

        # 2순위: 영수증 총액
        total_amt = receipt.get("total_amount")

        # 3순위: 공급가액 (필드 있으면 사용, 없으면 총액 / 1.1 추정)
        supply_amt = receipt.get("supply_amount")
        if supply_amt is None and total_amt is not None:
            supply_amt = round(total_amt / 1.1)

        candidates = [a for a in [best_item_amt, total_amt, supply_amt] if a is not None]

        gate_amount_passed = False
        best_diff_pct      = float("inf")
        best_pair          = (int(usage_amount), int(total_amt) if total_amt else 0)

        for cand in candidates:
            try:
                a1, a2 = int(usage_amount), int(cand)
                if max(a1, a2) == 0:
                    gate_amount_passed = True
                    break
                diff_pct = abs(a1 - a2) / max(a1, a2)
                if diff_pct <= GATE_AMOUNT_PCT:
                    gate_amount_passed = True
                    break
                if diff_pct < best_diff_pct:
                    best_diff_pct = diff_pct
                    best_pair     = (a1, a2)
            except (TypeError, ValueError):
                continue

        if not gate_amount_passed:
            if candidates:
                a1, a2 = best_pair
                failed.append(
                    f"금액 {best_diff_pct * 100:.1f}% 차이 "
                    f"(내역서: {a1:,}원 / 영수증 최근접: {a2:,}원, "
                    f"허용: ±{GATE_AMOUNT_PCT * 100:.0f}%)"
                )
            else:
                failed.append("영수증 금액 정보 없음")

    # ── Gate 3: 업체명 ────────────────────────────────────────
    # wage_statement(임금명세서)는 업체명 개념이 없으므로 Gate 3 자동 면제.
    # 임금명세서 외 문서라도 사용내역서에 업체명 미기재 시 비교 불가 → 면제.
    if receipt.get("doc_type") != "wage_statement":
        usage_vendor_raw  = usage_item.get("vendor", "") or ""
        receipt_store_raw = _extract_receipt_vendor(receipt)

        def _vendor_for_gate(text: str) -> str:
            """
            Gate 비교 전용 업체명 정규화.
            (주) / ㈜ / (유) 등 법인 단자(주·사·유)를 완전히 제거하고
            공백·특수문자를 없앤 뒤 소문자로 반환.
            예) "(주)한국건설안전기술원" → "한국건설안전기술원"
                "한국건설안전기술원(주)" → "한국건설안전기술원"
            """
            if not text:
                return ""
            t = _normalize_vendor(text)
            t = re.sub(r"^[주사유]\s*", "", t)
            t = re.sub(r"\s*[주사유]$", "", t)
            t = re.sub(r"[^가-힣a-zA-Z0-9]", "", t)
            return t.lower()

        u_clean = _vendor_for_gate(usage_vendor_raw)
        r_clean = _vendor_for_gate(receipt_store_raw)

        if u_clean:  # 사용내역서에 업체명이 기재된 경우에만 비교
            if not r_clean or u_clean != r_clean:
                failed.append(
                    f"업체명 불일치 "
                    f"(내역서: '{usage_vendor_raw}' / 영수증: '{receipt_store_raw}')"
                )

    return (len(failed) == 0), failed


# ══════════════════════════════════════════════════════════════
# 3-c. 영수증 품목 수준 금액 매칭 헬퍼
# ══════════════════════════════════════════════════════════════

def _find_best_receipt_item_amount(
    usage_desc: str,
    receipt_items: list,
    sim_threshold: float = 0.25,
) -> tuple[float, Optional[int]]:
    """
    사용내역 설명과 가장 유사한 영수증 품목을 찾아 그 금액을 반환.

    현실적 시나리오:
      - 하나의 영수증에 여러 품목이 있을 때
        사용내역서는 품목별로 분리 기재되지만,
        영수증 total_amount는 모든 품목의 합산임.
      - 이 함수는 사용내역 설명과 키워드 유사도가 가장 높은
        영수증 품목을 찾아 그 품목의 금액으로 비교하도록 도움.

    Args:
        usage_desc    : 사용내역서 description (예: "안전모 ABS형 구입")
        receipt_items : 영수증 items 리스트 (name, amount 포함)
        sim_threshold : 이 값 이상이어야 품목 매칭으로 간주

    Returns:
        (best_similarity, best_amount)  — 매칭 없으면 (0.0, None)
    """
    if not usage_desc or not receipt_items:
        return 0.0, None

    best_sim    = 0.0
    best_amount: Optional[int] = None

    for item in receipt_items:
        name = (item.get("name") or "").strip()
        if not name:
            continue
        sim = text_similarity(usage_desc, name)
        if sim > best_sim:
            best_sim    = sim
            best_amount = item.get("amount")

    if best_sim >= sim_threshold and best_amount is not None:
        return best_sim, best_amount
    return 0.0, None


# ══════════════════════════════════════════════════════════════
# 4. 세부 점수 계산
# ══════════════════════════════════════════════════════════════

def compute_component_scores(
    usage_item: dict,
    receipt: dict,
) -> dict[str, Optional[float]]:
    """
    4개 컴포넌트 점수를 계산해 dict로 반환.
    값이 None이면 해당 컴포넌트의 데이터가 부족하다는 의미 (가중치 재분배 대상).

    Components:
        date      : 날짜 근접도
        amount    : 금액 일치
        vendor    : 거래처/점포명 유사도
        item_desc : 품목명·내역 키워드 유사도

    ※ 현장사진 텍스트(photo_text) 항목은 비전 모델 파트로 분리됨.
    """
    # ── (1) 날짜 ──────────────────────────────────────────────
    usage_date   = usage_item.get("date")
    receipt_date = _extract_receipt_date(receipt)
    score_date   = date_score(usage_date, receipt_date)

    # ── (2) 금액 ──────────────────────────────────────────────
    # 영수증에 품목별 금액이 있으면 사용내역 설명과 가장 유사한 품목의 금액으로 비교.
    # 매칭되는 품목이 없으면 영수증 총액(total_amount)으로 비교.
    usage_amount  = usage_item.get("amount")
    receipt_items = receipt.get("items", [])

    best_item_sim, best_item_amount = _find_best_receipt_item_amount(
        usage_item.get("name") or usage_item.get("description", ""), receipt_items
    )

    # 품목 금액(부가세 미포함)과 영수증 총액(부가세 포함) 중 더 좋은 점수를 채택.
    # 카드 영수증처럼 항목은 공급가, total은 VAT 포함 금액인 경우를 커버.
    score_vs_item  = amount_score(usage_amount, best_item_amount) if best_item_amount is not None else None
    score_vs_total = amount_score(usage_amount, receipt.get("total_amount"))
    if score_vs_item is not None and score_vs_total is not None:
        score_amount = max(score_vs_item, score_vs_total)
    elif score_vs_item is not None:
        score_amount = score_vs_item
    else:
        score_amount = score_vs_total

    # ── (3) 거래처/점포명 유사도 ──────────────────────────────
    # 법인 단자(주·사·유) 및 특수문자를 완전히 제거한 뒤 비교.
    # gate 정규화와 동일한 수준을 적용하여 "(주)한국건설안전기술원" ==
    # "한국건설안전기술원(주)" 를 exact match로 인식.
    # exact match → 1.0,  아닌 경우 → text_similarity fallback
    def _vendor_normalized(text: str) -> str:
        """법인 단자·특수문자 제거 후 소문자 반환 (score 계산용)"""
        if not text:
            return ""
        t = _normalize_vendor(text)
        t = re.sub(r"^[주사유]\s*", "", t)
        t = re.sub(r"\s*[주사유]$", "", t)
        return re.sub(r"[^가-힣a-zA-Z0-9]", "", t).lower()

    usage_vendor_raw  = usage_item.get("vendor", "") or ""
    receipt_store_raw = _extract_receipt_vendor(receipt)
    u_vn = _vendor_normalized(usage_vendor_raw)
    r_vn = _vendor_normalized(receipt_store_raw)

    if u_vn and r_vn:
        if u_vn == r_vn:
            score_vendor = 1.0          # exact match (법인 단자 제거 후 동일)
        else:
            # 부분 일치는 기존 text_similarity 적용 (bigram + token + sequence)
            score_vendor = text_similarity(
                _normalize_vendor(usage_vendor_raw),
                _normalize_vendor(receipt_store_raw),
            )
    else:
        score_vendor = None

    # ── (4) 품목명·내역 유사도 ────────────────────────────────
    # usage: description + category 결합
    usage_desc_full = " ".join(filter(None, [
        usage_item.get("name") or usage_item.get("description", ""),
        usage_item.get("category", ""),
    ]))
    # receipt: 모든 품목명 결합
    receipt_items_text = " ".join(
        item.get("name", "")
        for item in receipt.get("items", [])
        if item.get("name")
    )
    score_item_desc = (
        text_similarity(usage_desc_full, receipt_items_text)
        if usage_desc_full and receipt_items_text
        else None
    )

    return {
        "date":      score_date,
        "amount":    score_amount,
        "vendor":    score_vendor,
        "item_desc": score_item_desc,
    }


# ══════════════════════════════════════════════════════════════
# 5. 가중 평균 최종 점수
# ══════════════════════════════════════════════════════════════

def weighted_aggregate(
    component_scores: dict[str, Optional[float]],
    weights: dict[str, float] = WEIGHTS,
) -> float:
    """
    None 컴포넌트의 가중치를 다른 컴포넌트에 비례 재분배해
    최종 유사도 점수(0.0 ~ 1.0)를 반환.
    """
    active = {k: v for k, v in component_scores.items() if v is not None}
    if not active:
        return 0.0

    # 활성 컴포넌트 가중치 합
    active_weight_sum = sum(weights.get(k, 0.0) for k in active)
    if active_weight_sum == 0.0:
        return 0.0

    total = sum(
        weights.get(k, 0.0) * v
        for k, v in active.items()
    )

    # 전체 가중치(1.0)에 대한 활성 비율로 정규화
    return round(total / active_weight_sum, 4)


# ══════════════════════════════════════════════════════════════
# 6. 영수증 요약 (출력용 — raw_clova 제외)
# ══════════════════════════════════════════════════════════════

def _receipt_summary(receipt: dict) -> dict:
    """raw_clova 등 디버그 전용 대용량 필드를 제거한 영수증 요약본"""
    exclude = {"raw_clova", "validation"}
    return {k: v for k, v in receipt.items() if k not in exclude}


# ══════════════════════════════════════════════════════════════
# 7. 단일 2-way 매칭
# ══════════════════════════════════════════════════════════════

def _decide_match_status(
    score: float,
    threshold_matched: float = THRESHOLD_MATCHED,
    threshold_review:  float = THRESHOLD_REVIEW,
) -> str:
    """
    유사도 점수를 3단계 매칭 상태로 변환.

    Returns:
        "matched"       — score >= threshold_matched (0.85 이상, 자동 통과)
        "review_needed" — threshold_review <= score < threshold_matched (0.75~0.85, 검토 필요)
        "unmatched"     — score < threshold_review (0.75 미만, 자동 반려)
    """
    if score >= threshold_matched:
        return "matched"
    elif score >= threshold_review:
        return "review_needed"
    else:
        return "unmatched"


def match_twoway(
    usage_item: dict,
    receipt: dict,
    threshold: float = MATCH_THRESHOLD,
    threshold_matched: float = THRESHOLD_MATCHED,
) -> dict:
    """
    단일 2-way 매칭 수행 (사용내역서 항목 ↔ 영수증).

    Args:
        usage_item        : parse_usage_statement.py 출력의 items[] 중 하나
        receipt           : clova_ocr_receipt.py 출력 JSON
        threshold         : review_needed 하한 임계값 (기본: 0.75)
        threshold_matched : matched 임계값 (기본: 0.85)

    Returns:
        {
          "match_id": str,
          "usage_item": dict,
          "receipt": dict,         # raw_clova 제외
          "similarity_score": float,
          "component_scores": dict,
          "match_status": "matched" | "review_needed" | "unmatched" | "rejected",
          "reject_reason": str | None,
          "matched_at": str (ISO)
        }
    """
    match_id    = str(uuid.uuid4())[:12]
    matched_at  = datetime.now().isoformat(timespec="seconds")

    # ── 한글/영문 키 정규화 ──────────────────────────────────
    usage_item = _normalize_usage_item(usage_item)

    # ── 반려 검사 ────────────────────────────────────────────
    reject_reason = _check_rejection(usage_item, receipt)
    if reject_reason:
        return {
            "match_id":         match_id,
            "usage_item":       usage_item,
            "receipt":          _receipt_summary(receipt),
            "similarity_score": 0.0,
            "component_scores": {},
            "match_status":     "rejected",
            "reject_reason":    reject_reason,
            "matched_at":       matched_at,
        }

    # ── 컴포넌트 점수 계산 ────────────────────────────────────
    component_scores = compute_component_scores(usage_item, receipt)

    # ── 최종 유사도 점수 ──────────────────────────────────────
    final_score = weighted_aggregate(component_scores)

    # ── 매칭 상태 결정 (3단계) ────────────────────────────────
    match_status = _decide_match_status(final_score, threshold_matched, threshold)

    # ── None 정리 (출력 가독성) ───────────────────────────────
    clean_scores = {
        k: (round(v, 4) if v is not None else None)
        for k, v in component_scores.items()
    }

    return {
        "match_id":         match_id,
        "usage_item":       usage_item,
        "receipt":          _receipt_summary(receipt),
        "similarity_score": final_score,
        "component_scores": clean_scores,
        "match_status":     match_status,
        "reject_reason":    None,
        "matched_at":       matched_at,
    }


# 하위 호환성 유지용 별칭
match_threeway = match_twoway


# ══════════════════════════════════════════════════════════════
# 8. 단일 사용내역서 항목 ↔ 후보 영수증 리스트 Best-Match
# ══════════════════════════════════════════════════════════════

def match_best(
    usage_item: dict,
    receipts: list[dict],
    threshold: float = MATCH_THRESHOLD,
    threshold_matched: float = THRESHOLD_MATCHED,
) -> dict:
    """
    하나의 사용내역서 항목에 대해 후보 영수증 중 최선의 매칭을 반환.

    [v4 Hard Gate 적용]
    1. 각 영수증에 대해 Hard Gate (날짜 ±1일 / 금액 ±1% / 업체명 일치) 검사
    2. Gate 통과 영수증만 scoring 대상으로 진행
    3. Gate 통과 영수증이 없으면 → 점수 가장 높은 gate 실패 케이스를
       warnings와 함께 unmatched로 반환

    Args:
        usage_item        : 사용내역서 단일 항목
        receipts          : 후보 영수증 리스트
        threshold         : review_needed 하한 임계값 (기본: 0.75)
        threshold_matched : matched 임계값 (기본: 0.85)

    Returns:
        match_twoway 결과 dict (gate_passed, gate_failed 필드 추가)
    """
    if not receipts:
        result = match_twoway(usage_item, {}, threshold, threshold_matched)
        result["gate_passed"] = False
        result["gate_failed"] = ["비교할 영수증 없음"]
        return result

    # ── 한글 키 정규화 (gate 검사용) ─────────────────────────
    usage_item_norm = _normalize_usage_item(usage_item)

    # ── 1단계: Hard Gate 필터링 ──────────────────────────────
    gate_passed_receipts: list[dict]             = []
    gate_failed_pairs:    list[tuple[dict, list]] = []

    for r in receipts:
        passed, failed_gates = _check_hard_gates(usage_item_norm, r)
        if passed:
            gate_passed_receipts.append(r)
        else:
            gate_failed_pairs.append((r, failed_gates))

    # ── 2단계: Gate 통과 영수증 scoring ─────────────────────
    if gate_passed_receipts:
        scored = [
            match_twoway(usage_item, r, threshold, threshold_matched)
            for r in gate_passed_receipts
        ]
        valid   = [r for r in scored if r["match_status"] != "rejected"]
        invalid = [r for r in scored if r["match_status"] == "rejected"]

        if valid:
            best = max(valid, key=lambda r: r["similarity_score"])
        else:
            best = max(invalid, key=lambda r: r["similarity_score"])

        best["gate_passed"] = True
        best["gate_failed"] = []
        return best

    # ── Gate 통과 영수증 없음 → 가장 근접한 케이스를 unmatched 반환 ──
    # 모든 gate 실패 영수증을 scoring해서 그 중 최고 점수를 참고용으로 첨부
    fallback_scored = [
        match_twoway(usage_item, r, threshold, threshold_matched)
        for r, _ in gate_failed_pairs
    ]
    best_fallback = max(
        fallback_scored, key=lambda r: r["similarity_score"]
    )

    # 해당 영수증의 gate 실패 사유 추출
    best_receipt_store = _extract_receipt_vendor(best_fallback.get("receipt") or {})
    failed_reasons: list[str] = []
    for r, gates in gate_failed_pairs:
        if _extract_receipt_vendor(r) == best_receipt_store:
            failed_reasons = gates
            break
    if not failed_reasons and gate_failed_pairs:
        failed_reasons = gate_failed_pairs[0][1]

    best_fallback["match_status"]     = "unmatched"
    best_fallback["gate_passed"]      = False
    best_fallback["gate_failed"]      = failed_reasons
    best_fallback["warnings"]         = (
        best_fallback.get("warnings") or []
    ) + [f"[Gate 실패] {r}" for r in failed_reasons]
    return best_fallback


# ══════════════════════════════════════════════════════════════
# 9. 배치 매칭: 사용내역서 전체 ↔ 영수증 리스트
# ══════════════════════════════════════════════════════════════

def match_all_usage_to_receipts(
    usage_statement: dict,
    receipts: list[dict],
    threshold: float = MATCH_THRESHOLD,
    threshold_matched: float = THRESHOLD_MATCHED,
) -> dict:
    """
    사용내역서 전체 항목을 영수증 리스트와 매칭.

    Args:
        usage_statement   : parse_usage_statement.py 출력 전체 JSON
        receipts          : clova_ocr_receipt.py 출력 JSON 리스트
        threshold         : review_needed 하한 임계값 (기본: 0.75)
        threshold_matched : matched 임계값 (기본: 0.85)

    Returns:
        {
          "batch_id": str,
          "source_usage": str,
          "thresholds": {"matched": float, "review": float},
          "results": [ {match_threeway 결과}, ... ],
          "summary": {
            "total":         int,
            "matched":       int,
            "review_needed": int,
            "unmatched":     int,
            "rejected":      int,
            "match_rate_pct":    float,
            "review_rate_pct":   float
          },
          "generated_at": str
        }
    """
    batch_id    = str(uuid.uuid4())[:8]
    # parse_usage_statement v2 출력 키: line_items (하위 호환: items)
    items       = usage_statement.get("line_items") or usage_statement.get("items", [])

    match_results = []
    for usage_item in items:
        best = match_best(usage_item, receipts, threshold, threshold_matched)
        match_results.append(best)

    # ── 요약 통계 (4단계) ────────────────────────────────────
    total        = len(match_results)
    matched      = sum(1 for r in match_results if r["match_status"] == "matched")
    review       = sum(1 for r in match_results if r["match_status"] == "review_needed")
    unmatched    = sum(1 for r in match_results if r["match_status"] == "unmatched")
    rejected     = sum(1 for r in match_results if r["match_status"] == "rejected")

    return {
        "batch_id":     batch_id,
        "source_usage": usage_statement.get("source_file", ""),
        "thresholds": {
            "matched": threshold_matched,
            "review":  threshold,
        },
        "results":      match_results,
        "summary": {
            "total":            total,
            "matched":          matched,
            "review_needed":    review,
            "unmatched":        unmatched,
            "rejected":         rejected,
            "match_rate_pct":   round(matched / total * 100, 1) if total else 0.0,
            "review_rate_pct":  round(review  / total * 100, 1) if total else 0.0,
        },
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


# ══════════════════════════════════════════════════════════════
# 10. 결과 저장
# ══════════════════════════════════════════════════════════════

def save_match_result(result: dict, output_path: str) -> str:
    """매칭 결과를 JSON 파일로 저장"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return str(output_path)


# ══════════════════════════════════════════════════════════════
# 11. 콘솔 요약 출력
# ══════════════════════════════════════════════════════════════

def print_match_result(result: dict):
    """단일 매칭 결과 콘솔 출력"""
    sep = "─" * 56
    status_icon = {
        "matched":       "✅",
        "review_needed": "🔍",
        "unmatched":     "❌",
        "rejected":      "🚫",
    }.get(result.get("match_status", ""), "?")
    print(f"\n{sep}")
    print(f"  match_id : {result.get('match_id')}")
    print(f"  상태     : {status_icon} {result.get('match_status')}")
    print(f"  유사도   : {result.get('similarity_score', 0):.4f}"
          f"  (matched≥{THRESHOLD_MATCHED} / review≥{THRESHOLD_REVIEW})")
    if result.get("reject_reason"):
        print(f"  반려사유 : {result['reject_reason']}")
    print(f"{sep}")

    usage = result.get("usage_item", {})
    rcpt  = result.get("receipt",     {})
    print(f"  [사용내역서]  {usage.get('date','-')}  "
          f"{usage.get('description','-')[:25]}  "
          f"{(usage.get('amount') or 0):,}원")
    print(f"  [영수증]      {rcpt.get('payment',{}).get('date','-')}  "
          f"{rcpt.get('store',{}).get('name','-')[:20]}  "
          f"{(rcpt.get('total_amount') or 0):,}원")
    comp = result.get("component_scores", {})
    if comp:
        print(f"{sep}")
        print(f"  [세부 점수]")
        labels = {"date": "날짜", "amount": "금액", "vendor": "거래처",
                  "item_desc": "품목내역"}
        for k, label in labels.items():
            v = comp.get(k)
            bar = "▓" * int((v or 0) * 20) + "░" * (20 - int((v or 0) * 20))
            val_str = f"{v:.4f}" if v is not None else " N/A  "
            print(f"    {label:6s}  [{bar}]  {val_str}")
    print(f"{sep}\n")


def print_batch_summary(batch_result: dict):
    """배치 매칭 요약 콘솔 출력"""
    sep  = "═" * 56
    s    = batch_result.get("summary", {})
    th   = batch_result.get("thresholds", {})
    print(f"\n{sep}")
    print(f"  배치 매칭 완료  (batch_id: {batch_result.get('batch_id')})")
    print(f"  사용내역서:    {batch_result.get('source_usage')}")
    print(f"  임계값:        matched≥{th.get('matched', THRESHOLD_MATCHED)}"
          f"  /  review≥{th.get('review', THRESHOLD_REVIEW)}")
    print(f"{sep}")
    print(f"  총 항목:          {s.get('total', 0):>4}개")
    print(f"  ✅ matched:       {s.get('matched', 0):>4}개  ({s.get('match_rate_pct', 0):.1f}%)")
    print(f"  🔍 review_needed: {s.get('review_needed', 0):>4}개  ({s.get('review_rate_pct', 0):.1f}%)")
    print(f"  ❌ unmatched:     {s.get('unmatched', 0):>4}개")
    print(f"  🚫 rejected:      {s.get('rejected', 0):>4}개")
    print(f"{sep}\n")


# ══════════════════════════════════════════════════════════════
# 12. CLI 진입점
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="2-way 매칭 엔진 — 산업안전관리비 검증 시스템"
    )
    parser.add_argument("--usage",   required=True,
                        help="사용내역서 파싱 결과 JSON 경로")
    parser.add_argument("--receipt", required=True, nargs="+",
                        help="영수증 OCR JSON 경로 (여러 개 가능)")
    parser.add_argument("--output",  default=None,
                        help="매칭 결과 저장 경로 (기본: match_results/)")
    parser.add_argument("--threshold", type=float, default=THRESHOLD_REVIEW,
                        help=f"review_needed 하한 임계값 (기본: {THRESHOLD_REVIEW})")
    parser.add_argument("--threshold-matched", type=float, default=THRESHOLD_MATCHED,
                        help=f"matched 임계값 (기본: {THRESHOLD_MATCHED})")
    parser.add_argument("--verbose", action="store_true",
                        help="개별 매칭 결과 상세 출력")
    args = parser.parse_args()

    # ── 입력 파일 로드 ─────────────────────────────────────
    with open(args.usage, encoding="utf-8") as f:
        usage_statement = json.load(f)

    receipts = []
    for rpath in args.receipt:
        with open(rpath, encoding="utf-8") as f:
            receipts.append(json.load(f))

    # ── 배치 매칭 ────────────────────────────────────────────
    batch = match_all_usage_to_receipts(
        usage_statement, receipts,
        threshold=args.threshold,
        threshold_matched=args.threshold_matched,
    )

    # ── 출력 ─────────────────────────────────────────────────
    if args.verbose:
        for r in batch["results"]:
            print_match_result(r)

    print_batch_summary(batch)

    # ── 저장 ─────────────────────────────────────────────────
    out_dir = Path(args.output) if args.output else (
        Path(args.usage).parent / "match_results"
    )
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"match_batch_{ts}.json"
    saved    = save_match_result(batch, str(out_path))
    print(f"  💾 저장: {saved}\n")


if __name__ == "__main__":
    main()
