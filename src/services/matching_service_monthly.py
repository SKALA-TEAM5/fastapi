"""
2-way 매칭 엔진 — 월 단위 날짜 비교 버전
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
산업안전관리비 AI 검증 시스템  —  matching_service_monthly.py  v2

[기존 matching_service.py 와의 차이점]
  ① 날짜 Gate 로직 교체
    · 구버전: |사용일자 − 영수증일자| ≤ 1일 (엄격한 일 단위 비교)
    · 이 버전: 사용내역서가 상시 업로드 가능하므로 used_on 기준 '같은 연월'이면 통과
              월 경계(±GATE_DATE_BOUNDARY_DAYS일) 도 허용 (예: 12/31 ↔ 01/01)

  ② 영수증 날짜 필드 이중 지원
    · 구버전 CLOVA 포맷 : receipt["payment"]["date"]
    · 신버전 통합 포맷   : receipt["date"]  (ReceiptOCRResponse)
    두 포맷 모두 자동 인식

  ③ date_score 기준
    · 같은 연월          → 1.0  (월 단위 완전 일치)
    · 월 경계 ±2일 이내  → 0.85 (인접 월, 경계 허용)
    · 그 외              → 0.0  (다른 월)

[원본 파일]
  src/services/matching_service.py (일 단위 비교, 보존됨)

[매칭 전략 — Hard Gate + 점수 보조]
  1단계 Hard Gate (3가지 조건, 모두 통과해야 후보 인정)
    · 날짜 Gate  : used_on 기준 같은 연월 (또는 월 경계 ±GATE_DATE_BOUNDARY_DAYS일)
    · 금액 Gate  : |사용금액 − 영수증금액| / max ≤ 1%
    · 업체명 Gate: 정규화 후 완전일치 (미기재 시 면제)

  2단계 보조 점수 (Gate 통과 영수증 중 최고 점수 선택)
    · matched / review_needed 구분 (0.85 / 0.75 임계값)
"""

from __future__ import annotations

import re
import json
import uuid
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from typing import Optional

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# 0. 설정 상수
# ══════════════════════════════════════════════════════════════

# ── 3단계 매칭 임계값 ───────────────────────────────────────────
THRESHOLD_MATCHED: float = 0.85
THRESHOLD_REVIEW:  float = 0.75
MATCH_THRESHOLD:   float = THRESHOLD_REVIEW

# ── Hard Gate 허용 오차 ─────────────────────────────────────────────
# 날짜: used_on 기준 같은 연월, 또는 월 경계 ±GATE_DATE_BOUNDARY_DAYS일 이내
# 금액: ±1%
# 업체명: 정규화 후 완전일치
GATE_DATE_BOUNDARY_DAYS: int   = 2     # 월 경계 허용 일수 (12/31 ↔ 01/01 등)
GATE_AMOUNT_PCT:         float = 0.01  # 금액 허용 오차 (1%)

# ── 품목명 합산 플래그 ───────────────────────────────────────────────
# MVP: 영수증·거래명세표 구분 없이 동일 적용 (True)
# 실제 현장 데이터 확보 후 doc_type별 분기 여부 재검토 → False로 끄거나 분기 추가
AGGREGATE_SAME_ITEM: bool = True

# ── 점수 가중치 ─────────────────────────────────────────────────────
# 날짜는 Gate에서만 사용 (같은 연월 여부 필터) → Score에서 제외
# 각 문서 유형(영수증/거래명세표/세금계산서)의 날짜가 서로 다른 이벤트를 가리키므로
# 날짜 점수 자체가 의미 없음. 금액 일치를 핵심 기준으로 사용.
WEIGHTS: dict[str, float] = {
    # "date": 제거 — Gate에서만 판별, Score에는 미포함
    "amount":    0.55,   # 금액 일치 (핵심 기준)
    "vendor":    0.25,   # 거래처/점포명 유사도
    "item_desc": 0.20,   # 품목명·내역 키워드 유사도
}

# ── 임금명세서 전용 가중치 ───────────────────────────────────────────
# 임금명세서(wage_statement)는 업체명 개념이 없어 vendor 점수가 구조적으로 0.
# vendor 비중(0.25)을 amount로 재배분해 최대 점수 상한이 0.75에서 막히는 문제 해소.
# [실제 현장 데이터 확보 후 재검토]
#   - item_desc: "안전관리자 임금"처럼 고정 표현이라 변별력 낮을 수 있음
#   - amount 비중 추가 상향(0.90) 여부는 데이터 확보 후 결정
WEIGHTS_WAGE: dict[str, float] = {
    "amount":    0.80,   # vendor 0.25 흡수 — 금액 일치가 사실상 유일한 기준
    "vendor":    0.00,   # 임금명세서는 업체명 없음 → 항상 0이므로 가중치 제거
    "item_desc": 0.20,   # 품목명 유사도 유지 (안전관리자 임금 vs 선임 수수료 등)
}

# ── 문서 유형별 가중치 매핑 ─────────────────────────────────────────
_WEIGHTS_BY_DOCTYPE: dict[str, dict[str, float]] = {
    "wage_statement": WEIGHTS_WAGE,
}

def _get_weights(doc_type: Optional[str]) -> dict[str, float]:
    """doc_type에 따른 가중치 반환. 미지정 또는 알 수 없는 유형은 기본값 사용."""
    return _WEIGHTS_BY_DOCTYPE.get(doc_type or "", WEIGHTS)


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
    text = text.replace("㈜", "주").replace("(주)", "주").replace("(株)", "주")
    text = text.replace("㈔", "사").replace("(사)", "사")
    text = text.replace("(유)", "유")
    return text.strip()


def _normalize(text: str) -> str:
    """
    텍스트 정규화:
    - 소문자 변환
    - 특수문자·괄호·조사 제거 (한글·숫자·영어 유지)
    - 연속 공백 단일화
    """
    if not text:
        return ""
    text = re.sub(r"[（）()\[\]{}<>【】「」『』""''·•※…]", " ", text)
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
    bigram 40% + token 30% + SequenceMatcher 30%
    """
    if not a or not b:
        return 0.0

    a_n = _normalize(a)
    b_n = _normalize(b)

    if not a_n or not b_n:
        return 0.0

    bigram_sim = _jaccard(_bigrams(a_n), _bigrams(b_n))
    token_sim  = _jaccard(_tokens(a_n),  _tokens(b_n))
    seq_sim    = SequenceMatcher(None, a_n, b_n).ratio()

    return round(0.40 * bigram_sim + 0.30 * token_sim + 0.30 * seq_sim, 4)


# ══════════════════════════════════════════════════════════════
# 2. 날짜 유틸리티 — 월 단위 비교
# ══════════════════════════════════════════════════════════════

def _parse_date_safe(date_str: Optional[str]) -> Optional[datetime]:
    """YYYY-MM-DD 파싱, 실패 시 None"""
    if not date_str:
        return None
    try:
        return datetime.strptime(str(date_str).strip()[:10], "%Y-%m-%d")
    except ValueError:
        return None


def _same_year_month(d1: datetime, d2: datetime) -> bool:
    """두 날짜가 같은 연·월인지 확인"""
    return d1.year == d2.year and d1.month == d2.month


def _near_month_boundary(
    d1: datetime,
    d2: datetime,
    boundary_days: int = GATE_DATE_BOUNDARY_DAYS,
) -> bool:
    """
    두 날짜가 서로 다른 월이지만 월 경계 ±boundary_days일 이내인지 확인.

    예) d1=12-31, d2=01-01 → 다른 월, 1일 차이 → True
        d1=01-15, d2=02-15 → 다른 월, 31일 차이 → False
    """
    if _same_year_month(d1, d2):
        return False
    return abs((d1 - d2).days) <= boundary_days


def _date_gate_monthly(usage_date: Optional[str], doc_date: Optional[str]) -> bool:
    """
    월 단위 날짜 Gate.

    사용내역서는 상시 업로드 가능하므로 고정 정산 사이클 대신
    used_on 기준 '같은 연월' 비교를 사용한다.

    통과 조건:
      ① 날짜 중 하나라도 없거나 파싱 실패 → 통과 (면제)
      ② 같은 연월                         → 통과
      ③ 월 경계 ±GATE_DATE_BOUNDARY_DAYS일 이내 → 통과 (예: 12/31 ↔ 01/01)
      ④ 그 외                             → 실패

    Returns:
        True  — Gate 통과
        False — Gate 실패
    """
    if not usage_date or not doc_date:
        return True

    d_usage = _parse_date_safe(usage_date)
    d_doc   = _parse_date_safe(doc_date)

    if d_usage is None or d_doc is None:
        return True

    if _same_year_month(d_usage, d_doc):
        return True

    return _near_month_boundary(d_usage, d_doc)


def date_score(date1: Optional[str], date2: Optional[str]) -> Optional[float]:
    """
    날짜 근접도 점수 — 월 단위 기준.

    - 두 날짜 모두 없으면 None (가중치 재분배)
    - 한쪽만 없으면 0.4 (부분 불확실)
    - 같은 연월           → 1.0   (월 완전 일치)
    - 인접 월 경계 ±2일   → 0.85  (경계 허용)
    - 그 외 다른 월        → 0.0   (월 불일치)

    ※ 같은 월 내에서 1일 vs 31일처럼 날짜 차이가 커도 1.0 부여.
       사용내역서는 월 단위 정산 문서이므로 같은 월이면 동일 거래로 간주.
    """
    d1 = _parse_date_safe(date1)
    d2 = _parse_date_safe(date2)

    if d1 is None and d2 is None:
        return None
    if d1 is None or d2 is None:
        return 0.4

    if _same_year_month(d1, d2):
        return 1.0

    if _near_month_boundary(d1, d2):
        return 0.85

    return 0.0


def amount_score(amount1: Optional[int], amount2: Optional[int]) -> Optional[float]:
    """
    금액 근접도 점수.
    1% 이내: 1.0 / 5%: 0.85 / 10%: 0.65 / 20%: 0.30 / 초과: 0.0
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
# 2-b. 영수증 날짜 필드 추출 — 이중 포맷 지원 (핵심 변경점)
# ══════════════════════════════════════════════════════════════

def _extract_receipt_date(receipt: dict) -> Optional[str]:
    """
    영수증 딕셔너리에서 날짜를 추출한다.

    두 가지 포맷을 자동 인식:
      ① 신버전 ReceiptOCRResponse (receipts.py 통합 응답):
           receipt["date"]  — 영수증: 카드 승인일시 / 거래명세표: 작성일자
      ② 구버전 clova_ocr_receipt.py 직접 출력:
           receipt["payment"]["date"]

    우선순위: ① → ②
    """
    # ① 신버전 통합 포맷 (ReceiptOCRResponse)
    date_top = receipt.get("date")
    if date_top:
        return date_top

    # ② 구버전 CLOVA 포맷
    date_payment = (receipt.get("payment") or {}).get("date")
    if date_payment:
        return date_payment

    return None


def _extract_receipt_vendor(receipt: dict) -> str:
    """
    영수증 딕셔너리에서 업체명을 추출한다.

    두 가지 포맷:
      ① 신버전: receipt["vendor"]
      ② 구버전: receipt["store"]["name"]
    """
    # ① 신버전
    vendor_top = receipt.get("vendor") or ""
    if vendor_top:
        return vendor_top

    # ② 구버전
    return (receipt.get("store") or {}).get("name") or ""


# ══════════════════════════════════════════════════════════════
# 2-c. 품목명 기준 합산 (MVP 핵심 전처리)
# ══════════════════════════════════════════════════════════════

def _aggregate_by_item_name(items: list[dict]) -> list[dict]:
    """
    같은 품목명(사용내역)이 날짜만 다르게 분할된 행을 합산한다.

    [배경]
      사용내역서는 실제 사용일 기준으로 날짜별로 기록되지만,
      영수증·거래명세표는 월 합산으로 발행되는 경우가 많다.

      예) 안전시설물 설치
            2022-12-21  5명 ×  170,000 =   850,000원
            2022-12-22  7명 ×  170,000 = 1,190,000원
            2022-12-23  6명 ×  170,000 = 1,020,000원
          ─────────────────────────────────────────
          합산          18명             3,060,000원  ← 거래명세표 한 줄과 비교

    [MVP 전략]
      AGGREGATE_SAME_ITEM = True 일 때 영수증·거래명세표 구분 없이 동일 적용.
      실제 현장 데이터 확보 후 doc_type별 분기 여부를 재검토한다.

    [grouping key]
      _normalize(품목명) — 특수문자·공백 제거, 소문자화 후 비교
      안전시설물 설치("추락","낙하"방지 시설) × 3행  →  1행으로 합산
      안전모 + 안전화 → 품목명이 다르므로 별도 유지

    [합산 대상]
      amount (계), count (수량) — 나머지 필드는 첫 번째 행 기준

    [감사 추적]
      _aggregated=True, _source_count=N, _source_dates=[...] 메타 추가
    """
    from collections import defaultdict

    # 정규화된 품목명 → 행 목록 (순서 유지)
    grouped: dict[str, list[dict]] = defaultdict(list)
    order:   list[str]             = []

    for item in items:
        name = (
            item.get("name")
            or item.get("description")
            or item.get("item_name")
            or item.get("category")
            or ""
        )
        key = _normalize(name)
        if key not in grouped:
            order.append(key)
        grouped[key].append(item)

    result: list[dict] = []
    for key in order:
        group = grouped[key]

        if len(group) == 1:
            result.append(group[0])
            continue

        # ── 합산 ──────────────────────────────────────────────
        total_amount = sum(
            (g.get("amount") or g.get("total_amount") or 0) for g in group
        )
        total_count = sum(
            (g.get("count") or g.get("quantity") or 0) for g in group
        )

        aggregated = dict(group[0])       # 첫 행을 기본값으로 복사
        aggregated.update({
            "amount":        total_amount,
            "total_amount":  total_amount,
            "count":         total_count,
            # 감사 추적용 메타데이터
            "_aggregated":   True,
            "_source_count": len(group),
            "_source_dates": [
                g.get("date") or g.get("used_on") for g in group
            ],
        })
        result.append(aggregated)

    return result


# ══════════════════════════════════════════════════════════════
# 2-d. 사용내역서 항목 키 정규화 (한글 ↔ 영문 통합)
# ══════════════════════════════════════════════════════════════

def _normalize_usage_item(item: dict) -> dict:
    """
    parse_usage_statement.py 버전별 키를 내부 표준 키(영문)로 정규화한다.
    v3 영문 키(used_on, total_amount, item_name, category_code, remark),
    구버전 영문/한글 키 모두 지원.
    """
    if "used_on" in item:
        vendor = ""
        remark = item.get("remark") or ""
        for part in remark.split("|"):
            part = part.strip()
            if ":" in part:
                vendor = part.split(":", 1)[1].strip()
                break
        return {
            **item,
            "date":        item.get("used_on", ""),
            "amount":      item.get("total_amount"),
            "description": item.get("item_name", ""),
            "name":        item.get("item_name", ""),
            "category":    item.get("category_code", ""),
            "vendor":      vendor,
        }

    if "date" in item or "amount" in item or "description" in item:
        return item

    extra = item.get("추가정보") or {}
    vendor_candidates = [
        extra.get("업체명"), extra.get("진단기관"), extra.get("지도기관"),
        extra.get("교육주관"), extra.get("진단병원"),
    ]
    vendor = next((v for v in vendor_candidates if v), "")

    return {
        **item,
        "date":        item.get("사용일자", ""),
        "amount":      item.get("금액"),
        "description": item.get("사용내역", ""),
        "name":        item.get("사용내역", ""),
        "category":    item.get("항목명", ""),
        "vendor":      vendor,
    }


# ══════════════════════════════════════════════════════════════
# 3. 반려 조건 검사
# ══════════════════════════════════════════════════════════════

# 문서 유형별 한국어 라벨 (반려 사유 문구 분기용)
_DOC_TYPE_LABELS = {
    "receipt":              "영수증",
    "delivery_statement":   "거래명세표",
    "transaction_statement": "거래명세표",
    "tax_invoice":          "세금계산서",
    "wage_statement":       "임금명세서",
}


def _doc_label(receipt: dict) -> str:
    """매칭된 증빙 문서 유형의 한국어 라벨을 반환한다(기본: 증빙)."""
    return _DOC_TYPE_LABELS.get(receipt.get("doc_type"), "증빙")


def _check_rejection(usage_item: dict, receipt: dict) -> Optional[str]:
    """
    반려(rejected) 조건 검사. 사유 문구는 매칭된 증빙의 문서 유형에 맞게 분기한다.
    1. 증빙 OCR 인식 실패
    2. 증빙에 품목명이 하나도 없음 (임금명세서는 면제)
       - 세금계산서: 품목 내역이 없어 매칭 불가 → 거래명세표/영수증 증빙 필요
       - 영수증·거래명세표: 품목 인식 실패 → 재제출 필요
    3. 사용내역서 항목에 설명이 없음

    [임금명세서 면제 이유]
    wage_statement는 근로자 임금 지급 내역이므로 품목 개념이 없어
    items가 빈 배열인 것이 구조적으로 정상이다. Gate 3(업체명)과 동일하게
    품목명 검사도 면제한다.
    """
    label = _doc_label(receipt)

    infer = receipt.get("infer_result", "")
    if infer not in ("SUCCESS", ""):
        if infer:
            return f"{label} OCR 인식 실패 (상태: {infer})"

    # 임금명세서는 items 비어있는 것이 정상 → 품목명 검사 면제
    if receipt.get("doc_type") != "wage_statement":
        items = receipt.get("items", [])
        has_named_item = any(
            item.get("name") and str(item["name"]).strip()
            or item.get("item_name") and str(item.get("item_name", "")).strip()
            for item in items
        )
        if not has_named_item:
            if receipt.get("doc_type") == "tax_invoice":
                # 세금계산서는 품목 내역이 없는 경우가 많음 → 진짜 필요한 것은
                # 품목이 기재된 거래명세표/영수증 증빙임을 안내한다.
                return "세금계산서에 품목 내역이 없어 매칭 불가 — 품목이 기재된 거래명세표/영수증 증빙 필요"
            return f"{label} 품목 인식 실패 — 증빙 재제출 필요"

    if not usage_item.get("name") and not usage_item.get("description") and not usage_item.get("category"):
        return "사용내역서 항목에 내용 설명 누락"

    return None


# ══════════════════════════════════════════════════════════════
# 3-a-2. Gate 2 금액 보정 (거래명세표 VAT)
# ══════════════════════════════════════════════════════════════

def _resolve_gate2_amount(receipt: dict, usage_amount) -> int | None:
    """
    Gate 2 금액 비교를 위한 거래명세표 VAT 보정 함수.

    delivery_statement의 경우 VLM이 total_amount를 공급가액(VAT 미포함)으로
    반환할 수 있어 사용내역서 금액(VAT 포함)과 ~9.1% 차이가 발생한다.
    3단계 보정을 시도해 Gate 2 통과 여부를 판정한다.

      1순위: raw total_amount 그대로 (이미 VAT 포함이거나 정상 금액)
      2순위: raw + tax_amount (VLM이 세액 필드를 별도 반환한 경우)
      3순위: raw × 1.1 (세액 필드 없을 때 10% VAT 추정)

    거래명세표·세금계산서·영수증 등 증빙은 공급가액(VAT 미포함)과 세액을 분리 제공할 수
    있으므로 doc_type에 관계없이 동일하게 VAT 보정한다. 보정이 게이트를 통과시키지 못하면
    raw를 그대로 반환하므로(1순위 raw 우선), 이미 VAT 포함인 영수증은 영향받지 않는다.
    """
    raw = receipt.get("total_amount")
    if raw is None:
        return None

    if usage_amount is None:
        return raw

    try:
        u, r = int(usage_amount), int(raw)
    except (TypeError, ValueError):
        return raw

    def _within_gate(a: int, b: int) -> bool:
        return max(a, b) > 0 and abs(a - b) / max(a, b) <= GATE_AMOUNT_PCT

    # 1순위: raw 그대로
    if _within_gate(u, r):
        return r

    # 2순위: raw + tax_amount
    tax = receipt.get("tax_amount")
    if tax is not None:
        candidate = r + int(tax)
        if _within_gate(u, candidate):
            return candidate

    # 3순위: raw × 1.1 추정
    candidate_vat = int(r * 1.1)
    if _within_gate(u, candidate_vat):
        return candidate_vat

    return r  # 보정 불가 → Gate 2 탈락


# ══════════════════════════════════════════════════════════════
# 3-b. Hard Gate 검사 (날짜·금액·업체명)
# ══════════════════════════════════════════════════════════════

def _check_hard_gates(
    usage_item: dict,
    receipt: dict,
) -> tuple[bool, list[str]]:
    """
    Hard Gate 3가지 검사.

    Gate 1 — 날짜  : 같은 연월 또는 월 경계 ±2일  [월 단위 비교]
    Gate 2 — 금액  : |사용금액 − 영수증금액| / max ≤ GATE_AMOUNT_PCT (1%)
    Gate 3 — 업체명: 정규화 후 완전일치 (사용내역서에 업체명 미기재 시 면제)

    영수증 날짜/업체명은 신·구 포맷 모두 지원 (_extract_receipt_date/vendor 사용)
    """
    failed: list[str] = []

    # ── Gate 1: 날짜 (월 단위, used_on 기준) ────────────────────
    usage_date   = usage_item.get("date")
    receipt_date = _extract_receipt_date(receipt)

    if usage_date and receipt_date:
        if not _date_gate_monthly(usage_date, receipt_date):
            failed.append(
                f"날짜 월 불일치 "
                f"(내역서: {usage_date} / 영수증: {receipt_date}, "
                f"허용: 같은 연월 또는 ±{GATE_DATE_BOUNDARY_DAYS}일)"
            )

    # ── Gate 2: 금액 ─────────────────────────────────────────
    # 사용내역서 금액(VAT 포함)과 비교할 영수증 금액 후보:
    #   ① 영수증 총액(_resolve_gate2_amount: VAT 보정)
    #   ② 사용내역과 가장 유사한 "품목 1건"의 VAT 포함 금액(_find_best_receipt_item_amount)
    # ②가 핵심 — 다품목 세금계산서/거래명세표는 총액(=여러 품목 합)이 단일 사용내역과
    # 맞지 않으므로, 해당 품목의 VAT 포함 소계로 비교해야 올바른 증빙이 Gate를 통과한다.
    usage_amount = usage_item.get("amount")
    candidates: list[int] = []
    total_cand = _resolve_gate2_amount(receipt, usage_amount)
    if total_cand is not None:
        candidates.append(total_cand)
    _, best_item_amount = _find_best_receipt_item_amount(
        usage_item.get("name") or usage_item.get("description", ""),
        receipt.get("items", []),
        doc_type=receipt.get("doc_type"),
    )
    if best_item_amount is not None:
        candidates.append(best_item_amount)

    if usage_amount is not None and candidates:
        try:
            a1 = int(usage_amount)
            passed_amount = any(
                max(a1, int(c)) > 0 and abs(a1 - int(c)) / max(a1, int(c)) <= GATE_AMOUNT_PCT
                for c in candidates
            )
            if not passed_amount:
                nearest = min(candidates, key=lambda c: abs(a1 - int(c)))
                diff_pct = abs(a1 - int(nearest)) / max(a1, int(nearest), 1)
                failed.append(
                    f"금액 {diff_pct * 100:.1f}% 차이 "
                    f"(내역서: {a1:,}원 / 영수증: {int(nearest):,}원, "
                    f"허용: ±{GATE_AMOUNT_PCT * 100:.0f}%)"
                )
        except (TypeError, ValueError):
            failed.append("금액 파싱 오류")

    # ── Gate 3: 업체명 ────────────────────────────────────────
    # wage_statement(임금명세서)는 업체명 개념이 없으므로 Gate 3 자동 면제
    if receipt.get("doc_type") != "wage_statement":
        usage_vendor_raw  = usage_item.get("vendor", "") or ""
        receipt_store_raw = _extract_receipt_vendor(receipt)

        def _vendor_for_gate(text: str) -> str:
            if not text:
                return ""
            t = _normalize_vendor(text)
            t = re.sub(r"^[주사유]\s*", "", t)
            t = re.sub(r"\s*[주사유]$", "", t)
            t = re.sub(r"[^가-힣a-zA-Z0-9]", "", t)
            return t.lower()

        u_clean = _vendor_for_gate(usage_vendor_raw)
        r_clean = _vendor_for_gate(receipt_store_raw)

        if u_clean:
            if not r_clean or u_clean != r_clean:
                failed.append(
                    f"업체명 불일치 "
                    f"(내역서: '{usage_vendor_raw}' / 영수증: '{receipt_store_raw}')"
                )

    return (len(failed) == 0), failed


# ══════════════════════════════════════════════════════════════
# 3-c. 영수증 품목 수준 금액 매칭 헬퍼
# ══════════════════════════════════════════════════════════════

VAT_RATE: float = 0.10  # 부가가치세율 10%

def _get_item_name(item: dict) -> str:
    """영수증 품목명 추출 — 신·구 포맷 공통 ('name' 또는 'item_name')"""
    return (item.get("name") or item.get("item_name") or "").strip()


def _get_item_vat_total(item: dict, doc_type: Optional[str] = None) -> Optional[int]:
    """
    영수증 품목 1건의 부가세 포함 실지출금액을 반환한다.

    [우선순위]
      ① 세금계산서(tax_invoice) : supply_amount + tax_amount (명시된 세액 사용)
      ② 거래명세서(delivery_statement): amount + tax_amount (있으면 사용, 없으면 × 1.1)
      ③ 일반 영수증: amount (이미 부가세 포함 금액으로 간주)

    [배경]
      거래명세서 items.amount = 공급가액(부가세 미포함).
      사용내역서 total_amount = 부가세 포함 실지출금액.
      두 금액이 일치하려면 영수증 품목 금액에 부가세를 더해야 한다.
    """
    supply = (
        item.get("supply_amount")   # 세금계산서 포맷
        or item.get("amount")       # 거래명세서 포맷
    )
    if supply is None:
        return None

    supply = int(supply)

    # 명시된 세액 우선 사용
    tax = item.get("tax_amount")
    if tax is not None:
        return supply + int(tax)

    # 세금계산서 / 거래명세서 → VAT_RATE 적용
    if doc_type in ("tax_invoice", "delivery_statement"):
        return round(supply * (1 + VAT_RATE))

    # 임금명세서: ③합계확인 보수총액은 VAT 없음 → amount 직접 사용
    if doc_type == "wage_statement":
        return supply

    # 일반 영수증: amount 자체가 부가세 포함
    return supply


def _verify_item_integrity(item: dict, doc_type: Optional[str] = None) -> Optional[str]:
    """
    품목별 단가 × 수량 = 공급가액 정합성을 검증한다.
    불일치 시 경고 문자열 반환, 정상이면 None.

    [검증 대상]
      거래명세서: unit_price × count ≈ amount (공급가액)
      세금계산서: unit_price × quantity ≈ supply_amount
    """
    if doc_type == "delivery_statement":
        up  = item.get("unit_price")
        cnt = item.get("count")
        amt = item.get("amount")
    elif doc_type == "tax_invoice":
        up  = item.get("unit_price")
        cnt = item.get("quantity")
        amt = item.get("supply_amount")
    else:
        return None

    if up is None or cnt is None or amt is None:
        return None

    calc = round(float(up) * float(cnt))
    diff = abs(calc - int(amt))
    if diff > max(1, int(amt) * 0.01):   # 1원 또는 1% 이내면 허용
        return (
            f"단가×수량 불일치: {up:,}×{cnt}={calc:,} ≠ 공급가액 {int(amt):,} "
            f"(차이 {diff:,})"
        )
    return None


def _expand_to_item_receipts(receipt: dict) -> list[dict]:
    """
    다품목 영수증(거래명세서·세금계산서)을 품목별 가상 영수증으로 분리한다.

    [적용 조건]
      doc_type이 delivery_statement 또는 tax_invoice이고
      items 배열에 2개 이상 품목이 있을 때만 분리.

    [가상 영수증 구조]
      - 날짜·업체명 등 헤더 필드는 부모 영수증 그대로 상속
      - total_amount = 품목별 부가세 포함 금액 (_get_item_vat_total)
      - items = [해당 품목만]
      - _expanded: True (감사 추적용)
      - _parent_source: 원본 source_file
      - _integrity_warning: 단가×수량 불일치 시 경고 문자열

    [단품 영수증 또는 미지원 doc_type]
      단일 원소 리스트로 반환 (그대로 사용).
    """
    doc_type = receipt.get("doc_type")
    items    = receipt.get("items", [])

    # 단품이거나 미지원 유형이면 확장 안 함
    if doc_type not in ("delivery_statement", "tax_invoice", "wage_statement") or len(items) <= 1:
        return [receipt]

    expanded: list[dict] = []
    for item in items:
        # 품목별 실지출금액 계산
        vat_total = _get_item_vat_total(item, doc_type)
        if vat_total is None:
            continue

        # 정합성 검증
        integrity_warn = _verify_item_integrity(item, doc_type)

        # 부모 영수증 복사 후 품목·금액 교체
        sub = {k: v for k, v in receipt.items() if k != "items"}
        sub.update({
            "items":                [item],
            "total_amount":         vat_total,
            "_expanded":            True,
            "_parent_source":       receipt.get("source_file", ""),
            "_item_name":           _get_item_name(item),
            "_item_supply_amount":  item.get("supply_amount") or item.get("amount"),
            "_item_tax_amount":     item.get("tax_amount"),
            "_integrity_warning":   integrity_warn,
        })
        if integrity_warn:
            logger.warning("품목 정합성 경고 [%s / %s]: %s",
                           receipt.get("source_file"), _get_item_name(item), integrity_warn)
        expanded.append(sub)

    return expanded if expanded else [receipt]


def _find_best_receipt_item_amount(
    usage_desc: str,
    receipt_items: list,
    doc_type: Optional[str] = None,
    sim_threshold: float = 0.25,
) -> tuple[float, Optional[int]]:
    """
    사용내역 설명과 가장 유사한 영수증 품목을 찾아 부가세 포함 금액을 반환.

    [변경 이력]
      v2: item.amount 대신 _get_item_vat_total(item, doc_type) 사용
          → 공급가액만 있는 거래명세서 품목에 VAT 자동 추가
    """
    if not usage_desc or not receipt_items:
        return 0.0, None

    best_sim    = 0.0
    best_amount: Optional[int] = None

    for item in receipt_items:
        name = _get_item_name(item)
        if not name:
            continue
        sim = text_similarity(usage_desc, name)
        if sim > best_sim:
            best_sim    = sim
            best_amount = _get_item_vat_total(item, doc_type)

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
    4개 컴포넌트 점수 계산.
    날짜 점수는 월 단위 date_score 사용 (이 파일의 핵심 변경점).
    영수증 날짜·업체명은 신·구 포맷 모두 지원.
    """
    # ── (1) 날짜 — Score에서 제외 ────────────────────────────
    # 날짜는 Hard Gate(_date_gate_monthly)에서만 사용.
    # 영수증·거래명세표·세금계산서 날짜가 서로 다른 이벤트(결제일/납품일/작성일)를
    # 가리키므로 점수 비교 자체가 의미 없음.
    # → WEIGHTS에서도 "date" 키 제거됨.

    # ── (2) 금액 ──────────────────────────────────────────────
    usage_amount  = usage_item.get("amount")
    receipt_items = receipt.get("items", [])

    best_item_sim, best_item_amount = _find_best_receipt_item_amount(
        usage_item.get("name") or usage_item.get("description", ""),
        receipt_items,
        doc_type=receipt.get("doc_type"),
    )

    score_vs_item  = amount_score(usage_amount, best_item_amount) if best_item_amount is not None else None
    # 총액 비교도 게이트와 동일하게 VAT 보정된 금액을 사용한다(영수증 총액이 공급가액인
    # 경우 raw로 비교하면 점수가 낮아 matched가 아니라 review로 빠지는 문제 방지).
    score_vs_total = amount_score(usage_amount, _resolve_gate2_amount(receipt, usage_amount))
    if score_vs_item is not None and score_vs_total is not None:
        score_amount = max(score_vs_item, score_vs_total)
    elif score_vs_item is not None:
        score_amount = score_vs_item
    else:
        score_amount = score_vs_total

    # ── (3) 업체명 유사도 ─────────────────────────────────────
    def _vendor_normalized(text: str) -> str:
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
        score_vendor = (
            1.0 if u_vn == r_vn
            else text_similarity(_normalize_vendor(usage_vendor_raw),
                                 _normalize_vendor(receipt_store_raw))
        )
    else:
        score_vendor = None

    # ── (4) 품목명·내역 유사도 ────────────────────────────────
    # category 코드(예: "CAT_03")는 숫자·영문자로 구성되어 bigram을 오염시키므로 제외.
    # name 또는 description 만 사용.
    usage_desc_full = (
        usage_item.get("name") or usage_item.get("description", "") or ""
    ).strip()
    receipt_items_text = " ".join(
        _get_item_name(item)
        for item in receipt.get("items", [])
        if _get_item_name(item)
    )
    score_item_desc = (
        text_similarity(usage_desc_full, receipt_items_text)
        if usage_desc_full and receipt_items_text
        else None
    )

    return {
        # "date" 제거 — Gate에서만 사용, Score 미포함
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
    """None 컴포넌트 가중치를 비례 재분배해 최종 점수(0.0~1.0) 반환"""
    active = {k: v for k, v in component_scores.items() if v is not None}
    if not active:
        return 0.0

    active_weight_sum = sum(weights.get(k, 0.0) for k in active)
    if active_weight_sum == 0.0:
        return 0.0

    total = sum(weights.get(k, 0.0) * v for k, v in active.items())
    return round(total / active_weight_sum, 4)


# ══════════════════════════════════════════════════════════════
# 6. 영수증 요약
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
    weights: Optional[dict[str, float]] = None,
) -> dict:
    """
    단일 2-way 매칭 수행 (사용내역서 항목 ↔ 영수증).
    날짜 점수는 월 단위 기준 적용.

    Args:
        weights: 가중치 딕셔너리. None이면 receipt의 doc_type에 따라 자동 선택.
                 명시적으로 전달하면 해당 가중치를 그대로 사용.
    """
    match_id   = str(uuid.uuid4())[:12]
    matched_at = datetime.now().isoformat(timespec="seconds")

    usage_item = _normalize_usage_item(usage_item)

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

    effective_weights = weights if weights is not None else _get_weights(receipt.get("doc_type"))
    component_scores  = compute_component_scores(usage_item, receipt)
    final_score       = weighted_aggregate(component_scores, effective_weights)
    match_status      = _decide_match_status(final_score, threshold_matched, threshold)

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
        "weights_used":     effective_weights,   # 어떤 가중치로 계산됐는지 추적용
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

    [Hard Gate 적용]
    1. 월 단위 날짜 Gate + 금액 Gate + 업체명 Gate 검사
    2. Gate 통과 영수증만 scoring
    3. Gate 통과 없으면 → 최근접 케이스를 unmatched로 반환
    """
    if not receipts:
        result = match_twoway(usage_item, {}, threshold, threshold_matched)
        result["gate_passed"] = False
        result["gate_failed"] = ["비교할 영수증 없음"]
        return result

    usage_item_norm = _normalize_usage_item(usage_item)

    gate_passed_receipts: list[dict]              = []
    gate_failed_pairs:    list[tuple[dict, list]] = []

    for r in receipts:
        passed, failed_gates = _check_hard_gates(usage_item_norm, r)
        if passed:
            gate_passed_receipts.append(r)
        else:
            gate_failed_pairs.append((r, failed_gates))

    if gate_passed_receipts:
        scored  = [match_twoway(usage_item, r, threshold, threshold_matched) for r in gate_passed_receipts]
        valid   = [r for r in scored if r["match_status"] != "rejected"]
        invalid = [r for r in scored if r["match_status"] == "rejected"]

        best = max(valid, key=lambda r: r["similarity_score"]) if valid \
               else max(invalid, key=lambda r: r["similarity_score"])

        best["gate_passed"] = True
        best["gate_failed"] = []
        return best

    fallback_scored = [
        match_twoway(usage_item, r, threshold, threshold_matched)
        for r, _ in gate_failed_pairs
    ]
    best_fallback = max(fallback_scored, key=lambda r: r["similarity_score"])

    best_vendor = _extract_receipt_vendor(
        best_fallback.get("receipt") or {}
    )
    failed_reasons: list[str] = []
    for r, gates in gate_failed_pairs:
        if _extract_receipt_vendor(r) == best_vendor:
            failed_reasons = gates
            break
    if not failed_reasons and gate_failed_pairs:
        failed_reasons = gate_failed_pairs[0][1]

    best_fallback["match_status"] = "unmatched"
    best_fallback["gate_passed"]  = False
    best_fallback["gate_failed"]  = failed_reasons
    best_fallback["warnings"]     = (
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
    aggregate_items: bool = AGGREGATE_SAME_ITEM,
    **kwargs,
) -> dict:
    """
    사용내역서 전체 항목을 영수증 리스트와 매칭.

    [3단계 처리 흐름]
      Step 1 — 문서 유형별 분리
        · doc_type="tax_invoice"    → 세금계산서 풀 (사전검증 도구)
        · doc_type="wage_statement" → 임금명세서 풀 (세금계산서 검증 제외, exempt)
        · 나머지                    → 영수증·거래명세표 풀 (사전검증 대상)

      Step 1-A — 세금계산서 사전 검증 (영수증·거래명세표만 대상)
        · 영수증·거래명세표 각각에 대해 세금계산서와 월·금액·업체명 비교
        · 결과: tax_invoice_status = "verified" | "unverified"
        · unverified 도 Step 2에 포함 (탈락 아님, 표시만)

      Step 2 — 사용내역서 ↔ 영수증 매칭 (기존 로직)
        · Gate(월·금액·업체명) + 점수 기반 매칭
        · 임금명세서: Gate 3(업체명) 자동 면제, tax_invoice_status="exempt"
        · 세금계산서 자체 증빙: tax_invoice_status="self"
        · 각 결과에 Step 1의 tax_invoice_status 포함

    Args:
        receipts        : 영수증·거래명세표·세금계산서·임금명세서 혼합 목록.
                          doc_type 필드로 자동 분리.
        aggregate_items : 같은 품목명 행을 합산할지 여부.
                          MVP: True (영수증·거래명세표 구분 없이 동일 적용)
        **kwargs        : 구버전 호환용 (photo_texts 등 추가 인자 무시)

    Returns:
        {
          "batch_id":       str,
          "source_usage":   str,
          "aggregated":     bool,
          "thresholds":     {"matched": float, "review": float},
          "tax_invoice_verification": {   # Step 1 요약
              "total_evidence_docs": int,
              "verified":            int,
              "unverified":          int,
              "exempt_wage_docs":    int,  # 임금명세서 건수
          },
          "results":        [ {match_twoway 결과 + tax_invoice_status}, ... ],
          "summary":        { total/matched/review_needed/unmatched/rejected/... },
          "generated_at":   str
        }
    """
    from src.services.tax_invoice_verifier import verify_receipts_against_tax_invoices

    batch_id = str(uuid.uuid4())[:8]

    # ── Step 1: 문서 유형별 분리 ──────────────────────────────
    tax_invoices  = [r for r in receipts if r.get("doc_type") == "tax_invoice"]
    wage_docs     = [r for r in receipts if r.get("doc_type") == "wage_statement"]
    evidence_docs = [
        r for r in receipts
        if r.get("doc_type") not in ("tax_invoice", "wage_statement")
    ]

    # ── Step 1-B: 다품목 영수증 → 품목별 가상 영수증으로 분리 ──────
    # 거래명세서·세금계산서에 2개 이상 품목이 있으면 품목별로 분리한다.
    # 각 가상 영수증의 total_amount = 해당 품목 공급가액 + 세액 (부가세 포함 실지출).
    # 사용내역서는 부가세 포함 금액으로 기재되므로 비교 기준을 일치시키기 위함.
    expanded_evidence: list[dict] = []
    for doc in evidence_docs:
        sub_receipts = _expand_to_item_receipts(doc)
        if len(sub_receipts) > 1:
            logger.info(
                "다품목 영수증 분리 [%s]: %d품목 → %d건",
                doc.get("source_file"), len(doc.get("items", [])), len(sub_receipts),
            )
        expanded_evidence.extend(sub_receipts)
    evidence_docs = expanded_evidence

    # ── Step 1-A: 세금계산서 사전 검증 (영수증·거래명세표만) ──
    verified_docs = verify_receipts_against_tax_invoices(evidence_docs, tax_invoices)

    ti_verified   = sum(1 for d in verified_docs if d["tax_invoice_status"] == "verified")
    ti_unverified = sum(1 for d in verified_docs if d["tax_invoice_status"] == "unverified")

    logger.info(
        "Step1 세금계산서 사전검증: 영수증·거래명세표 %d건 / "
        "세금계산서 %d건 / 임금명세서 %d건 → verified %d / unverified %d",
        len(evidence_docs), len(tax_invoices), len(wage_docs),
        ti_verified, ti_unverified,
    )

    # ── 세금계산서 → Step 2 풀에 포함 (직접 증빙 케이스) ──────
    # 일부 항목(예: 본사 전담조직 임금)은 세금계산서가 유일한 증빙이므로
    # 세금계산서도 Step 2 매칭 풀에 포함한다.
    # tax_invoice_status = "self" : 세금계산서 자체가 증빙인 경우
    ti_as_evidence = [
        {**ti, "tax_invoice_status": "self", "matched_tax_invoice": None, "ti_failed_gates": []}
        for ti in tax_invoices
    ]

    # ── 임금명세서 → Step 2 풀에 포함 (세금계산서 검증 제외) ──
    # 임금명세서는 세금계산서가 발행되지 않으므로 사전검증 대상에서 제외.
    # Gate 3(업체명)도 자동 면제 (_check_hard_gates 참조).
    # ③합계확인에 구분별 breakdown(items)이 있으면 품목별 가상 영수증으로 확장 →
    # 사용내역서 line_item(안전관리자 인건비 / 안전보건담당자 업무수당 등)과 1:1 매칭.
    # tax_invoice_status = "exempt" : 세금계산서 검증 대상이 아님을 명시
    expanded_wage: list[dict] = []
    for w in wage_docs:
        base = {**w, "tax_invoice_status": "exempt", "matched_tax_invoice": None, "ti_failed_gates": []}
        sub = _expand_to_item_receipts(base)
        if len(sub) > 1:
            logger.info(
                "임금명세서 분리 [%s]: %d구분 → %d건",
                w.get("source_file"), len(w.get("items", [])), len(sub),
            )
        expanded_wage.extend(sub)
    wage_as_evidence = expanded_wage

    step2_pool = verified_docs + ti_as_evidence + wage_as_evidence

    # ── Step 2: 사용내역서 ↔ 영수증 매칭 ─────────────────────
    raw_items = usage_statement.get("line_items") or usage_statement.get("items", [])
    items = [_normalize_usage_item(i) for i in raw_items]

    # ── 품목명 기준 합산 (MVP 전처리) ────────────────────────
    # ※ 실제 데이터 확보 후 재검토 포인트:
    #   - 영수증이 일자별로 별도 발행된다면 집계 없이 1:1 매칭 필요
    #   - 거래명세표는 월 합산이므로 집계 유지
    #   → doc_type별 분기: delivery_statement만 aggregate_items=True
    if aggregate_items:
        items = _aggregate_by_item_name(items)
        logger.debug(
            "품목 합산: 원본 %d행 → 합산 후 %d행",
            len(raw_items), len(items),
        )

    match_results = []
    for usage_item in items:
        best = match_best(usage_item, step2_pool, threshold, threshold_matched)
        # 매칭된 영수증의 세금계산서 검증 상태를 결과에 포함
        matched_receipt = best.get("receipt") or {}
        best["tax_invoice_status"]  = matched_receipt.get("tax_invoice_status", "unverified")
        best["matched_tax_invoice"] = matched_receipt.get("matched_tax_invoice")
        match_results.append(best)

    total     = len(match_results)
    matched   = sum(1 for r in match_results if r["match_status"] == "matched")
    review    = sum(1 for r in match_results if r["match_status"] == "review_needed")
    unmatched = sum(1 for r in match_results if r["match_status"] == "unmatched")
    rejected  = sum(1 for r in match_results if r["match_status"] == "rejected")

    return {
        "batch_id":     batch_id,
        "source_usage": usage_statement.get("source_file", ""),
        "aggregated":   aggregate_items,
        "thresholds": {
            "matched": threshold_matched,
            "review":  threshold,
        },
        "tax_invoice_verification": {
            "total_evidence_docs": len(evidence_docs),
            "verified":            ti_verified,
            "unverified":          ti_unverified,
            "exempt_wage_docs":    len(wage_docs),   # 임금명세서 (세금계산서 검증 대상 외)
        },
        "results":      match_results,
        "summary": {
            "total":           total,
            "matched":         matched,
            "review_needed":   review,
            "unmatched":       unmatched,
            "rejected":        rejected,
            "match_rate_pct":  round(matched / total * 100, 1) if total else 0.0,
            "review_rate_pct": round(review  / total * 100, 1) if total else 0.0,
        },
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


# ══════════════════════════════════════════════════════════════
# 10. 결과 저장 / 콘솔 출력
# ══════════════════════════════════════════════════════════════

def save_match_result(result: dict, output_path: str) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return str(output_path)


def print_match_result(result: dict):
    sep = "─" * 60
    status_icon = {
        "matched": "✅", "review_needed": "🔍",
        "unmatched": "❌", "rejected": "🚫",
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
    rcpt  = result.get("receipt", {})
    rcpt_date   = _extract_receipt_date(rcpt) or "-"
    rcpt_vendor = _extract_receipt_vendor(rcpt) or "-"
    usage_desc = usage.get('description') or '-'
    print(f"  [사용내역서]  {usage.get('date', '-')}  "
          f"{usage_desc[:25]}  "
          f"{(usage.get('amount') or 0):,}원")
    print(f"  [영수증]      {rcpt_date}  {rcpt_vendor[:20]}  "
          f"{(rcpt.get('total_amount') or 0):,}원  "
          f"[{rcpt.get('doc_type', 'receipt')}]")

    comp = result.get("component_scores", {})
    if comp:
        print(f"{sep}")
        print(f"  [세부 점수]  ※날짜=월단위(같은월→1.0)")
        labels = {"date": "날짜", "amount": "금액", "vendor": "거래처", "item_desc": "품목내역"}
        for k, label in labels.items():
            v = comp.get(k)
            bar = "▓" * int((v or 0) * 20) + "░" * (20 - int((v or 0) * 20))
            val_str = f"{v:.4f}" if v is not None else " N/A  "
            print(f"    {label:6s}  [{bar}]  {val_str}")
    print(f"{sep}\n")


def print_batch_summary(batch_result: dict):
    sep = "═" * 60
    s   = batch_result.get("summary", {})
    th  = batch_result.get("thresholds", {})
    print(f"\n{sep}")
    print(f"  배치 매칭 완료  (batch_id: {batch_result.get('batch_id')})")
    print(f"  사용내역서:    {batch_result.get('source_usage')}")
    print(f"  날짜 전략:     월 단위 기준 (같은 연월 + 경계 ±{GATE_DATE_BOUNDARY_DAYS}일)")
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
# 11. CLI 진입점
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="2-way 매칭 엔진 (월 단위 날짜 비교) — 산업안전관리비 검증 시스템"
    )
    parser.add_argument("--usage",   required=True)
    parser.add_argument("--receipt", required=True, nargs="+")
    parser.add_argument("--output",  default=None)
    parser.add_argument("--threshold",         type=float, default=THRESHOLD_REVIEW)
    parser.add_argument("--threshold-matched", type=float, default=THRESHOLD_MATCHED)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.usage, encoding="utf-8") as f:
        usage_statement = json.load(f)

    receipts = []
    for rpath in args.receipt:
        with open(rpath, encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                receipts.extend(data)
            else:
                receipts.append(data)

    batch = match_all_usage_to_receipts(
        usage_statement, receipts,
        threshold=args.threshold,
        threshold_matched=args.threshold_matched,
    )

    if args.verbose:
        for r in batch["results"]:
            print_match_result(r)

    print_batch_summary(batch)

    out_dir  = Path(args.output) if args.output else Path(args.usage).parent / "match_results"
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"match_batch_monthly_{ts}.json"
    saved    = save_match_result(batch, str(out_path))
    print(f"  💾 저장: {saved}\n")


if __name__ == "__main__":
    main()
