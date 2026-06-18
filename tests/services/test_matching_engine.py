"""
matching_service.py 테스트 케이스
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
실제 파싱 결과 파일을 로드해 다양한 시나리오를 검증.

실행:
    python -m pytest tests/services/test_matching_engine.py
    python tests/services/test_matching_engine.py --verbose
"""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.services.matching_service import (
    match_threeway,
    match_best,
    match_all_usage_to_receipts,
    print_match_result,
    print_batch_summary,
    text_similarity,
    date_score,
    amount_score,
)

# ──────────────────────────────────────────────────────────────
# 샘플 데이터 (실제 파서 출력과 동일한 구조)
# ──────────────────────────────────────────────────────────────

# ── 사용내역서 항목들 ─────────────────────────────────────────
USAGE_ITEMS = [
    {
        "seq": "1",
        "date": "2025-01-05",
        "category": "안전시설비",
        "description": "안전모 ABS형 구입",
        "amount": 120000,
        "evidence_type": "세금계산서",
        "vendor": "대한안전용품(주)",
    },
    {
        "seq": "2",
        "date": "2025-01-08",
        "category": "안전시설비",
        "description": "안전화 270mm 5켤레",
        "amount": 225000,
        "evidence_type": "세금계산서",
        "vendor": "대한안전용품(주)",
    },
    {
        "seq": "6",
        "date": "2025-02-03",
        "category": "안전교육비",
        "description": "건설현장 안전교육(15인)",
        "amount": 500000,
        "evidence_type": "세금계산서",
        "vendor": "(주)한국산업안전교육",
    },
]

# ── 영수증 OCR: 안전용품 카드영수증 (기존 실제 파일과 동일) ──────
RECEIPT_SAFETY_GEAR = {
    "ocr_type": "receipt",
    "source_file": "receipt_카드_안전용품_001.jpg",
    "ocr_engine": "clova_receipt_v2",
    "infer_result": "SUCCESS",
    "store": {
        "name": "대한안전용품(주)",
        "biz_num": "123-45-67890",
        "address": "서울 구로구 공단로 45",
        "tel": "0212345578",
    },
    "payment": {
        "date": "2025-01-05",
        "time": "10:22:38",
        "card_company": "국민카드(법인)",
        "card_number": "9410-1234",
    },
    "items": [
        {"name": "안전모 ABS", "count": 10, "unit_price": 12000, "amount": 120000},
        {"name": "안전화(270mm)", "count": 5, "unit_price": 45000, "amount": 225000},
        {"name": "안전벨트(X형)", "count": 2, "unit_price": 38000, "amount": 76000},
    ],
    "total_amount": 421000,
    "tax_amount": 38273,
}

# ── 영수증 OCR: 안전교육비 ──────────────────────────────────────
RECEIPT_EDUCATION = {
    "ocr_type": "receipt",
    "source_file": "receipt_현금_교육비_001.jpg",
    "ocr_engine": "clova_receipt_v2",
    "infer_result": "SUCCESS",
    "store": {
        "name": "(주)한국산업안전교육",
        "biz_num": "234-56-78901",
        "address": "서울 중구 을지로 100",
    },
    "payment": {
        "date": "2025-02-03",
        "time": "14:00:00",
        "card_company": None,
    },
    "items": [
        {"name": "건설현장 안전교육 15인", "count": 1, "unit_price": 500000, "amount": 500000},
    ],
    "total_amount": 500000,
    "tax_amount": 45455,
}

# ── 영수증: 품목명 없는 간이영수증 (반려 대상) ─────────────────
RECEIPT_NO_ITEMS = {
    "ocr_type": "receipt",
    "source_file": "receipt_간이_의약품_001.jpg",
    "ocr_engine": "clova_receipt_v2",
    "infer_result": "SUCCESS",
    "store": {"name": "우리약국"},
    "payment": {"date": "2025-01-22"},
    "items": [],           # ← 품목명 없음 → 반려 조건
    "total_amount": 64000,
}

# ── 영수증: OCR 실패 (반려 대상) ────────────────────────────────
RECEIPT_OCR_FAIL = {
    "ocr_type": "receipt",
    "source_file": "blurry_receipt.jpg",
    "infer_result": "FAILURE",
    "error": "이미지 품질 불량",
    "store": {},
    "payment": {},
    "items": [],
    "total_amount": None,
}

# ── 영수증: 날짜/금액 완전 불일치 (unmatched 대상) ──────────────
RECEIPT_WRONG = {
    "ocr_type": "receipt",
    "source_file": "receipt_다른업체_001.jpg",
    "infer_result": "SUCCESS",
    "store": {"name": "편의점GS25"},
    "payment": {"date": "2024-06-15"},
    "items": [
        {"name": "음료수", "count": 3, "unit_price": 2000, "amount": 6000},
    ],
    "total_amount": 6000,
}


# ──────────────────────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────────────────────

PASS  = "✅ PASS"
FAIL  = "❌ FAIL"

def check(condition: bool, label: str) -> bool:
    icon = PASS if condition else FAIL
    print(f"  {icon}  {label}")
    return condition


# ──────────────────────────────────────────────────────────────
# 테스트 케이스
# ──────────────────────────────────────────────────────────────

def test_text_similarity():
    """유사도 함수 기본 동작 검증"""
    print("\n[Test 1] text_similarity 기본 동작")

    s1 = text_similarity("안전모 ABS형 구입", "안전모 ABS")
    check(s1 > 0.5, f"유사 텍스트 → 높은 점수 기대 (got {s1:.4f})")

    s2 = text_similarity("안전모", "교육비 납부")
    check(s2 < 0.3, f"이질 텍스트 → 낮은 점수 기대 (got {s2:.4f})")

    s3 = text_similarity("대한안전용품(주)", "대한안전용품(주)")
    check(s3 > 0.9, f"동일 텍스트 → 1.0 근접 기대 (got {s3:.4f})")

    s4 = text_similarity("", "안전모")
    check(s4 == 0.0, f"빈 문자열 → 0.0 (got {s4})")


def test_date_score():
    """날짜 점수 경계값 검증"""
    print("\n[Test 2] date_score 경계값")

    check(date_score("2025-01-05", "2025-01-05") == 1.0,   "당일 → 1.0")
    check(date_score("2025-01-05", "2025-01-07") == 0.85,  "2일 차이 → 0.85")
    check(date_score("2025-01-05", "2025-01-10") == 0.60,  "5일 차이 → 0.60")
    check(date_score("2025-01-05", "2025-01-12") == 0.60,  "7일 차이 → 0.60")
    check(date_score("2025-01-05", "2025-03-01") == 0.0,   "55일 차이 → 0.0")
    check(date_score(None, "2025-01-05") == 0.4,           "한쪽 None → 0.4")
    check(date_score(None, None) is None,                  "둘 다 None → None")


def test_amount_score():
    """금액 점수 경계값 검증"""
    print("\n[Test 3] amount_score 경계값")

    check(amount_score(120000, 120000) == 1.0,  "동일 금액 → 1.0")
    check(amount_score(120000, 121000) == 1.0,  "0.8% 차이 → 1.0")
    check(amount_score(120000, 125000) == 0.85, "4.2% 차이 → 0.85")
    check(amount_score(120000, 132000) == 0.65, "10% 차이 → 0.65")
    check(amount_score(120000, 200000) == 0.0,  "66% 차이 → 0.0")
    check(amount_score(None, 120000) == 0.3,    "한쪽 None → 0.3")


def test_matched(verbose: bool = False):
    """정상 매칭 케이스"""
    print("\n[Test 4] 정상 매칭 — 안전모 구입 ↔ 안전용품 영수증")

    photo = "안전모와 안전화 대한안전용품 배달 현장 사진"
    result = match_threeway(USAGE_ITEMS[0], RECEIPT_SAFETY_GEAR, photo)

    check(result["match_status"] == "matched",
          f"match_status == 'matched' (got '{result['match_status']}')")
    check(result["similarity_score"] >= 0.75,
          f"유사도 >= 0.75 (got {result['similarity_score']:.4f})")
    check(result["reject_reason"] is None, "reject_reason == None")
    check("match_id" in result and len(result["match_id"]) > 0, "match_id 존재")

    if verbose:
        print_match_result(result)


def test_matched_education(verbose: bool = False):
    """교육비 매칭 케이스"""
    print("\n[Test 5] 정상 매칭 — 안전교육비 ↔ 교육 영수증")

    photo = "한국산업안전교육 건설현장 교육 수료증 현장사진"
    result = match_threeway(USAGE_ITEMS[2], RECEIPT_EDUCATION, photo)

    check(result["match_status"] == "matched",
          f"match_status == 'matched' (got '{result['match_status']}')")
    check(result["similarity_score"] >= 0.75,
          f"유사도 >= 0.75 (got {result['similarity_score']:.4f})")

    if verbose:
        print_match_result(result)


def test_rejected_no_items(verbose: bool = False):
    """품목명 없는 영수증 → 반려"""
    print("\n[Test 6] 반려 — 품목명 없는 영수증")

    result = match_threeway(USAGE_ITEMS[0], RECEIPT_NO_ITEMS)

    check(result["match_status"] == "rejected",
          f"match_status == 'rejected' (got '{result['match_status']}')")
    check(result["reject_reason"] is not None, "reject_reason 설명 존재")
    check("품목명" in (result.get("reject_reason") or ""),
          f"반려 사유에 '품목명' 포함 (got '{result['reject_reason']}')")

    if verbose:
        print_match_result(result)


def test_rejected_ocr_fail(verbose: bool = False):
    """OCR 실패 영수증 → 반려"""
    print("\n[Test 7] 반려 — OCR 인식 실패 영수증")

    result = match_threeway(USAGE_ITEMS[0], RECEIPT_OCR_FAIL)

    check(result["match_status"] == "rejected",
          f"match_status == 'rejected' (got '{result['match_status']}')")
    check("실패" in (result.get("reject_reason") or "") or
          "FAILURE" in (result.get("reject_reason") or ""),
          f"반려 사유에 실패 언급 (got '{result['reject_reason']}')")

    if verbose:
        print_match_result(result)


def test_unmatched(verbose: bool = False):
    """완전히 무관한 영수증 → unmatched"""
    print("\n[Test 8] 비매칭 — 무관한 영수증")

    result = match_threeway(USAGE_ITEMS[0], RECEIPT_WRONG)

    check(result["match_status"] in ("unmatched", "rejected"),
          f"match_status in {{unmatched, rejected}} (got '{result['match_status']}')")
    check(result["similarity_score"] < 0.75,
          f"유사도 < 0.75 (got {result['similarity_score']:.4f})")

    if verbose:
        print_match_result(result)


def test_match_best(verbose: bool = False):
    """여러 후보 중 최선 매칭"""
    print("\n[Test 9] match_best — 여러 후보 중 최선 선택")

    candidates = [RECEIPT_NO_ITEMS, RECEIPT_WRONG, RECEIPT_SAFETY_GEAR, RECEIPT_OCR_FAIL]
    result = match_best(USAGE_ITEMS[0], candidates, "안전모 현장사진")

    check(result["match_status"] == "matched",
          f"match_status == 'matched' (got '{result['match_status']}')")
    check(result["receipt"].get("source_file") == "receipt_카드_안전용품_001.jpg",
          f"올바른 영수증 선택 (got '{result['receipt'].get('source_file')}')")

    if verbose:
        print_match_result(result)


def test_batch_with_real_files(verbose: bool = False):
    """실제 파싱 결과 파일로 배치 매칭 (파일 존재 시만 실행)"""
    print("\n[Test 11] 실제 파일 배치 매칭")

    code_dir = Path(__file__).parent
    usage_path = code_dir / "parsed_results" / "sample_사용내역서_2025Q1_parsed_20260424_174307.json"
    receipt_paths = list((code_dir / "ocr_results").glob("*_ocr_*.json"))
    # raw 파일 제외
    receipt_paths = [p for p in receipt_paths if "_raw" not in p.name]

    if not usage_path.exists():
        print("  ⏭️  건너뜀 — 사용내역서 파싱 파일 없음")
        return
    if not receipt_paths:
        print("  ⏭️  건너뜀 — 영수증 OCR 파일 없음")
        return

    with open(usage_path, encoding="utf-8") as f:
        usage_statement = json.load(f)

    receipts = []
    for rp in receipt_paths:
        with open(rp, encoding="utf-8") as f:
            receipts.append(json.load(f))

    batch = match_all_usage_to_receipts(usage_statement, receipts)

    check(
        "batch_id" in batch and "results" in batch and "summary" in batch,
        "배치 결과 구조 올바름"
    )
    check(
        len(batch["results"]) == len(usage_statement.get("items", [])),
        f"결과 수 == 사용내역서 항목 수 ({len(batch['results'])}건)"
    )
    check(
        0 <= batch["summary"]["match_rate_pct"] <= 100,
        f"매칭률 범위 정상 ({batch['summary']['match_rate_pct']}%)"
    )

    print_batch_summary(batch)

    if verbose:
        for r in batch["results"]:
            print_match_result(r)

    # 결과 저장
    out_path = code_dir / "match_results" / "test_batch_result.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(batch, f, ensure_ascii=False, indent=2)
    print(f"  💾 저장: {out_path}")


def test_output_schema(verbose: bool = False):
    """출력 JSON 스키마 필드 존재 여부 검증"""
    print("\n[Test 12] 출력 스키마 필드 검증")

    result = match_threeway(USAGE_ITEMS[0], RECEIPT_SAFETY_GEAR, "안전모 구입")

    required_fields = [
        "match_id", "usage_item", "receipt", "photo_text",
        "similarity_score", "component_scores",
        "match_status", "reject_reason", "matched_at",
    ]
    for field in required_fields:
        check(field in result, f"필드 '{field}' 존재")

    check(
        result["match_status"] in ("matched", "unmatched", "rejected"),
        f"match_status 유효값 (got '{result['match_status']}')"
    )
    check(
        isinstance(result["similarity_score"], float) and
        0.0 <= result["similarity_score"] <= 1.0,
        f"similarity_score ∈ [0, 1] (got {result['similarity_score']})"
    )


# ──────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="세부 점수 출력")
    args = parser.parse_args()
    v = args.verbose

    print("=" * 56)
    print("  matching_engine.py 테스트 시작")
    print("=" * 56)

    tests = [
        test_text_similarity,
        test_date_score,
        test_amount_score,
        lambda: test_matched(v),
        lambda: test_matched_education(v),
        lambda: test_rejected_no_items(v),
        lambda: test_rejected_ocr_fail(v),
        lambda: test_unmatched(v),
        lambda: test_match_best(v),
        lambda: test_batch_with_real_files(v),
        lambda: test_output_schema(v),
    ]

    passed = failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  ❌ AssertionError: {e}")
            failed += 1
        except Exception as e:
            print(f"  ❌ Exception: {type(e).__name__}: {e}")
            failed += 1

    print("\n" + "=" * 56)
    total = passed + failed
    print(f"  테스트 완료: {total}개 실행 | {passed}개 성공 | {failed}개 실패")
    print("=" * 56 + "\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
