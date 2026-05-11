"""
산업안전관리비 AI 검증 시스템 — 테스트 픽스처 데이터
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
test_matching.py / test_matching_engine.py 에서 공통 사용하는
예시 사용내역서 항목·영수증 OCR 결과·예상 매칭 결과를 정의한다.

시나리오 (V5 목업 데이터와 동일):
    항목1  안전관리자 임금   3,000,000원  → unmatched (영수증 없음)
    항목2  안전모           150,000원    → matched   (R01, 1:N)
    항목3  안전화           150,000원    → matched   (R01, 1:N)
    항목4  안전조끼         200,000원    → unmatched (R02, 금액 불일치)
    항목5  스마트 안전조끼  180,000원    → matched   (R02, 유사품목)
    항목6  안전화 추가분    90,000원     → review_needed (R04, 날짜 3일 차이)
"""

from __future__ import annotations

# ══════════════════════════════════════════════════════════════
# 사용내역서 개별 항목 리스트
# ══════════════════════════════════════════════════════════════

usage_items: list[dict] = [
    # seq=0  안전관리자 임금 (영수증 없음 → unmatched)
    {
        "seq": 0,
        "category_code": "CAT_01",
        "used_on": "2025-04-30",
        "date": "2025-04-30",
        "description": "안전관리자 임금",
        "item_name": "안전관리자 임금",
        "unit": "명",
        "quantity": 1.0,
        "unit_price": 3000000.0,
        "total_amount": 3000000,
        "amount": 3000000,
        "remark": None,
        "page_no": 2,
        "line_no": 1,
    },
    # seq=1  안전모 (R01 → matched)
    {
        "seq": 1,
        "category_code": "CAT_03",
        "used_on": "2025-04-15",
        "date": "2025-04-15",
        "description": "안전모",
        "item_name": "안전모",
        "unit": "개",
        "quantity": 10.0,
        "unit_price": 15000.0,
        "total_amount": 150000,
        "amount": 150000,
        "remark": None,
        "page_no": 3,
        "line_no": 1,
    },
    # seq=2  안전화 (R01 → matched, 추락방지망 비교는 R03 시나리오용)
    {
        "seq": 2,
        "category_code": "CAT_03",
        "used_on": "2025-04-15",
        "date": "2025-04-15",
        "description": "안전화",
        "item_name": "안전화",
        "unit": "켤레",
        "quantity": 5.0,
        "unit_price": 30000.0,
        "total_amount": 150000,
        "amount": 150000,
        "remark": None,
        "page_no": 3,
        "line_no": 2,
    },
    # seq=3  안전교육비 (R05 → unmatched, 금액 3.5배 차이)
    {
        "seq": 3,
        "category_code": "CAT_02",
        "used_on": "2025-04-20",
        "date": "2025-04-20",
        "description": "안전교육비",
        "item_name": "안전교육비",
        "unit": "식",
        "quantity": 1.0,
        "unit_price": 350000.0,
        "total_amount": 350000,
        "amount": 350000,
        "remark": None,
        "page_no": 4,
        "line_no": 1,
    },
    # seq=4  스마트 안전조끼 (R02 → matched)
    {
        "seq": 4,
        "category_code": "CAT_03",
        "used_on": "2025-04-22",
        "date": "2025-04-22",
        "description": "스마트 안전조끼",
        "item_name": "스마트 안전조끼",
        "unit": "개",
        "quantity": 8.0,
        "unit_price": 22500.0,
        "total_amount": 180000,
        "amount": 180000,
        "remark": None,
        "page_no": 3,
        "line_no": 3,
    },
    # seq=5  안전화 추가분 (R04 → review_needed, 날짜 3일 차이)
    {
        "seq": 5,
        "category_code": "CAT_03",
        "used_on": "2025-04-10",
        "date": "2025-04-10",
        "description": "안전화 추가분",
        "item_name": "안전화 추가분",
        "unit": "켤레",
        "quantity": 3.0,
        "unit_price": 30000.0,
        "total_amount": 90000,
        "amount": 90000,
        "remark": None,
        "page_no": 3,
        "line_no": 4,
    },
]


# ══════════════════════════════════════════════════════════════
# 영수증 OCR 결과 리스트
# ══════════════════════════════════════════════════════════════

receipts: list[dict] = [
    # R01  보호구 거래명세표 (안전모 + 안전화)
    {
        "receipt_id": "R01",
        "source_file": "보호구_거래명세표_20250415.jpg",
        "infer_result": "SUCCESS",
        "vendor": "한국안전용품",
        "date": "2025-04-15",
        "total_amount": 300000,
        "items": [
            {"item_name": "안전모",  "count": 10, "unit_price": 15000, "amount": 150000},
            {"item_name": "안전화",  "count": 5,  "unit_price": 30000, "amount": 150000},
        ],
        "validation": {"is_valid": True, "items_sum_match": True, "warnings": []},
    },
    # R02  보호구 거래명세표 (안전조끼 — 금액 불일치)
    {
        "receipt_id": "R02",
        "source_file": "보호구_거래명세표_20250422.jpg",
        "infer_result": "SUCCESS",
        "vendor": "현장안전물자",
        "date": "2025-04-22",
        "total_amount": 180000,
        "items": [
            {"item_name": "안전조끼", "count": 8, "unit_price": 22500, "amount": 180000},
        ],
        "validation": {"is_valid": True, "items_sum_match": True, "warnings": []},
    },
    # R03  추락방지망 (거래처명 차이 → review_needed)
    {
        "receipt_id": "R03",
        "source_file": "추락방지망_20250415.jpg",
        "infer_result": "SUCCESS",
        "vendor": "안전시설공업사",
        "date": "2025-04-15",
        "total_amount": 150000,
        "items": [
            {"item_name": "추락방지망", "count": 1, "unit_price": 150000, "amount": 150000},
        ],
        "validation": {"is_valid": True, "items_sum_match": True, "warnings": []},
    },
    # R04  안전화 추가분 (날짜 3일 차이 → review_needed)
    {
        "receipt_id": "R04",
        "source_file": "안전화_20250413.jpg",
        "infer_result": "SUCCESS",
        "vendor": "한국안전용품",
        "date": "2025-04-13",
        "total_amount": 90000,
        "items": [
            {"item_name": "안전화", "count": 3, "unit_price": 30000, "amount": 90000},
        ],
        "validation": {"is_valid": True, "items_sum_match": True, "warnings": []},
    },
    # R05  교육비 영수증 (금액 3.5배 차이 → unmatched)
    {
        "receipt_id": "R05",
        "source_file": "안전교육_20250420.jpg",
        "infer_result": "SUCCESS",
        "vendor": "안전교육원",
        "date": "2025-04-20",
        "total_amount": 1225000,
        "items": [
            {"item_name": "안전교육비", "count": 1, "unit_price": 1225000, "amount": 1225000},
        ],
        "validation": {"is_valid": True, "items_sum_match": True, "warnings": []},
    },
    # R06  품목 불일치 영수증 (사무용품 → unmatched)
    {
        "receipt_id": "R06",
        "source_file": "사무용품_20250430.jpg",
        "infer_result": "SUCCESS",
        "vendor": "사무용품점",
        "date": "2025-04-30",
        "total_amount": 3000000,
        "items": [
            {"item_name": "사무용품", "count": 1, "unit_price": 3000000, "amount": 3000000},
        ],
        "validation": {"is_valid": True, "items_sum_match": True, "warnings": []},
    },
    # R07  품목명 없는 영수증 → rejected
    {
        "receipt_id": "R07",
        "source_file": "영수증_품목없음.jpg",
        "infer_result": "SUCCESS",
        "vendor": "알수없음",
        "date": "2025-04-15",
        "total_amount": 150000,
        "items": [],
        "validation": {"is_valid": False, "items_sum_match": False, "warnings": ["품목 없음"]},
    },
    # R08  OCR 실패 → rejected
    {
        "receipt_id": "R08",
        "source_file": "ocr_실패.jpg",
        "infer_result": "FAILED",
        "vendor": None,
        "date": None,
        "total_amount": None,
        "items": [],
        "validation": {"is_valid": False, "items_sum_match": False, "warnings": ["OCR 실패"]},
    },
]


# ══════════════════════════════════════════════════════════════
# 현장사진 텍스트 (seq → 사진 OCR 결과 텍스트)
# ══════════════════════════════════════════════════════════════

photo_texts: dict[int, str] = {
    1: "안전모 착용 확인 / 현장 작업자 10명 착용 완료",
    2: "안전화 지급 완료 / 보관함 사진",
    4: "안전조끼 착용 현황 / 작업자 8명 착용",
}


# ══════════════════════════════════════════════════════════════
# 사용내역서 전체 구조 (match_all_usage_to_receipts 입력용)
# ══════════════════════════════════════════════════════════════

usage_statement: dict = {
    "source_file": "2025년4월_안전관리비사용내역서.pdf",
    "parse_status": "SUCCESS",
    "meta": {
        "project_name": "한강뷰 아파트 신축공사",
        "report_month": "2025-04",
    },
    "items": usage_items,
}


# ══════════════════════════════════════════════════════════════
# 예상 매칭 결과 (테스트 검증용)
# ══════════════════════════════════════════════════════════════

expected_matches: list[dict] = [
    {"usage_seq": 0, "expected_status": "unmatched",     "note": "영수증 미제출"},
    {"usage_seq": 1, "expected_status": "matched",       "note": "안전모 R01"},
    {"usage_seq": 2, "expected_status": "matched",       "note": "안전화 R01"},
    {"usage_seq": 3, "expected_status": "unmatched",     "note": "금액 3.5배 차이"},
    {"usage_seq": 4, "expected_status": "matched",       "note": "스마트 안전조끼 R02"},
    {"usage_seq": 5, "expected_status": "review_needed", "note": "날짜 3일 차이"},
]
