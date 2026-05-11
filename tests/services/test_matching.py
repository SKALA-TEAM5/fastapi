"""
산업안전관리비 AI 검증 시스템 — 매칭 엔진 테스트
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
test_matching.py

test_data.py의 예시 데이터로 매칭 엔진을 테스트하고
각 케이스별 match_status 예상값과 실제값을 비교한다.
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.services.matching_service import (
    match_threeway,
    match_best,
    match_all_usage_to_receipts,
    print_match_result,
    print_batch_summary,
    THRESHOLD_MATCHED,
    THRESHOLD_REVIEW,
)
from tests.services.test_data import (
    usage_items,
    receipts,
    photo_texts,
    usage_statement,
    expected_matches,
)


# ══════════════════════════════════════════════════════════════
# 헬퍼 함수
# ══════════════════════════════════════════════════════════════

STATUS_ICON = {
    "matched":       "✅",
    "review_needed": "🔍",
    "unmatched":     "❌",
    "rejected":      "🚫",
}

def _icon(status: str) -> str:
    return STATUS_ICON.get(status, "?")


# ══════════════════════════════════════════════════════════════
# TEST 1: 개별 영수증 매칭 테스트 (match_threeway)
# ══════════════════════════════════════════════════════════════

def test_individual_receipts():
    print("\n" + "═" * 60)
    print("  TEST 1: 개별 영수증 매칭 (match_threeway)")
    print("  임계값: matched ≥ {:.2f}  /  review_needed ≥ {:.2f}".format(
        THRESHOLD_MATCHED, THRESHOLD_REVIEW))
    print("═" * 60)

    # 각 예상 매칭 케이스별로 직접 비교
    test_cases = [
        # (usage_item, receipt, 예상 상태, 설명)
        (usage_items[1], receipts[0],  "matched",       "안전모 구입 — R01"),  # seq2 vs R01
        (usage_items[4], receipts[1],  "matched",       "스마트 안전조끼 — R02"),  # seq5 vs R02
        (usage_items[2], receipts[2],  "review_needed", "추락방지망 — R03 (거래처명 차이)"),
        (usage_items[5], receipts[3],  "review_needed", "안전화 — R04 (날짜 3일 차이)"),
        (usage_items[3], receipts[4],  "unmatched",     "안전교육비 — R05 (금액 3.5배 차이)"),
        (usage_items[0], receipts[5],  "unmatched",     "안전관리자 인건비 — R06 (품목 불일치)"),
        (usage_items[1], receipts[6],  "rejected",      "안전모 vs R07 (품목명 없는 영수증)"),
        (usage_items[1], receipts[7],  "rejected",      "안전모 vs R08 (OCR 실패)"),
    ]

    passed = 0
    failed = 0
    fail_cases = []

    for usage, receipt, expected, description in test_cases:
        photo = photo_texts.get(usage["seq"], "")
        result = match_threeway(usage, receipt, photo)

        actual   = result["match_status"]
        score    = result["similarity_score"]
        matched  = (actual == expected)

        status_str = f"{_icon(actual)} {actual:<15}"
        mark       = "  PASS ✓" if matched else "  FAIL ✗"

        print(f"\n  [{receipt.get('receipt_id', '??')}] {description}")
        print(f"    점수: {score:.4f}   상태: {status_str}  예상: {_icon(expected)} {expected}  {mark}")

        comp = result.get("component_scores", {})
        if comp:
            parts = []
            labels = {"date": "날짜", "amount": "금액", "vendor": "거래처",
                      "item_desc": "품목", "photo_text": "사진"}
            for k, lbl in labels.items():
                v = comp.get(k)
                parts.append(f"{lbl}={v:.3f}" if v is not None else f"{lbl}=N/A")
            print(f"    세부: {' | '.join(parts)}")

        if result.get("reject_reason"):
            print(f"    반려사유: {result['reject_reason']}")

        if matched:
            passed += 1
        else:
            failed += 1
            fail_cases.append({
                "description": description,
                "expected": expected,
                "actual": actual,
                "score": score,
                "component_scores": comp,
                "reject_reason": result.get("reject_reason"),
            })

    print(f"\n  {'─' * 40}")
    print(f"  결과: PASS {passed}개 / FAIL {failed}개 / 전체 {passed + failed}개")
    return passed, failed, fail_cases


# ══════════════════════════════════════════════════════════════
# TEST 2: 배치 매칭 테스트 (match_all_usage_to_receipts)
# ══════════════════════════════════════════════════════════════

def test_batch_matching():
    print("\n" + "═" * 60)
    print("  TEST 2: 배치 매칭 (match_all_usage_to_receipts)")
    print("═" * 60)

    batch = match_all_usage_to_receipts(
        usage_statement,
        receipts,
        photo_texts,
    )

    print_batch_summary(batch)

    print("  [배치 매칭 상세 결과]")
    print(f"  {'seq':<5} {'사용내역':<26} {'최선영수증':<8} {'점수':>6}  {'상태':<18} {'예상'}")
    print(f"  {'─'*5} {'─'*26} {'─'*8} {'─'*6}  {'─'*18} {'─'*15}")

    # expected 딕셔너리로 변환
    exp_map = {e["usage_seq"]: e for e in expected_matches}

    passed = 0
    failed = 0
    fail_cases = []

    for r in batch["results"]:
        seq    = r["usage_item"].get("seq", "?")
        desc   = r["usage_item"].get("description", "")[:24]
        score  = r["similarity_score"]
        status = r["match_status"]
        rcpt_id = r["receipt"].get("receipt_id", "??")

        exp    = exp_map.get(seq, {})
        exp_st = exp.get("expected_status", "?")
        matched = (status == exp_st)
        mark    = "✓" if matched else "✗ FAIL"

        print(f"  {seq:<5} {desc:<26} {rcpt_id:<8} {score:>6.4f}  "
              f"{_icon(status)} {status:<15}  {_icon(exp_st)} {exp_st}  {mark}")

        if matched:
            passed += 1
        else:
            failed += 1
            fail_cases.append({
                "seq": seq, "expected": exp_st, "actual": status,
                "score": score, "receipt_id": rcpt_id,
            })

    print(f"\n  결과: PASS {passed}개 / FAIL {failed}개")
    return passed, failed, fail_cases


# ══════════════════════════════════════════════════════════════
# TEST 3: 임계값 경계 검증
# ══════════════════════════════════════════════════════════════

def test_threshold_boundaries():
    """임계값 경계(0.75, 0.85) 근처 케이스가 올바르게 분류되는지 확인"""
    from matching_engine import _decide_match_status

    print("\n" + "═" * 60)
    print("  TEST 3: 임계값 경계 검증 (_decide_match_status)")
    print("═" * 60)

    boundary_cases = [
        (0.90, "matched",       "0.90 → matched"),
        (0.85, "matched",       "0.85 → matched (경계값 포함)"),
        (0.84, "review_needed", "0.84 → review_needed"),
        (0.80, "review_needed", "0.80 → review_needed"),
        (0.75, "review_needed", "0.75 → review_needed (경계값 포함)"),
        (0.74, "unmatched",     "0.74 → unmatched"),
        (0.50, "unmatched",     "0.50 → unmatched"),
        (0.00, "unmatched",     "0.00 → unmatched"),
    ]

    passed = 0
    failed = 0

    for score, expected, desc in boundary_cases:
        actual  = _decide_match_status(score)
        matched = (actual == expected)
        mark    = "PASS ✓" if matched else "FAIL ✗"
        print(f"    score={score:.2f}  {_icon(actual)} {actual:<15}  예상: {_icon(expected)} {expected:<15}  {mark}")
        if matched:
            passed += 1
        else:
            failed += 1

    print(f"\n  결과: PASS {passed}개 / FAIL {failed}개")
    return passed, failed, []


# ══════════════════════════════════════════════════════════════
# 실패 케이스 원인 분석
# ══════════════════════════════════════════════════════════════

def analyze_failures(all_failures: list):
    if not all_failures:
        print("\n" + "═" * 60)
        print("  🎉 모든 테스트 통과! 실패 케이스 없음.")
        print("═" * 60)
        return

    print("\n" + "═" * 60)
    print(f"  ⚠️  실패 케이스 원인 분석 ({len(all_failures)}개)")
    print("═" * 60)

    for i, f in enumerate(all_failures, 1):
        print(f"\n  [{i}] {f.get('description', f.get('seq', '?'))}")
        print(f"    예상: {f['expected']}  /  실제: {f['actual']}  /  점수: {f.get('score', '?'):.4f}")
        comp = f.get("component_scores", {})
        if comp:
            labels = {"date": "날짜", "amount": "금액", "vendor": "거래처",
                      "item_desc": "품목", "photo_text": "사진"}
            for k, lbl in labels.items():
                v = comp.get(k)
                val_str = f"{v:.4f}" if v is not None else "N/A"
                print(f"    · {lbl:5s}: {val_str}")
        # 자동 원인 진단
        exp = f["expected"]
        act = f["actual"]
        score = f.get("score", 0)
        if exp == "review_needed" and act == "matched":
            print(f"    → 진단: 점수({score:.4f})가 matched 기준({THRESHOLD_MATCHED}) 이상으로 상향 분류됨")
            print(f"           임계값 조정 또는 데이터 재확인 권고")
        elif exp == "matched" and act == "review_needed":
            print(f"    → 진단: 점수({score:.4f})가 {THRESHOLD_REVIEW}~{THRESHOLD_MATCHED} 구간에 위치")
            print(f"           품목명 또는 거래처명 정규화 로직 개선 검토")
        elif exp == "unmatched" and act in ("matched", "review_needed"):
            print(f"    → 진단: 점수({score:.4f}) 과대평가 — 금액/품목 가중치 재검토 필요")
        elif exp in ("matched", "review_needed") and act == "unmatched":
            print(f"    → 진단: 점수({score:.4f}) 과소평가 — 텍스트 유사도 함수 점검 권고")


# ══════════════════════════════════════════════════════════════
# 메인 실행
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "★" * 60)
    print("  산업안전관리비 AI 검증 시스템 — 매칭 엔진 테스트")
    print(f"  3단계 임계값: matched≥{THRESHOLD_MATCHED} / review≥{THRESHOLD_REVIEW} / 미만=unmatched")
    print("★" * 60)

    all_failures = []

    p1, f1, fails1 = test_individual_receipts()
    all_failures.extend(fails1)

    p2, f2, fails2 = test_batch_matching()
    all_failures.extend(fails2)

    p3, f3, _ = test_threshold_boundaries()

    analyze_failures(all_failures)

    total_pass = p1 + p2 + p3
    total_fail = f1 + f2 + f3
    total      = total_pass + total_fail

    print("\n" + "★" * 60)
    print(f"  최종 결과: PASS {total_pass}/{total}  FAIL {total_fail}/{total}")
    if total_fail == 0:
        print("  ✅ 전체 테스트 통과!")
    else:
        print(f"  ⚠️  {total_fail}개 케이스 검토 필요")
    print("★" * 60 + "\n")
