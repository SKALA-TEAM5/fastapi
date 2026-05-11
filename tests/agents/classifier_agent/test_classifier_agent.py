"""
산안비 2번 Agent(카테고리 검토) 평가 스크립트.

입력 스키마:
  {
    "사용내역서ID": ...,
    "항목목록": [
      {"행ID": 1, "기존카테고리코드": "CAT_02", "항목명": "...", "금액": 1000}
    ]
  }

출력 스키마:
  {
    "사용내역서ID": ...,
    "검토결과": [
      {
        "행ID": 1,
        "항목명": "...",
        "기존카테고리코드": "CAT_02",
        "최종카테고리코드": "CAT_04",
        "판정상태": "카테고리변경",
        "검토필요여부": false,
        "사유": "..."
      }
    ]
  }
"""
import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from src.agents.classifier_agent.agent import review_usage_statement

_W = 92


def load_cases(cases_dir: str, filter_id: str | None = None) -> list[dict]:
    base = Path(cases_dir)
    with open(base / "inputs.json", encoding="utf-8") as f:
        inputs = {c["id"]: c for c in json.load(f)}
    with open(base / "expected.json", encoding="utf-8") as f:
        expected = {c["id"]: c for c in json.load(f)}

    merged = []
    for cid, inp in inputs.items():
        if filter_id and cid != filter_id:
            continue
        merged.append({"input": inp, "expected": expected[cid]})
    return merged


def run_evaluation(
    cases_dir: str = "tests/agents/classifier_agent/cases",
    output_dir: str = "tests/agents/classifier_agent/output",
    verbose: bool = False,
    filter_id: str | None = None,
) -> None:
    cases = load_cases(cases_dir, filter_id=filter_id)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    print(f"\n{'=' * _W}")
    print(f"  산안비 카테고리 검증 평가 (2번 Agent)  |  총 {len(cases)}케이스")
    print(f"{'=' * _W}\n")

    case_records: list[dict] = []
    correct_rows = 0
    total_rows = 0
    review_count = 0
    correct_cases = 0
    expected_keep_total = 0
    expected_keep_correct = 0
    expected_reclass_total = 0
    expected_reclass_correct = 0
    overchange_count = 0
    confusion_matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for i, case in enumerate(cases, 1):
        inp = case["input"]
        exp = case["expected"]
        print(f"[{i:02d}/{len(cases)}] {inp['description']}")

        try:
            response = review_usage_statement(payload=inp["payload"])
        except Exception as e:
            print(f"  ERROR: {e}\n")
            case_records.append({"id": inp["id"], "error": str(e)})
            continue

        exp_index = {
            item["행ID"]: item
            for item in exp["검토결과"]
        }

        result_records = []
        for row in response.results:
            expected_row = exp_index.get(row.row_id, {})
            expected_status = expected_row.get("기대판정상태")
            expected_category = expected_row.get("기대최종카테고리코드")
            status_match = row.decision_status == expected_row.get("기대판정상태")
            category_match = row.final_category_code == expected_category
            is_correct = status_match and category_match
            total_rows += 1
            if is_correct:
                correct_rows += 1
            if row.needs_human_review:
                review_count += 1

            if expected_status == "유지":
                expected_keep_total += 1
                if is_correct:
                    expected_keep_correct += 1
                if row.decision_status == "카테고리변경":
                    overchange_count += 1
            elif expected_status == "카테고리변경":
                expected_reclass_total += 1
                if is_correct:
                    expected_reclass_correct += 1

            if expected_category and row.final_category_code:
                confusion_matrix[expected_category][row.final_category_code] += 1

            mark = "O" if is_correct else "X"
            review_mark = "H" if row.needs_human_review else " "
            print(
                f"    [{mark}][{review_mark}] 행 {row.row_id:<2} {row.item_name[:26]:<26} "
                f"{row.decision_status:<8} 최종:{row.final_category_code}"
            )
            if verbose and row.reason:
                print(f"           사유: {row.reason}")

            result_records.append(
                {
                    "행ID": row.row_id,
                    "항목명": row.item_name,
                    "기존카테고리코드": row.given_category_code,
                    "최종카테고리코드": row.final_category_code,
                    "판정상태": row.decision_status,
                    "검토필요여부": row.needs_human_review,
                    "사유": row.reason,
                    "is_correct": is_correct,
                    "기대판정상태": expected_row.get("기대판정상태"),
                    "기대최종카테고리코드": expected_row.get("기대최종카테고리코드"),
                }
            )

        print()
        case_ok = all(r["is_correct"] for r in result_records)
        if result_records and case_ok:
            correct_cases += 1
        case_records.append(
            {
                "id": inp["id"],
                "description": inp["description"],
                "사용내역서ID": response.usage_statement_id,
                "검토결과": result_records,
            }
        )

    accuracy = correct_rows / total_rows if total_rows else 0.0
    case_accuracy = correct_cases / len(cases) if cases else 0.0
    keep_accuracy = expected_keep_correct / expected_keep_total if expected_keep_total else 0.0
    reclass_accuracy = expected_reclass_correct / expected_reclass_total if expected_reclass_total else 0.0
    overchange_rate = overchange_count / expected_keep_total if expected_keep_total else 0.0

    print(f"\n{'=' * _W}")
    print("  검증 메트릭 요약")
    print(f"{'=' * _W}")
    print(f"  Row Accuracy      : {accuracy:.1%}  ({correct_rows}/{total_rows})")
    print(f"  Case Accuracy     : {case_accuracy:.1%}  ({correct_cases}/{len(cases)})")
    print(f"  유지 정확도         : {keep_accuracy:.1%}  ({expected_keep_correct}/{expected_keep_total})")
    print(f"  변경 정확도         : {reclass_accuracy:.1%}  ({expected_reclass_correct}/{expected_reclass_total})")
    print(f"  과잉변경률          : {overchange_rate:.1%}  ({overchange_count}/{expected_keep_total})")
    print(f"  Human Review Rate : {review_count/total_rows:.1%}  ({review_count}/{total_rows})")
    print("\n  카테고리 혼동행렬 (실제 -> 예측):")
    category_order = sorted(confusion_matrix.keys())
    if category_order:
        header = "실제\\예측".ljust(12) + "".join(cat.rjust(10) for cat in category_order)
        print(f"  {header}")
        print("  " + "─" * max(len(header), _W - 2))
        for actual in category_order:
            row_text = actual.ljust(12)
            for predicted in category_order:
                row_text += str(confusion_matrix[actual].get(predicted, 0)).rjust(10)
            print(f"  {row_text}")
    print("\n  케이스별 요약:")
    print(f"  {'id':<10} {'설명':<40} {'결과':>4}  {'검토':>4}")
    print("  " + "─" * (_W - 2))
    for rec in case_records:
        if "error" in rec:
            print(f"  {rec['id']:<10} ERROR")
            continue
        case_ok = all(r["is_correct"] for r in rec["검토결과"])
        has_review = any(r["검토필요여부"] for r in rec["검토결과"])
        print(f"  {rec['id']:<10} {rec['description']:<40} {'[O]' if case_ok else '[X]':>4}  {'H' if has_review else '-':>4}")
    print("  " + "─" * (_W - 2))
    print(f"{'=' * _W}")

    results_path = out_dir / f"results_{ts}.json"
    metrics_path = out_dir / f"metrics_{ts}.json"
    metrics_payload = {
        "total_cases": len(cases),
        "correct_cases": correct_cases,
        "case_accuracy": round(case_accuracy, 4),
        "total_rows": total_rows,
        "correct_rows": correct_rows,
        "row_accuracy": round(accuracy, 4),
        "expected_keep_total": expected_keep_total,
        "expected_keep_correct": expected_keep_correct,
        "keep_accuracy": round(keep_accuracy, 4),
        "expected_reclass_total": expected_reclass_total,
        "expected_reclass_correct": expected_reclass_correct,
        "reclassification_accuracy": round(reclass_accuracy, 4),
        "overchange_count": overchange_count,
        "overchange_rate": round(overchange_rate, 4),
        "review_count": review_count,
        "review_rate": round(review_count / total_rows if total_rows else 0.0, 4),
        "confusion_matrix": {
            actual: {predicted: count for predicted, count in sorted(predicted_counts.items())}
            for actual, predicted_counts in sorted(confusion_matrix.items())
        },
        "evaluated_at": datetime.now().isoformat(),
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {
                    "total_cases": len(cases),
                    "total_rows": total_rows,
                    "case_accuracy": round(case_accuracy, 4),
                    "row_accuracy": round(accuracy, 4),
                    "evaluated_at": datetime.now().isoformat(),
                },
                "cases": case_records,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {results_path}")
    print(f"메트릭 저장: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="산안비 카테고리 검증 평가 (2번 Agent)")
    parser.add_argument("--cases", default="tests/agents/classifier_agent/cases")
    parser.add_argument("--output", default="tests/agents/classifier_agent/output")
    parser.add_argument("--id", default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    run_evaluation(
        cases_dir=args.cases,
        output_dir=args.output,
        verbose=args.verbose,
        filter_id=args.id,
    )
