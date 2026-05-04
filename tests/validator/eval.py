"""
산안비 집행 적정성 평가 스크립트.

테스트 케이스는 tests/validator/cases/ 에서 관리:
  inputs.json   - Validator API 요청과 동일한 입력 형태
  expected.json - 카테고리별 기대 판정/조항/참고사유

Metrics: DecisionAccuracy, SemScore, HitRate@K, ConstraintMatch

uv run python -m tests.validator.eval
uv run python -m tests.validator.eval --model gpt-4o
uv run python -m tests.validator.eval --verbose
uv run python -m tests.validator.eval --cases tests/validator/cases

결과:
  tests/validator/output/results_YYYYMMDD_HHMM.json
  tests/validator/output/metrics_YYYYMMDD_HHMM.json
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer

from src.agents.validator_agent.agent import (
    run_audit,
    to_validator_response,
    validate_document,
    validate_usage_statement,
)
from src.core import llm_config
from src.schemas.validator import (
    AuditResponse,
    CategoryAuditResult,
    CategoryAuditSummary,
    UsageStatementValidatorEmbeddedResponse,
    UsageStatementValidatorRequest,
    ValidatorCategoryDataWithResult,
)

load_dotenv()

_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_SEM_THRESHOLD_GOOD = 0.75
_SEM_THRESHOLD_OK = 0.55
_W = 92


# ── 케이스 로드 ────────────────────────────────────────────────────────────────


def load_cases(cases_dir: str) -> list[dict]:
    """inputs.json + expected.json을 id 기준으로 병합하여 반환."""
    base = Path(cases_dir)
    with open(base / "inputs.json", encoding="utf-8") as f:
        inputs = {c["id"]: c for c in json.load(f)}
    with open(base / "expected.json", encoding="utf-8") as f:
        expected = {c["id"]: c for c in json.load(f)}

    merged = []
    for cid, inp in inputs.items():
        if cid not in expected:
            raise ValueError(f"expected.json에 '{cid}'가 없습니다.")
        merged.append({"input": inp, "expected": expected[cid]})
    return merged


def _to_api_input(payload: dict) -> dict:
    return {k: v for k, v in payload.items() if k != "id"}


def _to_embedded_output(inp: dict, response: AuditResponse) -> dict:
    request = UsageStatementValidatorRequest.model_validate(_to_api_input(inp))
    summary = to_validator_response(
        response=response,
        usage_statement_id=request.usage_statement_id,
    ).result
    result_by_code: dict[str, CategoryAuditSummary] = {
        result.category_code: result for result in summary.results
    }
    categories: list[ValidatorCategoryDataWithResult] = []
    for category in request.categories:
        result = result_by_code.get(category.category_code)
        if result is None:
            continue
        categories.append(
            ValidatorCategoryDataWithResult(
                category_code=category.category_code,
                summary=category.summary,
                items=category.items,
                result=result,
            )
        )
    return UsageStatementValidatorEmbeddedResponse(
        usage_statement_id=request.usage_statement_id,
        basic_info=request.basic_info,
        categories=categories,
    ).model_dump(by_alias=True)


# ── 메트릭 계산 ────────────────────────────────────────────────────────────────


def _sem_score(model: SentenceTransformer, text_a: str, text_b: str) -> float:
    emb = model.encode([text_a, text_b], normalize_embeddings=True)
    return float(np.dot(emb[0], emb[1]))


def _hit_rate(result: CategoryAuditResult, expected_laws: list[str]) -> bool:
    """referenced_laws에 기대 법령 키워드가 하나라도 있으면 True."""
    all_laws = " ".join(
        list(result.referenced_laws)
        + [law for j in result.items for law in j.referenced_laws]
    )
    return any(kw in all_laws for kw in expected_laws)


def _constraint_match(result: CategoryAuditResult) -> bool:
    """reasoning 중 하나라도 '단,' 또는 '다만,'을 포함하면 True."""
    return any("단," in j.reasoning or "다만," in j.reasoning for j in result.items)


# ── 평가 실행 ─────────────────────────────────────────────────────────────────


def run_evaluation(
    model_name: str = "gpt-4o-mini",
    cases_dir: str = "tests/validator/cases",
    output_dir: str = "tests/validator/output",
    verbose: bool = False,
) -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY가 없어 hard-rule 중심 모드로 실행합니다.")
    else:
        llm_config.configure(ChatOpenAI(model=model_name, temperature=0))

    print(f"\n임베딩 모델 로드 중: {_EMBED_MODEL}")
    embed_model = SentenceTransformer(_EMBED_MODEL)

    cases = load_cases(cases_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    print(f"\n{'=' * _W}")
    print(f"  산안비 집행 적정성 평가  |  총 {len(cases)}케이스  |  LLM: {model_name}")
    print(f"{'=' * _W}\n")

    case_records: list[dict] = []

    for i, case in enumerate(cases, 1):
        inp = case["input"]
        exp = case["expected"]
        label = f"[{i:02d}/{len(cases)}]"
        print(f"{label} {exp['description']}  ({exp['case_type']})")

        try:
            if "categories" in inp:
                response: AuditResponse = run_audit(
                    base_amount=inp["base_amount"],
                    categories=inp["categories"],
                    basic_info_by_category=inp.get("basic_info_by_category"),
                )
            elif "카테고리별데이터" in inp:
                response = validate_usage_statement(document=inp)
            else:
                category_result = validate_document(
                    category=inp["category"],
                    items=inp.get("items"),
                    basic_info=inp.get("basic_info"),
                    base_amount=inp.get("base_amount"),
                    document=inp.get("document"),
                )
                response = AuditResponse(
                    base_amount=float(
                        inp.get("base_amount")
                        or inp.get("basic_info", {}).get("base_amount")
                        or 0
                    ),
                    categories={inp["category"]: category_result},
                )
        except Exception as e:
            print(f"  ERROR: {e}\n")
            case_records.append(
                {
                    "id": inp["id"],
                    "description": exp["description"],
                    "case_type": exp["case_type"],
                    "error": str(e),
                }
            )
            continue

        cat_records: list[dict] = []
        expected_blocks = exp.get("카테고리별기대결과", [])
        for expected_block in expected_blocks:
            cat = expected_block["카테고리명"]
            expected_status = expected_block["판정상태"]
            cat_result = response.categories.get(cat)
            if cat_result is None:
                print(f"  [{cat}] 결과 없음\n")
                continue

            decision_correct = cat_result.status == expected_status

            combined_reasoning = " ".join(
                [cat_result.rejection_reason] + [j.reasoning for j in cat_result.items]
            ).strip()
            ref_text = expected_block.get("참고사유", "")
            sem = (
                _sem_score(embed_model, combined_reasoning, ref_text)
                if ref_text
                else 0.0
            )

            hit = _hit_rate(cat_result, expected_block.get("기대조항", []))

            expect_cm = expected_block.get("제약표현기대여부", False)
            cm = _constraint_match(cat_result) if expect_cm else None

            d_icon = "O" if decision_correct else "X"
            s_icon = (
                "O"
                if sem >= _SEM_THRESHOLD_GOOD
                else ("~" if sem >= _SEM_THRESHOLD_OK else "X")
            )
            h_icon = "O" if hit else "X"
            cm_str = f" | CM: {'O' if cm else 'X'}" if cm is not None else ""

            print(
                f"  [{cat}] "
                f"Decision:{d_icon}({expected_status}/{cat_result.status}) | "
                f"Sem:{sem:.3f}{s_icon} | Hit@5:{h_icon}{cm_str}"
            )

            if verbose:
                print(f"    limit: {cat_result.limit_rule or '(없음)'}")
                print(f"    rejection: {cat_result.rejection_reason or '(없음)'}")
                for j in cat_result.items:
                    icon = "O" if j.allowed else "X"
                    print(f"    [{icon}] {j.item:20s}  {j.reasoning[:80]}...")

            cat_records.append(
                {
                    "category": cat,
                    "expected_status": expected_status,
                    "actual_status": cat_result.status,
                    "decision_correct": decision_correct,
                    "sem_score": round(sem, 4),
                    "hit_rate": hit,
                    "constraint_match": cm,
                    "total": cat_result.total,
                    "limit": cat_result.limit,
                    "exceeded": cat_result.exceeded,
                    "limit_rule": cat_result.limit_rule,
                    "rejection_reason": cat_result.rejection_reason,
                    "progress_rate": cat_result.progress_rate,
                    "required_usage_rate": cat_result.required_usage_rate,
                    "required_used_amount": cat_result.required_used_amount,
                    "cumulative_used_amount": cat_result.cumulative_used_amount,
                    "usage_shortfall_amount": cat_result.usage_shortfall_amount,
                    "items": [
                        {
                            "item": j.item,
                            "amount": j.amount,
                            "category": j.category,
                            "allowed": j.allowed,
                            "confidence": j.confidence,
                            "reasoning": j.reasoning,
                            "referenced_laws": j.referenced_laws,
                        }
                        for j in cat_result.items
                    ],
                }
            )

        embedded_output = _to_embedded_output(inp, response)
        if verbose:
            print("  API 출력:")
            print(json.dumps(embedded_output, ensure_ascii=False, indent=2))
            print()

        print()
        case_records.append(
            {
                "id": inp["id"],
                "description": exp["description"],
                "case_type": exp["case_type"],
                "base_amount": (
                    inp.get("base_amount")
                    or inp.get("기본정보", {}).get("산안비총액")
                    or 0
                ),
                "api_input": _to_api_input(inp),
                "api_output": embedded_output,
                "categories": cat_records,
            }
        )

    # ── 메트릭 집계 + 출력 ─────────────────────────────────────────────────────
    all_cats = [
        c for r in case_records for c in r.get("categories", []) if "error" not in r
    ]
    metrics = _compute_metrics(all_cats)
    _print_metrics(metrics, case_records)

    # ── 저장 ────────────────────────────────────────────────────────────────────
    results_path = out_dir / f"results_{ts}.json"
    metrics_path = out_dir / f"metrics_{ts}.json"

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {
                    "model": model_name,
                    "total_cases": len(cases),
                    "evaluated_at": datetime.now().isoformat(),
                },
                "cases": case_records,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n결과 저장:")
    print(f"  results → {results_path}")
    print(f"  metrics → {metrics_path}")


# ── 메트릭 계산 / 리포트 ──────────────────────────────────────────────────────


def _compute_metrics(cat_records: list[dict]) -> dict:
    if not cat_records:
        return {}

    decision_acc = float(np.mean([r["decision_correct"] for r in cat_records]))
    sem_avg = float(np.mean([r["sem_score"] for r in cat_records]))
    sem_good = sum(1 for r in cat_records if r["sem_score"] >= _SEM_THRESHOLD_GOOD)
    hit_avg = float(np.mean([r["hit_rate"] for r in cat_records]))

    cm_rows = [r for r in cat_records if r["constraint_match"] is not None]
    cm_avg = (
        float(np.mean([r["constraint_match"] for r in cm_rows])) if cm_rows else None
    )

    return {
        "total_category_evals": len(cat_records),
        "decision_accuracy": round(decision_acc, 4),
        "sem_score_avg": round(sem_avg, 4),
        "sem_score_good_pct": round(sem_good / len(cat_records), 4),
        "hit_rate_at_5": round(hit_avg, 4),
        "constraint_match_avg": round(cm_avg, 4) if cm_avg is not None else None,
        "sem_threshold_good": _SEM_THRESHOLD_GOOD,
        "sem_threshold_ok": _SEM_THRESHOLD_OK,
    }


def _print_metrics(metrics: dict, case_records: list[dict]) -> None:
    all_cats = [
        c for r in case_records for c in r.get("categories", []) if "error" not in r
    ]
    if not all_cats:
        return

    print(f"\n{'=' * _W}")
    print("  평가 메트릭 요약")
    print(f"{'=' * _W}")
    print(f"  총 카테고리 평가  : {metrics['total_category_evals']}건")
    print(
        f"  Decision Accuracy : {metrics['decision_accuracy']:.1%}  "
        f"({sum(c['decision_correct'] for c in all_cats)}/{len(all_cats)})"
    )
    print(
        f"  SemScore (avg)    : {metrics['sem_score_avg']:.3f}  "
        f"(Good>={_SEM_THRESHOLD_GOOD}: {metrics['sem_score_good_pct']:.1%})"
    )
    print(f"  HitRate@5         : {metrics['hit_rate_at_5']:.1%}")
    if metrics["constraint_match_avg"] is not None:
        print(f"  ConstraintMatch   : {metrics['constraint_match_avg']:.1%}")

    for ctype, label in (
        ("item_validity", "항목 적정성"),
        ("amount_error", "금액 오류  "),
    ):
        rows = [
            c
            for r in case_records
            for c in r.get("categories", [])
            if r.get("case_type") == ctype and "error" not in r
        ]
        if not rows:
            continue
        da = np.mean([r["decision_correct"] for r in rows])
        sa = np.mean([r["sem_score"] for r in rows])
        ha = np.mean([r["hit_rate"] for r in rows])
        print(
            f"\n  [{label}] n={len(rows)}  Decision={da:.1%}  Sem={sa:.3f}  Hit@5={ha:.1%}"
        )

    print(f"\n{'=' * _W}")
    print("  케이스별 상세")
    print(f"{'=' * _W}")
    hdr = f"  {'id':<10} {'설명':<42} {'유형':<14} {'DA':>4}  {'Sem':>6}  {'Hit':>4}"
    print(hdr)
    print("  " + "─" * (_W - 2))

    for r in case_records:
        if "error" in r:
            print(f"  {r['id']:<10} {r['description']:<42}  ERROR")
            continue
        cats = r.get("categories", [])
        if not cats:
            continue
        da = np.mean([c["decision_correct"] for c in cats])
        sa = np.mean([c["sem_score"] for c in cats])
        ha = np.mean([c["hit_rate"] for c in cats])
        print(
            f"  {r['id']:<10} {r['description']:<42} {r['case_type']:<14} "
            f"{'O' if da == 1.0 else 'X'} {da:.0%}  {sa:.3f}  {'O' if ha == 1.0 else 'X'}"
        )
    print("  " + "─" * (_W - 2))


# ── 진입점 ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="산안비 집행 적정성 평가")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument(
        "--cases", default="tests/validator/cases", help="케이스 디렉토리"
    )
    parser.add_argument(
        "--output", default="tests/validator/output", help="결과 저장 디렉토리"
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    run_evaluation(
        model_name=args.model,
        cases_dir=args.cases,
        output_dir=args.output,
        verbose=args.verbose,
    )
