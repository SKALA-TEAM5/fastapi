"""
산안비 agent_logs 생성 확인 스크립트.

테스트 케이스는 tests/agents/validator_agent/cases/ 에서 관리:
  inputs.json   - 기존 케이스 입력. 실행 시 migration/V1 row shape로 변환한다.
  expected.json - 케이스 설명과 기대 판정 참고값

uv run python -m tests.agents.validator_agent.test_validator_agent
uv run python -m tests.agents.validator_agent.test_validator_agent --model gpt-4o
uv run python -m tests.agents.validator_agent.test_validator_agent --verbose
uv run python -m tests.agents.validator_agent.test_validator_agent --cases tests/agents/validator_agent/cases

결과:
  tests/agents/validator_agent/output/results_YYYYMMDD_HHMM.json
"""

import argparse
import json
import os
from datetime import date, datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.agents.validator_agent.agent import validate_usage_statement
from src.core import llm_config
from src.schemas.validator import (
    AuditResponse,
    UsageStatementValidatorRequest,
)
from src.services import validator_service as validator_log

load_dotenv()

_W = 92
_DEFAULT_PROJECT_ID = 1
_DEFAULT_SOURCE_FILE_ID_BASE = 9000


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


def _first_day(value: str | None) -> str:
    if not value:
        return date.today().replace(day=1).isoformat()
    try:
        parsed = datetime.fromisoformat(value).date()
    except ValueError:
        return date.today().replace(day=1).isoformat()
    return parsed.replace(day=1).isoformat()


def _to_v1_input(payload: dict) -> dict:
    """
    Validator 케이스 입력을 migration/V1 DB row shape로 정규화한다.

    V1에서 validator가 실제로 읽는 핵심 테이블은 usage_statements,
    usage_statement_summaries, usage_statement_items이며, 산안비 총액은
    validator_context로 함께 둔다. 이미 v1_input 형태면 그대로 사용한다.
    """
    if "v1_input" in payload:
        return payload["v1_input"]

    api_input = _to_api_input(payload)
    request = UsageStatementValidatorRequest.model_validate(api_input)
    usage_statement_id = int(request.usage_statement_id)
    project_id = int(
        payload.get("project_id") or api_input.get("project_id") or _DEFAULT_PROJECT_ID
    )

    first_used_on = None
    for category in request.categories:
        if category.items:
            first_used_on = category.items[0].used_on
            break

    usage_statement = {
        "id": usage_statement_id,
        "project_id": project_id,
        "source_file_id": int(
            payload.get("source_file_id")
            or (_DEFAULT_SOURCE_FILE_ID_BASE + usage_statement_id)
        ),
        "report_month": _first_day(first_used_on),
        "revision_no": 1,
        "document_written_date": first_used_on or date.today().isoformat(),
        "cumulative_progress_rate": request.basic_info.progress_rate,
    }

    summaries: list[dict] = []
    items: list[dict] = []
    for category in request.categories:
        summaries.append(
            {
                "usage_statement_id": usage_statement_id,
                "category_code": category.category_code,
                "previous_amount": category.summary.previous_amount,
                "current_amount": category.summary.current_amount,
                "cumulative_amount": category.summary.cumulative_amount,
            }
        )
        for idx, item in enumerate(category.items, 1):
            row_id = item.row_id or idx
            items.append(
                {
                    "id": usage_statement_id * 1000 + int(row_id),
                    "usage_statement_id": usage_statement_id,
                    "category_code": category.category_code,
                    "used_on": item.used_on,
                    "item_name": item.item_name,
                    "unit": item.unit,
                    "quantity": item.quantity,
                    "unit_price": item.unit_price,
                    "total_amount": item.total_amount,
                    "remark": item.remark,
                    "page_no": 1,
                    "source_row_id": row_id,
                }
            )

    return {
        "project": {
            "id": project_id,
        },
        "usage_statement": usage_statement,
        "usage_statement_summaries": summaries,
        "usage_statement_items": items,
        "validator_context": {
            "safety_budget_total": request.basic_info.base_amount,
            "cumulative_progress_rate": request.basic_info.progress_rate,
        },
    }


def _validator_input_from_v1(v1_input: dict) -> dict:
    """migration/V1 row shape를 validator API document로 재구성한다."""
    usage_statement = v1_input["usage_statement"]
    context = v1_input.get("validator_context") or {}
    summaries_by_code = {
        row["category_code"]: row
        for row in v1_input.get("usage_statement_summaries", [])
    }
    items_by_code: dict[str, list[dict]] = {}
    for item in v1_input.get("usage_statement_items", []):
        items_by_code.setdefault(item["category_code"], []).append(item)

    categories = []
    for category_code, summary in summaries_by_code.items():
        rows = sorted(items_by_code.get(category_code, []), key=lambda row: row["id"])
        categories.append(
            {
                "카테고리코드": category_code,
                "집계정보": {
                    "전회사용금액": summary.get("previous_amount") or 0,
                    "금회사용금액": summary.get("current_amount") or 0,
                    "누적사용금액": summary.get("cumulative_amount") or 0,
                },
                "항목목록": [
                    {
                        "행ID": row.get("source_row_id") or row["id"],
                        "사용일자": row.get("used_on"),
                        "항목명": row.get("item_name"),
                        "단위": row.get("unit"),
                        "수량": row.get("quantity"),
                        "단가": row.get("unit_price"),
                        "금액": row.get("total_amount"),
                        "비고": row.get("remark") or "",
                    }
                    for row in rows
                ],
            }
        )

    return {
        "사용내역서ID": usage_statement["id"],
        "기본정보": {
            "산안비총액": context.get("safety_budget_total"),
            "누계공정률": context.get("cumulative_progress_rate")
            if context.get("cumulative_progress_rate") is not None
            else usage_statement.get("cumulative_progress_rate"),
        },
        "카테고리별데이터": categories,
    }


def _agent_logs_from_response(
    *,
    v1_input: dict,
    response: AuditResponse,
    model_name: str,
) -> list[dict]:
    """
    DB 없이 _write_legal_agent_log()와 같은 agent_logs row payload를 만든다.
    details는 jsonb로 들어갈 dict 그대로 둔다.
    """
    project_id = int(v1_input["project"]["id"])
    usage_statement_id = int(v1_input["usage_statement"]["id"])
    item_id_map = {
        (row["category_code"], row["item_name"]): row["id"]
        for row in v1_input.get("usage_statement_items", [])
    }

    logs: list[dict] = []
    for _, result in response.categories.items():
        for item in result.items:
            item_db_id = validator_log._resolve_item_db_id(item_id_map, item)
            logs.append(
                {
                    "project_id": project_id,
                    "usage_statement_id": usage_statement_id,
                    "usage_statement_item_id": item_db_id,
                    "agent_type_code": "legal",
                    "status_code": "success",
                    "result_code": validator_log._item_result_code(item, result),
                    "reason": validator_log._item_reason(item, result),
                    "details": validator_log._item_details(item, result),
                    "model_name": model_name,
                }
            )

    return logs


def _result_code_counts(agent_logs: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in agent_logs:
        code = row.get("result_code") or "unknown"
        counts[code] = counts.get(code, 0) + 1
    return counts


# ── 평가 실행 ─────────────────────────────────────────────────────────────────


def run_evaluation(
    model_name: str = "gpt-4o-mini",
    cases_dir: str = "tests/agents/validator_agent/cases",
    output_dir: str = "tests/agents/validator_agent/output",
    verbose: bool = False,
) -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY가 없어 hard-rule 중심 모드로 실행합니다.")
    else:
        llm_config.configure(ChatOpenAI(model=model_name, temperature=0))

    cases = load_cases(cases_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    print(f"\n{'=' * _W}")
    print(f"  산안비 agent_logs 생성 확인  |  총 {len(cases)}케이스  |  LLM: {model_name}")
    print(f"{'=' * _W}\n")

    case_records: list[dict] = []

    for i, case in enumerate(cases, 1):
        raw_inp = case["input"]
        exp = case["expected"]
        v1_input = _to_v1_input(raw_inp)
        validator_input = _validator_input_from_v1(v1_input)
        label = f"[{i:02d}/{len(cases)}]"
        print(f"{label} {exp['description']}  ({exp['case_type']})")

        try:
            response: AuditResponse = validate_usage_statement(document=validator_input)
        except Exception as e:
            print(f"  ERROR: {e}\n")
            case_records.append(
                {
                    "id": raw_inp["id"],
                    "description": exp["description"],
                    "case_type": exp["case_type"],
                    "error": str(e),
                    "v1_input": v1_input,
                }
            )
            continue

        agent_logs = _agent_logs_from_response(
            v1_input=v1_input,
            response=response,
            model_name=model_name,
        )
        expected_log_count = len(v1_input.get("usage_statement_items", []))
        log_count_ok = len(agent_logs) == expected_log_count
        result_code_counts = _result_code_counts(agent_logs)
        count_icon = "O" if log_count_ok else "X"
        print(
            f"  agent_logs:{len(agent_logs)}/{expected_log_count}{count_icon} "
            f"| result_codes:{result_code_counts}"
        )

        if verbose:
            print("  agent_logs 출력:")
            print(json.dumps(agent_logs, ensure_ascii=False, indent=2))
            print()

        print()
        case_records.append(
            {
                "id": raw_inp["id"],
                "description": exp["description"],
                "case_type": exp["case_type"],
                "base_amount": (
                    v1_input.get("validator_context", {}).get("safety_budget_total")
                    or 0
                ),
                "v1_input": v1_input,
                "agent_logs": agent_logs,
                "checks": {
                    "expected_log_count": expected_log_count,
                    "actual_log_count": len(agent_logs),
                    "log_count_ok": log_count_ok,
                    "result_code_counts": result_code_counts,
                },
            }
        )

    # ── 저장 ────────────────────────────────────────────────────────────────────
    results_path = out_dir / f"results_{ts}.json"

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

    print("\n결과 저장:")
    print(f"  results → {results_path}")


# ── 진입점 ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="산안비 agent_logs 생성 확인")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument(
        "--cases", default="tests/agents/validator_agent/cases", help="케이스 디렉토리"
    )
    parser.add_argument(
        "--output", default="tests/agents/validator_agent/output", help="결과 저장 디렉토리"
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    run_evaluation(
        model_name=args.model,
        cases_dir=args.cases,
        output_dir=args.output,
        verbose=args.verbose,
    )
