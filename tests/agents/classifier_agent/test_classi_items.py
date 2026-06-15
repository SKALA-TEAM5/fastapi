"""
classi agent 항목별 분류 결과 확인 스크립트.

DB에서 usage_statement 데이터를 가져오거나, 항목명을 직접 입력해서
classi 결과(카테고리, 유지/변경, 사유, 토큰)를 확인한다.

사용법:
  # DB 기반 — usage_statement 전체 항목
  cd fastapi
  python tests/agents/classifier_agent/test_classi_items.py --id 2

  # 항목명 직접 입력 (category_code 지정 가능)
  python tests/agents/classifier_agent/test_classi_items.py --items "안전벨트" "안전화" "안전모"
  python tests/agents/classifier_agent/test_classi_items.py --items "안전벨트" --category CAT_03

  # DB + 항목명 추가
  python tests/agents/classifier_agent/test_classi_items.py --id 2 --items "안전벨트" "안전화"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from time import perf_counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.agents.classifier_agent.agent import review_usage_statement
from src.core import llm_config

load_dotenv()

W = 70


def _get_conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "safety"),
        user=os.getenv("SERVICE_APP_USER", "safety_service_app"),
        password=os.getenv("SERVICE_APP_PASSWORD", "safety_service_app_password"),
        options="-c search_path=service",
    )


def fetch_rows(usage_statement_id: int) -> tuple[list[dict], dict]:
    """DB에서 항목 목록과 basic_info를 조회한다."""
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT us.id, us.project_id, us.report_month, us.revision_no,
                       us.document_written_date, us.cumulative_progress_rate,
                       p.appropriated_amount
                FROM usage_statements us
                JOIN projects p ON p.id = us.project_id
                WHERE us.id = %s
                """,
                (usage_statement_id,),
            )
            header = cur.fetchone()
            if not header:
                raise ValueError(f"usage_statement id={usage_statement_id} 없음")

            cur.execute(
                """
                SELECT id, category_code, item_name, total_amount
                FROM usage_statement_items
                WHERE usage_statement_id = %s
                ORDER BY id
                """,
                (usage_statement_id,),
            )
            items = [dict(r) for r in cur.fetchall()]

    rows = [
        {
            "row_id": item["id"],
            "given_category_code": item.get("category_code") or "",
            "item_name": item.get("item_name") or "",
            "total_amount": float(item["total_amount"]) if item.get("total_amount") else 0,
        }
        for item in items
    ]
    basic_info = {
        k: v for k, v in {
            "report_month": str(header["report_month"]) if header.get("report_month") else None,
            "cumulative_progress_rate": float(header["cumulative_progress_rate"]) if header.get("cumulative_progress_rate") is not None else None,
            "safety_budget_total": int(header["appropriated_amount"]) if header.get("appropriated_amount") else None,
        }.items() if v is not None
    }
    return rows, basic_info


def _icon(decision_status: str) -> str:
    if decision_status == "유지":
        return "✓"
    if decision_status == "카테고리변경":
        return "↕"
    return "?"


def run(
    *,
    usage_statement_id: int | None,
    extra_items: list[str],
    default_category: str,
    model: str,
    verbose: bool,
) -> None:
    print(f"\n{'=' * W}")
    print(f"  classi 항목 분류 확인  |  model: {model}")
    print(f"{'=' * W}\n")

    rows: list[dict] = []
    basic_info: dict = {}

    # DB 조회
    if usage_statement_id is not None:
        db_rows, basic_info = fetch_rows(usage_statement_id)
        rows.extend(db_rows)
        print(f"[DB] usage_statement_id={usage_statement_id}  항목 {len(db_rows)}건 조회")

    # 직접 입력 항목 추가
    extra_start_id = 90000
    for i, item_name in enumerate(extra_items):
        rows.append({
            "row_id": extra_start_id + i,
            "given_category_code": default_category,
            "item_name": item_name,
            "total_amount": 0,
        })
    if extra_items:
        print(f"[직접 입력] {len(extra_items)}건: {', '.join(extra_items)}")

    if not rows:
        print("❌ 항목 없음 — --id 또는 --items 지정 필요")
        sys.exit(1)

    print(f"\n항목 수: {len(rows)}\n")

    # classi 실행
    t0 = perf_counter()
    response = review_usage_statement(
        usage_statement_id=usage_statement_id or 0,
        rows=rows,
        basic_info=basic_info,
    )
    elapsed = perf_counter() - t0

    results = response.results

    # 토큰 집계 (review_usage_statement가 이제 내부에서 추적)
    # 단, 이 테스트에서는 project_id 없이 호출하므로 agent_logs INSERT는 안 됨.
    # 토큰 확인은 langchain callback으로 별도 측정.
    try:
        from langchain_community.callbacks import get_openai_callback as _goc
        # 이미 위에서 실행됐으므로 여기선 추정 불가 — review_usage_statement 내부에서 측정됨
        token_note = "(토큰은 review_usage_statement 내부에서 집계됨)"
    except ImportError:
        token_note = ""

    # 결과 출력
    changed = [r for r in results if r.decision_status == "카테고리변경"]
    kept = [r for r in results if r.decision_status == "유지"]

    print(f"[결과] {len(results)}건  |  유지 {len(kept)}건  |  카테고리변경 {len(changed)}건  |  {elapsed:.1f}s\n")

    for result in results:
        icon = _icon(result.decision_status)
        reason = result.reason or ""
        # 실제 UI 차단 여부 감지
        is_blocked = "llm(unclassified)" in reason
        is_llm_classified = "llm(classified)" in reason

        cat_before = result.given_category_code or "-"
        cat_after = result.final_category_code or "-"
        if cat_before == cat_after:
            cat_str = cat_before
        else:
            cat_str = f"{cat_before} → {cat_after}"

        block_tag = "  ⛔ UI차단" if is_blocked else ("  🔵 LLM분류" if is_llm_classified else "")
        print(f"  {icon} [{cat_str}] {result.item_name}{block_tag}")
        if reason:
            if len(reason) > 120:
                print(f"    사유: {reason[:120]}")
                print(f"         {reason[120:240]}")
                if len(reason) > 240:
                    print(f"         ... (총 {len(reason)}자)")
            else:
                print(f"    사유: {reason}")

    if changed:
        print(f"\n[카테고리변경 상세]")
        for result in changed:
            print(f"  {result.item_name}")
            print(f"    {result.given_category_code} → {result.final_category_code}")
            print(f"    {result.reason}")

    if verbose:
        print(f"\n[전체 JSON]")
        print(json.dumps(
            [
                {
                    "row_id": r.row_id,
                    "item_name": r.item_name,
                    "given": r.given_category_code,
                    "final": r.final_category_code,
                    "status": r.decision_status,
                    "reason": r.reason,
                }
                for r in results
            ],
            ensure_ascii=False,
            indent=2,
        ))


def main() -> None:
    parser = argparse.ArgumentParser(description="classi 항목 분류 결과 확인")
    parser.add_argument("--id", type=int, default=None, help="usage_statement_id (DB 조회)")
    parser.add_argument(
        "--items", nargs="+", default=[],
        help="직접 입력할 항목명 목록 (예: '안전벨트' '안전화')",
    )
    parser.add_argument(
        "--category", default="CAT_03",
        help="직접 입력 항목의 기본 카테고리코드 (기본값: CAT_03)",
    )
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.id is None and not args.items:
        parser.error("--id 또는 --items 중 하나는 필수")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY 없음 — .env 파일 확인")
        sys.exit(1)
    llm_config.configure(ChatOpenAI(model=args.model, temperature=0))

    run(
        usage_statement_id=args.id,
        extra_items=args.items,
        default_category=args.category,
        model=args.model,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
