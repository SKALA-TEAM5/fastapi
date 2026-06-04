# --------------------------------------------------------------------------
# 작성자   : 송상민(ss19801)
# 작성일   : 2026-06-04
#
# [ 주요 함수 정의 ]
#
# 1. run_evaluation() : chatbot_agent 전체 케이스 평가 실행
# 2. run_single()     : 단일 케이스 실행 및 결과 출력
#
# [ 실행 방법 ]
#   cd fastapi
#   # 전체 케이스 실행
#   python -m tests.agents.chatbot_agent.test_chatbot_agent
#
#   # 특정 케이스만 실행
#   python -m tests.agents.chatbot_agent.test_chatbot_agent --id chat_01
#
#   # 키워드 검증 없이 답변만 확인
#   python -m tests.agents.chatbot_agent.test_chatbot_agent --no-check
#
# [ 출력 형식 ]
#   - 각 케이스별 intent 분류, 전체 답변(스트리밍 조합), 출처, 키워드 검증 결과
#   - 최종 요약: 케이스 수, intent 정확도, 키워드 통과율
# --------------------------------------------------------------------------
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from src.services.chatbot_service import stream_chat

_W = 90  # 출력 구분선 너비
_CASES_PATH = Path("tests/agents/chatbot_agent/cases/inputs.json")
_OUTPUT_DIR = Path("tests/agents/chatbot_agent/output")


# ══════════════════════════════════════════════════════════════════════════════
# 단일 케이스 실행
# ══════════════════════════════════════════════════════════════════════════════

async def run_single(case: dict, check: bool = True) -> dict:
    """단일 케이스를 실행하고 결과를 반환한다.

    SSE 스트림을 모두 소비하여 전체 답변을 조합한 뒤 출력한다.

    Args:
        case  : inputs.json의 케이스 단위 dict
        check : True이면 expected_keywords 포함 여부 검증

    Returns:
        {
            "id", "description", "question",
            "intent", "answer", "sources",
            "intent_ok", "keywords_ok", "missing_keywords"
        }
    """
    question         = case["question"]
    expected_intent  = case.get("expected_intent", "")
    expected_kws     = case.get("expected_keywords", [])

    intent  = ""
    tokens  = []
    sources = []

    # ── 스트리밍 소비 ─────────────────────────────────────────────────────────
    async for raw in stream_chat(question=question, session_id=None):
        raw = raw.strip()
        if not raw or raw == "data: [DONE]":
            continue
        if not raw.startswith("data: "):
            continue
        try:
            payload = json.loads(raw[6:])  # "data: " 이후
        except json.JSONDecodeError:
            continue

        t = payload.get("type", "")
        v = payload.get("value", "")

        if t == "intent":
            intent = v
        elif t == "token":
            tokens.append(v)
        elif t == "sources":
            sources = v if isinstance(v, list) else [v]

    answer = "".join(tokens)

    # ── 검증 ──────────────────────────────────────────────────────────────────
    intent_ok        = (intent == expected_intent)
    missing_keywords = []
    if check and expected_kws:
        missing_keywords = [kw for kw in expected_kws if kw not in answer]
    keywords_ok = len(missing_keywords) == 0

    return {
        "id":               case["id"],
        "description":      case.get("description", ""),
        "question":         question,
        "expected_intent":  expected_intent,
        "intent":           intent,
        "answer":           answer,
        "sources":          sources,
        "intent_ok":        intent_ok,
        "keywords_ok":      keywords_ok,
        "missing_keywords": missing_keywords,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 결과 출력
# ══════════════════════════════════════════════════════════════════════════════

def print_result(i: int, total: int, result: dict, check: bool) -> None:
    intent_mark  = "O" if result["intent_ok"]   else "X"
    keyword_mark = "O" if result["keywords_ok"] else "X"

    print(f"\n{'─' * _W}")
    print(
        f"  [{i:02d}/{total}] {result['description']}"
        f"  │  intent [{intent_mark}]"
        + (f"  keyword [{keyword_mark}]" if check else "")
    )
    print(f"{'─' * _W}")
    print(f"  Q : {result['question']}")
    print(
        f"  intent  : {result['intent']}"
        + (f"  (기대: {result['expected_intent']})" if result['expected_intent'] else "")
    )
    print()
    # 답변 — 80자 단위 줄바꿈
    answer_lines = []
    buf = ""
    for ch in result["answer"]:
        buf += ch
        if ch == "\n" or len(buf) >= 80:
            answer_lines.append(buf)
            buf = ""
    if buf:
        answer_lines.append(buf)

    print("  ┌─ 답변 " + "─" * (_W - 8))
    for line in answer_lines:
        print(f"  │ {line}", end="")
    print()
    print("  └" + "─" * (_W - 2))

    if result["sources"]:
        print(f"  출처 : {' | '.join(result['sources'])}")
    if check and result["missing_keywords"]:
        print(f"  ⚠ 누락 키워드: {result['missing_keywords']}")


# ══════════════════════════════════════════════════════════════════════════════
# 전체 평가 실행
# ══════════════════════════════════════════════════════════════════════════════

async def run_evaluation(
    cases_path: str   = str(_CASES_PATH),
    output_dir: str   = str(_OUTPUT_DIR),
    filter_id:  str | None = None,
    check:      bool  = True,
) -> None:
    with open(cases_path, encoding="utf-8") as f:
        all_cases = json.load(f)

    cases = [c for c in all_cases if not filter_id or c["id"] == filter_id]
    if not cases:
        print(f"케이스를 찾을 수 없습니다: {filter_id}")
        return

    print(f"\n{'=' * _W}")
    print(f"  산안비 챗봇 에이전트 평가  |  총 {len(cases)}케이스")
    print(f"{'=' * _W}")

    results = []
    for i, case in enumerate(cases, 1):
        print(f"\n  실행 중 [{i:02d}/{len(cases)}] {case['id']} — {case['description']} ...", end="", flush=True)
        try:
            result = await run_single(case, check=check)
            results.append(result)
            print(" 완료")
            print_result(i, len(cases), result, check)
        except Exception as e:
            print(f" 오류: {e}")
            results.append({
                "id": case["id"], "description": case.get("description", ""),
                "question": case["question"], "error": str(e),
                "intent_ok": False, "keywords_ok": False,
            })

    # ── 요약 ──────────────────────────────────────────────────────────────────
    total          = len(results)
    intent_correct = sum(1 for r in results if r.get("intent_ok", False))
    keyword_pass   = sum(1 for r in results if r.get("keywords_ok", False))
    errors         = sum(1 for r in results if "error" in r)

    print(f"\n\n{'=' * _W}")
    print("  평가 요약")
    print(f"{'=' * _W}")
    print(f"  총 케이스         : {total}")
    print(f"  Intent 정확도    : {intent_correct}/{total}  ({intent_correct/total:.0%})")
    if check:
        print(f"  키워드 통과율    : {keyword_pass}/{total}  ({keyword_pass/total:.0%})")
    if errors:
        print(f"  오류             : {errors}건")
    print()
    print(f"  {'id':<10} {'설명':<36} {'intent':^6}" + ("  {'keyword':^7}" if check else ""))
    print("  " + "─" * (_W - 2))
    for r in results:
        if "error" in r:
            print(f"  {r['id']:<10} {r['description']:<36}  ERROR")
            continue
        im = "  [O]" if r["intent_ok"] else "  [X]"
        km = "  [O]" if r["keywords_ok"] else "  [X]" if check else ""
        print(f"  {r['id']:<10} {r['description']:<36}{im}{km}")
    print(f"{'=' * _W}")

    # ── 결과 저장 ──────────────────────────────────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts          = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = Path(output_dir) / f"results_{ts}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {
                    "total": total,
                    "intent_correct": intent_correct,
                    "intent_accuracy": round(intent_correct / total, 4) if total else 0,
                    "keyword_pass": keyword_pass,
                    "keyword_pass_rate": round(keyword_pass / total, 4) if total else 0,
                    "evaluated_at": datetime.now().isoformat(),
                },
                "cases": [
                    {k: v for k, v in r.items() if k != "answer"}
                    | {"answer_preview": r.get("answer", "")[:200]}
                    for r in results
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n  결과 저장: {output_path}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="산안비 챗봇 에이전트 평가")
    parser.add_argument(
        "--id", default=None,
        help="특정 케이스 ID만 실행 (예: --id chat_01)",
    )
    parser.add_argument(
        "--cases", default=str(_CASES_PATH),
        help=f"케이스 파일 경로 (기본: {_CASES_PATH})",
    )
    parser.add_argument(
        "--output", default=str(_OUTPUT_DIR),
        help=f"결과 저장 경로 (기본: {_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--no-check", dest="check", action="store_false",
        help="키워드 검증 비활성화 (답변만 확인)",
    )
    args = parser.parse_args()

    asyncio.run(
        run_evaluation(
            cases_path=args.cases,
            output_dir=args.output,
            filter_id=args.id,
            check=args.check,
        )
    )
