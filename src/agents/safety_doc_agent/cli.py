# --------------------------------------------------------------------------
# 작성자   : 한채윤
# 작성일   : 2026-06-04
# 수정일   : 2026-06-18
#
# [ 주요 함수 정의 ]
#
# 1. cmd_check_missing_evidence() : 항목 1건 필수 증빙 누락 CLI 실행
# 2. build_parser()               : CLI argument parser 구성
# 3. main()                       : 콘솔 스크립트 진입점
# --------------------------------------------------------------------------
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agents.safety_doc_agent.agent import check_missing_evidence


def cmd_check_missing_evidence(args: argparse.Namespace) -> None:
    """로컬 Postgres를 읽어 항목 1건의 필수 증빙 누락 여부를 판단한다."""

    result = check_missing_evidence(args.item_id, dry_run=args.dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    """DB 기반 증빙 판단 CLI를 구성한다."""

    parser = argparse.ArgumentParser(description="산업안전보건관리비 서류 점검 에이전트")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser(
        "check-missing-evidence",
        help="항목 1건의 필수 증빙을 추론하고 제출 증빙과 비교해 누락 여부를 판단합니다.",
    )
    check_parser.add_argument("--item-id", required=True, type=int, help="대상 사용내역서 항목 ID")
    check_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="evidence_requirements 저장과 agent_logs 기록은 생략하고 AI 입력/출력만 확인합니다.",
    )
    check_parser.set_defaults(func=cmd_check_missing_evidence)

    return parser


def main() -> None:
    """`safety-doc-agent` 콘솔 스크립트의 진입점."""

    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
