# --------------------------------------------------------------------------
# 작성자   : 송상민(ss19801)
# 작성일   : 2026-05-04
#
# [ 주요 클래스 및 함수 정의 ]
#
# 1. _parse_args() : classifier agent 로컬 실행 인자 파싱
# 2. _load_payload() : 입력 JSON 파일 로드 및 검증
# 3. main() : classifier agent 실행 결과를 JSON으로 출력
# --------------------------------------------------------------------------
import argparse
import json
from pathlib import Path

from src.agents.classifier_agent.agent import review_usage_statement
from src.schemas.classifier import UsageStatementReviewRequest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="classifier agent 로컬 실행")
    parser.add_argument(
        "--input",
        default="examples/classifier_agent/sample_input.json",
        help="UsageStatementReviewRequest 형식의 JSON 파일 경로",
    )
    parser.add_argument(
        "--collection",
        default="documents",
        help="조회에 사용할 벡터스토어 컬렉션명",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="출력 JSON을 보기 좋게 포맷한다.",
    )
    return parser.parse_args()


def _load_payload(path_str: str) -> dict:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    request = UsageStatementReviewRequest.model_validate(payload)
    return request.model_dump(by_alias=True)


def main() -> None:
    args = _parse_args()
    payload = _load_payload(args.input)
    response = review_usage_statement(payload=payload, collection=args.collection)
    print(
        json.dumps(
            response.model_dump(by_alias=True),
            ensure_ascii=False,
            indent=2 if args.pretty else None,
        )
    )


if __name__ == "__main__":
    main()
