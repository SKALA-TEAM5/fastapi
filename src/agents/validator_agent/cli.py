# --------------------------------------------------------------------------
# 작성자   : 송상민(ss19801)
# 작성일   : 2026-05-04
#
# [ 주요 클래스 및 함수 정의 ]
#
# 1. _parse_args() : validator agent 로컬 실행 인자 파싱
# 2. _load_payload() : 입력 JSON 파일 로드 및 검증
# 3. main() : validator agent 실행 결과를 JSON으로 출력
# --------------------------------------------------------------------------
import argparse
import json
from pathlib import Path

from src.agents.validator_agent.agent import to_validator_response, validate_usage_statement
from src.core.storage import DEFAULT_COLLECTION
from src.schemas.validator import UsageStatementValidatorRequest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="validator agent 로컬 실행")
    parser.add_argument(
        "--input",
        default="examples/validator_agent/sample_input.json",
        help="UsageStatementValidatorRequest 형식의 JSON 파일 경로",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help="조회에 사용할 벡터스토어 컬렉션명",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="출력 JSON을 보기 좋게 포맷한다.",
    )
    return parser.parse_args()


def _load_payload(path_str: str) -> UsageStatementValidatorRequest:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return UsageStatementValidatorRequest.model_validate(payload)


def main() -> None:
    args = _parse_args()
    request = _load_payload(args.input)
    response = validate_usage_statement(
        document=request.model_dump(by_alias=True),
        collection=args.collection,
    )
    summary = to_validator_response(
        response=response,
        usage_statement_id=request.usage_statement_id,
    )
    print(
        json.dumps(
            summary.model_dump(by_alias=True),
            ensure_ascii=False,
            indent=2 if args.pretty else None,
        )
    )


if __name__ == "__main__":
    main()
