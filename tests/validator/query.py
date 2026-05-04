"""
RAG 질의응답 실행 스크립트.

uv run python -m tests.validator.query
uv run python -m tests.validator.query --question "안전모 구입 비용은 산안비 항목인가?"
uv run python -m tests.validator.query --collection my_collection
"""

import argparse
import json
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.core import llm_config
from src.services.ingestion_service import run_query

load_dotenv()

_DEFAULT_QUESTIONS = [
    "안전관리자 인건비를 산안비로 집행하는 것이 적합한가?",
    "현장 근로자 식대 지원을 산안비로 집행할 수 있는가?",
    "안전모·안전화 구입 비용은 산안비 사용 항목에 해당하는가?",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="산안비 법령 적합성 RAG 쿼리")
    parser.add_argument("--question", "-q", default=None, help="판정할 질의")
    parser.add_argument("--collection", default="documents", help="ChromaDB 컬렉션 이름")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI 모델 이름")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY가 없어 fallback judge 모드로 실행합니다.")
    else:
        llm_config.configure(ChatOpenAI(model=args.model, temperature=0))

    questions = [args.question] if args.question else _DEFAULT_QUESTIONS

    for q in questions:
        print(f"\n{'=' * 60}")
        print(f"질의: {q}")
        result = run_query(q, collection=args.collection)
        print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
