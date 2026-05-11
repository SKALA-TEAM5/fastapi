from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agents.safety_doc_agent.config import load_settings, project_data_dir
from src.agents.safety_doc_agent.parser import parse_guide, save_parsed_guide
from src.agents.safety_doc_agent.vector_store import get_openai_client, index_chunks
from src.prompts.safety_doc_agent_evidence_requirement_prompt import build_user_prompt
from src.repositories.safety_doc_agent_postgres_evidence_repository import PostgresEvidenceRepository
from src.services.safety_doc_agent_evidence_check_service import EvidenceCheckService
from src.services.safety_doc_agent_evidence_requirement_service import EvidenceRequirementService


def _normalize_collection_name(value: str) -> str:
    """사용자가 넘긴 컬렉션명을 공백 없이 정리한다."""

    collection_name = value.strip()
    if not collection_name:
        raise ValueError("--collection must not be empty.")
    return collection_name


def cmd_ingest(args: argparse.Namespace) -> None:
    """가이드를 파싱하고 벡터 인덱스를 다시 만든다."""

    settings = load_settings()
    collection_name = _normalize_collection_name(args.collection)
    parsed = parse_guide(args.guide)
    parsed_path = project_data_dir() / "parsed_guide.json"
    save_parsed_guide(parsed, parsed_path)
    count = index_chunks(parsed, settings, collection_name)
    print(f"Indexed {count} chunks into Qdrant collection '{collection_name}'.")
    print(f"Saved parsed guide to {parsed_path}")


def cmd_run_db_flow(args: argparse.Namespace) -> None:
    """Python이 로컬 Postgres를 직접 읽어 필수 증빙 판단 전체 흐름을 실행한다."""

    settings = load_settings()
    repository = PostgresEvidenceRepository(settings)
    service = EvidenceRequirementService(
        repository=repository,
        openai_client=get_openai_client(settings),
        settings=settings,
    )

    item_context = repository.get_item_context(args.item_id)
    ai_input = service.build_ai_input(args.item_id)
    ai_output = service.infer_required_evidences(args.item_id)

    saved_requirements = None
    requirements_after_save = None
    evidence_status = None

    if not args.dry_run:
        saved_requirements = [
            asdict(row)
            for row in repository.replace_active_requirements(args.item_id, ai_output.required_evidences)
        ]
        repository.append_validation_log(
            project_id=item_context.project_id,
            usage_statement_id=item_context.usage_statement_id,
            usage_statement_item_id=item_context.item_id,
            validation_type_code="evidence_requirement_generation",
            result_code="success",
            details=asdict(ai_output),
            model_name=settings.chat_model,
        )
        requirements_after_save = [
            asdict(row)
            for row in repository.list_active_requirements(args.item_id)
        ]
        evidence_status = asdict(EvidenceCheckService(repository).run(args.item_id))

    result = {
        "db_target": {
            "host": settings.db_host,
            "port": settings.db_port,
            "db_name": settings.db_name,
            "db_user": settings.db_user,
        },
        "input_from_db_views": json.loads(build_user_prompt(ai_input)),
        "ai_response": asdict(ai_output),
        "saved_requirements": saved_requirements,
        "requirements_after_save": requirements_after_save,
        "evidence_status": evidence_status,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    """가이드 적재와 DB 실험용 CLI를 구성한다."""

    parser = argparse.ArgumentParser(description="산업안전보건관리비 서류 점검 에이전트")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="가이드 문서를 파싱하고 Qdrant에 인덱싱합니다.")
    ingest_parser.add_argument("--guide", required=True, help="원본 마크다운 가이드 파일 경로")
    ingest_parser.add_argument("--collection", required=True, help="이번 실행에서 사용할 Qdrant 컬렉션명")
    ingest_parser.set_defaults(func=cmd_ingest)

    db_flow_parser = subparsers.add_parser(
        "run-db-flow",
        help="로컬 Postgres를 직접 읽어 필수 증빙 판단과 저장/검증 흐름을 실험합니다.",
    )
    db_flow_parser.add_argument("--item-id", required=True, type=int, help="대상 사용내역서 항목 ID")
    db_flow_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="DB 저장과 충족 여부 계산은 생략하고 AI 입력/출력만 확인합니다.",
    )
    db_flow_parser.set_defaults(func=cmd_run_db_flow)

    return parser


def main() -> None:
    """`safety-doc-agent` 콘솔 스크립트의 진입점."""

    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
