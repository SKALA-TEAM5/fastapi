from __future__ import annotations

import json
from dataclasses import asdict

from langsmith.wrappers import wrap_openai
from openai import OpenAI

from src.agents.safety_doc_agent.config import Settings, load_settings
from src.prompts.safety_doc_agent_evidence_requirement_prompt import (
    ALLOWED_BATCH_EVIDENCE_TYPES,
    build_user_prompt,
)
from src.repositories.safety_doc_agent_evidence_repository import EvidenceRepository
from src.repositories.safety_doc_agent_postgres_evidence_repository import PostgresEvidenceRepository
from src.services.safety_doc_agent_evidence_check_service import EvidenceCheckService
from src.services.safety_doc_agent_evidence_requirement_service import EvidenceRequirementService


def check_missing_evidence(
    item_id: int,
    *,
    dry_run: bool = False,
    persist_log: bool = True,
    settings: Settings | None = None,
    repository: EvidenceRepository | None = None,
    openai_client: OpenAI | None = None,
) -> dict:
    """항목 1건의 필수 증빙을 추론하고 제출 증빙과 비교해 누락 여부를 판단한다.

    오케스트레이터는 `item_id`만 넘겨 호출할 수 있고, 테스트나 API 레이어에서는
    `settings`, `repository`, `openai_client`를 주입해 같은 흐름을 재사용할 수 있다.
    """

    settings = settings or load_settings()
    repository = repository or PostgresEvidenceRepository(settings)
    openai_client = openai_client or get_openai_client(settings)
    requirement_service = EvidenceRequirementService(
        repository=repository,
        openai_client=openai_client,
        settings=settings,
    )
    check_service = EvidenceCheckService(repository)

    item_context = repository.get_item_context(item_id)
    ai_input = requirement_service.build_ai_input(item_id)
    ai_output = requirement_service.infer_required_evidences(item_id)

    saved_requirements = None
    requirements_after_save = None
    evidence_status = None

    if not dry_run:
        saved_requirements = [
            asdict(row)
            for row in repository.replace_active_requirements(item_id, ai_output.required_evidences)
        ]
        evidence_status = asdict(check_service.run(item_id))
        requirements_after_save = [
            asdict(row)
            for row in repository.list_active_requirements(item_id)
        ]
        missing_codes = evidence_status["missing_evidences"]
        if persist_log:
            repository.append_agent_log(
                project_id=item_context.project_id,
                usage_statement_id=item_context.usage_statement_id,
                usage_statement_item_id=item_context.item_id,
                status_code="success",
                result_code="hil" if missing_codes else "success",
                reason=missing_evidence_reason(missing_codes),
                details={
                    "check_type": "missing_evidence",
                    "item_context": asdict(item_context),
                    "ai_response": asdict(ai_output),
                    "saved_requirements": saved_requirements,
                    "requirements_after_save": requirements_after_save,
                    "evidence_status": evidence_status,
                },
                model_name=settings.chat_model,
                token=total_tokens(ai_output.usage),
            )

    return {
        "db_target": {
            "host": settings.db_host,
            "port": settings.db_port,
            "db_name": settings.db_name,
            "db_user": settings.db_user,
        },
        "model_name": settings.chat_model,
        "input_from_db_views": json.loads(build_user_prompt(ai_input)),
        "ai_response": asdict(ai_output),
        "saved_requirements": saved_requirements,
        "requirements_after_save": requirements_after_save,
        "evidence_status": evidence_status,
    }


def check_missing_evidence_batch(
    item_ids: list[int],
    *,
    settings: Settings | None = None,
    repository: EvidenceRepository | None = None,
    openai_client: OpenAI | None = None,
) -> list[dict]:
    """전체 항목의 필수 증빙을 한 번에 추론하고 항목별 상태를 저장한다."""

    if not item_ids:
        return []

    settings = settings or load_settings()
    repository = repository or PostgresEvidenceRepository(settings)
    openai_client = openai_client or get_openai_client(settings)
    requirement_service = EvidenceRequirementService(
        repository=repository,
        openai_client=openai_client,
        settings=settings,
    )
    check_service = EvidenceCheckService(repository)
    ai_inputs, outputs = requirement_service.infer_required_evidences_batch(item_ids)
    input_by_item_id = {item.item_context.item_id: item for item in ai_inputs}
    results = []

    for item_id in item_ids:
        ai_input = input_by_item_id[item_id]
        ai_output = outputs[item_id]
        saved_requirements = [
            asdict(row)
            for row in repository.replace_active_requirements(item_id, ai_output.required_evidences)
        ]
        evidence_status = asdict(check_service.run(item_id))
        requirements_after_save = [
            asdict(row)
            for row in repository.list_active_requirements(item_id)
        ]
        results.append(
            {
                "db_target": {
                    "host": settings.db_host,
                    "port": settings.db_port,
                    "db_name": settings.db_name,
                    "db_user": settings.db_user,
                },
                "model_name": settings.chat_model,
                "input_from_db_views": {
                    "item_context": asdict(ai_input.item_context),
                    "linked_files": [asdict(linked_file) for linked_file in ai_input.linked_files],
                    "available_evidence_types": list(ALLOWED_BATCH_EVIDENCE_TYPES),
                },
                "ai_response": asdict(ai_output),
                "saved_requirements": saved_requirements,
                "requirements_after_save": requirements_after_save,
                "evidence_status": evidence_status,
            }
        )

    return results


def get_openai_client(settings: Settings) -> OpenAI:
    """검증된 설정으로 OpenAI 클라이언트를 만들고 필요 시 LangSmith를 연결한다."""

    client = OpenAI(api_key=settings.openai_api_key)
    if settings.langsmith_tracing:
        return wrap_openai(client)
    return client


def total_tokens(usage: dict[str, int] | None) -> int | None:
    if not usage:
        return None
    total = usage.get("total_tokens")
    return total if isinstance(total, int) else None


def missing_evidence_reason(missing_codes: list[str]) -> str:
    if not missing_codes:
        return "필수 증빙 누락 없음"
    return "필수 증빙 누락: " + ", ".join(missing_codes)
