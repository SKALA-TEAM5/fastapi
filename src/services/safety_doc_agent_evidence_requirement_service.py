from __future__ import annotations

import json
import re
from dataclasses import asdict

from langsmith import traceable
from openai import OpenAI

from src.agents.safety_doc_agent.config import Settings
from src.prompts.safety_doc_agent_evidence_requirement_prompt import SYSTEM_PROMPT, build_user_prompt
from src.repositories.safety_doc_agent_evidence_repository import EvidenceRepository
from src.schemas.safety_doc_agent_evidence import (
    AIEvidenceRequirementInput,
    AIEvidenceRequirementOutput,
)


class EvidenceRequirementService:
    """상세 항목 1건의 필수 증빙을 생성하고 저장한다."""

    def __init__(
        self,
        repository: EvidenceRepository,
        openai_client: OpenAI,
        settings: Settings,
    ) -> None:
        self.repository = repository
        self.openai_client = openai_client
        self.settings = settings

    def build_ai_input(self, item_id: int) -> AIEvidenceRequirementInput:
        """DB 조회 결과를 LLM 입력 구조로 변환한다."""

        item_context = self.repository.get_item_context(item_id)
        evidence_types = self.repository.list_evidence_types()
        linked_files = self.repository.list_linked_file_contexts(item_id)
        return AIEvidenceRequirementInput(
            item_context=item_context,
            linked_files=linked_files,
            available_evidence_types=[evidence_type.code for evidence_type in evidence_types],
            evidence_type_definitions=evidence_types,
        )

    @traceable(name="evidence_requirement_inference", run_type="chain")
    def infer_required_evidences(self, item_id: int) -> AIEvidenceRequirementOutput:
        """LLM을 호출하고 응답을 구조화된 결과로 정리한다."""

        ai_input = self.build_ai_input(item_id)
        response = self.openai_client.responses.create(
            model=self.settings.chat_model,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": build_user_prompt(ai_input)}],
                },
            ],
        )
        payload = _parse_json_payload(response)
        usage = _extract_usage(response)

        available_codes = set(ai_input.available_evidence_types)
        required_evidences = [
            code
            for code in payload.get("required_evidences", [])
            if code in available_codes
        ]

        return AIEvidenceRequirementOutput(
            required_evidences=required_evidences,
            confidence=payload.get("confidence"),
            reason=payload.get("reason"),
            usage=usage,
        )

    @traceable(name="evidence_requirement_generation", run_type="chain")
    def run(self, item_id: int, *, project_id: int, usage_statement_id: int | None) -> AIEvidenceRequirementOutput:
        """필수 증빙을 추론하고 active requirement 및 로그를 갱신한다."""

        result = self.infer_required_evidences(item_id)
        self.repository.replace_active_requirements(item_id, result.required_evidences)
        self.repository.append_agent_log(
            project_id=project_id,
            usage_statement_id=usage_statement_id,
            usage_statement_item_id=item_id,
            status_code="success",
            result_code="success",
            reason="필수 증빙 요구사항 생성 완료",
            details=asdict(result),
            model_name=self.settings.chat_model,
            token=_total_tokens(result.usage),
        )
        return result


def _total_tokens(usage: dict[str, int] | None) -> int | None:
    if not usage:
        return None
    total = usage.get("total_tokens")
    return total if isinstance(total, int) else None

def _parse_json_payload(response: object) -> dict:
    """Responses API 응답에서 JSON 객체만 안전하게 추출한다.

    모델이 JSON 코드블록이나 짧은 설명문을 함께 반환하는 경우가 있어,
    첫 번째 JSON 객체를 찾아 파싱하도록 처리한다.
    """

    output_text = getattr(response, "output_text", "") or ""
    text = output_text.strip()
    if not text:
        raise ValueError(f"Model returned empty output_text: {response!r}")

    candidates = [text]

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced_match:
        candidates.append(fenced_match.group(1).strip())

    object_match = re.search(r"\{.*\}", text, re.DOTALL)
    if object_match:
        candidates.append(object_match.group(0).strip())

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            continue

    raise ValueError(f"Could not parse JSON from model response: {text}")


def _extract_usage(response: object) -> dict[str, int] | None:
    """Responses API 응답에서 토큰 사용량을 최대한 안정적으로 꺼낸다."""

    usage = getattr(response, "usage", None)
    if usage is None:
        return None

    candidates = {}
    for key in (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "reasoning_tokens",
        "cached_tokens",
    ):
        value = getattr(usage, key, None)
        if isinstance(value, int):
            candidates[key] = value

    if candidates:
        return candidates

    if hasattr(usage, "model_dump"):
        dumped = usage.model_dump()
        return {key: value for key, value in dumped.items() if isinstance(value, int)} or dumped

    if hasattr(usage, "dict"):
        dumped = usage.dict()
        return {key: value for key, value in dumped.items() if isinstance(value, int)} or dumped

    return None
