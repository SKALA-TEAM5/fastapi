# --------------------------------------------------------------------------
# 작성자   : 한채윤
# 작성일   : 2026-06-04
# 수정일   : 2026-06-18
#
# [ 주요 클래스/함수 정의 ]
#
# 1. EvidenceRequirementService : LLM 기반 필수 증빙 추론 서비스
# 2. infer_required_evidences() : 항목 1건 필수 증빙 추론
# 3. infer_required_evidences_batch() : 여러 항목 필수 증빙 일괄 추론
# 4. total_tokens() : Responses API usage에서 total token 추출
# --------------------------------------------------------------------------
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict

from langsmith import traceable
from openai import OpenAI

from src.agents.safety_doc_agent.config import Settings
from src.core.metrics import (
    SAFETY_DOC_CONFIDENCE,
    SAFETY_DOC_INFERENCE_DURATION,
    SAFETY_DOC_LLM_FAILURES,
    SAFETY_DOC_REFERENCE_FAILURES,
    SAFETY_DOC_TOKENS,
)
from src.prompts.safety_doc_agent_evidence_requirement_prompt import (
    ALLOWED_BATCH_EVIDENCE_TYPES,
    BATCH_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_batch_user_prompt,
    build_user_prompt,
)
from src.repositories.safety_doc_agent_evidence_repository import EvidenceRepository
from src.schemas.safety_doc_agent_evidence import (
    AIEvidenceRequirementInput,
    AIEvidenceRequirementOutput,
)
from src.tools.safety_doc_reference_vector import search_reference_vector_db


log = logging.getLogger(__name__)


# 사진 증빙 규칙(결정적):
#   - 안전모·안전벨트(보호구) → '착용사진(wearing_photo)'만 요구. '현장사진(site_photo)'은 부적합.
#   - 그 외 품목 → 사진 증빙 요구 없음.
# LLM이 보호구에 site_photo를 붙이거나(현장사진 오류) 사진을 아예 빠뜨리는(누락) 비결정성을
# 후처리로 보정해, 보호구는 항상 착용사진만 요구하도록 고정한다.
_SITE_PHOTO_CODE: str = "site_photo"
_WEARING_PHOTO_CODE: str = "wearing_photo"
_PHOTO_EVIDENCE_CODES: frozenset[str] = frozenset({_SITE_PHOTO_CODE, _WEARING_PHOTO_CODE})
_PHOTO_ITEM_KEYWORDS: tuple[str, ...] = ("안전모", "헬멧", "안전벨트", "안전대", "안전띠")


def _photo_evidence_allowed(item_name: str | None) -> bool:
    """해당 품목이 사진 증빙(착용사진) 요구 대상(안전모·안전벨트)인지 판단."""
    name = item_name or ""
    return any(kw in name for kw in _PHOTO_ITEM_KEYWORDS)


def _filter_photo_codes(codes: set[str], item_name: str | None) -> set[str]:
    """
    사진 증빙 코드를 규칙에 맞게 보정한다.
      - 보호구(안전모·안전벨트): 현장사진 제거 + 착용사진 보장(항상 요구).
      - 그 외 품목: 사진 증빙(현장/착용) 모두 제거.
    """
    result = set(codes)
    if _photo_evidence_allowed(item_name):
        result.discard(_SITE_PHOTO_CODE)   # 보호구엔 현장사진 부적합 → 제거
        result.add(_WEARING_PHOTO_CODE)    # 착용사진은 항상 요구
        return result
    return result - _PHOTO_EVIDENCE_CODES


class EvidenceRequirementService:
    """상세 항목 1건의 필수 증빙을 생성하고 저장한다."""

    def __init__(
        self,
        repository: EvidenceRepository,
        openai_client: OpenAI,
        settings: Settings,
    ) -> None:
        """Initialize the requirement service with repository, client, and settings."""

        self.repository = repository
        self.openai_client = openai_client
        self.settings = settings

    def build_ai_input(self, item_id: int) -> AIEvidenceRequirementInput:
        """DB 조회 결과를 LLM 입력 구조로 변환한다."""

        item_context = self.repository.get_item_context(item_id)
        evidence_types = self.repository.list_evidence_types()
        linked_files = self.repository.list_linked_file_contexts(item_id)
        reference_contexts = self._search_reference_contexts(item_context)
        return AIEvidenceRequirementInput(
            item_context=item_context,
            linked_files=linked_files,
            available_evidence_types=[evidence_type.code for evidence_type in evidence_types],
            evidence_type_definitions=evidence_types,
            reference_contexts=reference_contexts,
        )

    def build_batch_ai_inputs(self, item_ids: list[int]) -> list[AIEvidenceRequirementInput]:
        """전체 항목 문맥을 모으고 증빙 정의·참고 문맥 조회를 한 번으로 제한한다."""

        if not item_ids:
            return []

        allowed_codes = set(ALLOWED_BATCH_EVIDENCE_TYPES)
        evidence_types = [
            evidence_type
            for evidence_type in self.repository.list_evidence_types()
            if evidence_type.code in allowed_codes
        ]
        item_contexts = [self.repository.get_item_context(item_id) for item_id in item_ids]
        reference_contexts = self._search_reference_contexts_for_items(item_contexts)

        return [
            AIEvidenceRequirementInput(
                item_context=item_context,
                linked_files=self.repository.list_linked_file_contexts(item_context.item_id),
                available_evidence_types=[evidence_type.code for evidence_type in evidence_types],
                evidence_type_definitions=evidence_types,
                reference_contexts=reference_contexts,
            )
            for item_context in item_contexts
        ]

    def _search_reference_contexts(self, item_context) -> list[dict]:
        """Search reference contexts for one usage-statement item."""

        query = _reference_query([item_context])
        return self._search_reference_contexts_for_query(query=query, mode="single")

    def _search_reference_contexts_for_items(self, item_contexts: list) -> list[dict]:
        """Search reference contexts for a batch of usage-statement items."""

        if not item_contexts:
            return []

        query = _reference_query(item_contexts)
        return self._search_reference_contexts_for_query(query=query, mode="batch")

    def _search_reference_contexts_for_query(self, *, query: str, mode: str) -> list[dict]:
        """Search and normalize safety-doc reference hits for a prepared query."""

        if not query or self.settings.reference_top_k <= 0:
            return []

        try:
            hits = search_reference_vector_db(
                query=query,
                collection_name=self.settings.reference_collection,
                top_k=self.settings.reference_top_k,
            )
        except Exception as exc:
            SAFETY_DOC_REFERENCE_FAILURES.labels(mode=mode).inc()
            log.warning(
                "safety-doc reference search skipped: mode=%s collection=%s error=%s",
                mode,
                self.settings.reference_collection,
                exc,
            )
            return []

        return [_reference_context_from_hit(hit) for hit in hits]

    @traceable(name="evidence_requirement_inference", run_type="chain")
    def infer_required_evidences(
        self,
        item_id: int,
        *,
        ai_input: AIEvidenceRequirementInput | None = None,
    ) -> AIEvidenceRequirementOutput:
        """LLM을 호출하고 응답을 구조화된 결과로 정리한다."""

        ai_input = ai_input or self.build_ai_input(item_id)
        started_at = time.perf_counter()
        try:
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
        except Exception:
            SAFETY_DOC_LLM_FAILURES.labels(mode="single").inc()
            raise
        finally:
            SAFETY_DOC_INFERENCE_DURATION.labels(
                mode="single",
                model=self.settings.chat_model,
            ).observe(time.perf_counter() - started_at)

        available_codes = set(ai_input.available_evidence_types)
        required_evidences = sorted(
            _filter_photo_codes(
                {
                    code
                    for code in payload.get("required_evidences", [])
                    if code in available_codes
                },
                ai_input.item_context.item_name,
            )
        )

        result = AIEvidenceRequirementOutput(
            required_evidences=required_evidences,
            confidence=payload.get("confidence"),
            reason=payload.get("reason"),
            usage=usage,
        )
        _record_model_observability(
            mode="single",
            model=self.settings.chat_model,
            result=result,
        )
        return result

    @traceable(name="batch_evidence_requirement_inference", run_type="chain")
    def infer_required_evidences_batch(
        self,
        item_ids: list[int],
    ) -> tuple[list[AIEvidenceRequirementInput], dict[int, AIEvidenceRequirementOutput]]:
        """사용내역서의 모든 항목을 한 번의 LLM 호출로 판단한다."""

        ai_inputs = self.build_batch_ai_inputs(item_ids)
        if not ai_inputs:
            return [], {}

        started_at = time.perf_counter()
        try:
            response = self.openai_client.responses.create(
                model=self.settings.chat_model,
                input=[
                    {"role": "system", "content": [{"type": "input_text", "text": BATCH_SYSTEM_PROMPT}]},
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": build_batch_user_prompt(ai_inputs)}],
                    },
                ],
            )
            payload = _parse_json_payload(response)
            usage = _extract_usage(response)
        except Exception:
            SAFETY_DOC_LLM_FAILURES.labels(mode="batch").inc()
            raise
        finally:
            SAFETY_DOC_INFERENCE_DURATION.labels(
                mode="batch",
                model=self.settings.chat_model,
            ).observe(time.perf_counter() - started_at)
        expected_item_ids = {item.item_context.item_id for item in ai_inputs}
        allowed_codes = set(ALLOWED_BATCH_EVIDENCE_TYPES)
        item_name_by_id = {
            item.item_context.item_id: item.item_context.item_name
            for item in ai_inputs
        }
        outputs: dict[int, AIEvidenceRequirementOutput] = {}

        for row in payload.get("results") or []:
            if not isinstance(row, dict):
                continue
            raw_item_id = row.get("item_id")
            if isinstance(raw_item_id, float) and not raw_item_id.is_integer():
                continue
            try:
                item_id = int(raw_item_id)
            except (TypeError, ValueError):
                continue
            if item_id not in expected_item_ids or item_id in outputs:
                continue
            raw_required_evidences = row.get("required_evidences")
            required_evidences = (
                raw_required_evidences
                if isinstance(raw_required_evidences, list)
                else []
            )
            outputs[item_id] = AIEvidenceRequirementOutput(
                required_evidences=sorted(
                    _filter_photo_codes(
                        {
                            code
                            for code in required_evidences
                            if code in allowed_codes
                        },
                        item_name_by_id.get(item_id),
                    )
                ),
                confidence=row.get("confidence"),
                reason=row.get("reason"),
                usage=None,
            )

        missing_item_ids = sorted(expected_item_ids - outputs.keys())
        if missing_item_ids:
            SAFETY_DOC_LLM_FAILURES.labels(mode="batch").inc()
            raise ValueError(f"Batch model response omitted item_ids: {missing_item_ids}")

        outputs[item_ids[0]].usage = usage
        for output in outputs.values():
            _record_model_observability(
                mode="batch",
                model=self.settings.chat_model,
                result=output,
            )
        return ai_inputs, outputs

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
            token=total_tokens(result.usage),
        )
        return result


def total_tokens(usage: dict[str, int] | None) -> int | None:
    """Return total token count from a Responses API usage dict."""

    if not usage:
        return None
    total = usage.get("total_tokens")
    return total if isinstance(total, int) else None


def _reference_query(item_contexts: list) -> str:
    """Build a deterministic reference-search query from item contexts."""

    return " ".join(
        filter(
            None,
            (
                " ".join(
                    str(part or "").strip()
                    for part in (
                        item_context.category_name,
                        item_context.item_name,
                        item_context.remark,
                    )
                    if str(part or "").strip()
                )
                for item_context in item_contexts
            ),
        )
    )


def _reference_context_from_hit(hit: dict) -> dict:
    """Normalize one Qdrant reference hit for prompt context."""

    payload = hit.get("payload") or {}
    return {
        "score": hit.get("score"),
        "title": payload.get("title") or payload.get("section") or payload.get("source"),
        "text": payload.get("text") or payload.get("content") or payload.get("body"),
        "metadata": payload.get("metadata") or {},
    }


def _record_model_observability(
    *,
    mode: str,
    model: str,
    result: AIEvidenceRequirementOutput,
) -> None:
    """Record model confidence and token metrics for safety-doc inference."""

    if isinstance(result.confidence, (int, float)):
        SAFETY_DOC_CONFIDENCE.labels(mode=mode).observe(float(result.confidence))
    if not result.usage:
        return
    for token_type in (
        "input_tokens",
        "output_tokens",
        "cached_tokens",
        "reasoning_tokens",
    ):
        value = result.usage.get(token_type)
        if isinstance(value, int) and value >= 0:
            SAFETY_DOC_TOKENS.labels(model=model, type=token_type).inc(value)


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
