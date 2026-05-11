from __future__ import annotations

"""보고서 문장 생성을 위한 OpenAI 어댑터입니다.

전체 보고서가 아니라 작은 JSON 패치만 요청합니다. ReportAgent는
금액, 판정, 법령 근거는 기존 초안 그대로 유지하고 문장 필드만 병합할 수 있습니다.
"""

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_OPENAI_MODEL = "gpt-5.2"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


class ReportLLMError(RuntimeError):
    """LLM 문장 생성이 유효한 JSON 패치를 만들지 못했을 때 발생합니다."""

    pass


class OpenAIReportLLMClient:
    """보고서 문장 생성을 위한 기본 LLM 어댑터입니다.

    이 클라이언트는 ReportAgent가 기존 초안에 병합해도 되는 JSON 필드만
    반환하도록 요청합니다.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str = OPENAI_RESPONSES_URL,
        timeout_seconds: int = 60,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_REPORT_MODEL") or DEFAULT_OPENAI_MODEL
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_environment(cls) -> "OpenAIReportLLMClient | None":
        """API 키가 설정된 경우에만 기본 LLM을 활성화합니다."""

        if not os.getenv("OPENAI_API_KEY"):
            return None
        return cls()

    def __call__(self, task_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Responses API를 호출하고 검증 가능한 사전 형태의 JSON을 반환합니다."""

        if task_name != "report_draft":
            raise ReportLLMError(f"Unsupported LLM task: {task_name}")
        if not self.api_key:
            raise ReportLLMError("OPENAI_API_KEY is not configured.")

        request_payload = {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": _read_prompt("report_agent_system.md")}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": _build_user_prompt(payload),
                        }
                    ],
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "report_draft_text_patch",
                    "strict": True,
                    "schema": _response_schema(),
                }
            },
        }

        data = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.base_url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ReportLLMError(f"OpenAI API HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ReportLLMError(f"OpenAI API request failed: {exc.reason}") from exc

        return _parse_response_json(raw)


def _build_user_prompt(payload: dict[str, Any]) -> str:
    instruction = _read_prompt("report_agent_draft_template.md")
    context_json = json.dumps(payload.get("context", {}), ensure_ascii=False, indent=2)
    draft_json = json.dumps(payload.get("draft", {}), ensure_ascii=False, indent=2)
    return (
        f"{instruction}\n\n"
        "## ReportContext\n"
        f"```json\n{context_json}\n```\n\n"
        "## Deterministic ReportDraft\n"
        f"```json\n{draft_json}\n```"
    )


def _parse_response_json(raw: str) -> dict[str, Any]:
    body = json.loads(raw)
    output_text = body.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return json.loads(output_text)

    for item in body.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                return json.loads(content["text"])

    raise ReportLLMError("OpenAI API response did not contain JSON output text.")


def _read_prompt(filename: str) -> str:
    return (Path(__file__).parents[2] / "prompts" / filename).read_text(encoding="utf-8")


def _response_schema() -> dict[str, Any]:
    """LLM이 반환할 수 있는 필드만 정의한 JSON 스키마입니다."""

    text_field = {"type": "string"}
    numbered_issue = {
        "type": "object",
        "additionalProperties": False,
        "required": ["no", "agent_conclusion", "required_action"],
        "properties": {
            "no": {"type": "integer"},
            "agent_conclusion": text_field,
            "required_action": text_field,
        },
    }
    numbered_action = {
        "type": "object",
        "additionalProperties": False,
        "required": ["no", "action"],
        "properties": {
            "no": {"type": "integer"},
            "action": text_field,
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["conclusion", "overall_opinion", "issue_details", "supplement_actions"],
        "properties": {
            "conclusion": text_field,
            "overall_opinion": text_field,
            "issue_details": {"type": "array", "items": numbered_issue},
            "supplement_actions": {"type": "array", "items": numbered_action},
        },
    }
