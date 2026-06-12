from dataclasses import dataclass
from types import SimpleNamespace

from src.agents.safety_doc_agent.config import Settings
from src.schemas.safety_doc_agent_evidence import (
    EvidenceRequirementItemContext,
    EvidenceType,
)
from src.services.safety_doc_agent_evidence_requirement_service import EvidenceRequirementService


@dataclass
class FakeResponse:
    output_text: str
    usage: object


class FakeResponses:
    def __init__(self, response: FakeResponse):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeRepository:
    def __init__(self):
        self.contexts = {
            1: _context(1, "CAT_02", "안전난간 설치"),
            2: _context(2, "CAT_03", "안전모 구입"),
        }

    def list_evidence_types(self):
        return [
            EvidenceType("site_photo", "현장사진", "설치 확인"),
            EvidenceType("wearing_photo", "착용사진", "착용 확인"),
            EvidenceType("tax_invoice", "세금계산서", "구매 확인"),
            EvidenceType("receipt", "영수증", "결제 확인"),
            EvidenceType("supply_ledger", "지급대장", "지급 확인"),
        ]

    def get_item_context(self, item_id):
        return self.contexts[item_id]

    def list_linked_file_contexts(self, item_id):
        return []


def _context(item_id: int, category_code: str, item_name: str):
    return EvidenceRequirementItemContext(
        project_id=1,
        project_name="테스트",
        item_id=item_id,
        usage_statement_id=10,
        report_month="2026-06-01",
        revision_no=1,
        category_code=category_code,
        category_name="테스트 분류",
        used_on="2026-06-01",
        item_name=item_name,
        unit=None,
        quantity=1,
        unit_price=1000,
        total_amount=1000,
        remark=None,
        page_no=1,
    )


def _settings():
    return Settings(openai_api_key="test", reference_top_k=0)


def test_batch_inference_uses_one_llm_call_and_filters_evidence_types():
    responses = FakeResponses(
        FakeResponse(
            output_text=(
                '{"results":['
                '{"item_id":1,"required_evidences":["site_photo","tax_invoice","supply_ledger"],'
                '"confidence":0.9,"reason":"설치 확인"},'
                '{"item_id":2,"required_evidences":["wearing_photo","receipt"],'
                '"confidence":0.8,"reason":"착용 확인"}]}'
            ),
            usage=SimpleNamespace(input_tokens=100, output_tokens=20, total_tokens=120),
        )
    )
    service = EvidenceRequirementService(
        FakeRepository(),
        SimpleNamespace(responses=responses),
        _settings(),
    )

    ai_inputs, outputs = service.infer_required_evidences_batch([1, 2])

    assert len(responses.calls) == 1
    assert len(ai_inputs) == 2
    assert outputs[1].required_evidences == ["site_photo", "tax_invoice"]
    assert outputs[2].required_evidences == ["receipt", "wearing_photo"]
    assert outputs[1].usage == {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
    assert outputs[2].usage is None


def test_batch_inference_rejects_missing_items():
    responses = FakeResponses(
        FakeResponse(
            output_text='{"results":[{"item_id":1,"required_evidences":["site_photo"]}]}',
            usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15),
        )
    )
    service = EvidenceRequirementService(
        FakeRepository(),
        SimpleNamespace(responses=responses),
        _settings(),
    )

    try:
        service.infer_required_evidences_batch([1, 2])
    except ValueError as exc:
        assert "omitted item_ids: [2]" in str(exc)
    else:
        raise AssertionError("Missing batch items must fail the inference")
