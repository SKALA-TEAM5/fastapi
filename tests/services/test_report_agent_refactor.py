from datetime import date, datetime
from decimal import Decimal

import pytest

from src.agents.report_agent.agent import MIN_LLM_OVERALL_OPINION_CHARS, ReportAgent
from src.agents.report_agent.llm import ReportLLMError
from src.agents.report_agent.schemas import (
    EvidenceRequirementContext,
    LegalCitationContext,
    LegalValidationResultContext,
    ProjectContext,
    ReportContext,
    ReviewerContext,
    UsageCategorySummary,
    UsageStatementContext,
    UsageStatementItemContext,
    ValidationLogContext,
)


def _long_opinion() -> str:
    base = (
        "이번 산업안전보건관리비 집행 검토에서는 사용내역서, 증빙자료, 법령 검토 결과를 함께 대조했습니다. "
        "일부 항목은 증빙 보완과 담당자 확인이 필요하지만, 보고서 초안은 시스템 판정 근거와 검토 필요 사유를 "
        "분리하여 후속 조치가 가능하도록 작성되었습니다. "
    )
    while len(base) < MIN_LLM_OVERALL_OPINION_CHARS:
        base += "최종 결재 전에는 원본 증빙과 현장 담당자 설명을 다시 확인해야 합니다. "
    return base


def _context() -> ReportContext:
    return ReportContext(
        project=ProjectContext(
            id=5,
            contract_no="CN-5",
            construction_company="산안건설",
            project_name="테스트 현장",
            site_location="서울",
            representative_name="홍길동",
            contract_amount=Decimal("100000000"),
            construction_start_date=date(2026, 1, 1),
            construction_end_date=date(2026, 12, 31),
            client_name="테스트 발주처",
            appropriated_amount=Decimal("5000000"),
        ),
        usage_statement=UsageStatementContext(
            id=3,
            report_month=date(2026, 6, 1),
            revision_no=1,
            document_written_date=date(2026, 6, 18),
            cumulative_progress_rate=Decimal("75.5"),
        ),
        summaries=[
            UsageCategorySummary(
                category_code="CAT_03",
                category_name="보호구",
                previous_amount=Decimal("100000"),
                current_amount=Decimal("200000"),
                cumulative_amount=Decimal("300000"),
            )
        ],
        items=[
            UsageStatementItemContext(
                id=11,
                category_code="CAT_03",
                category_name="보호구",
                used_on=date(2026, 6, 10),
                item_name="안전모 구입",
                unit="개",
                quantity=Decimal("10"),
                unit_price=Decimal("20000"),
                total_amount=Decimal("200000"),
                remark="보호구",
                page_no=1,
                evidence_requirements=[
                    EvidenceRequirementContext(evidence_type_code="receipt", is_satisfied=False)
                ],
                validation_logs=[
                    ValidationLogContext(
                        id=1,
                        validation_type_code="legal",
                        result_code="hil",
                        details={"summary": "법령 검토 필요", "required_action": "증빙 보완"},
                        model_name="validator",
                        created_at=datetime(2026, 6, 18, 12, 0),
                    )
                ],
                legal_validation_result=LegalValidationResultContext(
                    category_code="CAT_03",
                    status="검토필요",
                    reason="보호구 구매 증빙 확인이 필요합니다.",
                    citations=[
                        LegalCitationContext(
                            legal_basis="산업안전보건법 제72조",
                            summary="산안비 사용 근거",
                        )
                    ],
                ),
            )
        ],
        reviewer=ReviewerContext(name="검토자", department="안전팀", title="매니저"),
        report_no="R-202606-3",
        report_written_date=date(2026, 6, 18),
        report_period_label="2026년 06월",
    )


def test_report_agent_generate_preserves_public_shape_and_sections():
    llm_calls: list[tuple[str, dict]] = []

    def fake_llm(task_name: str, payload: dict) -> dict:
        llm_calls.append((task_name, payload))
        return {
            "conclusion": "안전모 구입 건은 증빙 보완 후 최종 인정 여부를 확정해야 합니다.",
            "overall_opinion": _long_opinion(),
            "issue_details": [
                {
                    "no": 1,
                    "agent_conclusion": "증빙 보완이 필요한 항목입니다.",
                    "required_action": "영수증 및 관련 증빙을 추가 제출해야 합니다.",
                }
            ],
        }

    draft = ReportAgent(llm_client=fake_llm).generate(_context())

    assert llm_calls and llm_calls[0][0] == "report_draft"
    assert draft.report_no == "R-202606-3"
    assert draft.site_name == "테스트 현장"
    assert draft.issue_details[0].required_action == "영수증 및 관련 증빙을 추가 제출해야 합니다."
    assert draft.report_sections
    assert draft.report_sections[0].section_id
    assert all(section.title for section in draft.report_sections)
    assert any(section.tables for section in draft.report_sections)
    encoded = draft.model_dump(mode="json")
    assert encoded["report_sections"] == [
        section.model_dump(mode="json") for section in draft.report_sections
    ]


def test_report_agent_rejects_short_llm_overall_opinion():
    def fake_llm(task_name: str, payload: dict) -> dict:
        return {
            "conclusion": "결론",
            "overall_opinion": "짧은 의견",
            "issue_details": [],
        }

    with pytest.raises(ReportLLMError, match="overall_opinion must be at least"):
        ReportAgent(llm_client=fake_llm).generate(_context())
