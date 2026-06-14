from datetime import date, datetime
from decimal import Decimal

from src.agents.report_agent.agent import ReportAgent
from src.agents.report_agent.schemas import (
    LegalValidationResultContext,
    ProjectContext,
    ReportContext,
    UsageCategorySummary,
    UsageStatementContext,
    UsageStatementItemContext,
    ValidationLogContext,
)


def _long_overall_opinion() -> str:
    return (
        "본 보고서는 검토 대상 기간의 산업안전보건관리비 집행 내역과 제출 증빙을 대상으로 "
        "법령 적정성 및 증빙 충족 여부를 확인한 결과입니다. 검토는 사용내역서, 세부 항목, "
        "agent 검증 로그에 포함된 판정 사유와 법령 근거를 바탕으로 수행되었으며, 입력 자료에 "
        "없는 금액이나 법령 조항은 임의로 추가하지 않았습니다. 일부 항목은 증빙 또는 사용 목적의 "
        "확인이 필요하여 정산 과정에서 추가 소명이나 보완 제출이 요구될 수 있습니다. 담당자는 "
        "본 초안을 참고하되 원 증빙과 내부 기준을 대조하여 최종 인정 여부를 확인해야 하며, 필요한 "
        "경우 사용내역 정정 또는 추가 증빙 확보 후 결재 절차를 진행해야 합니다. 최종 판단은 담당자 "
        "검토와 책임자 확인을 통해 확정되어야 합니다."
    )


def test_report_agent_generates_required_action_when_source_action_is_missing():
    def fake_llm(task_name: str, payload: dict) -> dict:
        issue = payload["draft"]["issue_details"][0]
        assert task_name == "report_draft"
        assert issue["required_action_fact"] is None
        assert issue["required_action"] is None
        return {
            "conclusion": "검토 필요 항목 1건에 대해 담당자 확인이 필요합니다.",
            "overall_opinion": _long_overall_opinion(),
            "issue_details": [
                {
                    "no": issue["no"],
                    "agent_conclusion": issue["agent_conclusion"],
                    "required_action": "해당 집행 항목의 사용 목적과 증빙 적정성을 담당자가 재확인하시기 바랍니다.",
                }
            ],
        }

    context = ReportContext(
        project=ProjectContext(
            id=1,
            contract_no="2026-0001",
            construction_company="스칼라건설",
            project_name="테스트 현장",
            site_location="서울",
            representative_name="대표자",
            contract_amount=Decimal("100000000"),
            construction_start_date=date(2026, 1, 1),
            construction_end_date=date(2026, 12, 31),
            client_name="발주처",
            appropriated_amount=Decimal("5000000"),
        ),
        usage_statement=UsageStatementContext(
            id=10,
            report_month=date(2026, 6, 1),
            revision_no=1,
            document_written_date=date(2026, 6, 30),
            cumulative_progress_rate=Decimal("10"),
        ),
        summaries=[
            UsageCategorySummary(
                category_code="SAFETY_FACILITY",
                category_name="안전시설비 등",
                previous_amount=Decimal("0"),
                current_amount=Decimal("1600000"),
                cumulative_amount=Decimal("1600000"),
                item_count=1,
            )
        ],
        items=[
            UsageStatementItemContext(
                id=100,
                category_code="SAFETY_FACILITY",
                category_name="안전시설비 등",
                used_on=date(2026, 6, 12),
                item_name="안전난간 설치비",
                quantity=Decimal("1"),
                unit_price=Decimal("1600000"),
                total_amount=Decimal("1600000"),
                page_no=1,
                validation_logs=[
                    ValidationLogContext(
                        id=1,
                        validation_type_code="legal",
                        result_code="needs_review",
                        details={"reason": "사용 목적 확인 필요"},
                        created_at=datetime(2026, 6, 13, 9, 0, 0),
                    )
                ],
                legal_validation_result=LegalValidationResultContext(
                    category_code="SAFETY_FACILITY",
                    status="검토필요",
                    reason="사용 목적 확인 필요",
                ),
            )
        ],
        report_no="R-2026-0001",
        report_written_date=date(2026, 6, 13),
        report_period_label="2026년 06월",
    )

    draft = ReportAgent(llm_client=fake_llm).generate(context)

    assert draft.issue_details[0].required_action == "해당 집행 항목의 사용 목적과 증빙 적정성을 담당자가 재확인하시기 바랍니다."
    issue_section = next(section for section in draft.report_sections if section.section_id == "issue_details")
    assert ["조치 사항", "해당 집행 항목의 사용 목적과 증빙 적정성을 담당자가 재확인하시기 바랍니다."] in issue_section.tables[0].rows
