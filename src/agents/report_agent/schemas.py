from __future__ import annotations

"""보고서 생성 경계에서 사용하는 타입 계약입니다.

ReportContext는 FastAPI나 worker가 DB row와 다른 agent 결과를 조립한 입력입니다.
ReportDraft는 화면에서 편집하고 API 응답으로 반환하는 JSON 보고서 산출물입니다.
"""

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, Field


DecisionCode = Literal["appropriate", "needs_review", "inappropriate"]
RiskLevel = Literal["low", "medium", "high"]
EvidenceTypeCode = Literal["usage_statement", "receipt", "site_photo", "tax_invoice", "other"]
ClassificationStatus = Literal["유지", "카테고리변경", "검토필요"]
LegalValidationStatus = Literal["적절", "부적절", "검토필요"]


class ProjectContext(BaseModel):
    id: int
    construction_company: str
    project_name: str
    site_location: str
    representative_name: str | None = None
    contract_amount: Decimal
    construction_start_date: date
    construction_end_date: date
    client_name: str | None = None
    appropriated_amount: Decimal


class UsageStatementContext(BaseModel):
    id: int
    report_month: date
    revision_no: int
    document_written_date: date
    cumulative_progress_rate: Decimal
    source_file_id: int | None = None


class UsageCategorySummary(BaseModel):
    category_code: str
    category_name: str
    previous_amount: Decimal
    current_amount: Decimal
    cumulative_amount: Decimal
    item_count: int = 0


class EvidenceFileContext(BaseModel):
    file_id: int
    original_filename: str
    evidence_type_code: EvidenceTypeCode | str
    evidence_detail_name: str | None = None
    mime_type: str | None = None
    captured_at: datetime | None = None
    uploaded_at: datetime | None = None


class EvidenceRequirementContext(BaseModel):
    evidence_type_code: EvidenceTypeCode | str
    is_satisfied: bool


class ValidationLogContext(BaseModel):
    id: int
    validation_type_code: str
    result_code: str
    details: dict[str, Any] = Field(default_factory=dict)
    model_name: str | None = None
    created_at: datetime


class ClassificationResultContext(BaseModel):
    """사용내역 항목에 붙는 분류 agent 결과입니다."""

    original_category_code: str
    final_category_code: str
    status: ClassificationStatus | str
    needs_review: bool = False
    reason: str = ""


class LegalCitationContext(BaseModel):
    """법령 validator가 제공한 단일 법령 근거입니다."""

    legal_basis: str
    summary: str | None = None
    citation_text: str | None = None
    source_id: str | None = None
    source_name: str | None = None
    article_no: str | None = None
    paragraph_no: str | None = None
    item_no: str | None = None


class LegalValidationResultContext(BaseModel):
    """법령 validator 결과이며, 보고서 생성은 이 값을 사실로 취급합니다."""

    category_code: str
    status: LegalValidationStatus | str
    reason: str
    citations: list[LegalCitationContext] = Field(default_factory=list)


class UsageStatementItemContext(BaseModel):
    id: int
    category_code: str
    category_name: str
    used_on: date
    item_name: str
    unit: str | None = None
    quantity: Decimal
    unit_price: Decimal
    total_amount: Decimal
    remark: str | None = None
    page_no: int
    evidence_files: list[EvidenceFileContext] = Field(default_factory=list)
    evidence_requirements: list[EvidenceRequirementContext] = Field(default_factory=list)
    validation_logs: list[ValidationLogContext] = Field(default_factory=list)
    classification_result: ClassificationResultContext | None = None
    legal_validation_result: LegalValidationResultContext | None = None


class ActionRequestContext(BaseModel):
    id: int
    usage_statement_item_id: int | None = None
    title: str
    reason: str | None = None
    status_code: str
    due_date: date | None = None
    assignee_name: str | None = None
    created_at: datetime
    resolved_at: datetime | None = None


class ReviewerContext(BaseModel):
    name: str
    department: str | None = None
    title: str | None = None


class ReportContext(BaseModel):
    """보고서 초안 하나를 만들기 위한 전체 입력 컨텍스트입니다."""

    project: ProjectContext
    usage_statement: UsageStatementContext
    summaries: list[UsageCategorySummary]
    items: list[UsageStatementItemContext]
    action_requests: list[ActionRequestContext] = Field(default_factory=list)
    reviewer: ReviewerContext | None = None
    report_no: str
    report_written_date: date
    report_period_label: str
    system_name: str = "AI 증빙 검토 시스템"


class AmountSummaryDraft(BaseModel):
    label: str
    amount: Decimal
    ratio_label: str
    count_label: str


class CategorySummaryDraft(BaseModel):
    category_code: str
    category_name: str
    amount: Decimal
    count: int
    note: str


class EvidenceValidationSummaryDraft(BaseModel):
    evidence_type_code: str
    evidence_type_name: str
    submitted_count: int
    passed_count: int
    error_count: int
    missing_count: int
    major_error: str


class ItemReviewDraft(BaseModel):
    no: int
    usage_statement_item_id: int
    category_code: str
    item_name: str
    amount: Decimal
    decision: DecisionCode
    decision_label: str
    summary_reason: str
    risk_level: RiskLevel


class LegalCitationDraft(BaseModel):
    """화면 검토와 감사 추적을 위해 ReportDraft에 보존하는 citation 형태입니다."""

    legal_basis: str
    summary: str | None = None
    citation_text: str | None = None
    source_id: str | None = None
    source_name: str | None = None


class IssueDetailDraft(BaseModel):
    issue_type: Literal["inappropriate", "needs_review"]
    no: int
    usage_statement_item_id: int | None = None
    title: str
    amount_label: str
    problem: str | None = None
    legal_basis: str
    legal_citations: list[LegalCitationDraft] = Field(default_factory=list)
    site_claim: str | None = None
    agent_conclusion: str
    required_action_fact: str | None = None
    required_action: str | None = None


class ReportTableDraft(BaseModel):
    """웹 화면과 DOCX 추출기가 같은 표를 그릴 수 있도록 보존하는 표 구조입니다."""

    title: str | None = None
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)


class ReportSectionDraft(BaseModel):
    """실제 보고서 형식의 섹션 단위 레이아웃입니다."""

    section_id: str
    title: str
    kind: Literal["cover", "table", "detail", "opinion"]
    paragraphs: list[str] = Field(default_factory=list)
    tables: list[ReportTableDraft] = Field(default_factory=list)


class ReportDraft(BaseModel):
    """화면에서 편집하고 저장하는 구조화 JSON 보고서 초안입니다."""

    layout_version: str = "safety_cost_report_v1"
    report_no: str
    title: str = "산업안전보건관리비 집행 증빙 검토 결과 보고서"
    site_name: str
    report_period_label: str
    written_date_label: str
    department_label: str
    reviewer_label: str
    basic_info: dict[str, str]
    amount_summary: list[AmountSummaryDraft]
    category_summaries: list[CategorySummaryDraft]
    evidence_validation_summaries: list[EvidenceValidationSummaryDraft]
    conclusion: str
    item_reviews: list[ItemReviewDraft]
    issue_details: list[IssueDetailDraft]
    overall_opinion: str
    report_sections: list[ReportSectionDraft] = Field(default_factory=list)
    needs_human_review: list[str] = Field(default_factory=list)
