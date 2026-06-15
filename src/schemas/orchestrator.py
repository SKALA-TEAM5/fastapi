"""
AI Review Orchestrator Pydantic 스키마
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Orchestrator API의 요청/응답 모델을 정의한다.

주요 모델:
  - 사용내역서 파싱, 증빙 검증, legal/report 실행 요청
  - 화면 상태 조회 응답
  - SHE 대시보드용 실행 로그 및 토큰 사용량 요약 응답
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

from pydantic import AliasChoices, BaseModel, Field


class UsageStatementParseRequest(BaseModel):
    file_id: int = Field(..., description="files.id. uploaded_evidence_type_code=usage_statement인 파일")
    report_year: int = Field(
        ...,
        description="사용자가 입력한 사용내역서 보고 연도. OCR 인식 연월과 대조해 불일치 시 409 거부.",
    )
    report_month: int = Field(
        ...,
        ge=1,
        le=12,
        description="사용자가 입력한 사용내역서 보고 월(1~12). report_year와 함께 OCR 인식 연월과 대조.",
    )


class UsageStatementClassifyRequest(BaseModel):
    project_id: int
    usage_statement_id: int
    item_name: str
    used_on: date | None = None
    unit: str | None = None
    quantity: Decimal | None = None
    unit_price: Decimal | None = None
    total_amount: int | None = None
    item_id: int | None = None
    category_code: str | None = None
    remark: str | None = None


class OrchestratorActionResponse(BaseModel):
    status: str = Field(..., description="업무 단계 처리 상태")
    message: str = Field("", description="프론트 표시용 메시지")
    usage_statement_id: int | None = None
    target_agents: list[str] = Field(default_factory=list)
    hil_agents: list[str] = Field(default_factory=list)
    result: dict[str, Any] = Field(default_factory=dict)


class AgentLogSnapshot(BaseModel):
    agent_type_code: str
    status_code: str | None = None
    result_code: str | None = None
    reason: str | None = None
    details: dict[str, Any] | None = None
    token: int | None = None


class SupplementTodoSnapshot(BaseModel):
    agent_type_code: str
    usage_statement_item_id: int | None = None
    category_code: str | None = None
    category_name: str | None = None
    usage_statement_item_name: str | None = None
    file_id: int | None = None
    reason: str
    status_code: str = "open"


class AgentDashboardSummary(BaseModel):
    agent_type_code: str
    status_code: str | None = None
    result_code: str | None = None
    usage_statement_id: int | None = None
    token: int = 0
    reason: str | None = None


class OrchestratorStatusResponse(BaseModel):
    project_id: int
    usage_statement_id: int
    has_usage_statement_items: bool
    has_receipts_or_tax_invoices: bool
    has_site_photos: bool
    classi_ready: bool
    evidence_review_ready: bool
    legal_ready: bool
    report_ready: bool
    logs: list[AgentLogSnapshot] = Field(default_factory=list)
    todos: list[SupplementTodoSnapshot] = Field(default_factory=list)


class OrchestratorDashboardResponse(BaseModel):
    project_id: int
    usage_statement_id: int | None = None
    total_logs: int
    total_token: int
    status_counts: dict[str, int] = Field(default_factory=dict)
    result_counts: dict[str, int] = Field(default_factory=dict)
    hil_agents: list[str] = Field(default_factory=list)
    agents: list[AgentDashboardSummary] = Field(default_factory=list)


class EvidenceReviewRequest(BaseModel):
    project_id: int
    usage_statement_id: int
    requested_by_user_id: int | None = Field(
        None,
        validation_alias=AliasChoices("requested_by_user_id", "triggered_by_user_id"),
        description="보완 TODO 요청자 사용자 ID",
    )


class LegalReviewRequest(BaseModel):
    project_id: int
    usage_statement_id: int
    she_user_id: int | None = Field(
        None,
        validation_alias=AliasChoices("she_user_id", "triggered_by_user_id"),
        description="SHE 담당자 사용자 ID",
    )


class ReportDraftRequest(BaseModel):
    project_id: int
    usage_statement_id: int
    she_user_id: int | None = Field(
        None,
        validation_alias=AliasChoices("she_user_id", "triggered_by_user_id"),
        description="SHE 담당자 사용자 ID",
    )
