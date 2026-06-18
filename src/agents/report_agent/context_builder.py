# --------------------------------------------------------------------------
# 작성자   : 차현주
# 작성일   : 2026-06-18
# 수정일   : 2026-06-18
#
# [ 주요 함수 정의 ]
#
# 1. build_report_context() : 저장소 row를 ReportContext로 정규화
# 2. decimal_ratio()        : Decimal 비율 계산
# --------------------------------------------------------------------------
from __future__ import annotations

"""저장소 row에서 ReportContext를 조립합니다.

이 모듈은 DB 접근 경계만 정의합니다. 보고서 문장 작성, LLM 호출,
JSON 보고서 파일 출력 로직은 여기에 두지 않습니다.
"""

from collections import defaultdict
from datetime import date
from decimal import Decimal
from typing import Protocol

from .schemas import (
    ActionRequestContext,
    EvidenceFileContext,
    EvidenceRequirementContext,
    ProjectContext,
    ReportContext,
    ReviewerContext,
    UsageCategorySummary,
    UsageStatementContext,
    UsageStatementItemContext,
    ValidationLogContext,
)


class ReportRepository(Protocol):
    """FastAPI 또는 작업자가 구현해야 하는 DB 접근 경계입니다."""

    def get_project(self, project_id: int) -> dict:
        """Return one project row."""
        ...

    def get_usage_statement(self, usage_statement_id: int) -> dict:
        """Return one usage-statement row."""
        ...

    def list_usage_categories(self) -> list[dict]:
        """Return all usage category rows."""
        ...

    def list_usage_statement_summaries(self, usage_statement_id: int) -> list[dict]:
        """Return category summary rows for a usage statement."""
        ...

    def list_usage_statement_items(self, usage_statement_id: int) -> list[dict]:
        """Return item rows for a usage statement."""
        ...

    def list_evidence_files_by_item(self, usage_statement_id: int) -> dict[int, list[dict]]:
        """Return evidence-file rows grouped by item id."""
        ...

    def list_evidence_requirements_by_item(self, usage_statement_id: int) -> dict[int, list[dict]]:
        """Return evidence-requirement rows grouped by item id."""
        ...

    def list_validation_logs_by_item(self, usage_statement_id: int) -> dict[int, list[dict]]:
        """Return validation-log rows grouped by item id."""
        ...

    def list_action_requests(self, project_id: int, usage_statement_id: int) -> list[dict]:
        """Return action request rows for the report scope."""
        ...


def build_report_context(
    repo: ReportRepository,
    *,
    project_id: int,
    usage_statement_id: int,
    report_no: str,
    report_written_date: date,
    report_period_label: str,
    reviewer: ReviewerContext | None = None,
) -> ReportContext:
    """관련 DB row를 모아 ReportContext로 정규화합니다."""

    project_row = repo.get_project(project_id)
    statement_row = repo.get_usage_statement(usage_statement_id)
    categories = {row["code"]: row["name"] for row in repo.list_usage_categories()}
    summaries = repo.list_usage_statement_summaries(usage_statement_id)
    item_rows = repo.list_usage_statement_items(usage_statement_id)
    files_by_item = defaultdict(list, repo.list_evidence_files_by_item(usage_statement_id))
    requirements_by_item = defaultdict(list, repo.list_evidence_requirements_by_item(usage_statement_id))
    logs_by_item = defaultdict(list, repo.list_validation_logs_by_item(usage_statement_id))
    action_rows = repo.list_action_requests(project_id, usage_statement_id)

    items = [
        UsageStatementItemContext(
            **row,
            # DB row에는 카테고리 코드가 있고, 보고서 초안에는 표시명이 필요합니다.
            category_name=categories.get(row["category_code"], row["category_code"]),
            evidence_files=[EvidenceFileContext(**file_row) for file_row in files_by_item[row["id"]]],
            evidence_requirements=[EvidenceRequirementContext(**req_row) for req_row in requirements_by_item[row["id"]]],
            validation_logs=[ValidationLogContext(**log_row) for log_row in logs_by_item[row["id"]]],
        )
        for row in item_rows
    ]

    return ReportContext(
        project=ProjectContext(**project_row),
        usage_statement=UsageStatementContext(**statement_row),
        summaries=[
            UsageCategorySummary(
                **row,
                category_name=categories.get(row["category_code"], row["category_code"]),
                item_count=sum(1 for item in item_rows if item["category_code"] == row["category_code"]),
            )
            for row in summaries
        ],
        items=items,
        action_requests=[ActionRequestContext(**row) for row in action_rows],
        reviewer=reviewer,
        report_no=report_no,
        report_written_date=report_written_date,
        report_period_label=report_period_label,
    )


def decimal_ratio(numerator: Decimal, denominator: Decimal) -> Decimal:
    """Run decimal ratio."""
    if denominator == 0:
        return Decimal("0")
    return (numerator / denominator * Decimal("100")).quantize(Decimal("0.1"))
