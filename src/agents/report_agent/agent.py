from __future__ import annotations

"""ReportContext에서 ReportDraft JSON을 생성합니다.

먼저 LLM 없이 입력 데이터를 보고서 표와 이슈 목록으로 바꿉니다. 이 단계에서
금액 합계, 판정 라벨, 법령 근거, 증빙 유형별 건수처럼 틀리면 안 되는 값을
확정합니다. 그 다음 LLM은 결론, 종합 의견, 필요 조치 같은 문장만 다듬습니다.
"""

import json
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

from .context_builder import decimal_ratio
from .llm import OpenAIReportLLMClient, ReportLLMError
from .schemas import (
    AmountSummaryDraft,
    CategorySummaryDraft,
    DecisionCode,
    EvidenceValidationSummaryDraft,
    IssueDetailDraft,
    ItemReviewDraft,
    LegalCitationDraft,
    ReportContext,
    ReportDraft,
    ReportSectionDraft,
    ReportTableDraft,
    SupplementActionDraft,
    TaxSettlementRowDraft,
)


LLMClient = Callable[[str, dict], dict]
REPORT_TEMPLATE_PATH = Path(__file__).parent / "templates" / "report_template.json"
TEMPLATE_TOKEN_PATTERN = re.compile(r"{{\s*([^}]+?)\s*}}")


RESULT_TO_DECISION: dict[str, DecisionCode] = {
    "ok": "appropriate",
    "appropriate": "appropriate",
    "pass": "appropriate",
    "warning": "needs_review",
    "needs_review": "needs_review",
    "conditional": "needs_review",
    "error": "inappropriate",
    "fail": "inappropriate",
    "inappropriate": "inappropriate",
    "적절": "appropriate",
    "부적절": "inappropriate",
    "검토필요": "needs_review",
}

DECISION_LABEL = {
    "appropriate": "적정",
    "needs_review": "검토필요",
    "inappropriate": "부적정",
}


class ReportAgent:
    """선택적 기본 LLM 문장 보강을 포함한 보고서 초안 생성기입니다."""

    def __init__(self, llm_client: LLMClient | None = None, *, use_default_llm: bool = True) -> None:
        self.llm_client = llm_client
        if self.llm_client is None and use_default_llm:
            self.llm_client = OpenAIReportLLMClient.from_environment()

    def generate(self, context: ReportContext) -> ReportDraft:
        """보고서 초안을 만들고, 필요하면 LLM 문장을 병합합니다."""

        base_draft = self._build_base_draft(context)
        if not self.llm_client:
            return base_draft

        try:
            llm_payload = self.llm_client(
                "report_draft",
                {
                    "context": context.model_dump(mode="json"),
                    "draft": base_draft.model_dump(mode="json"),
                },
            )
        except Exception as exc:
            return self._with_llm_failure_note(base_draft, exc)
        return self._merge_llm_text(base_draft, llm_payload)

    def _build_base_draft(self, context: ReportContext) -> ReportDraft:
        """LLM 없이 입력 데이터만 사용해 보고서 초안의 표와 이슈를 만듭니다."""

        project = context.project
        statement = context.usage_statement
        reviewer = context.reviewer
        reviewer_label = "미지정"
        department_label = project.construction_company
        if reviewer:
            reviewer_label = " ".join(part for part in [reviewer.name, reviewer.department, reviewer.title] if part)
            department_label = reviewer.department or department_label

        current_total = sum((summary.current_amount for summary in context.summaries), Decimal("0"))
        cumulative_total = sum((summary.cumulative_amount for summary in context.summaries), Decimal("0"))
        remaining_amount = project.appropriated_amount - cumulative_total
        item_count = len(context.items)

        item_reviews = [self._build_item_review(index, item) for index, item in enumerate(context.items, start=1)]
        issues = self._build_issues(item_reviews, context)
        evidence_summaries = self._build_evidence_summaries(context)
        supplement_actions = self._build_supplement_actions(context, issues)

        conclusion = self._build_conclusion(issues)
        overall_opinion = self._build_overall_opinion(
            context=context,
            current_total=current_total,
            item_count=item_count,
            issue_count=len(issues),
        )

        draft = ReportDraft(
            report_no=context.report_no,
            site_name=project.project_name,
            report_period_label=context.report_period_label,
            written_date_label=self._date_label(context.report_written_date),
            department_label=department_label,
            reviewer_label=reviewer_label,
            basic_info={
                "보고서 번호": context.report_no,
                "검토 일자": self._date_label(context.report_written_date),
                "현장명": project.project_name,
                "현장코드": str(project.id),
                "발주처": project.client_name or "",
                "시공사": project.construction_company,
                "계약금액": self._money(project.contract_amount),
                "공사기간": f"{project.construction_start_date:%Y.%m.%d} ~ {project.construction_end_date:%Y.%m.%d}",
                "검토 대상 기간": context.report_period_label,
                "검토자": reviewer_label,
                "법정 계상액": self._money(project.appropriated_amount),
                "검토 목적": "산업안전보건관리비 집행 증빙 적정성 확인",
            },
            amount_summary=[
                AmountSummaryDraft(label="법정 계상액 (기준)", amount=project.appropriated_amount, ratio_label="100.0%", count_label="-"),
                AmountSummaryDraft(label="당기 집행 누계액", amount=cumulative_total, ratio_label=f"{decimal_ratio(cumulative_total, project.appropriated_amount)}%", count_label=f"{item_count}건"),
                AmountSummaryDraft(label="잔액 (미집행)", amount=remaining_amount, ratio_label=f"{decimal_ratio(remaining_amount, project.appropriated_amount)}%", count_label="-"),
                AmountSummaryDraft(label="이번 검토 대상", amount=current_total, ratio_label=f"{decimal_ratio(current_total, project.appropriated_amount)}%", count_label=f"{item_count}건"),
            ],
            category_summaries=[
                CategorySummaryDraft(
                    category_code=summary.category_code,
                    category_name=summary.category_name,
                    amount=summary.current_amount,
                    count=summary.item_count,
                    note=self._category_note(summary.category_code, item_reviews),
                )
                for summary in context.summaries
            ],
            evidence_validation_summaries=evidence_summaries,
            tax_settlement_rows=self._build_tax_rows(context),
            conclusion=conclusion,
            item_reviews=item_reviews,
            issue_details=issues,
            supplement_actions=supplement_actions,
            overall_opinion=overall_opinion,
            needs_human_review=self._find_missing_critical_inputs(context),
        )
        return draft.model_copy(update={"report_sections": self._build_report_sections(draft)})

    def _build_item_review(self, no: int, item) -> ItemReviewDraft:
        worst_result = "ok"
        reasons: list[str] = []

        # 분류/법령 agent가 이미 판단한 결과이므로 여기서는 그대로 보고서 판정에 반영합니다.
        if item.classification_result:
            if item.classification_result.status == "검토필요":
                worst_result = "needs_review"
            if item.classification_result.status == "카테고리변경":
                reasons.append(item.classification_result.reason or "분류 agent가 카테고리 변경 필요성을 확인함")

        if item.legal_validation_result:
            decision = RESULT_TO_DECISION.get(item.legal_validation_result.status, "needs_review")
            if decision == "inappropriate":
                worst_result = "inappropriate"
            elif decision == "needs_review" and worst_result != "inappropriate":
                worst_result = "needs_review"
            if item.legal_validation_result.reason:
                reasons.append(item.legal_validation_result.reason)

        for log in item.validation_logs:
            decision = RESULT_TO_DECISION.get(log.result_code, "needs_review")
            if decision == "inappropriate":
                worst_result = "inappropriate"
            elif decision == "needs_review" and worst_result != "inappropriate":
                worst_result = "needs_review"
            summary = log.details.get("summary") or log.details.get("reason")
            if summary:
                reasons.append(str(summary))

        missing = [req.evidence_type_code for req in item.evidence_requirements if not req.is_satisfied]
        if missing and worst_result == "appropriate":
            worst_result = "needs_review"
            reasons.append(f"필수 증빙 미충족: {', '.join(map(str, missing))}")

        decision = RESULT_TO_DECISION.get(worst_result, "appropriate")
        return ItemReviewDraft(
            no=no,
            usage_statement_item_id=item.id,
            category_code=item.category_code,
            item_name=item.item_name,
            amount=item.total_amount,
            decision=decision,
            decision_label=DECISION_LABEL[decision],
            summary_reason=reasons[0] if reasons else "제출 증빙과 사용내역서 기준 검토 결과 특이사항 없음",
            risk_level="high" if decision == "inappropriate" else "medium" if decision == "needs_review" else "low",
        )

    def _build_issues(self, item_reviews: list[ItemReviewDraft], context: ReportContext) -> list[IssueDetailDraft]:
        items_by_id = {item.id: item for item in context.items}
        issues: list[IssueDetailDraft] = []
        for review in item_reviews:
            if review.decision == "appropriate":
                continue
            item = items_by_id[review.usage_statement_item_id]
            legal_citations = self._build_legal_citations(item)
            basis = self._legal_basis_label(legal_citations) or self._first_detail_value(item.validation_logs, ["legal_basis", "law_basis", "basis", "조항"]) or "검증 로그의 판정 근거 확인 필요"
            action = self._first_detail_value(item.validation_logs, ["required_action", "action", "recommendation", "필요조치", "필요 조치"])
            problem = self._issue_problem(item, review.summary_reason)
            issues.append(
                IssueDetailDraft(
                    issue_type="inappropriate" if review.decision == "inappropriate" else "needs_review",
                    no=review.no,
                    usage_statement_item_id=review.usage_statement_item_id,
                    title=item.item_name,
                    amount_label=f"{self._money(item.total_amount)} ({item.used_on:%Y.%m.%d})",
                    problem=problem,
                    legal_basis=str(basis),
                    legal_citations=legal_citations,
                    site_claim=None,
                    agent_conclusion=review.summary_reason,
                    required_action_fact=str(action) if action else None,
                    required_action=str(action) if action else "담당자 검토 후 증빙 보완 또는 사용내역 정정 필요",
                )
            )
        return issues

    def _build_evidence_summaries(self, context: ReportContext) -> list[EvidenceValidationSummaryDraft]:
        """증빙을 유형별로 집계하되 기타 서류는 세부명으로 분리합니다."""

        submitted: Counter[str] = Counter()
        missing: Counter[str] = Counter()
        errors: Counter[str] = Counter()
        major_errors: dict[str, list[str]] = {}
        for item in context.items:
            for file in item.evidence_files:
                submitted[self._evidence_summary_key(str(file.evidence_type_code), file.original_filename, file.evidence_detail_name)] += 1
            for req in item.evidence_requirements:
                if not req.is_satisfied:
                    missing[self._evidence_summary_key(str(req.evidence_type_code))] += 1
            for log in item.validation_logs:
                if RESULT_TO_DECISION.get(log.result_code) in {"needs_review", "inappropriate"}:
                    evidence_type = log.details.get("evidence_type_code")
                    if not evidence_type and not self._is_evidence_validation_log(log):
                        continue
                    evidence_key = self._evidence_summary_key(
                        str(evidence_type or "other"),
                        log.details.get("original_filename") or log.details.get("filename"),
                        self._evidence_detail_from_log(log),
                    )
                    errors[evidence_key] += 1
                    major_errors.setdefault(evidence_key, []).append(str(log.details.get("summary") or log.details.get("reason") or log.result_code))

        evidence_types = sorted(set(submitted) | set(missing) | set(errors) | {"usage_statement"}, key=self._evidence_sort_key)
        return [
            EvidenceValidationSummaryDraft(
                evidence_type_code=code,
                evidence_type_name=self._evidence_type_name(code),
                submitted_count=submitted[code],
                passed_count=max(submitted[code] - errors[code], 0),
                error_count=errors[code],
                missing_count=missing[code],
                major_error=major_errors.get(code, ["-"])[0],
            )
            for code in evidence_types
        ]

    def _build_tax_rows(self, context: ReportContext) -> list[TaxSettlementRowDraft]:
        rows: list[TaxSettlementRowDraft] = []
        for item in context.items:
            tax_logs = [log for log in item.validation_logs if "tax" in log.validation_type_code.lower() or log.details.get("evidence_type_code") == "tax_invoice"]
            for log in tax_logs:
                document_amount = Decimal(str(log.details.get("document_supply_amount", item.total_amount)))
                execution_amount = Decimal(str(log.details.get("execution_supply_amount", item.total_amount)))
                vat_amount = Decimal(str(log.details.get("vat_amount", 0)))
                diff = execution_amount - document_amount
                rows.append(
                    TaxSettlementRowDraft(
                        item_name=item.item_name,
                        document_supply_amount=document_amount,
                        execution_supply_amount=execution_amount,
                        vat_amount=vat_amount,
                        difference_label="일치" if diff == 0 else f"{diff:+,.0f}원 차이",
                    )
                )
        return rows

    def _build_supplement_actions(self, context: ReportContext, issues: list[IssueDetailDraft]) -> list[SupplementActionDraft]:
        if context.action_requests:
            return [
                SupplementActionDraft(
                    no=index,
                    title=request.title,
                    action=request.reason or request.title,
                    due_date_label=request.due_date.strftime("%Y.%m.%d") if request.due_date else "미지정",
                    assignee=request.assignee_name or "미지정",
                )
                for index, request in enumerate(context.action_requests, start=1)
            ]
        return [
            SupplementActionDraft(
                no=index,
                title=issue.title,
                action=issue.required_action or "보완 필요",
                due_date_label="담당자 지정 필요",
                assignee="미지정",
            )
            for index, issue in enumerate(issues, start=1)
        ]

    def _build_conclusion(self, issues: list[IssueDetailDraft]) -> str:
        if not issues:
            return "제출된 사용내역서와 증빙자료 검토 결과, 중대한 부적정 또는 보완 필요 사항은 확인되지 않았습니다."
        first = issues[0]
        return f"{first.title} 건에서 {first.problem or first.agent_conclusion} 사항이 확인되어 보완 또는 정정 처리가 필요합니다."

    def _build_overall_opinion(self, *, context: ReportContext, current_total: Decimal, item_count: int, issue_count: int) -> str:
        return (
            f"이번 검토 대상 기간({context.report_period_label}) 산업안전보건관리비 집행 내역 "
            f"{item_count}건, {self._money(current_total)}에 대한 증빙 검토를 완료했습니다. "
            f"검토 결과 {issue_count}건의 보완 또는 검토 필요 사항이 확인되었습니다. "
            "본 보고서는 시스템 검증 결과를 바탕으로 자동 생성된 초안이며, 최종 판단 및 결재는 담당자 확인을 통해 확정해야 합니다."
        )

    def _merge_llm_text(self, draft: ReportDraft, payload: dict) -> ReportDraft:
        """LLM 응답 중 결론/종합의견/필요조치 문장만 골라서 반영합니다."""

        allowed_text_fields = {"conclusion", "overall_opinion"}
        update = {}
        for field in allowed_text_fields:
            value = payload.get(field)
            if isinstance(value, str) and value.strip():
                update[field] = value.strip()

        issues = self._merge_llm_issues(draft, payload.get("issue_details"))
        if issues is not draft.issue_details:
            update["issue_details"] = issues

        actions = self._merge_llm_supplement_actions(draft, payload.get("supplement_actions"))
        if actions is not draft.supplement_actions:
            update["supplement_actions"] = actions

        merged = draft.model_copy(update=update)
        return merged.model_copy(update={"report_sections": self._build_report_sections(merged)})

    def _build_report_sections(self, draft: ReportDraft) -> list[ReportSectionDraft]:
        """구조 기반 JSON 템플릿을 ReportDraft 값으로 채워 report_sections를 만듭니다."""

        template = self._load_report_template()
        data = draft.model_dump(mode="python")
        data["department_full_label"] = self._department_full_label(draft)

        sections: list[ReportSectionDraft] = []
        for section_template in template.get("sections", []):
            section_context = {"draft": data}
            tables: list[ReportTableDraft] = []
            for table_template in section_template.get("tables", []):
                tables.extend(self._render_table_templates(table_template, section_context))
            sections.append(
                ReportSectionDraft(
                    section_id=str(section_template["section_id"]),
                    title=self._render_template_text(str(section_template["title"]), section_context),
                    kind=section_template["kind"],
                    paragraphs=[
                        self._render_template_text(str(paragraph), section_context)
                        for paragraph in section_template.get("paragraphs", [])
                    ],
                    tables=tables,
                )
            )
        return sections

    def _load_report_template(self) -> dict[str, Any]:
        return json.loads(REPORT_TEMPLATE_PATH.read_text(encoding="utf-8"))

    def _render_table_templates(self, table_template: dict[str, Any], context: dict[str, Any]) -> list[ReportTableDraft]:
        repeat_for = table_template.get("repeat_for")
        repeat_mode = table_template.get("repeat_mode", "rows")
        if repeat_for and repeat_mode == "tables":
            items = self._get_template_value(context, f"draft.{repeat_for}")
            if not isinstance(items, list) or not items:
                return [
                    ReportTableDraft(
                        title=self._render_template_text(str(table_template.get("empty_title", table_template.get("title", "-"))), context),
                        headers=[str(header) for header in table_template.get("headers", [])],
                        rows=[
                            [self._render_template_text(str(cell), context) for cell in row]
                            for row in table_template.get("empty_rows", [["-"]])
                        ],
                    )
                ]
            rendered_tables: list[ReportTableDraft] = []
            for index, item in enumerate(items, start=1):
                item_context = {**context, "item": self._augment_repeat_item(str(repeat_for), item, index), "index": index}
                rendered_tables.append(
                    ReportTableDraft(
                        title=self._render_template_text(str(table_template.get("title", "")), item_context) or None,
                        headers=[str(header) for header in table_template.get("headers", [])],
                        rows=[
                            [self._render_template_text(str(cell), item_context) for cell in row]
                            for row in table_template.get("rows", [])
                        ],
                    )
                )
            return rendered_tables

        rows = [
            [self._render_template_text(str(cell), context) for cell in row]
            for row in table_template.get("rows", [])
        ]
        if repeat_for:
            items = self._get_template_value(context, f"draft.{repeat_for}")
            if isinstance(items, list) and items:
                for index, item in enumerate(items, start=1):
                    item_context = {**context, "item": self._augment_repeat_item(str(repeat_for), item, index), "index": index}
                    rows.append(
                        [
                            self._render_template_text(str(cell), item_context)
                            for cell in table_template.get("row_template", [])
                        ]
                    )
            else:
                rows.append([str(cell) for cell in table_template.get("empty_row", ["-"])])

        return [
            ReportTableDraft(
                title=self._render_template_text(str(table_template.get("title", "")), context) or None,
                headers=[str(header) for header in table_template.get("headers", [])],
                rows=rows,
            )
        ]

    def _augment_repeat_item(self, repeat_for: str, item: object, index: int) -> dict[str, Any]:
        item_data = item if isinstance(item, dict) else {"value": item}
        augmented = {**item_data, "index": index}
        if repeat_for == "issue_details":
            issue_type = str(item_data.get("issue_type", ""))
            augmented["issue_label"] = "부적정" if issue_type == "inappropriate" else "검토 필요"
            augmented["problem_label"] = "확인된 문제" if issue_type == "needs_review" else "집행 경위"
            augmented["decision_label"] = "부적정" if issue_type == "inappropriate" else "검토필요"
            augmented["problem_text"] = item_data.get("problem") or item_data.get("agent_conclusion") or "-"
            augmented["action_text"] = item_data.get("required_action") or item_data.get("agent_conclusion") or "-"
        return augmented

    def _render_template_text(self, template_text: str, context: dict[str, Any]) -> str:
        def replace(match) -> str:
            expression = match.group(1)
            path, *filters = [part.strip() for part in expression.split("|")]
            value = self._get_template_value(context, path)
            for filter_name in filters:
                value = self._apply_template_filter(value, filter_name)
            return "-" if value in (None, "") else str(value)

        return TEMPLATE_TOKEN_PATTERN.sub(replace, template_text)

    def _get_template_value(self, context: dict[str, Any], path: str) -> Any:
        value: Any = context
        for part in path.split("."):
            if isinstance(value, dict):
                value = value.get(part)
            else:
                value = getattr(value, part, None)
            if value is None:
                return None
        return value

    def _apply_template_filter(self, value: Any, filter_name: str) -> Any:
        if filter_name == "money":
            try:
                return self._money(Decimal(str(value)))
            except Exception:
                return value
        if filter_name == "count":
            try:
                return self._count(int(value))
            except Exception:
                return value
        return value

    def _merge_llm_issues(self, draft: ReportDraft, payload: object):
        if not isinstance(payload, list):
            return draft.issue_details

        by_no = {issue.no: issue for issue in draft.issue_details}
        updates = {}
        for row in payload:
            if not isinstance(row, dict):
                continue
            no = row.get("no")
            if no not in by_no:
                continue
            allowed = {}
            for field in ["agent_conclusion", "required_action"]:
                value = row.get(field)
                if isinstance(value, str) and value.strip():
                    allowed[field] = value.strip()
            if allowed:
                updates[no] = by_no[no].model_copy(update=allowed)

        if not updates:
            return draft.issue_details
        return [updates.get(issue.no, issue) for issue in draft.issue_details]

    def _merge_llm_supplement_actions(self, draft: ReportDraft, payload: object):
        if not isinstance(payload, list):
            return draft.supplement_actions

        by_no = {action.no: action for action in draft.supplement_actions}
        updates = {}
        for row in payload:
            if not isinstance(row, dict):
                continue
            no = row.get("no")
            if no not in by_no:
                continue
            value = row.get("action")
            if isinstance(value, str) and value.strip():
                updates[no] = by_no[no].model_copy(update={"action": value.strip()})

        if not updates:
            return draft.supplement_actions
        return [updates.get(action.no, action) for action in draft.supplement_actions]

    def _with_llm_failure_note(self, draft: ReportDraft, exc: Exception) -> ReportDraft:
        if isinstance(exc, ReportLLMError):
            message = f"LLM 문장 보강 실패: {exc}"
        else:
            message = f"LLM 문장 보강 실패: {type(exc).__name__}"
        return draft.model_copy(update={"needs_human_review": [*draft.needs_human_review, message]})

    def _find_missing_critical_inputs(self, context: ReportContext) -> list[str]:
        missing: list[str] = []
        if not context.items:
            missing.append("사용내역서 상세항목이 없습니다.")
        if not context.summaries:
            missing.append("사용내역서 카테고리 요약이 없습니다.")
        if not any(item.validation_logs for item in context.items):
            missing.append("항목별 검증 로그가 없어 판정 사유가 제한됩니다.")
        return missing

    def _category_note(self, category_code: str, item_reviews: list[ItemReviewDraft]) -> str:
        related = [review for review in item_reviews if review.category_code == category_code]
        bad = [review for review in related if review.decision == "inappropriate"]
        review = [review for review in related if review.decision == "needs_review"]
        if bad:
            return f"부적정 {len(bad)}건 포함"
        if review:
            return f"검토 필요 {len(review)}건 포함"
        return "적정"

    def _first_detail_value(self, logs, keys: list[str]) -> str | None:
        for log in logs:
            for key in keys:
                value = log.details.get(key)
                if value:
                    return json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
        return None

    def _issue_problem(self, item, fallback: str) -> str:
        if item.legal_validation_result and item.legal_validation_result.reason:
            return item.legal_validation_result.reason
        if item.classification_result and item.classification_result.reason:
            return item.classification_result.reason
        return self._first_detail_value(item.validation_logs, ["problem", "error", "reason", "사유"]) or fallback

    def _build_legal_citations(self, item) -> list[LegalCitationDraft]:
        if item.legal_validation_result and item.legal_validation_result.citations:
            return [
                LegalCitationDraft(
                    legal_basis=citation.legal_basis,
                    summary=citation.summary,
                    citation_text=citation.citation_text,
                    source_id=citation.source_id,
                    source_name=citation.source_name,
                )
                for citation in item.legal_validation_result.citations
            ]

        citations: list[LegalCitationDraft] = []
        for log in item.validation_logs:
            raw_sources = log.details.get("sources") or log.details.get("citations") or log.details.get("출처")
            if not isinstance(raw_sources, list):
                continue
            for source in raw_sources:
                if not isinstance(source, dict):
                    continue
                legal_basis = source.get("legal_basis") or source.get("조항") or source.get("article_no")
                if not legal_basis:
                    continue
                citations.append(
                    LegalCitationDraft(
                        legal_basis=str(legal_basis),
                        summary=source.get("summary") or source.get("요지"),
                        citation_text=source.get("citation_text") or source.get("인용원문"),
                        source_id=source.get("source_id"),
                        source_name=source.get("source_name"),
                    )
                )
        return citations

    def _legal_basis_label(self, citations: list[LegalCitationDraft]) -> str | None:
        if not citations:
            return None
        bases = []
        for citation in citations:
            if citation.legal_basis not in bases:
                bases.append(citation.legal_basis)
        return ", ".join(bases)

    def _date_label(self, value) -> str:
        return value.strftime("%Y년 %m월 %d일")

    def _money(self, value: Decimal) -> str:
        return f"{value:,.0f}원"

    def _count(self, value: int) -> str:
        return "-" if value == 0 else f"{value}건"

    def _department_full_label(self, draft: ReportDraft) -> str:
        company = draft.basic_info.get("시공사", "").strip()
        department = draft.department_label.strip()
        if company and department and company not in department:
            return f"{company} {department}"
        return department or company or "-"

    def _evidence_type_name(self, code: str) -> str:
        if code.startswith("other:"):
            detail = code.split(":", 1)[1]
            return f"기타 서류({detail})" if detail else "기타 서류(세부명 미확인)"
        return {
            "usage_statement": "사용내역서",
            "receipt": "영수증·거래명세서",
            "site_photo": "현장사진",
            "tax_invoice": "세금계산서",
            "other_document": "기타 서류(세부명 미확인)",
            "other": "기타 서류(세부명 미확인)",
        }.get(code, code)

    def _evidence_summary_key(self, code: str, filename: object | None = None, detail_name: object | None = None) -> str:
        """증빙 요약 표에서 사용할 집계 키를 반환합니다."""

        normalized = "other" if code in {"other", "other_document"} else code
        if normalized != "other":
            return normalized

        detail = str(detail_name).strip() if detail_name else ""
        if not detail and filename:
            detail = self._other_document_detail_from_filename(str(filename))
        return f"other:{detail or '세부명 미확인'}"

    def _is_evidence_validation_log(self, log) -> bool:
        validation_type = str(log.validation_type_code).lower()
        return any(keyword in validation_type for keyword in ["evidence", "ocr", "vision", "receipt", "tax"])

    def _evidence_detail_from_log(self, log) -> str:
        for key in [
            "evidence_detail_name",
            "document_name",
            "detail_name",
            "document_type_name",
            "서류명",
            "문서명",
        ]:
            value = log.details.get(key)
            if value:
                return str(value)
        return ""

    def _other_document_detail_from_filename(self, filename: str) -> str:
        stem = Path(filename).stem.strip()
        if not stem:
            return ""

        normalized = stem.replace("_", " ").replace("-", " ")
        known_details = [
            "지급대장",
            "건강검진 계약서",
            "건강검진계약서",
            "계약서",
            "선임계",
            "설치확인서",
            "교육 이수 자료",
            "위험성평가 결과 보고 자료",
            "이체확인증",
        ]
        for detail in known_details:
            if detail in normalized:
                return "건강검진 계약서" if detail == "건강검진계약서" else detail
        return normalized

    def _evidence_sort_key(self, code: str) -> tuple[int, str]:
        order = {
            "usage_statement": 0,
            "receipt": 1,
            "site_photo": 2,
            "tax_invoice": 3,
            "other": 4,
        }
        base = "other" if code.startswith("other:") else code
        return (order.get(base, 99), self._evidence_type_name(code))
