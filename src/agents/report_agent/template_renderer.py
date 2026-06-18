# --------------------------------------------------------------------------
# 작성자   : 차현주
# 작성일   : 2026-06-18
# 수정일   : 2026-06-18
#
# [ 주요 클래스 정의 ]
#
# 1. ReportTemplateRenderer : report_template.json을 ReportDraft 값으로 렌더링
# --------------------------------------------------------------------------
from __future__ import annotations

import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

from .schemas import ReportDraft, ReportSectionDraft, ReportTableDraft


TEMPLATE_TOKEN_PATTERN = re.compile(r"{{\s*([^}]+?)\s*}}")


class ReportTemplateRenderer:
    """Render structured report sections from a ReportDraft and JSON template."""

    def __init__(
        self,
        *,
        template_path: Path,
        money_formatter: Callable[[Decimal], str],
        count_formatter: Callable[[int], str],
        department_label_formatter: Callable[[ReportDraft], str],
    ) -> None:
        """Initialize the renderer with its template path and value filters."""

        self.template_path = template_path
        self.money_formatter = money_formatter
        self.count_formatter = count_formatter
        self.department_label_formatter = department_label_formatter

    def build_report_sections(self, draft: ReportDraft) -> list[ReportSectionDraft]:
        """Fill the JSON template with ReportDraft values."""

        template = self._load_report_template()
        data = draft.model_dump(mode="python")
        data["department_full_label"] = self.department_label_formatter(draft)

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
        """Load the report template JSON from disk."""

        return json.loads(self.template_path.read_text(encoding="utf-8"))

    def _render_table_templates(self, table_template: dict[str, Any], context: dict[str, Any]) -> list[ReportTableDraft]:
        """Render one table template, expanding repeated rows or tables."""

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
        """Add display-only fields used by repeated table templates."""

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
        """Render all template tokens in one text value."""

        def replace(match) -> str:
            """Render one template token match."""

            expression = match.group(1)
            path, *filters = [part.strip() for part in expression.split("|")]
            value = self._get_template_value(context, path)
            for filter_name in filters:
                value = self._apply_template_filter(value, filter_name)
            return "-" if value in (None, "") else str(value)

        return TEMPLATE_TOKEN_PATTERN.sub(replace, template_text)

    def _get_template_value(self, context: dict[str, Any], path: str) -> Any:
        """Resolve a dotted template path against dicts or objects."""

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
        """Apply a named template filter to a resolved value."""

        if filter_name == "money":
            try:
                return self.money_formatter(Decimal(str(value)))
            except Exception:
                return value
        if filter_name == "count":
            try:
                return self.count_formatter(int(value))
            except Exception:
                return value
        return value
