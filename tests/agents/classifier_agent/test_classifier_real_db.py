"""Integration tests for classi against the local port-forwarded services."""

import pytest

from langchain_openai import ChatOpenAI

from src.agents.classifier_agent.agent import review_usage_statement
from src.core import llm_config

from .test_classifier_from_db import (
    _aggregate_agent_log_payload,
    _result_details,
    fetch_usage_statement_rows,
)


def test_usage_statement_3_classi_dry_run_matches_expected_db_case():
    """Run classi for project 5 / usage statement 3 without writing logs."""
    try:
        data = fetch_usage_statement_rows(3)
    except Exception as exc:
        pytest.skip(f"local PostgreSQL port-forward is unavailable: {exc}")

    assert data["project_id"] == 5
    assert data["usage_statement_id"] == 3
    assert len(data["rows"]) == 9

    try:
        llm_config.configure(ChatOpenAI(model="gpt-4.1-mini", temperature=0))
    except Exception:
        # The classifier has a non-writing fallback path when LLM setup is unavailable.
        pass

    response = review_usage_statement(
        usage_statement_id=data["usage_statement_id"],
        rows=data["rows"],
        basic_info=data["basic_info"],
        model_name="gpt-4.1-mini",
    )
    details = _result_details(response.results)
    agent_log = _aggregate_agent_log_payload(
        project_id=data["project_id"],
        usage_statement_id=data["usage_statement_id"],
        details=details,
    )

    changed_count = details["payload"]["changed_count"]
    kept_count = details["payload"]["kept_count"]

    assert details["event"] == (
        "classification_updated" if changed_count else "classification_checked"
    )
    assert details["summary"] == (
        f"세부내역 {changed_count}건을 올바른 항목으로 이동했습니다."
        if changed_count
        else "세부내역 분류 이동 없음"
    )
    assert changed_count + kept_count == 9
    assert len(details["payload"]["changes"]) == changed_count
    assert len(details["payload"]["results"]) == 9
    assert {
        row["row_id"] for row in details["payload"]["results"]
    } == {19, 20, 21, 22, 23, 24, 25, 26, 27}
    assert all(row["final_category_code"] for row in details["payload"]["results"])
    assert agent_log["project_id"] == 5
    assert agent_log["usage_statement_id"] == 3
    assert agent_log["agent_type_code"] == "classi"
    assert agent_log["status_code"] == "success"
    assert agent_log["result_code"] == "success"
    assert agent_log["reason"] == details["summary"]
