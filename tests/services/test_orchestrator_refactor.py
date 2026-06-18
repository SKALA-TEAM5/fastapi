from types import SimpleNamespace
from datetime import date

from src.schemas.orchestrator import UsageStatementClassifyRequest
from src.services import orchestrator_service as orchestrator


def test_orchestrator_duplicate_dict_helpers_match_current_behavior():
    payload = {"a": 1}

    assert orchestrator._dict_or_empty(payload) == payload
    assert orchestrator._as_dict(payload) == payload
    assert orchestrator._dict_or_empty(None) == {}
    assert orchestrator._as_dict(None) == {}
    assert orchestrator._dict_or_empty([("a", 1)]) == {}
    assert orchestrator._as_dict([("a", 1)]) == {}


def test_orchestrator_number_helpers_keep_numeric_contract():
    assert orchestrator._int_or_none("3") == 3
    assert orchestrator._int_or_none("3.2") is None
    assert orchestrator._number_or_none("3.2") == 3.2
    assert orchestrator._number_or_none(None) is None
    assert orchestrator._number_or_none("bad") is None


def test_legal_reason_sentence_filters_keep_expected_text():
    reason = (
        "공정률 75% 기준 별표 3 검토가 필요합니다. "
        "안전난간 설치는 사용 가능합니다. "
        "키워드가 감지되어 제한 검토 대상으로 분류되었습니다. "
        "카테고리 집행액이 법정 한도를 초과해 부적절합니다."
    )

    assert orchestrator._legal_without_progress_reason(reason) == (
        "안전난간 설치는 사용 가능합니다. "
        "키워드가 감지되어 제한 검토 대상으로 분류되었습니다. "
        "카테고리 집행액이 법정 한도를 초과해 부적절합니다."
    )
    assert orchestrator._legal_without_internal_match_reason(reason) == (
        "공정률 75% 기준 별표 3 검토가 필요합니다. "
        "안전난간 설치는 사용 가능합니다. "
        "카테고리 집행액이 법정 한도를 초과해 부적절합니다."
    )
    assert orchestrator._legal_remove_generic_limit_sentences(reason) == (
        "공정률 75% 기준 별표 3 검토가 필요합니다. "
        "안전난간 설치는 사용 가능합니다. "
        "키워드가 감지되어 제한 검토 대상으로 분류되었습니다."
    )
    assert orchestrator._legal_clean_final_reason(reason) == "안전난간 설치는 사용 가능합니다."


def test_classify_existing_usage_statement_success_shape(monkeypatch):
    calls: list[tuple[str, dict]] = []

    review = SimpleNamespace(
        row_id=1,
        final_category_code="CAT_03",
        decision_status="변경",
        reason="보호구 항목으로 판단됩니다.",
        item_name="안전모",
    )
    review_response = SimpleNamespace(results=[review])

    monkeypatch.setattr(orchestrator, "get_openai_callback", None)
    monkeypatch.setattr(
        orchestrator,
        "review_usage_statement",
        lambda **kwargs: review_response,
    )

    class FakeConnection:
        pass

    class FakeConnectionManager:
        def __enter__(self):
            return FakeConnection()

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(orchestrator, "get_connection", lambda: FakeConnectionManager())
    monkeypatch.setattr(
        orchestrator,
        "insert_usage_statement_item",
        lambda conn, **kwargs: 77,
    )
    monkeypatch.setattr(
        orchestrator,
        "upsert_agent_log",
        lambda **kwargs: calls.append(("upsert", kwargs)),
    )
    monkeypatch.setattr(
        orchestrator,
        "mark_orchestrator",
        lambda **kwargs: calls.append(("mark", kwargs)),
    )
    monkeypatch.setattr(
        orchestrator,
        "_record_agent_usage",
        lambda **kwargs: calls.append(("usage", kwargs)),
    )

    request = UsageStatementClassifyRequest(
        project_id=5,
        usage_statement_id=3,
        item_name="안전모",
        category_code="CAT_02",
        total_amount=10000,
    )

    response = orchestrator.classify_existing_usage_statement(request)

    assert response.status == "success"
    assert response.usage_statement_id == 3
    assert response.target_agents == ["classi"]
    assert response.result["event"] == "classification_updated"
    assert response.result["payload"]["changed_count"] == 1
    assert response.result["payload"]["changes"][0]["item_id"] == 77
    assert response.result["payload"]["changes"][0]["from_category_code"] == "CAT_02"
    assert response.result["payload"]["changes"][0]["to_category_code"] == "CAT_03"
    assert any(name == "upsert" for name, _ in calls)


def test_classify_existing_usage_statement_unclassified_shape(monkeypatch):
    calls: list[tuple[str, dict]] = []
    review = SimpleNamespace(
        row_id=1,
        final_category_code="CAT_02",
        decision_status="유지",
        reason="llm(unclassified): 분류 불가",
        item_name="애매한 물품",
    )

    monkeypatch.setattr(orchestrator, "get_openai_callback", None)
    monkeypatch.setattr(
        orchestrator,
        "review_usage_statement",
        lambda **kwargs: SimpleNamespace(results=[review]),
    )
    monkeypatch.setattr(
        orchestrator,
        "insert_usage_statement_item",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not insert")),
    )
    monkeypatch.setattr(orchestrator, "upsert_agent_log", lambda **kwargs: calls.append(("upsert", kwargs)))
    monkeypatch.setattr(orchestrator, "mark_orchestrator", lambda **kwargs: calls.append(("mark", kwargs)))
    monkeypatch.setattr(orchestrator, "_record_agent_usage", lambda **kwargs: calls.append(("usage", kwargs)))

    request = UsageStatementClassifyRequest(
        project_id=5,
        usage_statement_id=3,
        item_name="애매한 물품",
        category_code="CAT_02",
        item_id=88,
        total_amount=10000,
    )

    response = orchestrator.classify_existing_usage_statement(request)

    assert response.status == "success"
    assert response.result["event"] == "classification_checked"
    assert response.result["payload"]["changed_count"] == 0
    assert response.result["payload"]["kept_count"] == 1
    assert response.result["payload"]["results"][0]["reason"] == "llm(unclassified): 분류 불가"
    upsert = next(kwargs for name, kwargs in calls if name == "upsert")
    assert upsert["result_code"] == "fail"
    mark = [kwargs for name, kwargs in calls if name == "mark"][-1]
    assert mark["payload"]["item_id"] == 88


def test_classify_existing_usage_statement_failure_shape(monkeypatch):
    calls: list[tuple[str, dict]] = []

    monkeypatch.setattr(orchestrator, "get_openai_callback", None)

    def raise_review_error(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(orchestrator, "review_usage_statement", raise_review_error)
    monkeypatch.setattr(orchestrator, "upsert_agent_log", lambda **kwargs: calls.append(("upsert", kwargs)))
    monkeypatch.setattr(orchestrator, "mark_orchestrator", lambda **kwargs: calls.append(("mark", kwargs)))

    request = UsageStatementClassifyRequest(
        project_id=5,
        usage_statement_id=3,
        item_name="안전모",
        category_code="CAT_02",
        item_id=99,
    )

    response = orchestrator.classify_existing_usage_statement(request)

    assert response.status == "fail"
    assert response.target_agents == ["classi"]
    assert response.result["classi"]["status_code"] == "fail"
    assert response.result["classi"]["result_code"] == "fail"
    assert "RuntimeError" in response.message
    upsert = next(kwargs for name, kwargs in calls if name == "upsert")
    assert upsert["status_code"] == "fail"
    assert upsert["details"]["event"] == "classification_failed"
    mark = [kwargs for name, kwargs in calls if name == "mark"][-1]
    assert mark["event"] == "classi_failed"
    assert mark["payload"]["item_id"] == 99


def test_run_safety_doc_agent_builds_todos_and_usage(monkeypatch):
    calls: list[tuple[str, dict]] = []

    monkeypatch.setattr(orchestrator, "list_usage_statement_item_ids", lambda usage_statement_id: [11, 12])
    monkeypatch.setattr(
        orchestrator,
        "check_missing_evidence_batch",
        lambda item_ids: [
            {
                "evidence_status": {"missing_evidences": ["receipt"]},
                "ai_response": {"usage": {"input_tokens": 5, "output_tokens": 2, "cached_input_tokens": 1}},
                "model_name": "safety-test",
                "input_from_db_views": {
                    "item_context": {
                        "category_code": "CAT_01",
                        "category_name": "안전관리비",
                        "item_name": "안전 교육",
                    }
                },
            },
            {"evidence_status": {"missing_evidences": []}, "ai_response": {"usage": {"input_tokens": 3}}},
        ],
    )
    monkeypatch.setattr(orchestrator, "_evidence_type_name_map", lambda: {"receipt": "영수증"})
    monkeypatch.setattr(orchestrator, "upsert_agent_log", lambda **kwargs: calls.append(("upsert", kwargs)))
    monkeypatch.setattr(orchestrator, "_record_agent_usage", lambda **kwargs: calls.append(("usage", kwargs)))

    result = orchestrator._run_safety_doc_agent(project_id=5, usage_statement_id=3, requested_by_user_id=9)

    assert result["status_code"] == "success"
    assert result["result_code"] == "hil"
    assert result["reason"] == "필수 증빙 누락 항목 1건"
    assert result["todos"] == [
        {
            "usage_statement_item_id": 11,
            "category_code": "CAT_01",
            "category_name": "안전관리비",
            "usage_statement_item_name": "안전 교육",
            "title": "영수증",
            "evidence_type_code": "receipt",
            "evidence_type_codes": ["receipt"],
            "reason": "필수 증빙 누락: 영수증",
        }
    ]
    completed_log = [kwargs for name, kwargs in calls if name == "upsert"][-1]
    assert completed_log["details"]["event"] == "safety_doc_completed"
    assert completed_log["details"]["payload"]["hil_item_ids"] == [11]
    usage_call = next(kwargs for name, kwargs in calls if name == "usage")
    assert usage_call["token"] == 10
    assert usage_call["cached_input_tokens"] == 1
    assert usage_call["requested_by_user_id"] == 9


def test_run_link_agent_builds_review_todos_and_file_status(monkeypatch):
    calls: list[tuple[str, dict]] = []

    monkeypatch.setattr(
        orchestrator,
        "list_evidence_file_ids_by_type",
        lambda project_id, usage_statement_id: {"receipt": [21], "transaction_statement": [22], "tax_invoice": [23]},
    )
    monkeypatch.setattr(
        orchestrator,
        "_usage_statement_item_context_index",
        lambda usage_statement_id: {
            11: {
                "category_code": "CAT_02",
                "category_name": "안전시설비",
                "usage_statement_item_name": "안전망 설치",
            }
        },
    )
    monkeypatch.setattr(
        orchestrator,
        "run_link_pipeline",
        lambda **kwargs: {
            "summary": {"review_needed": 1, "unmatched": 1, "rejected": 0},
            "match_results": [
                {
                    "line_id": 11,
                    "match_status": "review_needed",
                    "amount_match": False,
                    "date_match": True,
                },
                {
                    "line_id": 12,
                    "match_status": "unmatched",
                    "candidate_count": 0,
                    "candidates": [],
                },
            ],
        },
    )
    monkeypatch.setattr(orchestrator, "upsert_agent_log", lambda **kwargs: calls.append(("upsert", kwargs)))
    monkeypatch.setattr(orchestrator, "update_file_statuses", lambda **kwargs: calls.append(("status", kwargs)))

    result = orchestrator._run_link_agent(project_id=5, usage_statement_id=3, requested_by_user_id=9)

    assert result["status_code"] == "success"
    assert result["result_code"] == "hil"
    assert result["reason"] == "매칭 검토 필요 2건"
    assert len(result["todos"]) == 2
    assert result["todos"][0]["usage_statement_item_id"] == 11
    assert result["todos"][0]["title"] == "증빙 매칭 검토"
    assert result["todos"][0]["reason"] == "안전망 설치 증빙 매칭: 검토 필요"
    assert result["todos"][1]["usage_statement_item_id"] == 12
    assert result["todos"][1]["match_status"] == "unmatched"
    status_call = next(kwargs for name, kwargs in calls if name == "status")
    assert status_call["file_ids"] == [21, 22, 23]
    assert status_call["status_code"] == "fail"
    completed_log = [kwargs for name, kwargs in calls if name == "upsert"][-1]
    assert completed_log["details"]["event"] == "link_completed"
    assert completed_log["details"]["payload"]["todos"] == result["todos"]


def test_build_validator_document_groups_rows(monkeypatch):
    class FakeCursor:
        def __init__(self):
            self.calls = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, query, params):
            self.calls += 1

        def fetchone(self):
            return {"appropriated_amount": "100000", "cumulative_progress_rate": "75.5"}

        def fetchall(self):
            if self.calls == 2:
                return [
                    {
                        "category_code": "CAT_02",
                        "previous_amount": "1000",
                        "current_amount": "2000",
                        "cumulative_amount": "3000",
                    }
                ]
            return [
                {
                    "id": 11,
                    "category_code": "CAT_02",
                    "used_on": date(2026, 6, 18),
                    "item_name": "안전망 설치",
                    "unit": "식",
                    "quantity": "1.5",
                    "unit_price": "2000",
                    "total_amount": "3000",
                    "remark": None,
                }
            ]

    class FakeConnection:
        def cursor(self, cursor_factory=None):
            return FakeCursor()

    class FakeConnectionManager:
        def __enter__(self):
            return FakeConnection()

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(orchestrator, "get_connection", lambda: FakeConnectionManager())

    document, category_rows = orchestrator._build_validator_document(project_id=5, usage_statement_id=3)

    assert document["사용내역서ID"] == 3
    assert document["기본정보"]["산안비총액"] == 100000
    assert document["기본정보"]["누계공정률"] == 75.5
    category = document["카테고리별데이터"][0]
    assert category["카테고리코드"] == "CAT_02"
    assert category["집계정보"]["누적사용금액"] == 3000
    assert category_rows["CAT_02"][0]["행ID"] == 11
    assert category_rows["CAT_02"][0]["사용일자"] == "2026-06-18"


def test_run_report_agent_generates_draft_and_records_usage(monkeypatch):
    calls: list[tuple[str, dict]] = []

    class FakeDraft:
        report_no = "R-3"
        site_name = "테스트 현장"
        needs_human_review = False

        def model_dump(self, mode=None):
            return {
                "report_no": self.report_no,
                "site_name": self.site_name,
                "needs_human_review": self.needs_human_review,
            }

    class FakeReportAgent:
        llm_client = SimpleNamespace(
            model="report-test",
            last_usage={"input_tokens": 9, "output_tokens": 4, "cached_input_tokens": 2},
        )

        def generate(self, context):
            calls.append(("generate", {"context": context}))
            return FakeDraft()

    class FakeRepo:
        def __init__(self, conn):
            self.conn = conn

        def get_project(self, project_id):
            return {"contract_no": "CN-5"}

        def get_usage_statement(self, usage_statement_id):
            return {"report_month": date(2026, 6, 1)}

    class FakeConnectionManager:
        def __enter__(self):
            return object()

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(orchestrator, "get_connection", lambda: FakeConnectionManager())
    monkeypatch.setattr(orchestrator, "PostgresReportRepository", FakeRepo)
    monkeypatch.setattr(orchestrator, "default_report_no", lambda contract_no, usage_statement_id, written_date: "R-3")
    monkeypatch.setattr(
        orchestrator,
        "build_report_context",
        lambda repo, **kwargs: {"context_kwargs": kwargs},
    )
    monkeypatch.setattr(orchestrator, "ReportAgent", FakeReportAgent)
    monkeypatch.setattr(orchestrator, "upsert_agent_log", lambda **kwargs: calls.append(("upsert", kwargs)))
    monkeypatch.setattr(orchestrator, "_record_agent_usage", lambda **kwargs: calls.append(("usage", kwargs)))

    result = orchestrator._run_report_agent(project_id=5, usage_statement_id=3, she_user_id=9)

    assert result["status_code"] == "success"
    assert result["result_code"] == "success"
    assert result["reason"] == "보고서 초안 생성 완료"
    assert result["result"]["reportDraft"]["report_no"] == "R-3"
    assert "run_id" in result["result"]
    generate_call = next(kwargs for name, kwargs in calls if name == "generate")
    assert generate_call["context"]["context_kwargs"]["report_period_label"] == "2026년 06월"
    completed_log = [kwargs for name, kwargs in calls if name == "upsert"][-1]
    assert completed_log["details"]["event"] == "report_completed"
    assert completed_log["details"]["payload"]["reportDraft"]["site_name"] == "테스트 현장"
    assert completed_log["model_name"] == "report-test"
    usage_call = next(kwargs for name, kwargs in calls if name == "usage")
    assert usage_call["model_name"] == "report-test"
    assert usage_call["input_tokens"] == 9
    assert usage_call["output_tokens"] == 4
    assert usage_call["cached_input_tokens"] == 2
    assert usage_call["requested_by_user_id"] == 9
