from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.core.metrics import (
    AGENT_REVIEW_ITEMS,
    AGENT_RUNS,
    AGENT_RUNS_IN_PROGRESS,
    AGENT_TOKENS,
    SAFETY_DOC_BATCH_SIZE,
    SAFETY_DOC_CONFIDENCE,
    SAFETY_DOC_LLM_FAILURES,
    SAFETY_DOC_MISSING_EVIDENCE,
    SAFETY_DOC_RUNS,
    SAFETY_DOC_TOKENS,
    configure_metrics,
    record_agent_tokens,
    track_agent_run,
)


def test_metrics_endpoint_exposes_http_metrics() -> None:
    app = FastAPI()
    configure_metrics(app)

    @app.get("/items/{item_id}")
    def get_item(item_id: int) -> dict[str, int]:
        return {"item_id": item_id}

    client = TestClient(app)
    assert client.get("/items/42").status_code == 200

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "fastapi_http_requests_total" in response.text
    assert 'method="GET",path="/items/{item_id}",status="200"' in response.text
    assert "fastapi_http_request_duration_seconds_bucket" in response.text


def test_metrics_use_unmatched_path_for_404_requests() -> None:
    app = FastAPI()
    configure_metrics(app)
    client = TestClient(app)

    assert client.get("/not-found").status_code == 404
    response = client.get("/metrics")

    assert 'method="GET",path="unmatched",status="404"' in response.text


def test_metrics_endpoint_exposes_safety_doc_metrics() -> None:
    app = FastAPI()
    configure_metrics(app)
    SAFETY_DOC_RUNS.labels(mode="batch", result="success").inc()
    SAFETY_DOC_LLM_FAILURES.labels(mode="batch").inc()
    SAFETY_DOC_MISSING_EVIDENCE.labels(evidence_type="receipt").inc()
    SAFETY_DOC_BATCH_SIZE.observe(3)
    SAFETY_DOC_CONFIDENCE.labels(mode="batch").observe(0.8)
    SAFETY_DOC_TOKENS.labels(model="test-model", type="input_tokens").inc(10)

    response = TestClient(app).get("/metrics")

    assert response.status_code == 200
    assert 'safety_doc_runs_total{mode="batch",result="success"}' in response.text
    assert 'safety_doc_llm_failures_total{mode="batch"}' in response.text
    assert 'safety_doc_missing_evidence_total{evidence_type="receipt"}' in response.text
    assert "safety_doc_batch_size_bucket" in response.text
    assert 'safety_doc_confidence_count{mode="batch"}' in response.text
    assert (
        'safety_doc_tokens_total{model="test-model",type="input_tokens"}'
        in response.text
    )


def test_agent_metrics_track_result_duration_review_items_and_tokens() -> None:
    @track_agent_run("legal")
    def run_agent() -> dict:
        return {
            "status_code": "success",
            "result_code": "hil",
            "todos": [{"reason": "review"}, {"reason": "review"}],
        }

    runs_before = AGENT_RUNS.labels(agent="legal", result="hil")._value.get()
    reviews_before = AGENT_REVIEW_ITEMS.labels(agent="legal")._value.get()

    result = run_agent()
    record_agent_tokens(
        agent="legal",
        model="test-model",
        token_type="input",
        value=12,
    )

    assert result["result_code"] == "hil"
    assert AGENT_RUNS.labels(agent="legal", result="hil")._value.get() == runs_before + 1
    assert AGENT_REVIEW_ITEMS.labels(agent="legal")._value.get() == reviews_before + 2
    assert AGENT_RUNS_IN_PROGRESS.labels(agent="legal")._value.get() == 0
    assert AGENT_TOKENS.labels(
        agent="legal",
        model="test-model",
        type="input",
    )._value.get() >= 12

    app = FastAPI()
    configure_metrics(app)
    response = TestClient(app).get("/metrics")
    assert 'ai_agent_runs_total{agent="legal",result="hil"}' in response.text
    assert "ai_agent_run_duration_seconds_bucket" in response.text
    assert 'ai_agent_review_items_total{agent="legal"}' in response.text
    assert (
        'ai_agent_tokens_total{agent="legal",model="test-model",type="input"}'
        in response.text
    )


def test_agent_metrics_record_exception_as_failure() -> None:
    @track_agent_run("vision")
    def run_agent() -> None:
        raise RuntimeError("vision unavailable")

    failures_before = AGENT_RUNS.labels(agent="vision", result="fail")._value.get()

    try:
        run_agent()
    except RuntimeError:
        pass
    else:
        raise AssertionError("The wrapped exception must be re-raised")

    assert AGENT_RUNS.labels(agent="vision", result="fail")._value.get() == failures_before + 1
    assert AGENT_RUNS_IN_PROGRESS.labels(agent="vision")._value.get() == 0
