from __future__ import annotations

from collections.abc import Callable
from functools import wraps
import time
from typing import Any, ParamSpec, TypeVar

from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.types import ASGIApp, Message, Receive, Scope, Send

HTTP_REQUESTS = Counter(
    "fastapi_http_requests_total",
    "Total number of HTTP requests handled by FastAPI.",
    ("method", "path", "status"),
)
HTTP_REQUEST_DURATION = Histogram(
    "fastapi_http_request_duration_seconds",
    "FastAPI HTTP request duration in seconds.",
    ("method", "path"),
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600),
)
HTTP_REQUESTS_IN_PROGRESS = Gauge(
    "fastapi_http_requests_in_progress",
    "Number of FastAPI HTTP requests currently being processed.",
)
SAFETY_DOC_RUNS = Counter(
    "safety_doc_runs_total",
    "Total number of Safety Doc agent runs.",
    ("mode", "result"),
)
SAFETY_DOC_INFERENCE_DURATION = Histogram(
    "safety_doc_inference_duration_seconds",
    "Safety Doc LLM inference duration in seconds.",
    ("mode", "model"),
    buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300),
)
SAFETY_DOC_LLM_FAILURES = Counter(
    "safety_doc_llm_failures_total",
    "Total number of Safety Doc LLM inference failures.",
    ("mode",),
)
SAFETY_DOC_REFERENCE_FAILURES = Counter(
    "safety_doc_reference_failures_total",
    "Total number of Safety Doc reference search failures.",
    ("mode",),
)
SAFETY_DOC_MISSING_EVIDENCE = Counter(
    "safety_doc_missing_evidence_total",
    "Total number of missing evidences detected by type.",
    ("evidence_type",),
)
SAFETY_DOC_BATCH_SIZE = Histogram(
    "safety_doc_batch_size",
    "Number of usage statement items in a Safety Doc batch.",
    buckets=(1, 2, 5, 10, 20, 50, 100, 200),
)
SAFETY_DOC_CONFIDENCE = Histogram(
    "safety_doc_confidence",
    "Confidence reported by the Safety Doc model.",
    ("mode",),
    buckets=(0, 0.25, 0.5, 0.7, 0.8, 0.9, 0.95, 1),
)
SAFETY_DOC_TOKENS = Counter(
    "safety_doc_tokens_total",
    "Total number of tokens used by the Safety Doc model.",
    ("model", "type"),
)
AGENT_RUNS = Counter(
    "ai_agent_runs_total",
    "Total number of orchestrated AI agent runs.",
    ("agent", "result"),
)
AGENT_RUN_DURATION = Histogram(
    "ai_agent_run_duration_seconds",
    "Orchestrated AI agent run duration in seconds.",
    ("agent",),
    buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600),
)
AGENT_RUNS_IN_PROGRESS = Gauge(
    "ai_agent_runs_in_progress",
    "Number of orchestrated AI agent runs currently in progress.",
    ("agent",),
)
AGENT_REVIEW_ITEMS = Counter(
    "ai_agent_review_items_total",
    "Total number of human review items produced by AI agents.",
    ("agent",),
)
AGENT_TOKENS = Counter(
    "ai_agent_tokens_total",
    "Total number of tokens used by orchestrated AI agents.",
    ("agent", "model", "type"),
)

P = ParamSpec("P")
R = TypeVar("R")


def track_agent_run(agent: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """오케스트레이터 Agent 함수의 실행 결과와 지연 시간을 공통 계측한다."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            started_at = start_agent_run(agent)
            try:
                result = func(*args, **kwargs)
            except Exception:
                finish_agent_run(agent=agent, started_at=started_at, result="fail")
                raise
            else:
                finish_agent_run(
                    agent=agent,
                    result=_agent_result_code(result),
                    started_at=started_at,
                    review_items=_agent_review_item_count(result),
                )
                return result

        return wrapper

    return decorator


def start_agent_run(agent: str) -> float:
    AGENT_RUNS_IN_PROGRESS.labels(agent=agent).inc()
    return time.perf_counter()


def finish_agent_run(
    *,
    agent: str,
    started_at: float,
    result: str,
    review_items: int = 0,
) -> None:
    AGENT_RUNS.labels(agent=agent, result=result).inc()
    AGENT_RUN_DURATION.labels(agent=agent).observe(time.perf_counter() - started_at)
    if review_items > 0:
        AGENT_REVIEW_ITEMS.labels(agent=agent).inc(review_items)
    AGENT_RUNS_IN_PROGRESS.labels(agent=agent).dec()


def record_agent_run(
    *,
    agent: str,
    result: str,
    duration_seconds: float = 0,
    review_items: int = 0,
) -> None:
    AGENT_RUNS.labels(agent=agent, result=result).inc()
    AGENT_RUN_DURATION.labels(agent=agent).observe(max(duration_seconds, 0))
    if review_items > 0:
        AGENT_REVIEW_ITEMS.labels(agent=agent).inc(review_items)


def record_agent_tokens(
    *,
    agent: str,
    model: str | None,
    token_type: str,
    value: int | None,
) -> None:
    if isinstance(value, int) and value >= 0:
        AGENT_TOKENS.labels(
            agent=agent,
            model=model or "unknown",
            type=token_type,
        ).inc(value)


def _agent_result_code(result: Any) -> str:
    if isinstance(result, dict):
        value = result.get("result_code") or result.get("status_code")
    else:
        value = getattr(result, "result_code", None) or getattr(result, "status", None)
    normalized = str(value or "unknown").strip().lower()
    if normalized == "failed":
        return "fail"
    if normalized in {"success", "hil", "fail", "skipped", "canceled"}:
        return normalized
    return "unknown"


def _agent_review_item_count(result: Any) -> int:
    if isinstance(result, dict):
        todos = result.get("todos")
    else:
        todos = getattr(result, "todos", None)
    return len(todos) if isinstance(todos, list) else 0


class PrometheusMiddleware:
    def __init__(self, app: ASGIApp, excluded_paths: set[str] | None = None) -> None:
        self.app = app
        self.excluded_paths = excluded_paths or {"/metrics"}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") in self.excluded_paths:
            await self.app(scope, receive, send)
            return

        started_at = time.perf_counter()
        status_code = 500
        HTTP_REQUESTS_IN_PROGRESS.inc()

        async def send_with_status(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_with_status)
        finally:
            route = scope.get("route")
            route_path = getattr(route, "path", None) or "unmatched"
            method = scope.get("method", "UNKNOWN")
            elapsed = time.perf_counter() - started_at

            HTTP_REQUESTS.labels(
                method=method,
                path=route_path,
                status=str(status_code),
            ).inc()
            HTTP_REQUEST_DURATION.labels(method=method, path=route_path).observe(elapsed)
            HTTP_REQUESTS_IN_PROGRESS.dec()


def configure_metrics(app: FastAPI) -> None:
    app.add_middleware(PrometheusMiddleware)

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
