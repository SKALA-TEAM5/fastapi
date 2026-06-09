from __future__ import annotations

import time

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
