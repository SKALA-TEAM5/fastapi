from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.core.metrics import configure_metrics


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
