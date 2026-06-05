import bentoml

from src.main import app


@bentoml.service(
    name="safety_vision",
    traffic={"timeout": 120},
)
@bentoml.asgi_app(app, path="/")
class VisionService:
    @bentoml.api(route="/bento/health")
    def bento_health(self) -> dict[str, str]:
        return {"status": "ok", "service": "safety_vision"}
