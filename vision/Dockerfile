FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:${PATH}" \
    VISION_MODEL_PATH=/models/ppe-detector.pt \
    SAFETY_NET_MODEL_PATH=/models/safety-net-classifier.pt \
    VISION_INPUT_DIR=/data/files \
    VISION_OUTPUT_DIR=/data/vision_results \
    BENTOML_HOST=0.0.0.0 \
    BENTOML_PORT=8002

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /uvx /bin/

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 fonts-noto-cjk fontconfig \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src ./src
COPY service.py ./service.py
COPY bentofile.yaml ./bentofile.yaml

EXPOSE 8002

CMD ["bentoml", "serve", "--host", "0.0.0.0", "--port", "8002", "service:VisionService"]
