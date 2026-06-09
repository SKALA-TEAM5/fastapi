FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    HF_HOME="/app/.cache/huggingface" \
    TRANSFORMERS_CACHE="/app/.cache/huggingface/transformers" \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /uvx /bin/

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        fonts-noto-cjk \
        fontconfig \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./

# Keep the largest CPU-only ML wheels in separate registry layers. This avoids
# one multi-gigabyte upload and prevents GPU libraries from entering the image.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv \
    && uv pip install --python .venv \
        --index https://download.pytorch.org/whl/cpu \
        "torch==2.9.1+cpu" \
        "torchvision==0.24.1+cpu"

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv "paddlepaddle==3.3.1"

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Preload local embedding and legal reranker models into the image so the first
# orchestrator legal run does not download them from HuggingFace at runtime.
RUN .venv/bin/python -c "\
from sentence_transformers import CrossEncoder, SentenceTransformer; \
SentenceTransformer('jhgan/ko-sroberta-multitask'); \
CrossEncoder('BAAI/bge-reranker-v2-m3', backend='onnx')"

COPY src ./src

EXPOSE 8001

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8001"]
