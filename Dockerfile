FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
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

COPY src ./src

EXPOSE 8001

CMD ["python", "-m", "src.main"]
