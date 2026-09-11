# faceymatch container image.
#
# This is the x86 Beelink deployment path only. onnxruntime in this Linux image
# has no CoreML execution provider, so it runs on CPUExecutionProvider. Apple
# Silicon hosts do NOT use this image: they run natively with `uv run` from
# backend/, where onnxruntime picks up the CoreML EP.
#
# Stage 1 builds the static frontend with bun. Stage 2 is the runtime: Python
# 3.12 plus uv, with no Node runtime present (D7).

# --- stage 1: frontend build -------------------------------------------------
FROM oven/bun:1.2 AS frontend

WORKDIR /build

# Manifest first so the dependency layer caches independently of source edits.
COPY frontend/package.json frontend/bun.lock ./
RUN bun install --frozen-lockfile

# Explicit sources only: never copy the host's frontend/node_modules (darwin
# binaries) or a stale frontend/dist over this stage's clean install.
COPY frontend/tsconfig.json frontend/tsconfig.node.json frontend/vite.config.ts frontend/index.html ./
COPY frontend/src ./src
# Emits /build/dist, which stage 2 copies to /app/frontend/dist.
RUN bun run build

# --- stage 2: backend runtime ------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

# libgomp1: onnxruntime's OpenMP runtime.
# libgl1 + libglib2.0-0: OpenCV's runtime deps (YuNet detector, alignment).
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        libgomp1 \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Pinned uv, copied from its official distroless image. Build-time only.
COPY --from=ghcr.io/astral-sh/uv:0.12.7 /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app/backend

# Dependency layer: resolve from backend/pyproject.toml without the project
# itself, so app source edits do not invalidate the installed dependencies.
COPY backend/pyproject.toml backend/uv.lock* ./
RUN uv sync --no-dev --no-install-project

# Only the package source: no host .venv, __pycache__, or test fixtures.
COPY backend/app ./app
RUN uv sync --no-dev

# The backend serves this at `/`, with the API under `/api`.
COPY --from=frontend /build/dist /app/frontend/dist

# Media store, crops, and SQLite DB live on volumes (see docker-compose.yml).
# Names match the unprefixed pydantic-settings fields in app/config.py.
RUN mkdir -p /app/media/crops /app/data /app/models
ENV DB_PATH=/app/data/facematch.db \
    MEDIA_DIR=/app/media \
    CROPS_DIR=/app/media/crops \
    MODELS_DIR=/app/models \
    FRONTEND_DIST=/app/frontend/dist \
    EXECUTION_PROVIDER=CPUExecutionProvider \
    ALLOW_NONCOMMERCIAL_MODELS=false

EXPOSE 8000

# --workers 1 is required, not a tuning choice: the audit log is hash-chained,
# so every append must read the previous hash and write the next under one
# single-writer process. A second worker would race the chain and fork it
# (invariant 6). The job worker is in-process for the same reason (D10).
#
# --host 127.0.0.1 keeps invariant 11 (loopback only). Note this binds the
# CONTAINER's loopback, so the published port cannot reach it; on the Beelink,
# run this image with `network_mode: host` or reach the app by exec'ing into
# the container. Remote access is always via `tailscale serve` on the host.
CMD ["uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000", "--workers", "1"]
