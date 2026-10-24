# syntax=docker/dockerfile:1
ARG BUILDKIT_CONTEXT_KEEP_GIT_DIR=1

FROM node:22-alpine AS frontend
WORKDIR /frontend
ARG VITE_COMMIT_HASH=""
ARG BUILD_COMMIT_HASH=""
ARG GIT_COMMIT=""
ARG SOURCE_COMMIT=""
ARG COMMIT_SHA=""
ARG GITHUB_SHA=""
ENV VITE_COMMIT_HASH=${VITE_COMMIT_HASH} \
    BUILD_COMMIT_HASH=${BUILD_COMMIT_HASH} \
    GIT_COMMIT=${GIT_COMMIT} \
    SOURCE_COMMIT=${SOURCE_COMMIT} \
    COMMIT_SHA=${COMMIT_SHA} \
    GITHUB_SHA=${GITHUB_SHA}
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit --fund=false
COPY .git* ./.git/
COPY backend/app/_build_info.py ./_build_info.py
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim-bookworm
WORKDIR /app
ARG VITE_COMMIT_HASH=""
ARG BUILD_COMMIT_HASH=""
ARG GIT_COMMIT=""
ARG SOURCE_COMMIT=""
ARG COMMIT_SHA=""
ARG GITHUB_SHA=""
ENV VITE_COMMIT_HASH=${VITE_COMMIT_HASH} \
    BUILD_COMMIT_HASH=${BUILD_COMMIT_HASH} \
    GIT_COMMIT=${GIT_COMMIT} \
    SOURCE_COMMIT=${SOURCE_COMMIT} \
    COMMIT_SHA=${COMMIT_SHA} \
    GITHUB_SHA=${GITHUB_SHA}

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        libglib2.0-0 \
        libgomp1 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.10.0 /uv /usr/local/bin/uv
RUN chmod +x /usr/local/bin/uv && ln -sf /usr/local/bin/uv /usr/local/bin/uvx

COPY vendor/familysearch-mcp /app/vendor/familysearch-mcp
COPY backend/pyproject.toml /app/backend/pyproject.toml
RUN mkdir -p /app/backend/app && touch /app/backend/app/__init__.py \
    && uv pip install --system --no-cache /app/vendor/familysearch-mcp /app/backend
COPY backend /app/backend
RUN uv pip install --system --no-cache --no-deps /app/backend

COPY --from=frontend /frontend/dist /app/frontend/dist
COPY .git* /app/.git/

RUN mkdir -p /app/data /app/samples

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

EXPOSE 8080
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers", "--forwarded-allow-ips", "*"]
