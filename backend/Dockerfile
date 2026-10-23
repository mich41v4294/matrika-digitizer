# syntax=docker/dockerfile:1

FROM node:22-alpine AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit --fund=false
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim-bookworm
WORKDIR /app

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

RUN mkdir -p /app/data /app/samples

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

EXPOSE 8080
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers", "--forwarded-allow-ips", "*"]
