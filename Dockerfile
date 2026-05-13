# syntax=docker/dockerfile:1

FROM dhi.io/bun:1-dev AS frontend-builder

WORKDIR /frontend

COPY package.json bun.lock tsconfig.json ./
RUN --mount=type=cache,target=/root/.bun bun ci --minimum-release-age=259200

COPY app/assets ./app/assets
COPY app/templates ./app/templates
RUN mkdir -p app/static/js app/static/css && bun run build

FROM python:3.14-slim-bookworm AS runtime

WORKDIR /service_root

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-compile --uploaded-prior-to=P3D -r requirements.txt

# Install curl for health checks
RUN apt-get update && apt-get install --no-install-recommends -y curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --system appgroup && \
    useradd --system --gid appgroup --no-create-home appuser

COPY --chown=appuser:appgroup app ./app
COPY --from=frontend-builder --chown=appuser:appgroup /frontend/app/static/ ./app/static/

USER appuser

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--forwarded-allow-ips", "*"]
