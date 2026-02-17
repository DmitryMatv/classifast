FROM python:3.14-slim-bookworm

WORKDIR /service_root

# Install Python deps directly (no venv for simplicity)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install curl for health checks
USER root
RUN apt-get update && apt-get install --no-install-recommends -y curl


# Create non-root user
RUN groupadd --system appgroup && \
    useradd --system --gid appgroup --no-create-home appuser

# Copy application code (entire app directory)
COPY --chown=appuser:appgroup ./app /service_root/app/

USER appuser

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--forwarded-allow-ips", "*"]
