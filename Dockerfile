# --- Stage 1: Builder ---
# This stage builds Python wheels for our dependencies
# and creates a virtual environment to keep things clean.
FROM python:3.13.3-slim-bookworm AS builder

# Set environment variables
ENV PYTHONUNBUFFERED=1

WORKDIR /service_root

# Create a virtual environment
RUN python -m venv /opt/venv

# Install dependencies directly into the venv path without activating
COPY requirements.txt .
RUN /opt/venv/bin/pip install --no-cache-dir -r requirements.txt


# --- Stage 2: Final ---
# This stage takes the installed dependencies and application code
# to create a lean production image.
FROM python:3.13.3-slim-bookworm AS final

# Set environment variables for the final image
ENV PYTHONUNBUFFERED=1 \
    # Path to the virtual environment's executables
    PATH="/opt/venv/bin:$PATH"

WORKDIR /service_root

# Install curl for health checks as root user
USER root
RUN apt-get update && \
    apt-get install --no-install-recommends -y curl && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user and group
RUN groupadd --system appgroup && \
    useradd --system --gid appgroup --no-create-home appuser

# Copy the virtual environment from the builder stage
COPY --from=builder --chown=appuser:appgroup /opt/venv /opt/venv

# Copy application code
# Ensure your app code is in a subdirectory (e.g., ./app) for cleaner COPY
COPY --chown=appuser:appgroup ./app ./app

USER appuser

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "3", "--forwarded-allow-ips", "fddf:4d:bfa1::1:0"]
