# Build stage
FROM python:3.13-alpine AS builder
WORKDIR /service_root
COPY requirements.txt .

# Add build dependencies for Alpine. These are needed if your requirements.txt has packages with C extensions.
#RUN apk add --no-cache build-base python3-dev
RUN pip install --no-cache-dir -r requirements.txt


# Final stage
FROM python:3.13-alpine
WORKDIR /service_root

# Install curl (for healthcheck) and remove wget
RUN apk update && \
    apk add --no-cache curl && \
    rm -rf /var/cache/apk/*

# Create a non-root user and group
RUN addgroup -S appgroup && adduser -S -G appgroup appuser

# Copy installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
# Copy executables (like uvicorn) from the builder stage's bin directory
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the entire local 'app' directory (which should contain main.py, classifier.py, static, templates, __init__.py)
# to a directory named 'app' inside the WORKDIR.
# Structure inside container: /service_root/app/main.py, /service_root/app/classifier.py etc.
# Also set permissions
COPY --chown=appuser:appgroup ./app ./app

# Set the user to the non-root user
USER appuser

# Make port 6009 available to the world outside this container
EXPOSE 6009

# Command to run the application using Uvicorn
# Uvicorn will look for 'app.main:app' relative to the WORKDIR /service_root
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "6009", "--workers", "4"]