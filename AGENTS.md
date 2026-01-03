# AGENTS.md

This file provides guidance for agentic coding assistants working on the classifast repository.

## Build & Development Commands

### Running the Application

- **Development server**: `uvicorn app.main:app --reload --port 8001`
- **Docker**: `docker build -t classifast . && docker run -p 8001:8001 classifast`
- **Docker Compose**: `docker-compose up -d`

### CSS Development

- **Watch**: `npx @tailwindcss/cli -i ./app/static/css/input.css -o ./app/static/css/styles.css --watch`
- **Build**: `npx @tailwindcss/cli -i ./app/static/css/input.css -o ./app/static/css/styles.css`

### Testing

- **Run specific test**: `python utilities/test_<name>.py`
  - Example: `python utilities/test_rapidapi.py`
  - Example: `python utilities/test_embedding_ordering.py`
- Tests use simple Python scripts with manual assertions, not pytest

### Health Checks

- `curl http://localhost:8001/health`
- `curl http://localhost:8001/api/v1/rapid/ping`

## Code Style Guidelines

### Import Order

1. Standard library (os, sys, logging, asyncio, typing, etc.)
2. Third-party dependencies (fastapi, qdrant_client, redis, etc.)
3. Local imports (use relative imports for app modules: `from .classifier import ...`)

### Formatting & Types

- Use type hints consistently (typing module for generic types)
- Python 3.10+ union syntax: `str | None` instead of `Optional[str]`
- Async/await patterns for all I/O operations (Redis, Qdrant, HTTP)
- 4 spaces indentation

### Naming Conventions

- Classes: PascalCase (`JsonFormatter`, `UsageStatus`, `ClassificationResult`)
- Functions: snake_case (`get_embedding`, `perform_classification`)
- Constants: UPPER_SNAKE_CASE (`ANON_LIMIT`, `TRACKING_COOKIE_NAME`)
- Private functions: prefix with underscore (`_is_suspicious_encoding`)
- Pydantic models: PascalCase with descriptive names

### Logging

- Initialize per module: `logger = logging.getLogger(__name__)`
- Structured JSON logging in production (JsonFormatter class in main.py)
- Log levels: `logger.info()` for user actions, `logger.debug()` for details, `logger.error()` for errors, `logger.warning()` for non-critical issues
- Include timing info for external API calls: `logger.debug("API call: %.3fs", duration)`

### Error Handling

- API errors: raise `HTTPException` with appropriate status code and detail message
- External service failures: log error details with elapsed time, then raise HTTPException
- Input validation: use `HTTPException(status_code=400, detail="...")`
- Always wrap external API calls in try/except with logging

### API Endpoints

- Use FastAPI dependency injection (`Depends()`) for auth/rate limiting
- Decorate endpoints with `@router.get()` or `@app.get()`
- Include docstrings with multi-line descriptions using triple quotes
- Use Pydantic models for request/response validation
- Return `JSONResponse` for complex responses, otherwise rely on FastAPI serialization

### Configuration & Constants

- Load environment variables with `from dotenv import load_dotenv` at module level
- Store configuration in `classifier_config.py` dictionary structure
- Define constants at module top after imports
- Use `os.getenv("KEY", "default_value")` pattern with sensible defaults

### Async Client Management

- Initialize async clients (Qdrant, Redis, Gemini) in `lifespan()` context manager
- Store clients in `app.state` for request access
- Always close clients in lifespan shutdown handler
- Use `asyncio.gather()` for parallel independent operations

### Security

- Never log secrets, tokens, or sensitive headers
- Use `hashlib.sha256` for IP hashing before storage
- Sanitize all user input before processing (see `sanitize_query_text()`)
- Verify JWT signatures before trusting claims (see `payments.py`)
- Set security headers in middleware (CSP, HSTS, X-Frame-Options)

### Redis Patterns

- Use `redis.asyncio as redis` for async Redis operations
- Use `incr()` + `expire()` for counters with TTL
- Cache tier lookups with different TTLs for positive (60s) vs negative (10s) results
- Handle `redis.RedisError` exceptions gracefully (fail open)

### Vector Database (Qdrant)

- Use `AsyncQdrantClient` for async operations
- Check quantization config at startup: `collection.config.quantization_config`
- For quantized collections: internal top_k=100, rescore=True, oversampling=3.0
- Hybrid search: exact text matches first (score=0.999), then semantic results

### Testing Philosophy

- Utilities in `utilities/` are integration tests, not unit tests
- Tests should be self-contained scripts with main() entry point
- Use f-strings for readable output in test scripts
- Tests verify external API integrations (RapidAPI, Clerk, Qdrant, Redis)

### File Structure

- `app/main.py`: FastAPI app, middleware, lifespan management
- `app/classifier.py`: Embedding generation, Qdrant search, classification logic
- `app/classifier_config.py`: All classifier configuration (standards, versions, collections)
- `app/api.py`: RapidAPI endpoints
- `app/web.py`: Web interface routes, HTMX form handling
- `app/payments.py`: Polar checkout/webhooks, Clerk JWT verification
- `app/usage_tracker.py`: Redis-based usage quotas, user tier caching
- `app/dependencies.py`: Rate limiters, Jinja2 templates
- `utilities/`: Integration test scripts

### Middleware Stack Order (Critical)

1. PerformanceMiddleware (adds X-Process-Time)
2. GZipMiddleware (compression > 1000 bytes)
3. URLEncodingValidationMiddleware (blocks encoded attacks)
4. SecurityHeadersMiddleware (CSP, HSTS, etc.)

### Common Patterns

- Timing: `start_time = time.time()` or `time.perf_counter()` at function start, log `elapsed = time.time() - start_time` on completion
- Request validation: check client availability first (`if not embed_client: raise HTTPException`), then sanitize input
- Response formatting: extract payload from Qdrant results, format for API response with proper base URLs
- Dual tracking: anonymous users tracked by both cookie AND IP hash to prevent quota bypass
