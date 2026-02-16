# Classifast - AGENTS.md

A FastAPI-based classification service with semantic search powered by Qdrant vector database.

## Quick Start

```bash
# Development server
uvicorn app.main:app --reload --port 8001

# Health check
curl http://localhost:8001/health
```

## Build, Lint & Test Commands

```bash
# Development server with auto-reload
uvicorn app.main:app --reload --port 8001

# Run with docker-compose (includes Redis, Qdrant)
docker-compose up -d

# Build and run Docker container
docker build -t classifast . && docker run -p 8001:8001 classifast

# Run specific integration test
python utilities/test_<name>.py

# Examples
python utilities/test_rapidapi.py
python utilities/test_embedding_ordering.py

# CSS development (watch mode)
npx @tailwindcss/cli -i ./app/static/css/input.css -o ./app/static/css/styles.css --watch
```

## Project Structure

```
app/
├── main.py           # FastAPI app, middleware, lifespan
├── api.py            # RapidAPI endpoints
├── web.py            # Web interface (HTMX)
├── classifier.py     # Embedding & classification logic
├── classifier_config.py  # Configuration
├── payments.py       # Polar checkout & Clerk JWT
├── usage_tracker.py  # Redis-based quotas
└── dependencies.py   # Templates, Clerk JWT
utilities/            # Integration test scripts
```

## Critical Rules

1. **Always use relative imports** for app modules: `from .classifier import ...`
2. **Middleware order matters** (see docs/ARCHITECTURE.md#middleware-stack)
3. **Use Pydantic models** for all API responses (JSONResponse only for custom headers)

---

## Code Style Guidelines

### Import Order

```python
1. Standard library (os, sys, logging, asyncio, typing, etc.)
2. Third-party (fastapi, qdrant_client, redis, httpx, tenacity, etc.)
3. Local imports (always relative: `from .classifier import ...`)
```

### Formatting & Types

- **Type hints**: Use Python 3.10+ syntax (`str | None`, not `Optional[str]`)
- **Indentation**: 4 spaces
- **Line length**: Keep lines under 120 chars when possible
- **Async/await**: Required for all I/O (Redis, Qdrant, HTTP)
- **Dataclasses**: Use `@dataclass` for structured data models

### Naming Conventions

| Type            | Convention       | Examples                                  |
| --------------- | ---------------- | ----------------------------------------- |
| Classes         | PascalCase       | `JsonFormatter`, `UsageStatus`            |
| Functions       | snake_case       | `get_embedding`, `perform_classification` |
| Constants       | UPPER_SNAKE_CASE | `ANON_LIMIT`, `TRACKING_COOKIE_NAME`      |
| Private         | `_prefix`        | `_is_suspicious_encoding`                 |
| Pydantic models | PascalCase       | `ClassificationResult`                    |

### Logging

```python
logger = logging.getLogger(__name__)
# logger.info() - User actions
# logger.debug() - Details + timing for external API calls
# logger.error() - Errors
# logger.warning() - Non-critical issues
```

## Error Handling

```python
from fastapi import HTTPException

# Input validation
if not valid:
    raise HTTPException(status_code=400, detail="Invalid input")

# External service failures
try:
    result = await external_api.call()
except ExternalError as e:
    logger.error("External API failed: %s", e)
    raise HTTPException(status_code=502, detail="External service unavailable")

# External API retries - use Tenacity
from tenacity import retry, stop_after_attempt, wait_exponential
@retry(stop_after_attempt=3, wait_exponential(multiplier=4, min=4, max=10))
async def call_external_api(): ...
```

## Async Patterns

```python
# Client initialization in lifespan()
@app.on_event("startup")
async def lifespan():
    app.state.qdrant = AsyncQdrantClient(...)
    app.state.redis = redis.from_url(...)

@app.on_event("shutdown")
async def shutdown():
    await app.state.qdrant.close()
    await app.state.redis.close()

# Parallel operations
results = await asyncio.gather(redis_operation(), qdrant_operation())
```

## Redis Patterns

```python
import redis.asyncio as redis
# Counters with TTL
await redis.incr(key)
await redis.expire(key, ttl)
# Cache tier lookups - ttl = 60 if result else 10
# Fail open on Redis errors
try:
    result = await redis.get(key)
except redis.RedisError:
    result = None
```

## Security

- **Never log**: secrets, tokens, sensitive headers
- **IP hashing**: Use `hashlib.sha256` before storage
- **Input sanitization**: See `sanitize_query_text()` in classifier.py
- **JWT verification**: Verify signatures with PyJWKClient before trusting claims

## Documentation

- [Code Style](docs/STYLE.md) - Imports, formatting, naming, logging
- [Architecture](docs/ARCHITECTURE.md) - Async patterns, Redis, Qdrant, security
- [API Patterns](docs/API_PATTERNS.md) - Endpoints, error handling
- [Testing](docs/TESTING.md) - Integration test utilities
