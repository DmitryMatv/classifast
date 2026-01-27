# Code Style Guide

## Import Order

```python
1. Standard library (os, sys, logging, asyncio, typing, etc.)
2. Third-party (fastapi, qdrant_client, redis, httpx, etc.)
3. Local imports (always relative: `from .classifier import ...`)
```

## Formatting & Types

- **Type hints**: Use consistently with Python 3.10+ syntax (`str | None`, not `Optional[str]`)
- **Indentation**: 4 spaces
- **Async/await**: Required for all I/O (Redis, Qdrant, HTTP)
- **Dataclasses**: Use `@dataclass` for structured data models
- **Tenacity**: External API calls must use retry:

  ```python
  @retry(stop_after_attempt=3, wait_exponential(multiplier=4, min=4, max=10))
  ```

## Naming Conventions

| Type | Convention | Examples |
|------|-----------|----------|
| Classes | PascalCase | `JsonFormatter`, `UsageStatus` |
| Functions | snake_case | `get_embedding`, `perform_classification` |
| Constants | UPPER_SNAKE_CASE | `ANON_LIMIT`, `TRACKING_COOKIE_NAME` |
| Private | `_prefix` | `_is_suspicious_encoding` |
| Pydantic models | PascalCase | `ClassificationResult` |

## Logging

```python
logger = logging.getLogger(__name__)
```

- `logger.info()` - User actions
- `logger.debug()` - Details + timing for external API calls
- `logger.error()` - Errors
- `logger.warning()` - Non-critical issues

**Timing pattern**:

```python
start_time = time.time()
# ... API call ...
logger.debug("API call: %.3fs", time.time() - start_time)
```
