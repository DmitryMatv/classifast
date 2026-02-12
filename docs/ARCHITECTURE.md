# Architecture Guide

## File Structure

| File                       | Purpose                                             |
| -------------------------- | --------------------------------------------------- |
| `app/main.py`              | FastAPI app, middleware, lifespan management        |
| `app/classifier.py`        | Embedding generation, Qdrant search, classification |
| `app/classifier_config.py` | Standards, versions, collections config             |
| `app/api.py`               | RapidAPI endpoints                                  |
| `app/web.py`               | Web interface, HTMX forms, SEO                      |
| `app/payments.py`          | Polar checkout/webhooks, Clerk JWT verification     |
| `app/usage_tracker.py`     | Redis-based usage quotas, user tier caching         |
| `app/dependencies.py`      | Jinja2 templates, Clerk JWT utilities               |
| `utilities/`               | Integration test scripts                            |

## Middleware Stack (Critical - Do Not Reorder)

1. **PerformanceMiddleware** - Adds X-Process-Time header
2. **GZipMiddleware** - Compression for responses > 1000 bytes
3. **URLEncodingValidationMiddleware** - Blocks encoded attacks
4. **SecurityHeadersMiddleware** - CSP, HSTS, X-Frame-Options

## Async Client Management

Initialize in `lifespan()` context manager:

```python
@app.on_event("startup")
async def lifespan():
    app.state.qdrant = AsyncQdrantClient(...)
    app.state.redis = redis.from_url(...)
    # ...

@app.on_event("shutdown")
async def shutdown():
    await app.state.qdrant.close()
    await app.state.redis.close()
```

Use `asyncio.gather()` for parallel independent operations.

## Redis Patterns

```python
import redis.asyncio as redis

# Counters with TTL
await redis.incr(key)
await redis.expire(key, ttl)

# Cache tier lookups
ttl = 60 if result else 10  # Positive vs negative caching

# Fail open on Redis errors
try:
    result = await redis.get(key)
except redis.RedisError:
    result = None
```

## Vector Database (Qdrant)

```python
from qdrant_client import AsyncQdrantClient

# Always use async client
client = AsyncQdrantClient(...)

# Quantized collections config
top_k=100, rescore=True, oversampling=3.0

# Hybrid search: exact matches first (score=0.999), then semantic
```

## Security

- **Never log**: secrets, tokens, sensitive headers
- **IP hashing**: Use `hashlib.sha256` before storage
- **Input sanitization**: See `sanitize_query_text()` implementation
- **JWT verification**: Verify signatures with PyJWKClient before trusting claims
- **Dual tracking**: Anonymous users tracked by cookie + IP hash (prevents quota bypass)

## Configuration

```python
from dotenv import load_dotenv

load_dotenv()  # At module level

# Pattern: os.getenv with defaults
TIMEOUT = int(os.getenv("TIMEOUT", "30"))
```

Store app config in `classifier_config.py` dictionary structure.

## Common Patterns

**Timing**:

```python
start_time = time.perf_counter()
# ... operation ...
elapsed = time.perf_counter() - start_time
```

**Request validation**:

```python
if not embed_client:
    raise HTTPException(status_code=503, detail="Service unavailable")
sanitized_input = sanitize_query_text(user_input)
```

**SEO URLs**:

```python
# Use slugify() from web.py for URL-safe identifiers
slug = slugify("My Category Name")  # "my-category-name"
```
