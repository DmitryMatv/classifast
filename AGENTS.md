# Classifast - AGENTS.md

A FastAPI-based classification service with semantic search powered by Qdrant vector database.

## Quick Start

```bash
# Development server
uvicorn app.main:app --reload --port 8001

# Health check
curl http://localhost:8001/health
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
└── dependencies.py   # Rate limiters, templates

utilities/            # Integration test scripts
```

## Critical Rules

1. **Always use relative imports** for app modules: `from .classifier import ...`
2. **Middleware order matters** (see [Architecture](docs/ARCHITECTURE.md#middleware-stack))
3. **Use Pydantic models** for all API responses (JSONResponse only for custom headers/non-model data)

## Documentation

- [Code Style](docs/STYLE.md) - Imports, formatting, naming, logging
- [Architecture](docs/ARCHITECTURE.md) - Async patterns, Redis, Qdrant, security
- [API Patterns](docs/API_PATTERNS.md) - Endpoints, error handling, HTTP client
- [Testing](docs/TESTING.md) - Integration test utilities
