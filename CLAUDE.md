# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application

- **Development server**: `uvicorn app.main:app --reload --port 8001`
- **Docker**: `docker build -t classifast . && docker run -p 8001:8001 classifast`
- **Docker Compose**: `docker-compose up -d` (includes health checks)

### Environment Setup

Required environment variables (use `.env` file for local development):

- `GEMINI_API_KEY`: Google Gemini API key for embeddings
- `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_API_KEY`: Qdrant vector database connection
- `REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD`, `REDIS_USERNAME`: Redis for usage tracking
- `CLERK_SECRET_KEY`, `CLERK_FRONTEND_API`, `CLERK_PERMITTED_ORIGINS`: Clerk authentication
- `POLAR_ACCESS_TOKEN`, `POLAR_WEBHOOK_SECRET`: Polar subscription payments
- `RAPIDAPI_SECRET`: RapidAPI proxy authentication
- `DEBUG_MODE`: Enable test classifier and debug endpoints

Install dependencies:

- Python: `pip install -r requirements.txt`
- Node (for CSS): `npm install`

### CSS Development

- **Watch**: `npx @tailwindcss/cli -i ./app/static/css/input.css -o ./app/static/css/styles.css --watch`
- **Build**: `npx @tailwindcss/cli -i ./app/static/css/input.css -o ./app/static/css/styles.css`

### Testing & Debugging

- **Health check**: `curl http://localhost:8001/health`
- **RapidAPI health**: `curl http://localhost:8001/api/v1/rapid/ping`
- **Test utilities** in `utilities/`:
  - `test_rapidapi.py`: Test RapidAPI endpoints
  - `test_embedding_ordering.py`: Validate embedding generation order
  - `check_match.py`: Verify classification accuracy for specific queries
  - `count_codes.py`: Analyze collection statistics

## Architecture Overview

### Core Modules

| Module | Purpose |
|--------|---------|
| `app/main.py` | FastAPI app, lifespan management, middleware stack, client initialization |
| `app/classifier.py` | Embedding generation, Qdrant search, `classify_string_batch()` and `perform_classification()` |
| `app/classifier_config.py` | `CLASSIFIER_CONFIG` dict with all classification standards and versions |
| `app/api.py` | RapidAPI endpoints (`/api/v1/rapid/classify`, `/standards`, `/ping`) |
| `app/web.py` | Web interface routes, HTMX form handling |
| `app/payments.py` | Polar checkout/webhooks, Clerk JWT verification, subscription tier management |
| `app/usage_tracker.py` | Redis-based usage quotas, user tier caching, IP/cookie tracking |
| `app/dependencies.py` | Rate limiters, Jinja2 templates |

### Classification System

- **10 standards**: UNSPSC, ETIM, NAICS, ISIC, HS, CN, NACE, CPV, NSN, HTS (plus TEST in debug mode)
- **Embeddings**: Google Gemini (`gemini-embedding-001` at 3072 dims, `text-embedding-004` at 768 dims)
- **Vector DB**: Qdrant with INT8 scalar quantization, rescore=true, oversampling=3.0, hnsw_ef=256
- **Flow**: Query → Gemini embedding → Qdrant search → top_k results with confidence scores

### User Tiers & Usage Tracking

Three user types with Redis-based quota tracking:

- **Pro** (authenticated, `tier=pro` in Clerk metadata): Unlimited access
- **Free** (authenticated, no pro tier): 30 requests/month
- **Anonymous**: 10 requests/month (tracked by both cookie and IP hash)

Usage tracking uses dual counters (cookie + IP hash) for anonymous users to prevent quota bypass via cookie clearing.

### Payment Flow

1. User authenticates via Clerk (JWT with RS256 signature verification)
2. Frontend calls `/api/create-checkout` with Clerk JWT
3. Backend creates Polar checkout with `user_id` in metadata
4. On payment, Polar webhook (`/api/webhooks/polar`) updates Clerk `public_metadata.tier`
5. `usage_tracker.py` checks tier from JWT, with Redis cache fallback to Clerk API

### Middleware Stack (order matters)

1. `PerformanceMiddleware`: Adds X-Process-Time header
2. `GZipMiddleware`: Compression for responses > 1000 bytes
3. `URLEncodingValidationMiddleware`: Blocks triple-encoded URLs and HTML injection attempts
4. `SecurityHeadersMiddleware`: CSP, HSTS, X-Frame-Options, etc.

### Request Flow

```
Request → Middleware → Rate Limiter → Usage Check → Classification → Usage Increment → Response
```

For web routes: `check_usage()` → `perform_classification()` → `increment_usage()` → render template

### Key Implementation Details

- **Retry logic**: Tenacity with 3 attempts, exponential backoff (4-10s) for Gemini API
- **Quantization search**: Internal top_k=100, then rescore and return user-requested top_k
- **Client initialization**: All async clients (Qdrant, Redis, Gemini) initialized in `lifespan()` context manager
- **Collection validation**: Startup verifies all configured collections exist with correct vector dimensions
- **Rate limiting**: SlowAPI with 20/min for web, 600/min for RapidAPI
- **Caching strategy**: Static files 1hr browser/4hr CDN, Redis tier cache 10s TTL
