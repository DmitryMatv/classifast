# AGENTS.md - Agentic Coding Guidelines for Classifast

This document provides guidelines for agents working on the Classifast codebase.

## Project Overview

Classifast is a classification API service that uses embeddings and vector search (Qdrant) to classify products into UNSPSC/CPV categories. It has:

- **Backend**: Python FastAPI with async/await patterns
- **Frontend**: TypeScript with Tailwind CSS, served via FastAPI
- **Infrastructure**: Redis (usage tracking), Qdrant (vector database), Clerk (authentication), ZeroEntropy (reranking)

---

## Build/Lint/Test Commands

### Frontend (TypeScript + CSS)

```bash
# Install dependencies
bun install

# Build production assets
bun run build          # Builds both TS and CSS
bun run build:ts      # Build TypeScript only
bun run build:css      # Build CSS with Tailwind

# Development
bun run watch          # Watch TypeScript changes
bun run watch:css     # Watch CSS changes
bun run dev           # Run full dev server (Tailwind + TS watch + uvicorn)

# Type checking
bun run typecheck      # Run TypeScript type checker (tsc --noEmit)
```

### Backend (Python)

```bash
# Install Python dependencies
pip install -r requirements.txt

# Run the application
uvicorn app.main:app --reload --port 8001

# Run a single Python test file
python utilities/test_title_functionality.py
python utilities/test_subscription_events.py
python utilities/test_rapidapi.py
python utilities/test_embedding_ordering.py

# Linting (if ruff is installed)
ruff check .
ruff check app/        # Check specific directory
```

### Docker

```bash
# Build and run with docker-compose
docker-compose up --build

# Or build manually
docker build -t classifast .
docker run -p 8001:8001 classifast
```

---

## Code Style Guidelines

### Python (Backend)

**Imports**

- Standard library first, then third-party, then local
- Use absolute imports from package root (e.g., `from app.classifier import ...`)
- Group imports with blank lines between groups

**Naming Conventions**

- Functions/variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private methods: prefix with `_`

**Type Hints**

- Use explicit type hints for function parameters and return values
- Use `Optional[X]` instead of `X | None` for compatibility
- Enable strict typing where practical

**Error Handling**

- Use specific exception types when possible
- Always chain exceptions with `from e` for debugging
- Log errors before re-raising
- Use custom error responses for API errors

**Async/Await**

- Use `async def` for all route handlers
- Use `await asyncio.to_thread()` for blocking operations
- Initialize async clients in lifespan context

**Logging**

- Use module-level logger: `logger = logging.getLogger(__name__)`
- Use JSON formatter for production (already configured)
- Log at appropriate levels: ERROR for failures, WARNING for recoverable issues, INFO for important events

**Example Function:**

```python
async def classify_product(
    client: QdrantClient,
    text: str,
    collection: str,
) -> list[ClassificationResult]:
    """Classify a product description against a collection.

    Args:
        client: Qdrant client instance
        text: Product description to classify
        collection: Name of the collection to search

    Returns:
        List of classification results sorted by score
    """
    try:
        embeddings = await get_embeddings(client, text)
        results = await search_similar(client, embeddings, collection)
        return results
    except Exception as e:
        logger.error("Classification failed: %s", e)
        raise ClassificationError(f"Failed to classify: {e}") from e
```

### TypeScript (Frontend)

**Configuration**

- Strict mode is enabled in `tsconfig.json`
- Module resolution: `bundler`
- Target: `ESNext`

**Naming Conventions**

- Variables/functions: `camelCase`
- Classes: `PascalCase`
- Components: `PascalCase`
- Constants: `UPPER_SNAKE_CASE` or `camelCase` with prefix

**Types**

- Use explicit types for function parameters and returns
- Use interfaces for object shapes
- Avoid `any` - use `unknown` when type is truly unknown

**Error Handling**

- Use try/catch with typed error variables
- Log errors to console with descriptive messages
- Show user-friendly feedback in UI

**Example:**

```typescript
interface SearchResult {
  id: string;
  score: number;
  className: string;
}

async function searchProducts(query: string): Promise<SearchResult[]> {
  try {
    const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
    if (!response.ok) {
      throw new Error(`Search failed: ${response.statusText}`);
    }
    return (await response.json()) as SearchResult[];
  } catch (err) {
    console.error("Search error:", err);
    throw err;
  }
}
```

**HTML/CSS**

- Use Tailwind CSS utility classes
- Keep HTML templates in `app/templates/`
- Use semantic HTML elements

### General

**Security**

- Never commit secrets to git (use `.env` files, gitignored)
- Validate and sanitize all user inputs
- Use parameterized queries for database operations
- Apply security headers (already configured in middleware)

**Git**

- Create feature branches for new work
- Write meaningful commit messages
- Run typecheck before committing

**File Organization**

```
app/
  main.py           # FastAPI app entry, middleware, lifespan
  api.py            # API routes
  web.py            # Web UI routes
  classifier.py     # Core classification logic
  classifier_config.py  # Configuration
  payments.py       # Payment handling
  dependencies.py   # FastAPI dependencies
  usage_tracker.py # Redis usage tracking
  static/           # Built assets
  templates/        # Jinja2 templates
app/assets/ts/     # TypeScript source
app/assets/css/    # CSS source (Tailwind input)
utilities/         # Utility scripts and tests
mapping/           # UNSPSC/CPV mapping scripts
embedders/         # Embedding-related code
```

---

## Common Development Tasks

### Running a Single Test

```bash
python utilities/test_title_functionality.py
python utilities/test_rapidapi.py
```

### Adding a New API Endpoint

1. Add route to `app/api.py` or create new route file
2. Use dependency injection for clients from `app.state`
3. Return appropriate response types (JSONResponse, FileResponse, etc.)
4. Add error handling with proper HTTP status codes

### Modifying Frontend

1. Edit TypeScript in `app/assets/ts/`
2. Run `bun run watch` to auto-rebuild
3. Refresh browser to see changes

### Adding Environment Variables

1. Add to `.env` file (gitignored)
2. Document in `.env` with comments
3. Load with `from dotenv import load_dotenv; load_dotenv()`
4. Access via `os.getenv("VARIABLE_NAME")`

---

## Dependencies

### Key Packages

- **Backend**: FastAPI, uvicorn, qdrant-client, google-genai, redis, python-jose, cryptography
- **Frontend**: TypeScript, Tailwind CSS, Bun (build tool)
- **Testing**: Manual test scripts in `utilities/`

### External Services

- **Qdrant**: Vector database for embeddings
- **Redis**: Session and usage tracking
- **Clerk**: Authentication
- **ZeroEntropy**: Reranking
- **Google Gemini**: Text embeddings

---

## Deployment Infrastructure

### Hardware & Hosting

- **Hardware**: Raspberry Pi 4 (4GB RAM)
- **Location**: Self-hosted on-premises
- **Public Access**: Cloudflare Tunnel (`cloudflared` service) handles:
  - DDoS protection
  - Dynamic IP resolution
  - SSL termination at the edge

### Container Management

- **PaaS**: Coolify running on the Raspberry Pi
- **All services run as Docker containers**:
  - Classifast (this application)
  - Qdrant (vector database)
  - Redis (caching/sessions)
  - WordPress (blog)

### Deployment Notes

- When modifying the app, rebuild the Docker image and redeploy via Coolify
- Ensure memory usage stays within 4GB (Qdrant can be memory-intensive)
- Cloudflare tunnel forwards traffic to the container's internal port (8001)
- Health checks are configured at `/health` endpoint
