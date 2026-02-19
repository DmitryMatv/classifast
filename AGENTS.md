# AGENTS.md - Agentic Coding Guidelines for Classifast

Classifast is a classification API service using embeddings and vector search (Qdrant) to classify products into UNSPSC/CPV categories.

- **Backend**: Python FastAPI
- **Frontend**: TypeScript with Tailwind CSS, served via FastAPI
- **Infrastructure**: Redis (usage tracking), Qdrant (vector database), ZeroEntropy (reranking), Clerk (authentication), Polar (payments)

## Code Style Guidelines

### TypeScript (Frontend)

- Strict mode in `tsconfig.json`, module resolution: `bundler`, target: `ESNext`
- Naming: `camelCase` (vars/functions), `PascalCase` (classes/components)
- Types: Explicit params/returns, interfaces for objects, avoid `any` (use `unknown`)
- Error handling: try/catch with typed errors, descriptive logging, user-friendly UI
- CSS: Tailwind utilities, templates in `app/templates/`

### General

- **Security**: Never commit secrets (.env files - ensure in .gitignore), validate/sanitize inputs, parameterized queries
- **Git**: Feature branches, meaningful commits, run typecheck before commit

## File Organization

```
app/
  main.py            # FastAPI app entry, middleware, lifespan
  api.py             # API routes
  web.py             # Web UI routes
  classifier.py      # Core classification logic
  classifier_config.py  # Configuration
  payments.py        # Payment handling
  dependencies.py    # FastAPI dependencies
  usage_tracker.py  # Redis usage tracking
  static/           # Built assets
  templates/         # Jinja2 templates
app/assets/ts/      # TypeScript source
app/assets/css/     # CSS source (Tailwind)
utilities/          # Test scripts
```

## Development Tasks

**Frontend**: Edit TS in `app/assets/ts/`, don't run build (I do it), don't edit JS files.

**Backend**: `pip install -r requirements.txt`, set up `.env`, run `uvicorn app.main:app --reload --port 8001`.

**Adding Tests**: Create script in `utilities/`

## Dependencies

- **Backend**: FastAPI, uvicorn, qdrant-client, google-genai, redis, polar-sdk, cryptography, zeroentropy
- **Frontend**: TypeScript, Tailwind CSS, Bun
- **External**: Qdrant (vectors), Redis (caching), Clerk (auth), Polar (payments), ZeroEntropy (reranking), Google Gemini (embeddings)

## Deployment

- **Host**: Raspberry Pi 4 (4GB) self-hosted via Coolify
- **Access**: Cloudflare Tunnel (DDoS protection, SSL termination)
- **Containers**: Classifast, Qdrant, Redis, WordPress (Docker)
- **Notes**: Rebuild Docker image on changes, stay within 4GB memory, tunnel routes to port 8001, health check at `/health`
