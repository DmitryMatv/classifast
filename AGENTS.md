# AGENTS.md - Agentic Coding Guidelines for Classifast

The role of this file is to describe common mistakes and confusion points that agents might encounter as they work in this project. If you ever encounter something in the project that surprises you, please alert the developer working with you and indicate that this is the case in the AGENTS.md file to help prevent future agents from having the same issue.

Classifast is a code search & match service that is using embeddings and vector search (Qdrant) to classify any text input (mostly products descriptions) into categories of various industry standard classifications, like UNSPSC, NAICS, CN/HS codes, ISIS, ETIM, CPV, etc.

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
  api.py             # API routes for RapidAPI (unrelated to Web)
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

## Development

**Frontend**: Edit TS in `app/assets/ts/`, don't run build (I will do it), don't edit JS files.

**Backend**: `pip install -r requirements.txt`, set up `.env`, run `uvicorn app.main:app --reload --port 8001`.

**Adding Tests**: Create script in `utilities/`

### Dependencies

- **Backend**: FastAPI, uvicorn, qdrant-client, google-genai, redis, polar-sdk, cryptography, zeroentropy
- **Frontend**: TypeScript, Tailwind CSS, Bun
- **External**: Qdrant (vectors), Redis (caching), Clerk (auth), Polar (payments), ZeroEntropy (reranking), Google Gemini (embeddings)

### Deployment

- **Host**: Raspberry Pi 4 (4GB) self-hosted via Coolify
- **Access**: Cloudflare Tunnel (DDoS protection, SSL termination)
- **Containers**: Classifast, Qdrant, Redis, WordPress (Docker)
- **Notes**: Rebuild Docker image on changes, stay within 4GB memory, tunnel routes to port 8001, health check at `/health`

### Cloudflare CDN Caching

The app uses Cloudflare's CDN edge caching to reduce API costs and improve performance. Classification results (Gemini embeddings, ZeroEntropy reranking, Qdrant search) are cached at edge for 7 days.

#### CF Cache Rule

- **Rule**: Cache everything respecting headers (no path-based filtering)
- **Edge TTL**: Use `Cache-Control` header if present, bypass if not
- **Key**: CF does NOT cache responses with `Set-Cookie` header (regardless of `Cache-Control: public`)

#### Cache Headers

- **Browser**: `Cache-Control: public, max-age=14400` (4 hours)
- **CDN**: `Cloudflare-CDN-Cache-Control: max-age=604800` (7 days)
- **Vary**: `Accept-Encoding`
- **Paywall**: `Cache-Control: no-store` (never cached)

#### Cookie & Caching Architecture

| Endpoint                      | Sets Cookie | CF Cached              |
| ----------------------------- | ----------- | ---------------------- |
| `/{type}/fragment?...`        | No          | Yes                    |
| `/{type}/`, `/{type}/{query}` | Yes         | No                     |
| Paywall response              | Yes         | No (explicit no-store) |

#### Usage Tracking with Cached Fragments

Since fragments don't set cookies, usage tracking works via:

1. **Cookie from request header** - Returning visitors already have cookie from full page load
2. **IP hash fallback** - New visitors tracked by hashed IP (handles cookie absence)

Redis stores both `anon:{tracking_id}:usage_count` and `anon:ip:{ip_hash}:usage_count`, using the higher value to prevent abuse via cookie clearing.

#### When Modifying Cache Behavior

- **Never add `Set-Cookie` to fragment responses** - breaks CDN caching
- **Paywalls must use `no-store`** - prevents serving cached paywall to allowed users
- **Full pages can set cookies** - they're browser-only anyway due to `Set-Cookie` header
