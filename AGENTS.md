# AGENTS.md

The role of this file is to describe common mistakes and confusion points that agents might encounter as they work in this project. If you ever encounter something in this project that surprises you, please alert the developer working with you and indicate that this is the case in the AGENTS.md file to help prevent future agents from having the same issue.

## Ask Before You Assume

Never guess at intent. If a task leaves anything open - which screen, which endpoint, what happens on failure, whether it needs a migration, whether this is user-facing - stop and ask. If an important decision is unresolved, stop and ask. One question up front is cheaper than half-a-day of work in the wrong direction.

- Ask when the request could reasonably mean two different things.
- Ask before changing a public API shape.
- Do not invent product decisions, copy, or acceptance criteria.
- Do not widen scope past what was asked. Note the adjacent thing you spotted; don't fix it unprompted.
- If you had to assume something you couldn't resolve, list it explicitly at the top of your summary.

## Testing

Always use `npm test` or `npm run test:watch` for frontend tests.

Always use `pytest` for backend tests. The suite retains `unittest`-compatible
test classes and standard-library mocks, but pytest is the official runner.

Keep pytest collection scoped to `tests/`. The `utilities/test_*.py` files are
manual live/debug helpers, and the ignored `embedders/tests/` tree contains
separate experimental tests that are not part of the maintained backend suite.

Use Virtual Environment `source .venv/bin/activate` because all dependencies are installed there already.

`utilities/qdrant_config.py` is an executable migration-style script, not passive configuration. Importing or running it updates a hardcoded Qdrant collection, so review it carefully before execution.

`utilities/sync_payload_indexes.py check` is read-only.
`utilities/sync_payload_indexes.py apply` is a migration-style command that
backfills payloads and creates or replaces indexes in configured Qdrant
collections. Runtime startup must validate Qdrant without creating, replacing,
or deleting indexes.

## Project Snapshot

Classifast is a classification service web application that uses embeddings and vector search (Qdrant) to classify any text input (mostly product descriptions) into categories of various industry standard classifications, like UNSPSC, NAICS, CN/HS codes, ISIS, ETIM, CPV, etc.

## Tech Stack

- Backend: Python FastAPI
- Frontend: TypeScript with Tailwind CSS (built with Node/npm and Vite, served via FastAPI)
- Infrastructure: Redis (usage tracking), Qdrant (vector database), Hugging Face Inference (embeddings), OpenRouter (deployed-service reranking), Clerk (authentication), Polar (payments)

## Hardware, Deployment, Cache

Self-hosted from Raspberry Pi 4 (4GB) via Coolify behind Cloudflare Tunnel (Full HTTPS/TLS Setup for All Resources). The app uses Cloudflare's CDN edge caching to reduce API costs and improve performance. Classification results are cached at edge for 7 days.

## When Modifying Cache Behavior

- Never add `Set-Cookie` to fragment responses - breaks CDN caching
- Paywalls must use `no-store` - prevents serving cached paywall to allowed users
- Full pages can set cookies - but prefer client-side JavaScript to prevent CDN cache pollution
- Generate per-user state client-side when possible (e.g., tracking IDs via `crypto.randomUUID()`) instead of server-side templating - keeps HTML cacheable across all users
- `Cloudflare-CDN-Cache-Control` controls Cloudflare independently of browser
  `Cache-Control`. Responses that may be requested with `Authorization` must
  explicitly include `public` (or another authorization-compatible shared-cache
  directive) in the Cloudflare-specific header.

## Cloudflare Tunnel Access Logs

Uvicorn's `fddf:...` client address is the Cloudflare Tunnel peer, not the end
visitor. Do not correlate adjacent access-log entries as one user without using
`CF-Connecting-IP`, a Cloudflare Ray ID, or equivalent structured request data.

## Rapid API (API.py)

`app/api.py` is specifically made for the Rapid API platform. It contains endpoints that make the classification service accessible on that platform. Ignore api.py unless explicitly asked to work on Rapid API service integration.
