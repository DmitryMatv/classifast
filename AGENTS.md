# AGENTS.md

The role of this file is to describe common mistakes and confusion points that agents might encounter as they work in this project. If you ever encounter something in this project that surprises you, please alert the developer working with you and indicate that this is the case in the AGENTS.md file to help prevent future agents from having the same issue.

## Ask Before You Assume

Never guess at intent. If a task leaves anything open - which screen, which endpoint, what happens on failure, whether it needs a migration, whether this is user-facing - stop and ask. If an important decision is unresolved, stop and ask. One question up front is cheaper than half a day of work in the wrong direction.

- Ask when the request could reasonably mean two different things.
- Ask before changing a public API shape.
- Do not invent product decisions, copy, or acceptance criteria.
- Do not widen scope past what was asked. Note the adjacent thing you spotted; don't fix it unprompted.
- If you had to assume something you couldn't resolve, list it explicitly at the top of your summary.

## Testing

Always use `npm test` or `npm run test:watch` for frontend tests.

Always use `pytest` for backend tests. The suite retains `unittest`-compatible
test classes and standard-library mocks, but pytest is the official runner.

pytest.ini scopes pytest collection to `tests/`. The `utilities/test_*.py` files are
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

## When Modifying Cache Behaviour

Cache headers are defined in `app/cache_profiles.py` - edit the profiles there instead of hand-rolling headers.

- Never add `Set-Cookie` to fragment responses - breaks CDN caching
- Paywalls must use `no-store` - prevents serving cached paywall to allowed users
- Never set cookies on full pages either - the server has a blanket no-cookie
  policy; set per-user state from client-side JavaScript instead
- Generate per-user state client-side when possible (e.g., tracking IDs via `crypto.randomUUID()`) instead of server-side templating - keeps HTML cacheable across all users
- `Cloudflare-CDN-Cache-Control` controls Cloudflare independently of browser
  `Cache-Control`. Responses that may be requested with `Authorization` must
  explicitly include `public` (or another authorization-compatible shared-cache
  directive) in the Cloudflare-specific header.

## Built Frontend Files

`app/static/js/*.js` and `app/static/css/styles.css` are Vite build outputs
from `app/assets/ts/` and `app/assets/css/`. Never hand-edit them; edit the TS
sources and run `npm run build`.

## Rapid API (API.py)

`app/api.py` is specifically made for the Rapid API platform. It contains endpoints that make the classification service accessible on that platform. Ignore api.py unless explicitly asked to work on Rapid API service integration.

## Gotchas and Non-Obvious Behaviors

- `data/`, `embedders/`, and `mapping/` are gitignored and exist only on this
  machine. A fresh clone will not have them, and ripgrep-based searches
  silently return zero hits inside them (they respect `.gitignore`). Use
  `--no-ignore` or explicit paths when searching them.
- `app/classifier_page_delivery.py` parses `app/static/sitemap.xml` at import
  time to build `SITEMAP_QUERY_PATHS`, which gates SSR eligibility and homepage
  anchor links. Editing the sitemap only changes app behavior after a restart.
- `asset_url` hashes are cached per process (`app/dependencies.py`). After
  `npm run build`, restart the FastAPI process or the browser keeps loading
  the old JS with stale `?v=` values.
- The server never sets cookies on any response, cacheable or not. `cf_track`
  is set by client-side JavaScript. On HTML or fragment routes a server-side
  cookie additionally breaks the `HTML_PAGE`/`CLASSIFICATION_RESULT` CDN cache
  profiles; on other routes it is still forbidden by design, so use
  client-side JavaScript for any per-user state.
- Two client-IP trust policies coexist: `app/usage_tracker.py` always trusts
  `CF-Connecting-IP`; `app/google_crawlers.py` requires the explicit
  `GOOGLE_CRAWLER_TRUST_CF_CONNECTING_IP` opt-in. Pick one policy deliberately
  for new IP-dependent code.
- `paywall.ts` is wrapped in a parse guard on purpose (class declarations
  re-execute on bfcache/history-restore re-parsing). Do not remove the guard.
- `htmx.min.js` is vendored in `app/static` and is the one asset loaded
  WITHOUT the `?v=` cache-busting param. `emptyOutDir: false` in
  `vite.config.ts` protects it from build cleanup.
- The app assumes a single uvicorn worker: module-level caches (JWKS client,
  asset versions, crawler IP ranges) and the process-randomized ETag fallback
  depend on it. Scaling workers changes their semantics.
- `tests/integration/` is empty. The suite is fully unit-level with mocks and
  says nothing about live Qdrant/Redis/HF connectivity; manual live helpers
  live in `utilities/test_*.py` (excluded from pytest collection).
- `/health` requires initialized embedding and Qdrant clients, then probes
  Qdrant only. If `HF_TOKEN` is missing, public pages can still serve while
  `/health` returns 503, provided Qdrant startup succeeds. It does not check
  Redis or live Hugging Face requests. A Qdrant startup failure stops the app.
- `.env` is a personal cross-project secrets file (it contains keys unrelated
  to classifast too). Never print, copy, or commit it.
- Checkout endpoints are rate limited per IP via `app/rate_limit.py`
  (fixed-window Redis counter, fails closed with 503). Checkout grace
  (`checkout_grace:*` in Redis) is activated only by the signature-verified
  Polar webhook; the success URL carries no token and grants nothing.
- The homepage and classifier templates load production Clerk scripts and
  Google Analytics. Production Clerk rejects localhost, while analytics may
  still send traffic. The UI can render a fallback `Sign In` link when Clerk
  fails; that link does not prove authentication works. Use a Clerk test
  configuration that accepts the local origin to verify sign-in.
- Template `url_for` links render as absolute URLs with the request origin.
  Browser checks should use accessible names or inspect the URL pathname,
  rather than match an exact relative `href`.
