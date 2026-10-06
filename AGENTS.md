# AGENTS.md

The role of this file is to describe common mistakes and confusion points that agents might encounter as they work in this project. If you ever encounter something in this project that surprises you, please alert the developer working with you and indicate that this is the case in the AGENTS.md file to help prevent future agents from having the same issue.

## Testing

Always use `npm test` or `npm run test:watch` for frontend tests.

Always use `pytest` for backend tests. The suite retains `unittest`-compatible
test classes and standard-library mocks, but pytest is the official runner.

Fresh checkouts may lack `.venv`, `node_modules`, and generated frontend assets
because they are ignored. If `.venv` is absent, run `python -m venv .venv` and
`.venv/bin/pip install -r requirements-dev.txt` before `pytest`. If
`node_modules` is absent, run `npm ci` before frontend tests or builds. The
verification driver (`.agents/skills/verify/`) also requires a frontend build
(`npm run build`).

pytest.ini scopes pytest collection to `tests/`. The `utilities/test_*.py` files are
manual live/debug helpers, and the ignored `embedders/tests/` tree contains
separate experimental tests that are not part of the maintained backend suite.
`tests/test_emdn_embedder.py` imports the ignored
`embedders/embedder_remote_EMDN_hf.py`; a checkout without that local file cannot
collect the full suite. Report that limitation when excluding this test.

Activate the Python environment with `source .venv/bin/activate` before backend
tests or the verification driver.

jsdom prints an exception thrown inside an event listener, but the test still
passes and `npm test` exits 0. Read the test output as well as the exit status.
To prove a listener does not throw, capture `window` `error` events in the test,
as `common.test.ts` does for `ResultCopier`.

`utilities/qdrant_config.py` is an executable migration-style script, not passive configuration. Importing or running it updates a hardcoded Qdrant collection, so review it carefully before execution.

`utilities/sync_payload_indexes.py check` is read-only.
`utilities/sync_payload_indexes.py apply` is a migration-style command that
backfills payloads and creates or replaces indexes in configured Qdrant
collections. Runtime startup must validate Qdrant without creating, replacing,
or deleting indexes.

## Project Snapshot

Classifast is a classification service web application that uses embeddings and vector search (Qdrant) to classify any text input (mostly product descriptions) into categories of various industry standard classifications, like UNSPSC, NAICS, CN/HS codes, ISIC, ETIM, CPV, etc.

## Tech Stack

- Backend: Python FastAPI
- Frontend: TypeScript with Tailwind CSS (built with Node/npm and Vite, served via FastAPI)
- Infrastructure: Redis (usage tracking), Qdrant (vector database), Hugging Face Inference (embeddings), OpenRouter (deployed-service reranking and opt-in query enhancement), Clerk (authentication), Polar (payments)

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

- Polar SDK version range is declared in `requirements.txt`.
  SDK 1.x uses versioned `polar` imports, direct checkout keyword arguments, and
  webhook dataclasses with `event.type` instead of `event.TYPE`.
  The `polar.v2026_10` webhook parser requires `api_version` and complete
  subscription fields even when their values are null. Check the dashboard
  endpoint version when migrating older webhook payloads.
  Invalid signatures return 403. Verified unknown event types are acknowledged
  without changing entitlements; malformed webhook payloads return 400.
- `data/`, `embedders/`, and `mapping/` are gitignored and may be absent from
  a checkout. Ripgrep silently returns zero hits inside them because it
  respects `.gitignore`; use `--no-ignore` or explicit paths. If `embedders/`
  is absent, `pytest` fails while collecting `tests/test_emdn_embedder.py`.
  Run `pytest --ignore=tests/test_emdn_embedder.py` for the remaining suite.
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
- HTMX 4 history restore preserves `document.body` and replaces its children.
  Initialize restored controls after the BODY `htmx:after:swap` with
  `HX-History-Restore-Request`, and retire handlers for the previous form.
  Restored result autoload must use `push_url=false` to preserve Forward.
- `common.ts` can load through both a versioned template URL and the classifier's
  unversioned module import. Both instances must share one document owner for
  Clerk bootstrap, token refresh, and global handlers.
- `htmx.min.js` is vendored in `app/static` and must use `asset_url` like the
  application scripts. An unversioned URL can pair cached HTMX with incompatible
  event handlers after an upgrade. `emptyOutDir: false` in `vite.config.ts`
  protects it from build cleanup.
- The app assumes a single uvicorn worker. The JWKS client, asset versions,
  crawler IP ranges, and the classification queue capacity are all
  process-local. Scaling workers changes their semantics.
- There is no integration test suite. Most tests use mocks, including the checkout
  rate-limit tests. The suite does not prove deployed Qdrant, Redis, or Hugging
  Face connectivity. Manual live helpers live in `utilities/test_*.py`, which
  pytest excludes from collection.
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
- `CachedStaticFiles` must not replace the `ETag` that Starlette sets.
  Starlette compares `If-None-Match` with its own ETag inside `file_response`,
  before `CachedStaticFiles.get_response` adds headers. A replaced ETag turns
  every revalidation into a 200. `tests/test_static_headers.py` covers 304
  responses and changed-file revalidation on the real mount.
- Checkout rate limiting requires Redis 7+ for `EXPIRE NX`. Queue `INCR` and
  `EXPIRE NX` in one transactional pipeline. This assigns missing TTLs, including
  on stranded counters, while preserving existing deadlines. Redis errors must
  still return 503, and counters above the allowance must still return 429.
- Tier-cache lookup fills use `SET EX NX` and reread the winning value.
  Webhook tier updates use authoritative `SETEX`; a lookup completing afterward
  must preserve and return that tier. Upgrade and downgrade overlap tests also
  verify quota behavior without checkout grace.
- `ClassificationExecutor` holds one turn per complete classification, not per
  thread stage. Asynchronous query enhancement between stages keeps the turn.
  Each process admits one active and four waiting classifications
  (`QUEUE_CAPACITY`). The next request raises `ClassificationQueueFull`. The
  fragment and RapidAPI routes log a warning and return 503 with `no-store`.
  An overloaded SSR page falls back to client-side loading with `no-store`
  but stays indexable. Verified Google crawlers get that page as a 503 with
  `Retry-After` instead. Only genuine SSR failures send `noindex`, because
  Google drops a 200 page marked `noindex` but retries a 503.
- A fragment request runs crawler verification, Clerk caller resolution
  (`resolve_signed_in_caller`), queue admission, the turn, the Redis quota
  charge (the `authorize` callback), and then the pipeline, in that order.
  Overflowed and cancelled waiting requests are never charged. Keep slow or
  networked checks out of `authorize`, because it runs while holding the only
  turn. A Clerk signing-key fetch can take 30 seconds and a tier lookup 5.
- Cancelling a waiting classification frees its slot without running it.
  Cancelling the active one returns immediately, but the job keeps its slot
  until the running thread stage finishes. Shutdown cancels waiting jobs and
  drains the active one before shared clients close.
- Python 3.14's `asyncio.shield` logs a late failure of the shielded future
  after its waiter is cancelled, even when code retrieves that failure. The
  executor waits with `_wait_through_cancellation` (`asyncio.wait` plus
  explicit retrieval) instead. Keep the raced worker-error regression tests
  when changing these waits.
- Checkout recovery retries two seconds after each completed request and has a
  60-second deadline. An input or change inside the form cancels recovery,
  switches to the existing manual-retry message, and enables Try again.
  Keep displayed status consistent with whether automatic checks are running;
  update only the recovery status paragraph, preserving a fresh ordinary paywall.
  Successful recovery must leave results visible without recreating a warning.
