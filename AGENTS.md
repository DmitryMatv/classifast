# AGENTS.md

The role of this file is to describe common mistakes and confusion points that agents might encounter as they work in this project. If you ever encounter something in this project that surprises you, please alert the developer working with you and indicate that this is the case in the AGENTS.md file to help prevent future agents from having the same issue.

Never run `bun test` (runs Bun's built-in test runner). Always use `bun run test` (runs Vitest).

## Project Snapshot

Classifast is a classification service web application that uses embeddings and vector search (Qdrant) to classify any text input (mostly product descriptions) into categories of various industry standard classifications, like UNSPSC, NAICS, CN/HS codes, ISIS, ETIM, CPV, etc.

## Tech Stack

- Backend: Python FastAPI
- Frontend: TypeScript with Tailwind CSS (built with Bun, served via FastAPI)
- Infrastructure: Redis (usage tracking), Qdrant (vector database), ZeroEntropy (reranking), Clerk (authentication), Polar (payments)

## Hardware, Deployment, Cache

Self-hosted from Raspberry Pi 4 (4GB) via Coolify behind Cloudflare Tunnel (Full HTTPS/TLS Setup for All Resources). The app uses Cloudflare's CDN edge caching to reduce API costs and improve performance. Classification results (Gemini embeddings, ZeroEntropy reranking, Qdrant search) are cached at edge for 7 days.

## When Modifying Cache Behavior

- Never add `Set-Cookie` to fragment responses - breaks CDN caching
- Paywalls must use `no-store` - prevents serving cached paywall to allowed users
- Full pages can set cookies - but prefer client-side JavaScript to prevent CDN cache pollution
- Generate per-user state client-side when possible (e.g., tracking IDs via `crypto.randomUUID()`) instead of server-side templating - keeps HTML cacheable across all users
- `app/templates/classifier_page.html` currently does not have a hidden first-party `push_url` form input. The cache-key split came from the initial loader's `hx-vals`, while the form itself needed an explicit hidden `track_usage=true` input to keep first-party behavior consistent.
- Do not infer first-party `HX-Push-Url` server-side from `HX-Current-URL`. That makes a cacheable fragment vary on browser-local state. Keep first-party history decisions in client-side `hx-push-url` attributes instead.

## Qdrant Index Contract Gotcha

- Do not add a Qdrant full-text index to `original_id`.
- Exact ID lookup in `app/classifier.py` uses `MatchValue` on `original_id`, so that field should use a `keyword` payload index.
- Partial ID lookup in `app/classifier.py` uses `MatchText` plus Python-side normalization/filtering, and it relies on `original_id` not having a full-text index so Qdrant keeps substring-style matching behavior for that field.
- Use `text(word)` payload indexes only for human-readable fields such as `class_name`.

## Rapid API (API.py)

`app/api.py` is specifically made for the Rapid API platform. It contains endpoints that make the classification service accessible on that platform. Ignore api.py unless explicitly asked to work on Rapid API service integration.
