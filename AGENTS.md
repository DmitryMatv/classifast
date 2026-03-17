# AGENTS.md

The role of this file is to describe common mistakes and confusion points that agents might encounter as they work in this project. If you ever encounter something in this project that surprises you, please alert the developer working with you and indicate that this is the case in the AGENTS.md file to help prevent future agents from having the same issue.

Never run `bun test`. Always use `bun run test` (runs Vitest).

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

## SEO Gotchas

- Generated classifier search pages can look indexable because they have clean canonicals and titles, but if the unique results are only HTMX-loaded, bots may still see mostly boilerplate and defer indexing.
- Time-sensitive SEO claims such as the "current version" of a classifier must come from `CLASSIFIER_CONFIG`, not hardcoded template prose or JSON-LD.

## Qdrant Index Contract Gotcha

- Do not add a Qdrant full-text index to `original_id`.
- Exact ID lookup in `app/classifier.py` uses `MatchValue` on `original_id`, so that field should use a `keyword` payload index.
- Partial ID lookup in `app/classifier.py` uses `MatchText` plus Python-side normalization/filtering, and it relies on `original_id` not having a full-text index so Qdrant keeps substring-style matching behavior for that field.
- Use `text(word)` payload indexes only for human-readable fields such as `class_name`.

## Rapid API (API.py)

`app/api.py` is specifically made for the Rapid API platform. It contains endpoints that make the classification service accessible on that platform. Ignore api.py unless explicitly asked to work on Rapid API service integration.
