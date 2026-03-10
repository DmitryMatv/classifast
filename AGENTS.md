# AGENTS.md

The role of this file is to describe common mistakes and confusion points that agents might encounter as they work in this project. If you ever encounter something in this project that surprises you, please alert the developer working with you and indicate that this is the case in the AGENTS.md file to help prevent future agents from having the same issue.

## Project Snapshot

Classifast is a classification service web application that uses embeddings and vector search (Qdrant) to classify any text input (mostly product descriptions) into categories of various industry standard classifications, like UNSPSC, NAICS, CN/HS codes, ISIS, ETIM, CPV, etc.

## Agent Notes

- `app/web.py` still uses the older `templates.TemplateResponse(name, context)` signature. Route-level tests emit a Starlette deprecation warning until it is updated to `TemplateResponse(request, name, context)` style.

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

## Rapid API (API.py)

`app/api.py` is specifically made for the Rapid API platform. It contains endpoints that make the classification service accessible on that platform. Ignore api.py unless explicitly asked to work on Rapid API service integration.

## Docker Build Context Gotcha

This repo already has a `.dockerignore`. Before changing Docker/Coolify build behavior, inspect it first instead of assuming the Docker build context matches the git tree. It may already exclude files you expect to be available during image build.

## Compose Config Secret Leakage Gotcha

`docker compose config` expands values from `.env` and prints them in plaintext. Do not paste its full output into chat or logs when validating Docker/Compose changes in this repo unless secrets are redacted first.

## Browser History / HTMX

- `app/web.py` fragment requests used to overload `url_change` for three unrelated concerns: pushing browser history, updating the page title, and deciding whether usage tracking/paywalls applied. This coupling caused confusing Back/Forward behavior, especially for direct search URLs that auto-loaded results via HTMX.
- Keep history concerns (`push_url`) separate from quota concerns (`track_usage`) when touching classifier fragments or the initial hidden HTMX loader in `classifier_page.html`.

## Template Path Checks Gotcha

- `app/web.py` normalizes classifier routes to uppercase before rendering. Do not make template behavior depend on case-sensitive lowercase path sniffing like `'unspsc' in request.url.path`; pass explicit state or use data-driven formatting instead.
