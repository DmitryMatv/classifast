# AGENTS.md

The role of this file is to describe common mistakes and confusion points that agents might encounter as they work in this project. If you ever encounter something in this project that surprises you, please alert the developer working with you and indicate that this is the case in the AGENTS.md file to help prevent future agents from having the same issue.

## Coding Agent Guidelines for Classifast

Classifast is a code search service web application that is using embeddings and vector search (Qdrant) to classify any text input (mostly product descriptions) into categories of various industry standard classifications, like UNSPSC, NAICS, CN/HS codes, ISIS, ETIM, CPV, etc.

### Tech Stack

- Backend: Python FastAPI
- Frontend: TypeScript with Tailwind CSS, served via FastAPI, Bun
- Infrastructure: Redis (usage tracking), Qdrant (vector database), ZeroEntropy (reranking), Clerk (authentication), Polar (payments)

### Hardware & Deployment

Self-hosted from Raspberry Pi 4 (4GB) via Coolify behind full Cloudflare Tunnel. The app uses Cloudflare's CDN edge caching to reduce API costs and improve performance. Classification results (Gemini embeddings, ZeroEntropy reranking, Qdrant search) are cached at edge for 7 days.

### When Modifying Cache Behavior

- Never add `Set-Cookie` to fragment responses - breaks CDN caching
- Paywalls must use `no-store` - prevents serving cached paywall to allowed users
- Full pages can set cookies - they're browser-only anyway due to `Set-Cookie` header
