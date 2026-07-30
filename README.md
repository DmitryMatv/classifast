# [Classifast.com](https://classifast.com)

## Accurate classifier of UNSPSC, NAICS, HS/CN codes, and more

### Get the right category codes from the most widely used classification standards

Classifast is a web application that provides easy classification of any text input according to international product and service standards like UNSPSC, NAICS, ISIC, ETIM, HS, CN. Built with FastAPI and modern web technologies, it offers fast, accurate semantic search capabilities for automated yet intelligent categorization.

<a href="https://www.producthunt.com/products/classifast-com?embed=true&amp;utm_source=badge-featured&amp;utm_medium=badge&amp;utm_campaign=badge-classifast-com-2-0" target="_blank" rel="noopener noreferrer"><img alt="Classifast.com 2.0 - Introducing Mappings + More accurate results (re-ranking) | Product Hunt" width="250" height="54" src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1189360&amp;theme=light&amp;t=1783342139980"></a>

![CodeRabbit Pull Request Reviews](https://img.shields.io/coderabbit/prs/github/DmitryMatv/classifast?utm_source=oss&utm_medium=github&utm_campaign=DmitryMatv%2Fclassifast&labelColor=171717&color=FF570A&link=https%3A%2F%2Fcoderabbit.ai&label=CodeRabbit+Reviews)

## Features

- **Fast Classification**: Semantic search using advanced embedding models
- **High Accuracy**: Confidence scores for each classification result
- **Multiple Standards**: Support for UNSPSC, ETIM, and NAICS classification standards
- **Modern Interface**: Clean, responsive design built with Tailwind CSS
- **SEO Optimized**: Structured data, meta tags, and performance optimized

## Supported classification standards (Top 3)

### UNSPSC (United Nations Standard Products and Services Codes)

- Global standard for product and service categorization
- Improves spend analytics and procurement processes
- Version: UNv260801 (August 14, 2023)

### ETIM (European Technical Information Model)

- B2B open standard for technical product classification
- Specialized for electrical and technical products
- Version: 10.0 (2024-12-10)

### NAICS 2022 (North American Industry Classification System)

- Industry classification for business activities
- Essential for government contracting and reporting

## Stack

- **Backend**: FastAPI with Python
- **Frontend**: TypeScript, Tailwind CSS, HTMX
- **Vector Database**: Qdrant for semantic search
- **Embedding Models**: Hugging Face Inference with Qwen embeddings
- **Deployment**: Docker containerized

## Demo

Visit the working instance at [classifast.com](https://classifast.com) to try.

## Installation

1. Clone the repository
2. Install Python dependencies: `pip install -r requirements.txt`
3. Install frontend dependencies: `npm install`
4. Set environment variables for embedding inference, API keys, and payments/quota behavior
5. Run frontend watchers and the app: `npm run dev`

Relevant embedding environment variables:

- `HF_TOKEN`
- `HF_INFERENCE_PROVIDER` defaults to `auto`
- `HF_EMBEDDING_MODEL` defaults to `Qwen/Qwen3-Embedding-8B`
- `HF_EMBEDDING_DIMS` defaults to `2048`

Reranking is served by OpenRouter and uses its own API key and model:

- `OPENROUTER_API_KEY`
- `OPENROUTER_RERANK_MODEL` defaults to `nvidia/llama-nemotron-rerank-vl-1b-v2:free`
- `OPENROUTER_RERANK_TIMEOUT_SECONDS` defaults to `30`

Embedding inference uses the configured `HF_INFERENCE_PROVIDER`. Reranking is
sent to OpenRouter's `/api/v1/rerank` endpoint with the `OPENROUTER_API_KEY`.

Relevant Qdrant environment variables:

- `QDRANT_URL` is preferred when set. Full HTTP(S) URLs are preserved; a bare
  hostname is treated as HTTPS.
- `QDRANT_HOST` and `QDRANT_PORT` (default `localhost:6333`) are the fallback
  when `QDRANT_URL` is unset.
- `QDRANT_API_KEY` is optional for unprotected local Qdrant deployments.

Relevant payment and quota environment variables:

- `POLAR_ACCESS_TOKEN`
- `POLAR_WEBHOOK_SECRET`
- `POLAR_PRO_PRODUCT_ID`
- `QUOTA_FAIL_OPEN` defaults to `true`
- `CHECKOUT_GRACE_TTL` defaults to `300`

Frontend JS/CSS files under `app/static/` are build artifacts. They are generated from `app/assets/` and are intentionally not committed to git.

## Deployment

For production, deploy from the repo root with the provided `Dockerfile`. The image builds the frontend assets in a Node/npm builder stage and copies only the compiled files into the final Python runtime image.

For Coolify, use:

- Source directory: repo root
- Build pack: Dockerfile
- Dockerfile path: `./Dockerfile`
- Include Source Commit in Build: disabled

`docker-compose.yaml` can still be used for local or manual container runs, but the recommended Coolify production path is the Dockerfile build directly.

Application startup validates every configured Qdrant collection, vector size,
and required payload index without modifying Qdrant. Startup fails when the
schema contract is invalid. Prepare and verify Qdrant explicitly before a
deployment:

```bash
source .venv/bin/activate
python utilities/sync_payload_indexes.py apply
python utilities/sync_payload_indexes.py check
```

`apply` is a live migration operation: it backfills normalized ID payloads and
creates or replaces payload indexes. `check` is read-only. Run `apply` from a
controlled maintenance environment, verify the resolved Qdrant target before
confirming the operation, then deploy only after `check` succeeds. The utility
loads the repository `.env`; values already exported by the shell or supplied
by the container take precedence. A Qdrant client cleanup failure is reported
as an operational failure and makes the command exit nonzero.

Classification itself remains synchronous and is executed by one dedicated
background worker per application process. Each process admits one active and
up to four waiting classifications; requests above that fixed capacity receive
HTTP 503. This serializes vendor calls while bounding queued work and allowing
health checks, webhooks, and cached-page handling to remain responsive.

## API

- `GET /` Homepage
- `GET /{classifier_type}` Classification page (unspsc, etim, naics)
- `POST /{classifier_type}` Submit classification request
- `GET /health` Health check endpoint

## Testing

Automated backend tests live under `tests/` and use pytest as the test runner.
The existing tests retain their `unittest.TestCase`-compatible structure.

- Run the backend suite: `pytest`
- Run the frontend typecheck: `npm run typecheck`
- Run the Python typecheck: `python -m mypy`

`utilities/test_*.py` scripts are manual smoke/debug helpers for live integrations and are intentionally separate from the main regression suite.

## SEO

- Structured data markup (JSON-LD)
- FAQ schema for common questions
- Optimized meta descriptions and titles
- Breadcrumb navigation
- Semantic HTML structure
- Performance optimized loading
