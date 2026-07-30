import asyncio
import inspect
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote, quote_from_bytes, urlencode, urlparse, urlunparse

import redis.asyncio as redis
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import InferenceClient
from qdrant_client import QdrantClient
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

from . import api, payments, web
from .cache_profiles import (
    STATIC_CODE,
    STATIC_MEDIA,
    STATIC_TEXT,
    CacheProfile,
    build_cache_headers,
)
from .classification_executor import ClassificationExecutor
from .qdrant_connection import create_qdrant_client, resolve_qdrant_url
from .qdrant_schema import (
    QdrantSchemaValidationError,
    validate_configured_collections,
)
from .reranker import HuggingFaceReranker
from .usage_tracker import (
    REDIS_HOST,
    REDIS_PASSWORD,
    REDIS_PORT,
    REDIS_USERNAME,
)

BASE_DIR = Path(__file__).parent.parent


# Configure logging with Dozzle-friendly JSON formatter
class JsonFormatter(logging.Formatter):
    def format(self, record) -> str:
        log = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname.lower(),
            "msg": record.getMessage(),
            "logger": record.name,
        }
        if record.exc_info:
            log["error"] = self.formatException(record.exc_info)
        return json.dumps(log)


handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
logger = logging.getLogger(__name__)

load_dotenv()


@dataclass
class StartupClients:
    embed_client: Any | None = None
    qdrant_client: QdrantClient | None = None
    collection_quantization_cache: dict[str, bool] = field(default_factory=dict)
    redis_client: redis.Redis | None = None
    reranker: HuggingFaceReranker | None = None


def initialize_embed_client() -> Any | None:
    """Initialize the Hugging Face embedding client if credentials are present."""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.error("Error: HF_TOKEN not found in environment variables.")
        return None

    try:
        provider: Any = os.getenv("HF_INFERENCE_PROVIDER", "").strip() or "auto"
        embed_client = InferenceClient(provider=provider, api_key=hf_token)
        logger.info(
            "Hugging Face Inference client initialized successfully with provider=%s.",
            provider,
        )
        return embed_client
    except Exception as e:
        logger.error("Error initializing Hugging Face Inference client: %s", e)
        return None


def _close_qdrant_client_after_startup_failure(client: QdrantClient) -> None:
    """Close Qdrant without allowing cleanup to hide a startup failure."""
    try:
        client.close()
    except Exception:
        logger.exception("Error closing Qdrant client after startup failure.")


def initialize_qdrant_client() -> tuple[QdrantClient, dict[str, bool]]:
    """Initialize Qdrant and validate configured collections without mutation."""
    qdrant_client: QdrantClient | None = None
    try:
        logger.info("Connecting to Qdrant at %s...", resolve_qdrant_url())
        qdrant_client = create_qdrant_client(timeout=30)
        collection_quantization_cache = validate_configured_collections(qdrant_client)
        logger.info(
            "Qdrant schema validation succeeded for %d configured collections.",
            len(collection_quantization_cache),
        )
        return qdrant_client, collection_quantization_cache
    except QdrantSchemaValidationError as exc:
        for issue in exc.issues:
            logger.error("Qdrant contract violation: %s", issue)
        if qdrant_client is not None:
            _close_qdrant_client_after_startup_failure(qdrant_client)
        raise RuntimeError(
            "Failed to initialize Qdrant client: invalid schema"
        ) from exc
    except Exception as e:
        if qdrant_client is not None:
            _close_qdrant_client_after_startup_failure(qdrant_client)
        logger.error("Error initializing Qdrant client: %s", e)
        raise RuntimeError(f"Failed to initialize Qdrant client: {e}") from e


async def initialize_redis_client() -> redis.Redis | None:
    """Initialize Redis for usage tracking when available."""
    redis_client = None
    try:
        logger.info("Connecting to Redis at %s:%d...", REDIS_HOST, REDIS_PORT)
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD or None,
            username=REDIS_USERNAME if REDIS_PASSWORD else None,
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5,
        )
        ping_result = redis_client.ping()
        if inspect.isawaitable(ping_result):
            await ping_result
        logger.info("Redis client initialized successfully.")
        return redis_client
    except asyncio.CancelledError:
        if redis_client:
            try:
                await redis_client.close()
            except Exception:
                logger.exception(
                    "Error closing Redis client after startup cancellation."
                )
        raise
    except Exception as e:
        logger.warning("Redis not available, usage tracking disabled: %s", e)
        if redis_client:
            try:
                await redis_client.close()
            except Exception:
                pass
        return None


def initialize_huggingface_reranker() -> HuggingFaceReranker | None:
    """Initialize Hugging Face reranking when shared credentials are present."""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN not found - reranking disabled")
        return None

    try:
        model_name = (
            os.getenv("HF_RERANK_MODEL", "").strip() or "BAAI/bge-reranker-v2-m3"
        )
        timeout_seconds = float(os.getenv("HF_RERANK_TIMEOUT_SECONDS", "30"))
        if timeout_seconds <= 0:
            raise ValueError("HF_RERANK_TIMEOUT_SECONDS must be greater than zero")
        reranker = HuggingFaceReranker(
            api_key=hf_token,
            model_name=model_name,
            timeout_seconds=timeout_seconds,
        )
        logger.info(
            "Hugging Face reranker initialized successfully with model=%s provider=hf-inference.",
            model_name,
        )
        return reranker
    except Exception as e:
        logger.error("Error initializing Hugging Face reranker: %s", e)
        return None


async def initialize_startup_clients() -> StartupClients:
    """Initialize all startup clients using the existing fatal/non-fatal semantics."""
    clients = StartupClients()
    try:
        clients.embed_client = initialize_embed_client()
        (
            clients.qdrant_client,
            clients.collection_quantization_cache,
        ) = initialize_qdrant_client()
        clients.redis_client = await initialize_redis_client()
        clients.reranker = initialize_huggingface_reranker()
        return clients
    except BaseException:
        await close_startup_clients(clients)
        raise


def assign_startup_clients(
    app: FastAPI,
    clients: StartupClients,
    classification_executor: ClassificationExecutor,
) -> None:
    """Store initialized clients and caches on FastAPI app state."""
    app.state.embed_client = clients.embed_client
    app.state.reranker = clients.reranker
    app.state.qdrant_client = clients.qdrant_client
    app.state.collection_quantization_cache = clients.collection_quantization_cache
    app.state.redis_client = clients.redis_client
    app.state.classification_executor = classification_executor


async def close_startup_clients(clients: StartupClients) -> None:
    """Close startup clients that expose shutdown hooks."""
    if clients.reranker:
        try:
            clients.reranker.close()
            logger.info("Hugging Face reranker closed.")
        except Exception as e:
            logger.error("Error closing Hugging Face reranker: %s", e)
    if clients.qdrant_client:
        try:
            clients.qdrant_client.close()
            logger.info("Qdrant client closed.")
        except Exception as e:
            logger.error("Error closing Qdrant client: %s", e)
    if clients.redis_client:
        try:
            await clients.redis_client.close()
            logger.info("Redis client closed.")
        except Exception as e:
            logger.error("Error closing Redis client: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FastAPI application startup...")
    classification_executor = ClassificationExecutor()
    clients: StartupClients | None = None
    try:
        clients = await initialize_startup_clients()
        assign_startup_clients(app, clients, classification_executor)
        yield
    finally:
        logger.info("FastAPI application shutdown...")
        try:
            await classification_executor.close()
        finally:
            if clients is not None:
                await close_startup_clients(clients)


app = FastAPI(lifespan=lifespan)


# Performance monitoring middleware
class PerformanceMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response


app.add_middleware(PerformanceMiddleware)


# Add Gzip compression middleware, excluding sitemap.xml and robots.txt
# Googlebot may not handle gzipped sitemaps properly
GZIP_EXCLUDED_PATHS = frozenset({"/sitemap.xml", "/robots.txt", "/llms.txt"})


class GZipMiddlewareExcludingSitemap(GZipMiddleware):
    def __init__(
        self, app: ASGIApp, minimum_size: int = 500, compresslevel: int = 9
    ) -> None:
        super().__init__(app, minimum_size, compresslevel)
        self.exclude_paths = GZIP_EXCLUDED_PATHS

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope.get("path") in self.exclude_paths:
            await self.app(scope, receive, send)
        else:
            await super().__call__(scope, receive, send)


app.add_middleware(GZipMiddlewareExcludingSitemap, minimum_size=1000)


# URL Encoding Validation Middleware
class URLEncodingValidationMiddleware(BaseHTTPMiddleware):
    _spam_signatures = [
        "cfRLUnblockHandlers",
        "UnblockHandlers",
        "copyOriginalId",
    ]

    _attack_patterns = re.compile(
        r"(%25){3,}"  # Triple+ encoded % signs
        r"|"  # OR
        r"(\d{2,4})\1{15,}"  # 15+ repetitions of 2-4 digit pattern (removed unnecessary grouping)
        r"|"  # OR
        r"\d{50,}"  # 50+ consecutive digits
        r"|"  # OR
        r"%3c%3c|%3e%3e"  # HTML injection attempts
        r"|"  # OR
        r"(?<![a-zA-Z0-9])[0-9A-Fa-f]{64,}(?![a-zA-Z0-9])"  # 64+ hex chars at word boundaries (catches excessively long hex strings)
    )

    async def dispatch(self, request: Request, call_next):
        decoded_values = (
            list(request.query_params.values()) if request.query_params else []
        )

        # Length checks should count the raw URL once so oversized keys and encoded
        # query text are still bounded without double-counting decoded values.
        length_checked_content = "".join(
            [request.url.path or "", request.url.query or ""]
        )
        if length_checked_content and len(length_checked_content) > 4000:
            logger.warning(
                "Suspicious URL encoding detected: %s...",
                length_checked_content[:100],
            )
            return self._create_error_response()

        pattern_checked_content = "".join(
            [request.url.path or "", request.url.query or "", *decoded_values]
        )

        # Check for attack patterns
        if self._attack_patterns.search(pattern_checked_content):
            logger.warning(
                "Suspicious pattern detected: %s...", pattern_checked_content[:100]
            )
            return self._create_error_response()

        # Check for known spam signatures (Cloudflare bypass attempts)
        for sig in self._spam_signatures:
            if sig in pattern_checked_content:
                logger.warning("Known spam signature detected: %s...", sig)
                return self._create_error_response()

        response = await call_next(request)
        return response

    def _create_error_response(self) -> JSONResponse:
        """Create standardized error response"""
        return JSONResponse(
            status_code=400,
            content={
                "detail": "Request rejected due to suspicious URL encoding patterns",
                "error": "INVALID_ENCODING",
            },
        )


app.add_middleware(URLEncodingValidationMiddleware)


# Query Parameter Normalization Middleware
QUERY_WHITESPACE_RE = re.compile(r"\s+")
QUERY_COMPONENT_EXTRA_SAFE_CHARS = "()*,:"


def _normalize_query_value(value: str) -> str:
    return QUERY_WHITESPACE_RE.sub(" ", value).strip()


def _normalize_query_items(
    query_items: list[tuple[str, str]],
) -> tuple[list[tuple[str, str]], bool]:
    normalized_items = []
    needs_redirect = False

    for key, value in query_items:
        normalized = _normalize_query_value(value)
        normalized_items.append((key, normalized))
        if normalized != value:
            needs_redirect = True

    return normalized_items, needs_redirect


def _quote_query_component(
    s: str | bytes,
    safe: str | bytes,
    encoding: str | None = None,
    errors: str | None = None,
) -> str:
    if isinstance(s, bytes):
        safe_bytes = safe if isinstance(safe, bytes) else safe.encode("ascii")
        return quote_from_bytes(
            s, safe=QUERY_COMPONENT_EXTRA_SAFE_CHARS.encode("ascii") + safe_bytes
        )

    safe_str = safe.decode("ascii") if isinstance(safe, bytes) else safe
    return quote(
        s,
        safe=QUERY_COMPONENT_EXTRA_SAFE_CHARS + safe_str,
        encoding=encoding,
        errors=errors,
    )


def _build_canonical_query(normalized_items: list[tuple[str, str]]) -> str:
    return urlencode(
        normalized_items,
        doseq=True,
        quote_via=_quote_query_component,
    )


def _build_canonical_url(request_url: str, canonical_query: str) -> str:
    parsed = urlparse(request_url)
    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            canonical_query,
            parsed.fragment,
        )
    )


def _query_redirect_response(canonical_url: str) -> Response:
    return Response(status_code=308, headers={"Location": canonical_url})


class QueryNormalizationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        """Normalize query parameters by stripping whitespace and redirect to canonical URL."""
        query_items = list(request.query_params.multi_items())
        normalized_items, needs_redirect = _normalize_query_items(query_items)

        if needs_redirect:
            canonical_query = _build_canonical_query(normalized_items)
            canonical_url = _build_canonical_url(str(request.url), canonical_query)
            logger.info("Redirecting to normalized URL for path: %s", request.url.path)
            return _query_redirect_response(canonical_url)

        return await call_next(request)


app.add_middleware(QueryNormalizationMiddleware)


# Security Headers Middleware
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Set security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=()"
        )

        # Hardened CSP following Google's recommendations and best practices
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://unpkg.com https://www.googletagmanager.com https://www.google-analytics.com https://static.cloudflareinsights.com https://*.clerk.com https://clerk.classifast.com https://accounts.google.com https://challenges.cloudflare.com https://ajax.cloudflare.com; "
            "script-src-elem 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://unpkg.com https://www.googletagmanager.com https://www.google-analytics.com https://static.cloudflareinsights.com https://*.clerk.com https://clerk.classifast.com https://accounts.google.com https://challenges.cloudflare.com https://ajax.cloudflare.com; "
            "worker-src 'self' blob:; "
            "style-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://fonts.googleapis.com; "
            "style-src-elem 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://fonts.googleapis.com https://accounts.google.com/gsi/style; "
            "img-src 'self' data: https: https://*.googleapis.com https://*.gstatic.com https://*.clerk.com https://clerk.classifast.com; "
            "font-src 'self' https://fonts.gstatic.com https://*.googleapis.com https://*.gstatic.com; "
            "connect-src 'self' https: https://*.clerk.com https://accounts.google.com https://accounts.google.com/gsi/ https://*.googleapis.com https://challenges.cloudflare.com; "
            "frame-src 'self' https://accounts.google.com https://challenges.cloudflare.com; "
            "base-uri 'self'; "
            "form-action 'self'; "
            "manifest-src 'self'; "
            "object-src 'none'; "
            "frame-ancestors 'none'; "
            "upgrade-insecure-requests;"
        )

        return response


app.add_middleware(SecurityHeadersMiddleware)


# Mount static files with cache-profile-based browser and Cloudflare TTLs.
def get_static_cache_profile(path: str) -> CacheProfile:
    """Map a static asset path to the cache profile used for the response."""
    if path.endswith((".png", ".jpg", ".ico", ".pdf", ".zip", ".xlsx")):
        return STATIC_MEDIA
    if path.endswith((".css", ".js", ".min.js")):
        return STATIC_CODE
    if path.endswith(".csv"):
        return STATIC_TEXT
    return STATIC_TEXT


class CachedStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if isinstance(response, Response):
            response.headers.update(build_cache_headers(get_static_cache_profile(path)))

            # ETag for conditional requests (CF uses this for revalidation)
            try:
                if self.directory:
                    file_path = Path(self.directory) / path
                    file_stat = file_path.stat()
                    response.headers["ETag"] = (
                        f'"{int(file_stat.st_mtime)}-{file_stat.st_size}"'
                    )
                else:
                    response.headers["ETag"] = f'"{hash(path)}"'
            except (OSError, FileNotFoundError):
                response.headers["ETag"] = f'"{hash(path)}"'

            # Let CF handle compression and vary cache by encoding
            response.headers["Vary"] = "Accept-Encoding"
            # CF-specific: tag for cache purging via API
            response.headers["Cache-Tag"] = "static-files"
        return response


app.mount(
    "/static", CachedStaticFiles(directory=BASE_DIR / "app" / "static"), name="static"
)


# Root-level static files (browsers/crawlers expect these at root).
def static_file_response(
    path: str, cache_profile: CacheProfile = STATIC_TEXT
) -> FileResponse:
    """Serve a static file with Cloudflare-optimized cache headers."""
    file_path = BASE_DIR / "app" / "static" / path
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    response = FileResponse(file_path)
    response.headers.update(build_cache_headers(cache_profile))
    response.headers["Cache-Tag"] = "static-files"
    return response


@app.get("/favicon.ico", response_class=FileResponse, include_in_schema=False)
async def favicon():
    return static_file_response("images/favicon.ico", cache_profile=STATIC_MEDIA)


@app.get("/robots.txt", response_class=FileResponse)
async def robots_txt():
    return static_file_response("robots.txt", cache_profile=STATIC_TEXT)


@app.get("/sitemap.xml", response_class=FileResponse)
async def sitemap_xml():
    return static_file_response("sitemap.xml", cache_profile=STATIC_TEXT)


@app.get("/llms.txt", response_class=FileResponse)
async def llms_txt():
    return static_file_response("llms.txt", cache_profile=STATIC_TEXT)


# Healthcheck
@app.get("/health")
async def health_check(request: Request):
    """
    Health check endpoint for Docker/Kubernetes.
    Returns generic status to avoid information disclosure.
    """
    embed_client = getattr(request.app.state, "embed_client", None)
    qdrant_client = getattr(request.app.state, "qdrant_client", None)

    if not embed_client or not qdrant_client:
        raise HTTPException(
            status_code=503,
            detail="Service Unavailable",
        )

    try:
        await asyncio.wait_for(
            asyncio.to_thread(qdrant_client.get_collections), timeout=5
        )
        return {"status": "healthy"}
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Service Unavailable",
        )


# Include routers
app.include_router(api.router, prefix="/api/v1/rapid", tags=["rapidapi"])
app.include_router(payments.router, prefix="/api", tags=["payments"])
app.include_router(web.router)
