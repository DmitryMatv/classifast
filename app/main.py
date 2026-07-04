import asyncio
import inspect
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, quote_from_bytes, urlencode, urlparse, urlunparse

import redis.asyncio as redis
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from qdrant_client import QdrantClient, models
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send
from zeroentropy import ZeroEntropy

from . import api, payments, web
from .cache_profiles import (
    STATIC_CODE,
    STATIC_MEDIA,
    STATIC_TEXT,
    CacheProfile,
    build_cache_headers,
)
from .classifier_config import CLASSIFIER_CONFIG
from .id_lookup import (
    ORIGINAL_ID_FIELD,
    ORIGINAL_ID_NORMALIZED_FIELD,
    ORIGINAL_ID_NORMALIZED_REVERSED_FIELD,
)
from .usage_tracker import (
    QDRANT_API_KEY,
    QDRANT_HOST,
    QDRANT_PORT,
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
    embed_client: Any | None
    qdrant_client: QdrantClient | None
    collection_quantization_cache: dict[str, bool]
    redis_client: redis.Redis | None
    zclient: ZeroEntropy | None


def build_original_id_index_params() -> models.KeywordIndexParams:
    """Return the exact-match payload index settings for classification IDs."""
    return models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD)


def build_class_name_text_index_params() -> models.TextIndexParams:
    """Return the text-search payload index settings for class names."""
    return models.TextIndexParams(
        type=models.TextIndexType.TEXT,
        tokenizer=models.TokenizerType.WORD,
        min_token_len=1,
        max_token_len=30,
        lowercase=True,
    )


def build_normalized_original_id_text_index_params() -> models.TextIndexParams:
    """Return prefix text-search settings for normalized classification IDs."""
    return models.TextIndexParams(
        type=models.TextIndexType.TEXT,
        tokenizer=models.TokenizerType.PREFIX,
        min_token_len=1,
        max_token_len=64,
        lowercase=True,
    )


def get_payload_index_schema(
    field_name: str,
) -> models.KeywordIndexParams | models.TextIndexParams:
    """Return the expected payload index schema for a classifier field."""
    if field_name == ORIGINAL_ID_FIELD:
        return build_original_id_index_params()
    if field_name in {
        ORIGINAL_ID_NORMALIZED_FIELD,
        ORIGINAL_ID_NORMALIZED_REVERSED_FIELD,
    }:
        return build_normalized_original_id_text_index_params()
    if field_name == "class_name":
        return build_class_name_text_index_params()
    raise KeyError(f"Unsupported payload index field: {field_name}")


def provision_payload_indexes(
    qdrant_client: QdrantClient,
    collection_name: str,
) -> None:
    """
    Provision Qdrant payload indexes required by classifier lookups.

    Args:
        qdrant_client: The Qdrant client instance
        collection_name: The name of the collection to index
    """
    for field_name in (
        ORIGINAL_ID_FIELD,
        ORIGINAL_ID_NORMALIZED_FIELD,
        ORIGINAL_ID_NORMALIZED_REVERSED_FIELD,
        "class_name",
    ):
        try:
            qdrant_client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=get_payload_index_schema(field_name),
                wait=True,
            )
            logger.info(
                "Created payload index for field '%s' in collection '%s'",
                field_name,
                collection_name,
            )
        except Exception as e:
            error_message = str(e).lower()
            if "already exists" in error_message:
                logger.warning(
                    "Payload index for field '%s' in collection '%s' already exists. Existing collections may need utilities/sync_payload_indexes.py if payload indexes or normalized ID payload fields are out of date.",
                    field_name,
                    collection_name,
                )
                continue

            logger.warning(
                "Could not create payload index for field '%s' in collection '%s': %s",
                field_name,
                collection_name,
                e,
            )


def initialize_embed_client() -> Any | None:
    """Initialize the OpenRouter embedding client if credentials are present."""
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        logger.error("Error: OPENROUTER_API_KEY not found in environment variables.")
        return None

    try:
        base_url = (
            os.getenv("OPENROUTER_BASE_URL", "").strip()
            or "https://openrouter.ai/api/v1"
        )
        embed_client = OpenAI(
            base_url=base_url,
            api_key=openrouter_api_key,
            max_retries=0,
            timeout=60,  # tune to your SLA; avoids multi-minute hangs per attempt
        )
        logger.info(
            "OpenRouter embedding client initialized successfully with base_url=%s.",
            base_url,
        )
        return embed_client
    except Exception as e:
        logger.error("Error initializing OpenRouter embedding client: %s", e)
        return None


def get_existing_qdrant_collections(qdrant_client: QdrantClient) -> set[str]:
    """Return collection names visible to Qdrant, or an empty set on list failure."""
    try:
        collections_result = qdrant_client.get_collections()
        existing_collections = {col.name for col in collections_result.collections}
        collection_names = sorted(list(existing_collections))
        logger.info(
            "Qdrant client initialized. Found collections: %s",
            collection_names,
        )
        return existing_collections
    except Exception as e:
        logger.error(
            "Qdrant client initialized, but could not list collections: %s",
            e,
        )
        return set()


def validate_qdrant_collections(
    qdrant_client: QdrantClient,
    existing_collections: set[str],
) -> dict[str, bool]:
    """Validate configured Qdrant collections and provision required payload indexes."""
    collection_quantization_cache = {}

    for classifier_type, config in CLASSIFIER_CONFIG.items():
        embed_dims = config.get("embed_dims")
        for version, version_config in config["versions"].items():
            collection_name = version_config.get("collection_name")
            if not collection_name:
                continue

            if collection_name not in existing_collections:
                logger.warning(
                    "Warning: Collection %s for %s version %s does not exist.",
                    collection_name,
                    classifier_type,
                    version,
                )
                continue

            collection_info = qdrant_client.get_collection(collection_name)
            vector_params = collection_info.config.params.vectors

            if isinstance(vector_params, dict) and "size" in vector_params:
                vector_size = vector_params["size"]
                if vector_size != embed_dims:
                    logger.warning(
                        "Warning: Collection %s has vector size %d but config specifies %d",
                        collection_name,
                        vector_size,
                        embed_dims,
                    )

            has_quantization = collection_info.config.quantization_config is not None
            collection_quantization_cache[collection_name] = has_quantization
            provision_payload_indexes(qdrant_client, collection_name)

    return collection_quantization_cache


def initialize_qdrant_client() -> tuple[QdrantClient, dict[str, bool]]:
    """Initialize Qdrant and validate configured collections."""
    try:
        logger.info("Connecting to Qdrant at %s:%d...", QDRANT_HOST, QDRANT_PORT)
        qdrant_client = QdrantClient(
            url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
            api_key=QDRANT_API_KEY or None,
            timeout=30,
        )
        existing_collections = get_existing_qdrant_collections(qdrant_client)
        collection_quantization_cache = validate_qdrant_collections(
            qdrant_client,
            existing_collections,
        )
        return qdrant_client, collection_quantization_cache
    except Exception as e:
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
    except Exception as e:
        logger.warning("Redis not available, usage tracking disabled: %s", e)
        if redis_client:
            try:
                await redis_client.close()
            except Exception:
                pass
        return None


def initialize_zeroentropy_client() -> ZeroEntropy | None:
    """Initialize ZeroEntropy reranking when credentials are present."""
    zeroentropy_api_key = os.getenv("ZEROENTROPY_API_KEY")
    if not zeroentropy_api_key:
        logger.warning("ZEROENTROPY_API_KEY not found - reranking disabled")
        return None

    try:
        zclient = ZeroEntropy()
        logger.info("ZeroEntropy client initialized successfully.")
        return zclient
    except Exception as e:
        logger.error("Error initializing ZeroEntropy client: %s", e)
        return None


async def initialize_startup_clients() -> StartupClients:
    """Initialize all startup clients using the existing fatal/non-fatal semantics."""
    embed_client = initialize_embed_client()
    qdrant_client, collection_quantization_cache = initialize_qdrant_client()
    redis_client = await initialize_redis_client()
    zclient = initialize_zeroentropy_client()
    return StartupClients(
        embed_client=embed_client,
        qdrant_client=qdrant_client,
        collection_quantization_cache=collection_quantization_cache,
        redis_client=redis_client,
        zclient=zclient,
    )


def assign_startup_clients(app: FastAPI, clients: StartupClients) -> None:
    """Store initialized clients and caches on FastAPI app state."""
    app.state.embed_client = clients.embed_client
    app.state.zclient = clients.zclient
    app.state.qdrant_client = clients.qdrant_client
    app.state.collection_quantization_cache = clients.collection_quantization_cache
    app.state.redis_client = clients.redis_client


async def close_startup_clients(clients: StartupClients) -> None:
    """Close startup clients that expose shutdown hooks."""
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
    clients = await initialize_startup_clients()
    assign_startup_clients(app, clients)

    yield

    logger.info("FastAPI application shutdown...")
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
