import asyncio
import inspect
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

import redis.asyncio as redis
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from google import genai
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
    def format(self, record):
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


def get_payload_index_schema(
    field_name: str,
) -> models.KeywordIndexParams | models.TextIndexParams:
    """Return the expected payload index schema for a classifier field."""
    if field_name == "original_id":
        return build_original_id_index_params()
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
    for field_name in ("original_id", "class_name"):
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
                    "Payload index for field '%s' in collection '%s' already exists. Existing collections may need utilities/create_text_indexes.py if 'original_id' was previously indexed as TEXT.",
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs when the application starts
    logger.info("FastAPI application startup...")

    # Initialize Embedding Client (Google GenAI)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    embed_client = None
    if not GEMINI_API_KEY:
        logger.error("Error: GEMINI_API_KEY not found in environment variables.")
    else:
        try:
            embed_client = genai.Client(api_key=GEMINI_API_KEY)
            embed_client.models.list()  # Test connection
            logger.info("Google GenAI Client initialized successfully.")
        except Exception as e:
            logger.error("Error initializing Google GenAI Client: %s", e)
            embed_client = None

    # Initialize Qdrant Client with connection pooling
    qdrant_client = None
    collection_quantization_cache = {}  # Cache quantization config per collection
    try:
        logger.info("Connecting to Qdrant at %s:%d...", QDRANT_HOST, QDRANT_PORT)
        qdrant_client = QdrantClient(
            url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
            api_key=QDRANT_API_KEY or None,
            timeout=30,
        )

        # Check if Qdrant client can list collections as a health check
        existing_collections = set()
        try:
            collections_result = qdrant_client.get_collections()
            existing_collections = {col.name for col in collections_result.collections}
            collection_names = sorted(list(existing_collections))
            logger.info(
                "Qdrant client initialized. Found collections: %s", collection_names
            )
        except Exception as e:
            logger.error(
                "Qdrant client initialized, but could not list collections: %s", e
            )

        # Verify collections exist and store their vector sizes
        for classifier_type, config in CLASSIFIER_CONFIG.items():
            embed_dims = config.get("embed_dims")
            for version, version_config in config["versions"].items():
                collection_name = version_config.get("collection_name")
                if not collection_name:
                    continue

                # Check against the set of existing collections instead of making a new API call
                if collection_name not in existing_collections:
                    logger.warning(
                        "Warning: Collection %s for %s version %s does not exist.",
                        collection_name,
                        classifier_type,
                        version,
                    )
                    continue

                # Get collection info and check vector configuration
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

                # Cache quantization config for this collection
                has_quantization = (
                    collection_info.config.quantization_config is not None
                )
                collection_quantization_cache[collection_name] = has_quantization

                # Ensure text-search indexes exist for MatchText lookups.
                provision_payload_indexes(qdrant_client, collection_name)

    except Exception as e:
        logger.error("Error initializing Qdrant client: %s", e)
        raise RuntimeError(f"Failed to initialize Qdrant client: {e}") from e

    # Initialize Redis client for usage tracking
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
    except Exception as e:
        logger.warning("Redis not available, usage tracking disabled: %s", e)
        if redis_client:
            try:
                await redis_client.close()
            except Exception:
                pass
        redis_client = None

    # Initialize ZeroEntropy client for reranking
    ZEROENTROPY_API_KEY = os.getenv("ZEROENTROPY_API_KEY")
    zclient = None
    if not ZEROENTROPY_API_KEY:
        logger.warning("ZEROENTROPY_API_KEY not found - reranking disabled")
    else:
        try:
            zclient = ZeroEntropy()
            logger.info("ZeroEntropy client initialized successfully.")
        except Exception as e:
            logger.error("Error initializing ZeroEntropy client: %s", e)
            zclient = None

    # Store clients and caches in app state
    app.state.embed_client = embed_client
    app.state.zclient = zclient
    app.state.qdrant_client = qdrant_client
    app.state.collection_quantization_cache = collection_quantization_cache
    app.state.redis_client = redis_client

    yield

    # Runs when the application is shutting down
    logger.info("FastAPI application shutdown...")
    if qdrant_client:
        try:
            qdrant_client.close()
            logger.info("Qdrant client closed.")
        except Exception as e:
            logger.error("Error closing Qdrant client: %s", e)
    if redis_client:
        try:
            await redis_client.close()
            logger.info("Redis client closed.")
        except Exception as e:
            logger.error("Error closing Redis client: %s", e)


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
class GZipMiddlewareExcludingSitemap(GZipMiddleware):
    def __init__(self, app: ASGIApp, minimum_size: int = 500, compresslevel: int = 9):
        super().__init__(app, minimum_size, compresslevel)
        self.exclude_paths = {"/sitemap.xml", "/robots.txt", "/llms.txt"}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope.get("path") in self.exclude_paths:
            await self.app(scope, receive, send)
        else:
            await super().__call__(scope, receive, send)


app.add_middleware(GZipMiddlewareExcludingSitemap, minimum_size=1000)


# URL Encoding Validation Middleware
class URLEncodingValidationMiddleware(BaseHTTPMiddleware):
    # Simplified regex combining all attack patterns
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
class QueryNormalizationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        """Normalize query parameters by stripping whitespace and redirect to canonical URL."""
        query_items = list(request.query_params.multi_items())
        normalized_items = []
        needs_redirect = False

        for key, value in query_items:
            # Normalize: replace multiple spaces/newlines with single space and strip
            normalized = re.sub(r"\s+", " ", value).strip()
            normalized_items.append((key, normalized))
            if normalized != value:
                needs_redirect = True

        if needs_redirect:
            # Build canonical URL with normalized parameters
            from urllib.parse import quote, urlencode, urlparse, urlunparse

            parsed = urlparse(str(request.url))
            # Use quote to preserve %20 encoding (not +) for consistency with Cloudflare cache
            # Use safe='()' to preserve parentheses as they were in the original URL
            canonical_query = urlencode(
                normalized_items,
                doseq=True,
                quote_via=lambda s, safe, encoding=None, errors=None: quote(
                    s, safe="()*,:" + safe, encoding=encoding, errors=errors
                ),
            )
            canonical_url = urlunparse(
                (
                    parsed.scheme,
                    parsed.netloc,
                    parsed.path,
                    parsed.params,
                    canonical_query,
                    parsed.fragment,
                )
            )
            logger.info("Redirecting to normalized URL for path: %s", parsed.path)
            return Response(status_code=308, headers={"Location": canonical_url})

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
    if path.endswith((".woff", ".woff2", ".png", ".jpg", ".ico", ".pdf", ".zip", ".xlsx")):
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

    # Basic check: if we can reach here, the app is running.
    if embed_client and qdrant_client:
        # Optionally, perform a quick check on clients
        try:
            # Test embed client (run in thread pool to avoid blocking)
            await asyncio.to_thread(embed_client.models.list)
            # Test qdrant client (run in thread pool to avoid blocking)
            await asyncio.to_thread(qdrant_client.get_collections)
            return {"status": "healthy"}
        except Exception:
            raise HTTPException(
                status_code=503,
                detail="Service Unavailable",
            )
    else:
        raise HTTPException(
            status_code=503,
            detail="Service Unavailable",
        )


# Include routers
app.include_router(api.router, prefix="/api/v1/rapid", tags=["rapidapi"])
app.include_router(payments.router, prefix="/api", tags=["payments"])
app.include_router(web.router)
