import asyncio
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
from zeroentropy import ZeroEntropy

from . import api, payments, web
from .classifier_config import CLASSIFIER_CONFIG
from .dependencies import limiter
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


def ensure_text_search_indexes(
    qdrant_client: QdrantClient,
    collection_name: str,
) -> None:
    """
    Ensure keyword payload indexes exist for text search fields.
    Creates indexes for 'original_id' and 'class_name' fields if they don't exist.

    These indexes significantly speed up exact text matching (scroll API with MatchValue filter)
    on large collections by avoiding full collection scans.

    Args:
        qdrant_client: The Qdrant client instance
        collection_name: The name of the collection to index
    """
    fields_to_index = ["original_id", "class_name"]

    for field_name in fields_to_index:
        try:
            qdrant_client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )
            logger.info(
                "Created payload index for field '%s' in collection '%s'",
                field_name,
                collection_name,
            )
        except Exception as e:
            logger.warning(
                "Could not create payload index for field '%s': %s",
                field_name,
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
            for version, version_config in config.get("versions", {}).items():
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

                # Ensure text search indexes exist for optimal performance
                ensure_text_search_indexes(qdrant_client, collection_name)

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
        await redis_client.ping()
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


# Add Gzip compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)


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
    )

    async def dispatch(self, request: Request, call_next):
        # Combine all URL parts for single check
        url_parts = [request.url.path or "", request.url.query or ""]

        # Add query parameters
        if request.query_params:
            url_parts.extend(request.query_params.values())

        # Check combined URL content
        combined_url = "".join(url_parts)
        if combined_url and len(combined_url) > 4000:
            logger.warning(
                "Suspicious URL encoding detected: %s...", combined_url[:100]
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
            canonical_query = urlencode(normalized_items, doseq=True, quote_via=quote)
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
            "script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://unpkg.com https://www.googletagmanager.com https://www.google-analytics.com https://static.cloudflareinsights.com https://*.clerk.com https://clerk.classifast.com https://accounts.google.com https://challenges.cloudflare.com; "
            "script-src-elem 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://unpkg.com https://www.googletagmanager.com https://www.google-analytics.com https://static.cloudflareinsights.com https://*.clerk.com https://clerk.classifast.com https://accounts.google.com https://challenges.cloudflare.com; "
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


# Mount static files with caching
# Optimized for Cloudflare Tunnel (full): CF edge caches based on s-maxage, origin sees tunnel traffic
class CachedStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if isinstance(response, Response):
            # Long cache for versioned assets (fonts, images), shorter for CSS/JS that might change
            if path.endswith((".woff", ".woff2", ".png", ".jpg", ".ico")):
                # Immutable assets - cache for 1 year at edge, 1 week in browser
                response.headers["Cache-Control"] = "public, max-age=604800, immutable"
                response.headers["Cloudflare-CDN-Cache-Control"] = "max-age=31536000"
            elif path.endswith((".css", ".js")):
                # CSS/JS - 1 hour browser, 24 hours CF edge
                response.headers["Cache-Control"] = "public, max-age=3600"
                response.headers["Cloudflare-CDN-Cache-Control"] = "max-age=86400"
            else:
                # Other files - 4 hours browser, 7 days edge
                response.headers["Cache-Control"] = "public, max-age=14400"
                response.headers["Cloudflare-CDN-Cache-Control"] = "max-age=604800"

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


# Root-level static files (browsers/crawlers expect these at root)
# Cached at CF edge for 24h since these rarely change but get frequent requests
def static_file_response(
    path: str, max_age: int = 14400, cdn_max_age: int = 604800
) -> FileResponse:
    """Serve a static file with Cloudflare-optimized cache headers."""
    file_path = BASE_DIR / "app" / "static" / path
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    response = FileResponse(file_path)
    response.headers["Cache-Control"] = f"public, max-age={max_age}"
    response.headers["Cloudflare-CDN-Cache-Control"] = f"max-age={cdn_max_age}"
    response.headers["Cache-Tag"] = "static-files"
    return response


@app.get("/favicon.ico", response_class=FileResponse, include_in_schema=False)
async def favicon():
    return static_file_response("images/favicon.ico")


@app.get("/robots.txt", response_class=FileResponse)
async def robots_txt():
    return static_file_response("robots.txt")


@app.get("/sitemap.xml", response_class=FileResponse)
async def sitemap_xml():
    return static_file_response("sitemap.xml")


@app.get("/llms.txt", response_class=FileResponse)
async def llms_txt():
    return static_file_response("llms.txt")


# Set state for limiter
app.state.limiter = limiter


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
