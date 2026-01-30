import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager

import redis.asyncio as redis
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from qdrant_client import AsyncQdrantClient, models
from starlette.middleware.base import BaseHTTPMiddleware

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

# Base directory for resolving static file paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()


async def ensure_text_search_indexes(
    qdrant_client: AsyncQdrantClient,
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
            await qdrant_client.create_payload_index(
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
        qdrant_client = AsyncQdrantClient(
            url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
            api_key=QDRANT_API_KEY or None,
            timeout=30,
        )

        # Check if Qdrant client can list collections as a health check
        existing_collections = set()
        if qdrant_client:
            try:
                collections_result = await qdrant_client.get_collections()
                existing_collections = {
                    col.name for col in collections_result.collections
                }
                collection_names = sorted(list(existing_collections))
                logger.info(
                    "Qdrant client initialized. Found collections: %s", collection_names
                )
            except Exception as e:
                logger.error(
                    "Qdrant client initialized, but could not list collections: %s", e
                )
        else:
            logger.error("Qdrant client could not be initialized.")

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
                collection_info = await qdrant_client.get_collection(collection_name)
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
                await ensure_text_search_indexes(qdrant_client, collection_name)

    except Exception as e:
        logger.error("Error initializing Qdrant client: %s", e)

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

    # Store clients and caches in app state
    app.state.embed_client = embed_client
    app.state.qdrant_client = qdrant_client
    app.state.collection_quantization_cache = collection_quantization_cache
    app.state.redis_client = redis_client

    yield

    # Runs when the application is shutting down
    logger.info("FastAPI application shutdown...")
    if qdrant_client:
        try:
            await qdrant_client.close()
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
        if combined_url and self._is_suspicious_encoding(combined_url):
            logger.warning(
                "Suspicious URL encoding detected: %s...", combined_url[:100]
            )
            return self._create_error_response()

        response = await call_next(request)
        return response

    def _is_suspicious_encoding(self, text: str) -> bool:
        """Simplified check for obvious encoding attacks"""
        if not text or len(text) > 4000:
            return len(text) > 4000

        return bool(self._attack_patterns.search(text.lower()))

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
class CachedStaticFiles(StaticFiles):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if isinstance(response, Response):
            # Cache for static files
            if path.endswith(
                (".css", ".js", ".png", ".jpg", ".ico", ".woff", ".woff2")
            ):
                response.headers["Cache-Control"] = (
                    "public, max-age=3600, s-maxage=14400, "  # 1 hour browser, 4 hours CDN
                    "stale-while-revalidate=1800, "  # Allow serving stale for 30 min while revalidating
                    "stale-if-error=3600"  # Serve stale if origin is down
                )
            else:
                response.headers["Cache-Control"] = (
                    "public, max-age=300, s-maxage=3600, "  # 5 min browser, 1 hour CDN
                    "stale-while-revalidate=300, "  # Allow serving stale for 5 min while revalidating
                    "stale-if-error=1800"  # Serve stale if origin is down
                )

            # Add ETag for caching based on file modification time
            try:
                file_path = os.path.join(BASE_DIR, "app", "static", path.lstrip("/"))
                file_stat = os.stat(file_path)
                response.headers["ETag"] = (
                    f'"{int(file_stat.st_mtime)}-{file_stat.st_size}"'
                )
            except (OSError, FileNotFoundError):
                response.headers["ETag"] = f'"{hash(path)}"'

            # Add Vary header to correctly cache compressed responses
            response.headers["Vary"] = "Accept-Encoding"

            # Cloudflare-specific headers for better edge caching
            response.headers["Cache-Tag"] = "static-files"

            # Let Cloudflare handle compression - don't set Content-Encoding
        return response


# Explicit routes for static JS files (must be before StaticFiles mount)
@app.get("/static/js/htmx.min.js", response_class=FileResponse)
async def htmx_js():
    file_path = os.path.join(BASE_DIR, "app", "static", "js", "htmx.min.js")
    response = FileResponse(file_path)
    response.headers["Cache-Control"] = (
        "public, max-age=3600, s-maxage=14400, stale-if-error=3600"  # 1 hour browser, 4 hours CDN
    )
    return response


@app.get("/static/js/classifier.js", response_class=FileResponse)
async def classifier_js():
    file_path = os.path.join(BASE_DIR, "app", "static", "js", "classifier.js")
    response = FileResponse(file_path)
    response.headers["Cache-Control"] = (
        "public, max-age=3600, s-maxage=14400, stale-if-error=3600"  # 1 hour browser, 4 hours CDN
    )
    return response


@app.get("/static/js/paywall.js", response_class=FileResponse)
async def paywall_js():
    file_path = os.path.join(BASE_DIR, "app", "static", "js", "paywall.js")
    response = FileResponse(file_path)
    response.headers["Cache-Control"] = (
        "public, max-age=3600, s-maxage=14400, stale-if-error=3600"  # 1 hour browser, 4 hours CDN
    )
    return response


@app.get("/static/js/common.js", response_class=FileResponse)
async def common_js():
    file_path = os.path.join(BASE_DIR, "app", "static", "js", "common.js")
    response = FileResponse(file_path)
    response.headers["Cache-Control"] = (
        "public, max-age=3600, s-maxage=14400, stale-if-error=3600"  # 1 hour browser, 4 hours CDN
    )
    return response


app.mount(
    "/static",
    CachedStaticFiles(directory=os.path.join(BASE_DIR, "app", "static")),
    name="static",
)


@app.get("/favicon.ico", response_class=FileResponse, include_in_schema=False)
async def favicon():
    file_path = os.path.join(BASE_DIR, "app", "static", "images", "favicon.ico")
    response = FileResponse(file_path)
    response.headers["Cache-Control"] = (
        "public, max-age=3600, s-maxage=14400, stale-if-error=3600"  # 1 hour browser, 4 hours CDN
    )
    return response


@app.get("/robots.txt", response_class=FileResponse)
async def robots_txt():
    file_path = os.path.join(BASE_DIR, "app", "static", "robots.txt")
    response = FileResponse(file_path)
    response.headers["Cache-Control"] = (
        "public, max-age=3600, s-maxage=14400, stale-if-error=3600"  # 1 hour browser, 4 hours CDN
    )
    return response


@app.get("/sitemap.xml", response_class=FileResponse)
async def sitemap_xml():
    file_path = os.path.join(BASE_DIR, "app", "static", "sitemap.xml")
    response = FileResponse(file_path)
    response.headers["Cache-Control"] = (
        "public, max-age=3600, s-maxage=14400, stale-if-error=3600"  # 1 hour browser, 4 hours CDN
    )
    return response


@app.get("/llms.txt", response_class=FileResponse)
async def llms_txt():
    file_path = os.path.join(BASE_DIR, "app", "static", "llms.txt")
    response = FileResponse(file_path)
    response.headers["Cache-Control"] = (
        "public, max-age=3600, s-maxage=14400, stale-if-error=3600"  # 1 hour browser, 4 hours CDN
    )
    return response


@app.get("/static/css/styles.css", response_class=FileResponse)
async def styles_css():
    file_path = os.path.join(BASE_DIR, "app", "static", "css", "styles.css")
    response = FileResponse(file_path)
    response.headers["Cache-Control"] = (
        "public, max-age=3600, s-maxage=14400, stale-if-error=3600"  # 1 hour browser, 4 hours CDN
    )
    return response


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
            # Test embed client
            embed_client.models.list()
            # Test qdrant client
            await qdrant_client.get_collections()
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
