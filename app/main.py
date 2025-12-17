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
from qdrant_client import AsyncQdrantClient
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

load_dotenv()


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
    # Pre-compile regex patterns for performance
    # Only detect truly malicious patterns: multiple levels of encoding (%25%25%25+)
    _triple_encoded_percent = re.compile(r"(%25){3,}")

    async def dispatch(self, request: Request, call_next):
        # Check URL path for suspicious encoding (catches path-based attacks)
        url_path = request.url.path or ""
        if url_path and self._is_suspicious_encoding(url_path):
            logger.warning(
                "Suspicious URL encoding detected in path: %s...", url_path[:100]
            )
            return self._create_error_response()

        # Early check for URL query parameters (most efficient first)
        if request.query_params:
            for param_name, param_value in request.query_params.items():
                if self._is_suspicious_encoding(param_value):
                    logger.warning(
                        "Suspicious URL encoding detected in query param '%s': %s...",
                        param_name,
                        param_value[:100],
                    )
                    return self._create_error_response()

        # Check URL query string for suspicious encoding
        # Use query part only for better performance and accuracy
        url_query = request.url.query or ""
        if url_query and self._is_suspicious_encoding(url_query):
            logger.warning(
                "Suspicious URL encoding detected in query string: %s...",
                url_query[:100],
            )
            return self._create_error_response()

        # All checks passed - proceed with request
        response = await call_next(request)
        return response

    def _is_suspicious_encoding(self, text: str) -> bool:
        """Lenient check for only obvious encoding attacks"""
        if not text or len(text) > 4000:
            return len(text) > 4000  # Reject overlong strings

        text_lower = text.lower()

        # Only reject triple-encoded percent signs or higher
        # This catches %25%25%25 (decoded to %25) which indicates attack intent
        # but allows %2525 and %252X which are legitimate encoded characters
        if self._triple_encoded_percent.search(text):
            return True

        # Reject only extremely malicious patterns
        # Multiple consecutive %3C or %3E (< or >) suggest HTML injection attempts
        if "%3c%3c" in text_lower or "%3e%3e" in text_lower:
            return True

        return False

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
            "script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://unpkg.com https://www.googletagmanager.com https://www.google-analytics.com https://static.cloudflareinsights.com https://*.clerk.com https://clerk.classifast.com https://accounts.google.com; "
            "script-src-elem 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://unpkg.com https://www.googletagmanager.com https://www.google-analytics.com https://static.cloudflareinsights.com https://*.clerk.com https://clerk.classifast.com https://accounts.google.com; "
            "worker-src 'self' blob:; "
            "style-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://fonts.googleapis.com; "
            "style-src-elem 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://fonts.googleapis.com https://accounts.google.com/gsi/style; "
            "img-src 'self' data: https: https://*.googleapis.com https://*.gstatic.com https://*.clerk.com https://clerk.classifast.com; "
            "font-src 'self' https://fonts.gstatic.com https://*.googleapis.com https://*.gstatic.com; "
            "connect-src 'self' https: https://*.clerk.com https://accounts.google.com https://accounts.google.com/gsi/ https://*.googleapis.com; "
            "frame-src 'self' https://accounts.google.com; "
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
                file_path = os.path.join("app/static", path.lstrip("/"))
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


app.mount("/static", CachedStaticFiles(directory="app/static"), name="static")


@app.get("/favicon.ico", response_class=FileResponse, include_in_schema=False)
async def favicon():
    response = FileResponse("app/static/images/favicon.ico")
    response.headers["Cache-Control"] = (
        "public, max-age=3600, s-maxage=14400, stale-if-error=3600"  # 1 hour browser, 4 hours CDN
    )
    return response


@app.get("/robots.txt", response_class=FileResponse)
async def robots_txt():
    response = FileResponse("app/static/robots.txt")
    response.headers["Cache-Control"] = (
        "public, max-age=3600, s-maxage=14400, stale-if-error=3600"  # 1 hour browser, 4 hours CDN
    )
    return response


@app.get("/sitemap.xml", response_class=FileResponse)
async def sitemap_xml():
    response = FileResponse("app/static/sitemap.xml")
    response.headers["Cache-Control"] = (
        "public, max-age=3600, s-maxage=14400, stale-if-error=3600"  # 1 hour browser, 4 hours CDN
    )
    return response


@app.get("/llms.txt", response_class=FileResponse)
async def llms_txt():
    response = FileResponse("app/static/llms.txt")
    response.headers["Cache-Control"] = (
        "public, max-age=3600, s-maxage=14400, stale-if-error=3600"  # 1 hour browser, 4 hours CDN
    )
    return response


@app.get("/static/css/styles.css", response_class=FileResponse)
async def styles_css():
    response = FileResponse("app/static/css/styles.css")
    response.headers["Cache-Control"] = (
        "public, max-age=3600, s-maxage=14400, stale-if-error=3600"  # 1 hour browser, 4 hours CDN
    )
    return response


@app.get("/static/js/htmx.min.js", response_class=FileResponse)
async def htmx_js():
    response = FileResponse("app/static/js/htmx.min.js")
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


"""
npm install tailwindcss @tailwindcss/cli
npx @tailwindcss/cli -i ./app/static/css/input.css -o ./app/static/css/styles.css
uvicorn app.main:app --reload --port 8001
pkill -f "uvicorn"
"""
