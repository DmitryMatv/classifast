import logging
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from urllib.parse import unquote_plus

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, FastAPI, Form, HTTPException, Query, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from google import genai
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from .classifier import classify_string_batch
from .classifier_config import CLASSIFIER_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

# Global client variables with proper type annotations
embed_client: Optional[genai.Client] = None
qdrant_client: Optional[AsyncQdrantClient] = None
embed_model_name: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs when the application starts
    global embed_client, embed_model_name, qdrant_client

    logger.info("FastAPI application startup...")

    # Initialize Embedding Client (Google GenAI)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        logger.error("Error: GEMINI_API_KEY not found in environment variables.")
        # In a real app, you might raise an exception or handle this more gracefully
    else:
        try:
            embed_client = genai.Client(api_key=GEMINI_API_KEY)
            embed_client.models.list()  # Test connection
            logger.info("Google GenAI Client initialized successfully.")
        except Exception as e:
            logger.error("Error initializing Google GenAI Client: %s", e)
            embed_client = None  # Ensure it's None if init fails

    # Initialize Qdrant Client with connection pooling
    QDRANT_URL = os.getenv("QDRANT_URL", "qdrant.classifast.com")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    try:
        logger.info("Connecting to Qdrant...")
        qdrant_client = AsyncQdrantClient(
            api_key=QDRANT_API_KEY,
            host=QDRANT_URL,
            port=443,
            https=True,
            prefer_grpc=False,
            timeout=30,  # Lower timeout
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
                # Depending on severity, you might still want to set qdrant_client to None or raise
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

    except Exception as e:
        logger.error("Error initializing Qdrant client: %s", e)

    # --- Pre-loading removed ---

    yield

    # Runs when the application is shutting down
    logger.info("FastAPI application shutdown...")
    if qdrant_client:
        try:
            await qdrant_client.close()
            logger.info("Qdrant client closed.")
        except Exception as e:
            logger.error("Error closing Qdrant client: %s", e)


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
    _double_encoding_pattern = re.compile(r"%25{3,}")
    _encoding_pattern = re.compile(r"%[0-9A-Fa-f]{2}")
    _overlong_25_pattern = re.compile(r"(%25){2,}")
    _consecutive_encoding_pattern = re.compile(r"%[0-9A-Fa-f]{2}(%[0-9A-Fa-f]{2}){5,}")
    _suspicious_sequences = {
        r"%2525",  # Double-encoded %
        r"%2520",  # Double-encoded space
        r"%2522",  # Double-encoded quote
        r"%253c",  # Double-encoded <
        r"%253e",  # Double-encoded >
    }

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

        # Check URL query string for suspicious encoding (especially for POST requests with spam patterns)
        # Use query part only for better performance and accuracy
        url_query = request.url.query or ""
        if url_query and self._is_suspicious_encoding(url_query):
            logger.warning(
                "Suspicious URL encoding detected in query string: %s...",
                url_query[:100],
            )
            return self._create_error_response()

        # Special handling for POST/PUT/PATCH - only check content-type headers
        # Avoid reading body to prevent consuming it for downstream handlers
        if request.method in ["POST", "PUT", "PATCH"]:
            # Additional check: reject very large content lengths (DoS prevention)
            content_length = request.headers.get("content-length")
            if content_length:
                try:
                    length = int(content_length)
                    if length > 10000000:  # 10MB limit
                        logger.warning(
                            "Suspicious: Very large content length %d bytes", length
                        )
                        return self._create_error_response()
                except ValueError:
                    pass  # Invalid content-length header, let downstream handle it

        # All checks passed - proceed with request
        response = await call_next(request)
        return response

    def _is_suspicious_encoding(self, text: str) -> bool:
        """Optimized check for suspicious URL encoding patterns"""
        if not text or len(text) > 4000:
            return len(text) > 4000  # Reject overlong strings

        # Fast path: Check for most obvious spam patterns first
        if "%2525" in text.lower():
            return True

        # Check for decoded double-encoding patterns (e.g., "2525", "2520")
        text_lower = text.lower()
        decoded_spam_patterns = ["252525", "252520", "253c", "253e", "2522"]
        for pattern in decoded_spam_patterns:
            if pattern in text_lower:
                return True

        # Check for repeated %25 patterns (double encoding)
        if self._double_encoding_pattern.search(text):
            return True

        # Check for excessive URL encoding
        encoding_count = len(self._encoding_pattern.findall(text))
        if encoding_count > 10:
            total_chars = len(text)
            if total_chars > 0 and encoding_count / total_chars > 0.3:
                return True

        # Check for long sequences of consecutive digits (15+)
        digit_sequence = re.search(r"\d{15,}", text)
        if digit_sequence:
            return True

        # Check for repeating character patterns (any char repeated 10+ times)
        if re.search(r"(.)\1{9,}", text):
            return True

        # Check digit density (>60% digits is suspicious)
        digit_count = sum(c.isdigit() for c in text)
        if len(text) > 0 and digit_count / len(text) > 0.6:
            return True

        # Check for other suspicious patterns
        if self._overlong_25_pattern.search(text):
            return True

        if self._consecutive_encoding_pattern.search(text):
            return True

        # Check for suspicious sequences
        for pattern in self._suspicious_sequences:
            if pattern in text_lower:
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
            # Longer cache for static files
            if path.endswith(
                (".css", ".js", ".png", ".jpg", ".ico", ".woff", ".woff2")
            ):
                response.headers["Cache-Control"] = (
                    "public, max-age=604800, s-maxage=604800, "  # 1 week for both browser and CDN
                    "immutable, "  # Tell browsers it never changes
                    "stale-while-revalidate=86400"  # Allow serving stale for 1 day while revalidating
                )
            else:
                response.headers["Cache-Control"] = (
                    "public, max-age=86400, s-maxage=86400, "  # 1 day for both browser and CDN
                    "stale-while-revalidate=3600"  # Allow serving stale for 1 hour while revalidating
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
        "public, max-age=604800, s-maxage=604800"  # 1 week
    )
    return response


@app.get("/robots.txt", response_class=FileResponse)
async def robots_txt():
    response = FileResponse("app/static/robots.txt")
    response.headers["Cache-Control"] = "public, max-age=86400, s-maxage=86400"  # 1 day
    return response


@app.get("/sitemap.xml", response_class=FileResponse)
async def sitemap_xml():
    response = FileResponse("app/static/sitemap.xml")
    response.headers["Cache-Control"] = "public, max-age=86400, s-maxage=86400"  # 1 day
    return response


@app.get("/llms.txt", response_class=FileResponse)
async def llms_txt():
    response = FileResponse("app/static/llms.txt")
    response.headers["Cache-Control"] = "public, max-age=86400, s-maxage=86400"  # 1 day
    return response


@app.get("/static/css/styles.css", response_class=FileResponse)
async def styles_css():
    response = FileResponse("app/static/css/styles.css")
    response.headers["Cache-Control"] = (
        "public, max-age=604800, s-maxage=604800"  # 1 week, both browser and Cloudflare
    )
    return response


@app.get("/static/js/htmx.min.js", response_class=FileResponse)
async def htmx_js():
    response = FileResponse("app/static/js/htmx.min.js")
    response.headers["Cache-Control"] = (
        "public, max-age=604800, s-maxage=604800"  # 1 week, both browser and Cloudflare
    )
    return response


# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

app.state.limiter = limiter


async def custom_rate_limit_exceeded_handler(request, exc: Exception):
    if isinstance(exc, RateLimitExceeded):
        # Use global templates instance instead of creating a new one
        return templates.TemplateResponse(
            "rate_limit_warning.html", {"request": request}, status_code=429
        )
    return HTMLResponse(content="Internal Server Error", status_code=500)


app.add_exception_handler(RateLimitExceeded, custom_rate_limit_exceeded_handler)


# Healthcheck
@app.get("/health")
async def health_check():
    """
    Health check endpoint for Docker/Kubernetes.
    Returns generic status to avoid information disclosure.
    """
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


# Setup Jinja2 templates
templates = Jinja2Templates(directory="app/templates")


# Serve the main homepage
@app.get("/", response_class=HTMLResponse)
@app.head("/")  # Add HEAD support
async def read_root(request: Request):
    """Serves the main homepage with Cloudflare-friendly caching."""

    # For HEAD requests, return just headers
    if request.method == "HEAD":
        headers = {
            "Cache-Control": "public, max-age=86400, s-maxage=604800",
            "Vary": "Accept-Encoding",
            "Content-Type": "text/html; charset=utf-8",
            "Link": '<https://classifast.com/>; rel="canonical"',
        }
        return Response(headers=headers)

    response = templates.TemplateResponse("index.html", {"request": request})

    # Cloudflare-friendly cache headers (same as classifier pages)
    response.headers["Cache-Control"] = "public, max-age=86400, s-maxage=604800"
    response.headers["Vary"] = "Accept-Encoding"
    response.headers["Link"] = '<https://classifast.com/>; rel="canonical"'
    response.headers["X-Robots-Tag"] = "index, follow"

    return response


# Dictionary to map classifier types to their configurations
# CLASSIFIER_CONFIG is imported from .classifier_config


async def perform_classification(
    query: str,
    classifier_type: str,
    version: Optional[str] = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Shared classification service that handles all common logic between web form and API endpoints.

    Args:
        classifier_type: The classification standard (e.g., 'unspsc', 'etim', etc.)
        query: The product/service description to classify
        version: Optional specific version to use
        top_k: Number of results to return

    Returns:
        Dict containing classification results and metadata
    """
    # Validate classifier type
    config = CLASSIFIER_CONFIG.get(classifier_type.lower())
    if not config:
        raise HTTPException(
            status_code=404, detail=f"Classifier '{classifier_type}' not found"
        )

    # Validate version or use default
    versions = config.get("versions", {})
    if version:
        if version not in versions:
            raise HTTPException(
                status_code=404,
                detail=f"Version '{version}' for classifier '{classifier_type}' not found",
            )
        version_name = version
    else:
        version_name = next(iter(versions.keys())) if versions else ""

    version_config = versions[version_name]
    collection_name = version_config["collection_name"]
    embed_model_name = config["embed_model_name"]

    # Validate clients
    if not embed_client or not qdrant_client:
        raise HTTPException(
            status_code=503,
            detail="Backend services not available. Please check server logs.",
        )

    # Validate and normalize query - remove trailing slashes and replace with spaces
    normalized_query = query.replace("/", " ").strip()
    if not normalized_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if len(normalized_query) > 4000:
        raise HTTPException(
            status_code=400, detail="Query too long (max 4000 characters)"
        )

    try:
        # Perform classification with normalized query
        results_for_single_query = await classify_string_batch(
            qdrant_client=qdrant_client,
            embed_client=embed_client,
            embed_model_name=embed_model_name,
            query_texts=[normalized_query],
            collection_name=collection_name,
            embed_dims=config.get("embed_dims"),
            top_k=top_k,
        )

        classification_results = (
            results_for_single_query[0] if results_for_single_query else []
        )

        return {
            "results": classification_results,
            "collection_name": collection_name,
            "version_name": version_name,
            "version_config": version_config,
            "config": config,
            "query": normalized_query,
        }

    except Exception as e:
        logger.error("Classification error for '%s': %s", classifier_type, e)
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}"
        )


# ===== RAPIDAPI INTEGRATION =====


# Pydantic models for RapidAPI
class RapidAPIRequest(BaseModel):
    query: str = Field(..., description="Product or service description to classify")
    standard: str = Field(
        ..., description="Classification standard (unspsc, etim, naics, isic, hs)"
    )
    version: Optional[str] = Field(
        None, description="Specific version of the standard to use"
    )
    top_k: Optional[int] = Field(
        5, ge=1, le=100, description="Number of results to return"
    )


class ClassificationResult(BaseModel):
    code: str = Field(..., description="Classification code")
    name: str = Field(..., description="Classification name/description")
    score: float = Field(..., description="Similarity score (0-1)")
    url: Optional[str] = Field(None, description="External URL for more information")


class RapidAPIResponse(BaseModel):
    query: str = Field(..., description="Original query")
    standard: str = Field(..., description="Classification standard used")
    version: str = Field(..., description="Version of the standard used")
    results: List[ClassificationResult] = Field(
        ..., description="Classification results"
    )
    processing_time: float = Field(..., description="Processing time in seconds")


class RapidAPIError(BaseModel):
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")


# RapidAPI configuration
RAPIDAPI_SECRET = os.getenv("RAPIDAPI_SECRET")
RAPIDAPI_SECRET_HEADER = "X-RapidAPI-Proxy-Secret"


# Separate limiter for RapidAPI endpoints
rapid_limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["60/minute"],  # Stricter limits for external API
)


async def verify_rapidapi_key(request: Request) -> bool:
    """Verify RapidAPI key from header."""
    api_key = request.headers.get("X-RapidAPI-Key")
    if api_key:
        return True  # API key provided - valid
    return False


async def verify_rapidapi_proxy(request: Request) -> bool:
    """Verify RapidAPI proxy secret."""
    # Security fix: Do not default to True if secret is missing
    if not RAPIDAPI_SECRET:
        return False

    proxy_secret = request.headers.get("X-RapidAPI-Proxy-Secret")

    if proxy_secret != RAPIDAPI_SECRET:
        raise HTTPException(status_code=401, detail="Invalid proxy secret")

    return True


async def verify_rapidapi_auth(request: Request) -> bool:
    """Combined authentication function that accepts either API key or proxy authentication."""
    # First, check for proxy authentication (RapidAPI playground mode)
    try:
        proxy_valid = await verify_rapidapi_proxy(request)
        if proxy_valid:
            # Proxy authentication succeeded - allow access regardless of API key
            return True
    except HTTPException:
        # Proxy authentication failed, check if we have API key
        pass

    # Then, check for direct API key authentication
    api_key_valid = await verify_rapidapi_key(request)
    if api_key_valid:
        return True

    # Neither authentication method succeeded
    raise HTTPException(
        status_code=401,
        detail="Authentication required - provide either X-RapidAPI-Key or valid proxy secret",
        headers={"WWW-Authenticate": "ApiKey"},
    )


# Create RapidAPI router
rapid_router = APIRouter(
    tags=["rapidapi"],
    dependencies=[Depends(verify_rapidapi_auth)],
)


@rapid_router.get("/classify", response_model=RapidAPIResponse)
@rapid_limiter.limit("600/minute")
async def rapid_classify(
    request: Request,
    query: str = Query(..., description="Product or service description to classify"),
    standard: str = Query(
        ..., description="Classification standard (unspsc, etim, naics, isic, hs)"
    ),
    top_k: int = Query(3, ge=1, le=100, description="Number of results to return"),
    version: Optional[str] = Query(
        None, description="Specific version of the standard to use"
    ),
):
    """
    Classify a product or service description using the specified standard.

    This endpoint provides programmatic access to classification services via RapidAPI.
    """
    normalized_query = query.strip()
    logger.info("RapidAPI classification request: %s <- %s", standard, normalized_query)

    start_time = time.perf_counter()

    try:
        # Use shared classification service
        result = await perform_classification(
            query=normalized_query,
            classifier_type=standard,
            version=version,
            top_k=top_k or 1,
        )

        classification_results = result["results"]

        # Format results for API response
        formatted_results = []
        for r in classification_results:
            payload = r.get("payload", {})
            base_url = result["version_config"].get("base_url", "")
            code = payload.get("original_id", "")

            formatted_result = ClassificationResult(
                code=code,
                name=payload.get("class_name", ""),
                score=r.get("score", 0.0),
                url=f"{base_url}{code}" if base_url and code else None,
            )
            formatted_results.append(formatted_result)

        processing_time = time.perf_counter() - start_time

        return RapidAPIResponse(
            query=normalized_query,
            standard=standard.lower(),
            version=result["version_name"],
            results=formatted_results,
            processing_time=processing_time,
        )

    except HTTPException:
        # Let HTTP exceptions propagate to the handler
        raise
    except Exception as e:
        logger.error("RapidAPI classification error: %s", e)
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@rapid_router.get("/standards")
@rapid_limiter.limit("600/minute")
async def rapid_standards(request: Request):
    """List available classification standards and their versions."""
    standards_info = {}

    for standard_key, config in CLASSIFIER_CONFIG.items():
        standards_info[standard_key] = {
            "title": config["title"],
            "description": config["description"],
            "versions": list(config.get("versions", {}).keys()),
            "example": config["example"].replace("Example:", "").strip(),
        }

    return JSONResponse(content={"standards": standards_info, "timestamp": time.time()})


@rapid_router.get("/debug-headers")
async def debug_headers(request: Request):
    """Debug endpoint to show all received headers for Cloudflare troubleshooting."""
    # Only allow in development mode
    if os.getenv("DEBUG_MODE", "false").lower() != "true":
        raise HTTPException(status_code=404, detail="Debug endpoint not available")

    # Sanitize sensitive headers
    headers = dict(request.headers)
    sensitive_headers = {
        "authorization",
        "cookie",
        "x-api-key",
        "x-rapidapi-key",
        "x-rapidapi-proxy-secret",
    }

    sanitized_headers = {}
    for key, value in headers.items():
        key_lower = key.lower()
        if key_lower in sensitive_headers:
            sanitized_headers[key] = "[REDACTED]"
        else:
            sanitized_headers[key] = value

    return JSONResponse(
        content={
            "received_headers": sanitized_headers,
            "timestamp": time.time(),
            "host": request.headers.get("host"),
            "user_agent": request.headers.get("user-agent"),
        }
    )


# Include the RapidAPI router
app.include_router(rapid_router, prefix="/api/v1/rapid")  # is prefix needed here?


# Public API health check endpoint (bypasses RapidAPI authentication)
@app.get("/api/v1/rapid/ping")
@rapid_limiter.limit("600/minute")
async def rapid_health_public(request: Request):
    """Public health check endpoint for RapidAPI consumers."""
    health_status = {"status": "healthy", "timestamp": time.time(), "services": {}}

    # Check embedding service
    if embed_client:
        try:
            embed_client.models.list()
            health_status["services"]["embedding"] = "healthy"
        except Exception as e:
            health_status["services"]["embedding"] = f"unhealthy: {str(e)}"
    else:
        health_status["services"]["embedding"] = "unhealthy: not initialized"

    # Check Qdrant service
    if qdrant_client:
        try:
            await qdrant_client.get_collections()
            health_status["services"]["database"] = "healthy"
        except Exception as e:
            health_status["services"]["database"] = f"unhealthy: {str(e)}"
    else:
        health_status["services"]["database"] = "unhealthy: not initialized"

    # Overall health
    all_healthy = all(v == "healthy" for v in health_status["services"].values())
    status_code = 200 if all_healthy else 503

    return JSONResponse(content=health_status, status_code=status_code)


# ===== END RAPIDAPI INTEGRATION =====


@app.get("/{classifier_type}", response_class=HTMLResponse)
@app.head("/{classifier_type}")
async def show_classifier_page(
    request: Request,
    classifier_type: str,
    version: str | None = None,
    top_k: int = 10,
):
    """
    Serves the base classifier page.
    Redirects URLs without trailing slash to versions with trailing slash for SEO consistency.
    """
    # Redirect classifier URLs without trailing slash to versions with trailing slash
    if classifier_type in CLASSIFIER_CONFIG:
        # Check if this is a direct access without trailing slash (not a redirect)
        original_path = request.url.path
        if not original_path.endswith("/"):
            # Preserve query parameters in the redirect
            query_string = f"?{request.url.query}" if request.url.query else ""
            return RedirectResponse(
                url=f"/{classifier_type}/{query_string}", status_code=301
            )

    return await show_classifier_page_with_query(
        request, classifier_type, "", version, top_k
    )


@app.get("/{classifier_type}/{search_query:path}", response_class=HTMLResponse)
@app.head("/{classifier_type}/{search_query:path}")
async def show_classifier_page_with_query(
    request: Request,
    classifier_type: str,
    search_query: str = "",
    version: str | None = None,
    top_k: int = 10,
):
    """
    Serves the specific classifier page with clean URL structure.
    Handles both base URLs like /naics and search URLs like /naics/gamedev-studio
    """
    config = CLASSIFIER_CONFIG.get(classifier_type)
    if not config:
        raise HTTPException(
            status_code=404, detail=f"Classifier '{classifier_type}' not found"
        )

    # For HEAD requests, return just headers
    if request.method == "HEAD":
        headers = {
            "Cache-Control": "public, max-age=86400, s-maxage=604800",
            "Vary": "Accept-Encoding",
            "Content-Type": "text/html; charset=utf-8",
            "Link": f'<https://classifast.com/{classifier_type}/{search_query}>; rel="canonical"',
        }
        return Response(headers=headers)

    # Validate top_k parameter
    if top_k < 1 or top_k > 100:
        top_k = 3

    # Get first version for default handling
    versions_list = list(config.get("versions", {}).keys())
    first_version = versions_list[0] if versions_list else ""

    # Handle empty search query for base URLs
    decoded_search_query = ""
    if search_query and search_query.strip():
        decoded_search_query = (
            unquote_plus(search_query)
            .rstrip("/")
            .replace("/", " ")
            .replace("-", " ")
            .strip()
        )
        # Sanitize the decoded query
        # Relaxed sanitization: allow characters like apostrophes, but keep length limit
        if len(decoded_search_query) > 4000:
            decoded_search_query = decoded_search_query[:4000]

        decoded_search_query = decoded_search_query.strip()

    # Initialize results data structure
    results_data = {
        "results_for_query": [],
        "query": decoded_search_query,
        "base_url": "",
        "tooltip": "",
        "total_request_time": 0,
    }

    # If no search query (base URL), load example results on-demand
    if not decoded_search_query:
        example_query = config.get("example", "").replace("Example:", "").strip()
        if example_query:
            try:
                start_time = time.perf_counter()
                # Use shared classification service for example
                # We use the first version implicitly if not specified, which matches original behavior
                result = await perform_classification(
                    query=example_query,
                    classifier_type=classifier_type,
                    version=version,  # Use requested version if any, or default
                    top_k=10,
                )

                results_data["results_for_query"] = result["results"]
                results_data["query"] = example_query
                results_data["base_url"] = result["version_config"].get("base_url", "")
                results_data["tooltip"] = result["version_config"].get("tooltip", "")
                results_data["total_request_time"] = time.perf_counter() - start_time

            except Exception as e:
                logger.error(
                    "Failed to load example results for '%s': %s", classifier_type, e
                )
                # Keep defaults (empty results)

    # Slugify utility for SEO-friendly URLs
    def slugify(text):
        if not text:
            return ""
        # Sanitize input: limit length and remove harmful characters
        text = str(text)[:200]  # Limit to 200 chars max
        # Preserve periods and commas while removing other special characters
        text = re.sub(r"[^\w\s.,-]", "", text.lower())
        text = re.sub(r"[-\s]+", "-", text)
        return text.strip("-")

    # Build canonical URL
    canonical_url = f"https://classifast.com/{classifier_type}"
    if decoded_search_query:
        slug = slugify(decoded_search_query)
        canonical_url += f"/{slug}"

    response = templates.TemplateResponse(
        "classifier_page.html",
        {
            "request": request,
            "classifier_type": classifier_type,
            "title": config["title"],
            "heading": config["heading"],
            "description": config["description"],
            "versions": list(config.get("versions", {}).keys()),
            "example": config["example"],
            "url_params": {
                "search": decoded_search_query,
                "version": version if version and version != first_version else "",
                "top_k": top_k,
            },
            **results_data,
        },
    )

    # Cloudflare-friendly cache headers (aligned with homepage)
    response.headers["Cache-Control"] = "public, max-age=86400, s-maxage=604800"
    response.headers["Vary"] = "Accept-Encoding"
    response.headers["Link"] = f'<{canonical_url}>; rel="canonical"'
    response.headers["X-Robots-Tag"] = "index, follow"

    return response


@app.post("/{classifier_type}", response_class=HTMLResponse)
@limiter.limit("20/minute")  # Apply rate limit to this endpoint
async def handle_classify(
    request: Request,
    classifier_type: str,
    product_description: str = Form(...),
    top_k: int = Form(10),
    version: str = Form(...),
):
    """
    Receives product description for a specific classifier type,
    classifies it using the correct Qdrant collection,
    and returns HTML partial with results.
    """
    logger.info(
        "Received query for '%s' classification with version '%s'.",
        classifier_type,
        version,
    )

    # Handle empty query gracefully - also remove trailing slashes and replace with spaces
    normalized_description = product_description.replace("/", " ").strip()
    if not normalized_description:
        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "query": product_description,
                "results_for_query": [],
            },
        )

    start_total_time = time.perf_counter()

    try:
        # Use shared classification service
        result = await perform_classification(
            query=normalized_description,
            classifier_type=classifier_type,
            version=version,
            top_k=top_k,
        )

        classification_results = result["results"]

        result_lines = [
            f"{r['payload'].get('original_id', 'N/A')} - {r['payload'].get('class_name', 'N/A')}"
            for r in classification_results
        ]
        logger.info(
            "Results for '%s' in '%s':\n%s",
            product_description,
            result["collection_name"],
            "\n".join(result_lines),
        )

    except HTTPException:
        # Let HTTP exceptions propagate to the handler
        raise
    except Exception as e:
        logger.error("Error during '%s' classification: %s", classifier_type, e)
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}"
        )

    end_total_time = time.perf_counter()
    total_request_time = end_total_time - start_total_time
    logger.info("Total request processing time was %.4fs", total_request_time)

    # Render the results partial
    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "query": product_description,
            "results_for_query": classification_results,
            "base_url": result["version_config"].get("base_url", ""),
            "tooltip": result["version_config"].get("tooltip", ""),
            "total_request_time": total_request_time,
        },
    )


"""
npm install tailwindcss @tailwindcss/cli
npx @tailwindcss/cli -i ./app/static/css/input.css -o ./app/static/css/styles.css
uvicorn app.main:app --reload --port 8001
pkill -f "uvicorn"
"""
