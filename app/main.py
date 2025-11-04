import os
import re
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from urllib.parse import unquote_plus

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, FastAPI, Form, HTTPException, Query, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
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

load_dotenv()

PRELOADED_RESULTS_CACHE: Dict[str, Dict[str, Any]] = {}

# Global client variables with proper type annotations
embed_client: Optional[genai.Client] = None
qdrant_client: Optional[AsyncQdrantClient] = None
embed_model_name: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs when the application starts
    global embed_client, embed_model_name, qdrant_client

    print("FastAPI application startup...")

    # Initialize Embedding Client (Google GenAI)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not found in environment variables.")
        # In a real app, you might raise an exception or handle this more gracefully
    else:
        try:
            embed_client = genai.Client(api_key=GEMINI_API_KEY)
            embed_client.models.list()  # Test connection
            print("Google GenAI Client initialized successfully.")
        except Exception as e:
            print(f"Error initializing Google GenAI Client: {e}")
            embed_client = None  # Ensure it's None if init fails

    # Initialize Qdrant Client with connection pooling
    QDRANT_URL = os.getenv("QDRANT_URL", "qdrant.classifast.com")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    try:
        print("Connecting to Qdrant...")
        qdrant_client = AsyncQdrantClient(
            api_key=QDRANT_API_KEY,
            host=QDRANT_URL,
            port=443,
            https=True,
            prefer_grpc=False,
            timeout=30,  # Lower timeout
        )

        # Check if Qdrant client can list collections as a health check
        if qdrant_client:
            try:
                collections_result = await qdrant_client.get_collections()
                collection_names = sorted(
                    [col.name for col in collections_result.collections]
                )
                print("Qdrant client initialized. Found collections:")
                for name in collection_names:
                    print(f"💿 {name}")
            except Exception as e:
                print(f"Qdrant client initialized, but could not list collections: {e}")
                # Depending on severity, you might still want to set qdrant_client to None or raise
        else:
            print("Qdrant client could not be initialized.")

        # Verify collections exist and store their vector sizes
        for classifier_type, config in CLASSIFIER_CONFIG.items():
            embed_dims = config.get("embed_dims")
            for version, version_config in config.get("versions", {}).items():
                collection_name = version_config.get("collection_name")
                if not collection_name:
                    continue
                if not await qdrant_client.collection_exists(collection_name):
                    print(
                        f"Warning: Collection {collection_name} for {classifier_type} version {version} does not exist."
                    )
                    continue

                # Get collection info and check vector configuration
                collection_info = await qdrant_client.get_collection(collection_name)
                vector_params = collection_info.config.params.vectors

                if isinstance(vector_params, dict) and "size" in vector_params:
                    vector_size = vector_params["size"]
                    if vector_size != embed_dims:
                        print(
                            f"Warning: Collection {collection_name} has vector size {vector_size} but config specifies {embed_dims}"
                        )

    except Exception as e:
        print(f"Error initializing Qdrant client: {e}")

    # --- Pre-load and cache results for all example queries on startup ---
    if embed_client and qdrant_client:
        print("Pre-loading example query results for all classifiers...")
        for classifier_type, config in CLASSIFIER_CONFIG.items():
            query = config.get("example", "").replace("Example:", "").strip()
            if not query:
                continue

            try:
                start_total_time = time.perf_counter()
                version_name = next(iter(config.get("versions", {})))

                # Use shared classification service for pre-loading
                result = await perform_classification(
                    query=query,
                    classifier_type=classifier_type,
                    version=version_name,
                    top_k=10,
                )

                results_for_query = result["results"]
                end_total_time = time.perf_counter()
                total_request_time = end_total_time - start_total_time

                base_url = result["version_config"].get("base_url", "")
                tooltip = result["version_config"].get("tooltip", "")

                # Store all necessary data for the template in the cache
                PRELOADED_RESULTS_CACHE[classifier_type] = {
                    "results_for_query": results_for_query,
                    "query": query,
                    "base_url": base_url,
                    "tooltip": tooltip,
                    "total_request_time": total_request_time,
                }

                if results_for_query:
                    print(
                        f"Successfully pre-loaded and cached results for '{classifier_type}' in {total_request_time:.4f}s"
                    )
                else:
                    print(
                        f"Pre-loaded empty results for '{classifier_type}' in {total_request_time:.4f}s"
                    )

            except Exception as e:
                print(f"Failed to pre-load results for '{classifier_type}': {e}")
                PRELOADED_RESULTS_CACHE[classifier_type] = {
                    "results_for_query": [],
                    "query": query,
                    "base_url": "",
                    "tooltip": "",
                    "total_request_time": 0,
                }
    else:
        print(
            "Skipping pre-loading of example queries because clients are not initialized."
        )
        print(
            "❌ Critical Error: One or more clients failed to initialize. The application might not function correctly."
        )
    # --- End pre-loading ---

    yield

    # Runs when the application is shutting down
    print("FastAPI application shutdown...")
    if qdrant_client:
        try:
            await qdrant_client.close()
            print("Qdrant client closed.")
        except Exception as e:
            print(f"Error closing Qdrant client: {e}")


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


# Add this middleware to log user agents and help debug bot access
class BotDetectionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        user_agent = request.headers.get("user-agent", "")

        # Log bot visits
        if any(
            bot in user_agent.lower() for bot in ["googlebot", "bingbot", "crawler"]
        ):
            print(f"Bot detected: {user_agent} accessing {request.url}")

        response = await call_next(request)
        return response


app.add_middleware(BotDetectionMiddleware)


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
            print(f"Suspicious URL encoding detected in path: {url_path[:100]}...")
            return self._create_error_response()

        # Early check for URL query parameters (most efficient first)
        if request.query_params:
            for param_name, param_value in request.query_params.items():
                if self._is_suspicious_encoding(param_value):
                    print(
                        f"Suspicious URL encoding detected in query param '{param_name}': {param_value[:100]}..."
                    )
                    return self._create_error_response()

        # Check URL query string for suspicious encoding (especially for POST requests with spam patterns)
        # Use query part only for better performance and accuracy
        url_query = request.url.query or ""
        if url_query and self._is_suspicious_encoding(url_query):
            print(
                f"Suspicious URL encoding detected in query string: {url_query[:100]}..."
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
                        print(f"Suspicious: Very large content length {length} bytes")
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

        # Cloudflare-optimized security headers
        # Simplified CSP since Cloudflare handles most security
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://unpkg.com https://www.googletagmanager.com https://www.google-analytics.com https://static.cloudflareinsights.com https://darling-seagull-34.clerk.accounts.dev https://*.clerk.accounts.dev https://*.clerk.com; "
            "worker-src 'self' blob:; "
            "style-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://fonts.googleapis.com; "
            "style-src-elem 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://fonts.googleapis.com; "
            "img-src 'self' data: https:; "
            "font-src 'self' https://fonts.gstatic.com; "
            "connect-src 'self' https: https://darling-seagull-34.clerk.accounts.dev https://*.clerk.accounts.dev https://*.clerk.com; "
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
        templates = Jinja2Templates(directory="app/templates")
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
CLASSIFIER_CONFIG = {
    "etim": {
        "title": "ETIM International Classifier",
        "heading": "Get relevant EC classes from the ETIM standard",
        "description": "ETIM (ETIM Technical Information Model) is a format to share and exchange product data based on taxonomic identification. This widely used classification standard for technical products was developed to structure the information flow between B2B professionals.",
        "example": "Example: SH203-C20 Miniature Circuit Breaker 6kA 20A 3P",
        "embed_model_name": "gemini-embedding-001",
        "embed_dims": 3072,
        "versions": {
            "ETIM version 10.0 (2024-12-10)": {
                "collection_name": "ETIM_10_eng_new001_v4",
                "base_url": "https://prod.etim-international.com/Class/Details?classId=",
            },
        },
    },
    "unspsc": {
        "title": "UNSPSC Code Finder",
        "heading": "Get right UNSPSC codes for your products and services",
        "description": "The United Nations Standard Products and Services Code (UNSPSC) is a comprehensive, global classification system developed by the United Nations Development Programme (UNDP). This open, multi-sector standard enables organizations worldwide to classify products and services with precision and consistency. UNSPSC is essential for e-procurement platforms, supply chain optimization, spend analysis, vendor management, and facilitating B2B commerce across industries and borders.",
        "example": "Example: Office supplies",
        "embed_model_name": "gemini-embedding-001",
        "embed_dims": 3072,
        "versions": {
            "UNSPSC UNv260801.1 (18 March 2025)": {
                "collection_name": "UNSPSC_UNv260801-1-eng_new001-3072_v1",
                "base_url": "https://usa.databasesets.com/unspsc/search?keywords=",
            },
        },
    },
    "naics": {
        "title": "NAICS Code Finder",
        "heading": "Get appropriate codes from the NAICS standard",
        "description": "The North American Industry Classification System (NAICS) is the official industry classification system used by the United States, Canada, and Mexico to collect, analyze, and publish statistical data about their business economies. Developed jointly by these three countries, NAICS provides a standardized framework for measuring economic activity and is essential for business registration, tax reporting, government contracting, market research, and economic analysis across North America.",
        "example": "Example: Gamedev studio",
        "embed_model_name": "gemini-embedding-001",
        "embed_dims": 3072,
        "versions": {
            "2022 NAICS": {
                "collection_name": "NAICS_2022_eng_new001_v1",
                "base_url": "https://www.naics.com/code-search/?trms=",
                "tooltip": "T = Canadian, Mexican, and United States industries are comparable",
            },
            "2022 NAICS (only 6-digit codes)": {
                "collection_name": "NAICS_2022_SIXdigits_new001_v3",
                "base_url": "https://www.naics.com/naics-code-description/?code=",
            },
        },
    },
    "isic": {
        "title": "ISIC Classifier",
        "heading": "Instantly classify economic activities using the UN's ISIC",
        "description": "The International Standard Industrial Classification of All Economic Activities (ISIC) is the global reference classification for economic activities developed by the United Nations Statistics Division. Used by national statistical offices worldwide, ISIC provides a comprehensive framework for organizing economic data by type of productive activity. It serves as the foundation for compiling national accounts, analyzing industrial statistics, and facilitating international comparisons of economic structure and performance across countries.",
        "example": "Example: Manufacture of motor vehicles",
        "embed_model_name": "gemini-embedding-001",
        "embed_dims": 3072,
        "versions": {
            "ISIC Rev. 4": {
                "collection_name": "ISIC_4_new001_3corr",
                "base_url": "https://unstats.un.org/unsd/classifications/Econ/Structure/Detail/EN/27/",
            },
            "ISIC Rev. 5": {
                "collection_name": "ISIC_5_new001_v3corr",
            },
        },
    },
    "hs": {
        "title": "HS Code Finder",
        "heading": "Search HS codes for your goods",
        "description": "The Harmonized Commodity Description and Coding System (HS) is a globally standardized nomenclature developed by the World Customs Organization (WCO) for classifying traded products. Used by over 200 countries and territories, the HS serves as the foundation for international trade statistics, customs tariffs, and trade negotiations. This six-digit classification system is essential for importers, exporters, customs brokers, and logistics professionals to determine applicable duties, taxes, trade restrictions, and regulatory requirements for goods crossing international borders.",
        "example": "Example: Electric motor",
        "embed_model_name": "gemini-embedding-001",
        "versions": {
            "HS 2022": {
                "collection_name": "H6-HS_2022_new001_v4",
                "base_url": "https://www.tariffnumber.com/2025/",
            },
        },
    },
    "cn": {
        "title": "CN Code Finder",
        "heading": "Get CN codes for EU customs and trade",
        "description": "The Combined Nomenclature (CN) is the European Union's integrated tariff and statistical classification system, extending the international Harmonized System (HS) with EU-specific provisions. This 8-digit code structure is mandatory for all customs declarations, import/export documentation, and intra-EU trade statistics, serving as the legal basis for the EU's Common Customs Tariff and providing detailed classification for goods traded within the single market.",
        "example": "Example: Stainless steel sheets, 304 grade, 2mm thickness",
        "embed_model_name": "gemini-embedding-001",
        "versions": {
            "CN 2025": {
                "collection_name": "CN2025_v2",
                "base_url": "https://www.tariffnumber.com/2025/",
            },
        },
    },
    "nace": {
        "title": "NACE Business Activity Classifier",
        "heading": "Classify economic activities with EU's NACE standard",
        "description": "NACE (Nomenclature statistique des activités économiques) is the European Union's statistical classification of economic activities, developed by Eurostat to ensure harmonized economic analysis across all EU member states. This comprehensive framework enables consistent business registration, national accounts compilation, employment statistics, and cross-country economic comparisons, serving as the foundation for EU policy-making, regional development planning, and structural business statistics.",
        "example": "Example: Nuclear power plant (NPP)",
        "embed_model_name": "gemini-embedding-001",
        "versions": {
            "NACE Rev. 2.1": {
                "collection_name": "NACErev2-1_v2",
                "base_url": "https://showvoc.op.europa.eu/#/datasets/ESTAT_Statistical_Classification_of_Economic_Activities_in_the_European_Community_Rev._2.1._%28NACE_2.1%29/data?resId=http:%2F%2Fdata.europa.eu%2Fux2%2Fnace2.1%2F",
            },
        },
    },
    "cpv": {
        "title": "CPV Code Finder",
        "heading": "Find CPV codes for EU public procurement",
        "description": "The Common Procurement Vocabulary (CPV) is the European Union's standardized classification system for public procurement, established to ensure transparency and equal access to public contracts across the single market. This 9-digit hierarchical code structure is mandatory for all EU public procurement procedures, enabling consistent tender documentation, contract award notices in the TED system, and comprehensive market analysis while facilitating cross-border bidding and ensuring compliance with EU procurement directives.",
        "example": "Example: Indie gamedev studio",
        "embed_model_name": "gemini-embedding-001",
        "versions": {
            "CPV 2008 (ver. 2013)": {
                "collection_name": "cpv_2008_ver_2013_v3",
                # "base_url": "https://www.tariffnumber.com/2025/",
            },
            "CPV 2008 Supplementary codes": {
                "collection_name": "cpv_2008_ver_2013_Supplementary_codes_v2",
                # "base_url": "https://www.tariffnumber.com/2025/",
            },
        },
    },
    "nsn": {
        "title": "NATO Stock Number (NSN) Classifier",
        "heading": "Find NSN codes for military procurement",
        "description": "The NATO Stock Number (13 digits) consists of material group, material class, country code, and NIIN (National Item Identification Number). The NSN is a unique identifier for items of supply recognized by all NATO countries. The NSN is used to identify and manage supplies, ensuring that all member nations can effectively procure and utilize military equipment and materials. This classification system facilitates logistics, inventory management, and standardization across NATO forces.",
        "example": "Example: 1000W 120V AC power supply",
        "embed_model_name": "gemini-embedding-001",
        "versions": {
            "NSN extract (February 22, 2023)": {
                "collection_name": "nsn-extract-2-21-23_v3",
                # "base_url": "https://www.tariffnumber.com/2025/",
            },
        },
    },
    "hts": {
        "title": "HTS Code Finder",
        "heading": "Get Harmonized Tariff Schedule codes for US imports",
        "description": "Harmonized Tariff Schedule (HTS) is the United States comprehensive customs classification system for imported goods, extending the international Harmonized System (HS) with country-specific provisions. This 10-digit hierarchical code structure is mandatory for all US customs declarations and serves as the legal basis for determining applicable duties, taxes, trade restrictions, and regulatory requirements. The HTS is essential for importers, customs brokers, freight forwarders, and compliance professionals to ensure accurate classification and smooth customs clearance for goods entering the United States.",
        "example": "Example: Smartphone",
        "embed_model_name": "gemini-embedding-001",
        "embed_dims": 3072,
        "versions": {
            "HTS 2024": {
                "collection_name": "HTS_v4",
                "base_url": "https://hts.usitc.gov/search?query=",
            },
        },
    },
    "test": {
        "title": "Embedding Test Classifier",
        "heading": "Get codes for your goods",
        "description": "Is this really necessary here?",
        "example": "Example: Electric motor",
        "embed_model_name": "text-embedding-004",
        "embed_dims": 768,
        "versions": {
            "Old UNSPSC collection (text-embedding-004, 768)": {
                "collection_name": "UNSPSC_eng_UNv260801-1_768",
                "base_url": "https://usa.databasesets.com/unspsc/search?keywords=",
            },
        },
    },
}


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

    # Validate and normalize query
    normalized_query = query.strip()
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
        print(f"❌ Classification error for '{classifier_type}': {e}")
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
    if not RAPIDAPI_SECRET:
        return True

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
    print(f"🚀 RapidAPI classification request: {standard} <- {normalized_query}")

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
        print(f"❌ RapidAPI classification error: {e}")
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
    """
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
            unquote_plus(search_query).rstrip("/").replace("-", " ").strip()
        )
        # Sanitize the decoded query
        decoded_search_query = re.sub(r'[<>&"\']', "", decoded_search_query)[
            :4000
        ].strip()

    # Use pre-cached results if the query matches the example or is empty (base URL)
    preloaded_data = None
    example_text = config.get("example", "").replace("Example:", "").strip()

    if not decoded_search_query or decoded_search_query == example_text:
        preloaded_data = PRELOADED_RESULTS_CACHE.get(classifier_type)

    if not preloaded_data:
        preloaded_data = {
            "results_for_query": [],
            "query": decoded_search_query,
            "base_url": "",
            "tooltip": "",
            "total_request_time": 0,
        }

    # Slugify utility for SEO-friendly URLs
    def slugify(text):
        if not text:
            return ""
        # Sanitize input: limit length and remove harmful characters
        text = str(text)[:200]  # Limit to 200 chars max
        text = re.sub(r"[^\w\s-]", "", text.lower())
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
            **preloaded_data,
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
    print(
        f"❓ Received query for '{classifier_type}' classification with version '{version}'."
    )

    # Handle empty query gracefully
    normalized_description = product_description.strip()
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
        print(
            f"👇 Results for '{product_description}' in '{result['collection_name']}':\n"
            + "\n".join(result_lines)
        )

    except HTTPException:
        # Let HTTP exceptions propagate to the handler
        raise
    except Exception as e:
        print(f"❌ Error during '{classifier_type}' classification: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}"
        )

    end_total_time = time.perf_counter()
    total_request_time = end_total_time - start_total_time
    print(f"Total request processing time was {total_request_time:.4f}s")

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
"""
