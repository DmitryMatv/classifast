import os
import time
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.gzip import GZipMiddleware
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.requests import Request

from google import genai
from qdrant_client import AsyncQdrantClient

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

    # Initialize Qdrant Client
    QDRANT_URL = os.getenv("QDRANT_URL")
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
                for i, name in enumerate(collection_names, 1):
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
                version_config = config["versions"][version_name]
                collection_name = version_config["collection_name"]
                embed_model_name = config["embed_model_name"]
                base_url = version_config.get("base_url", "")
                tooltip = version_config.get("tooltip", "")

                results_for_single_query = await classify_string_batch(
                    qdrant_client=qdrant_client,
                    embed_client=embed_client,
                    embed_model_name=str(embed_model_name),
                    query_texts=[query],
                    collection_name=collection_name,
                    embed_dims=config.get("embed_dims"),
                    top_k=10,
                )

                results_for_query = (
                    results_for_single_query[0] if results_for_single_query else []
                )
                end_total_time = time.perf_counter()
                total_request_time = end_total_time - start_total_time

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

        # CSP header - updated for your specific needs
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://unpkg.com https://www.googletagmanager.com https://www.google-analytics.com https://static.cloudflareinsights.com; "
            "script-src-elem 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://unpkg.com https://www.googletagmanager.com https://www.google-analytics.com https://static.cloudflareinsights.com; "
            "script-src-attr 'unsafe-inline'; "
            "style-src 'self' https://cdn.tailwindcss.com 'unsafe-inline'; "
            "style-src-elem 'self' https://cdn.tailwindcss.com 'unsafe-inline'; "
            "style-src-attr 'unsafe-inline'; "
            "img-src 'self' https://*.classifast.com data: https://www.googletagmanager.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "connect-src 'self' https://www.google-analytics.com https://www.googletagmanager.com https://static.cloudflareinsights.com; "
            "object-src 'none'; "
            "base-uri 'self'; "
            "form-action 'self'; "
            "frame-ancestors 'none'; "
            "frame-src 'none'; "
            "media-src 'none'; "
            "manifest-src 'self'; "
            "worker-src 'none'; "
            "upgrade-insecure-requests;"
        )
        return response


app.add_middleware(SecurityHeadersMiddleware)

# Mount static files with caching
from fastapi.responses import Response


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


async def custom_rate_limit_exceeded_handler(_request, exc: Exception):
    if isinstance(exc, RateLimitExceeded):
        return HTMLResponse(
            content="<p>Rate limit exceeded. Please try again later.</p>",
            status_code=429,
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
        "title": "ETIM Classifier",
        "heading": "Get relevant EC classes from the ETIM International standard",
        "description": "ETIM (ETIM Technical Information Model) is a format to share and exchange product data based on taxonomic identification. This widely used classification standard for technical products was developed to structure the information flow between B2B professionals.",
        "example": """Example:
SH203-C20 Miniature Circuit Breaker 6kA 20A 3P
Characteristic curve: C-curve
Mounting: DIN rail""",
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
        "title": "UNSPSC Classifier",
        "heading": "Get right UNSPSC codes for your products and services",
        "description": "The United Nations Standard Products and Services Code (UNSPSC) is a comprehensive, global classification system developed by the United Nations Development Programme (UNDP). This open, multi-sector standard enables organizations worldwide to classify products and services with precision and consistency. UNSPSC is essential for e-procurement platforms, supply chain optimization, spend analysis, vendor management, and facilitating B2B commerce across industries and borders.",
        "example": "Example: Laptop computer, 15 inch screen, 8GB RAM",
        "embed_model_name": "text-embedding-004",
        "embed_dims": 768,
        "versions": {
            "UNSPSC UNv260801 (August 14, 2023)": {
                "collection_name": "UNSPSC_eng_UNv260801-1_768",
                "base_url": "https://usa.databasesets.com/unspsc/search?keywords=",
            },
        },
    },
    "naics": {
        "title": "NAICS Business Classifier",
        "heading": "Get appropriate codes from the NAICS standard",
        "description": "The North American Industry Classification System (NAICS) is the official industry classification system used by the United States, Canada, and Mexico to collect, analyze, and publish statistical data about their business economies. Developed jointly by these three countries, NAICS provides a standardized framework for measuring economic activity and is essential for business registration, tax reporting, government contracting, market research, and economic analysis across North America.",
        "example": "Example: Gamedev studio",
        "embed_model_name": "gemini-embedding-001",
        "embed_dims": 3072,
        "versions": {
            "2022 NAICS (only 6-digit categories)": {
                "collection_name": "NAICS_2022_SIXdigits_new001_v3",
                "base_url": "https://www.naics.com/naics-code-description/?v=2022&code=",
            },
            "2022 NAICS (all 2-to-6-digit categories)": {
                "collection_name": "NAICS_2022_eng_3072_exp",
                "base_url": "https://www.naics.com/naics-code-description/?v=2022&code=",
                "tooltip": "T = Canadian, Mexican, and United States industries are comparable",
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
                "collection_name": "ISIC_4_new001_2",
                "base_url": "https://unstats.un.org/unsd/classifications/Econ/Structure/Detail/EN/27/",
            },
            "ISIC Rev. 5": {
                "collection_name": "ISIC5_v7",
            },
        },
    },
    "hs": {
        "title": "HS Code Finder",
        "heading": "Get HS codes for your goods",
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
}


@app.get("/{classifier_type}", response_class=HTMLResponse)
@app.head("/{classifier_type}")
async def show_classifier_page(
    request: Request,
    classifier_type: str,
    search: str | None = None,
    version: str | None = None,
    top_k: int = 10,
):
    """
    Serves the specific classifier page, with optional URL parameters for search, version, and top_k.
    Supports both regular classifier pages and parameterized searches.
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
            "Link": f'<https://classifast.com/{classifier_type}>; rel="canonical"',
        }
        return Response(headers=headers)

    # Validate top_k parameter - always use default from frontend
    if top_k < 1 or top_k > 50:
        top_k = 10  # Default fallback

    # Get first version for default handling
    versions_list = list(config.get("versions", {}).keys())
    first_version = versions_list[0] if versions_list else ""

    # Use pre-cached results by default - skip classification on GET to avoid double-processing
    # URL parameters will be passed to the template for HTMX to handle via POST
    preloaded_data = PRELOADED_RESULTS_CACHE.get(
        classifier_type,
        {
            "results_for_query": [],
            "query": config.get("example", "").replace("Example:", "").strip(),
            "base_url": "",
            "tooltip": "",
            "total_request_time": 0,
        },
    )
    # --- End using pre-cached results ---

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
            # Determine if version should be included in URL params
            "url_params": {
                "search": search,
                "version": version if version and version != first_version else "",
                "top_k": top_k,
            },
            # Use the pre-loaded data from the cache or parameterized search
            **preloaded_data,
        },
    )

    # Cloudflare-friendly cache headers (aligned with homepage)
    response.headers["Cache-Control"] = "public, max-age=86400, s-maxage=604800"
    response.headers["Vary"] = "Accept-Encoding"
    response.headers["Link"] = (
        f'<https://classifast.com/{classifier_type}>; rel="canonical"'
    )
    response.headers["X-Robots-Tag"] = "index, follow"

    return response


@app.post("/{classifier_type}", response_class=HTMLResponse)
@limiter.limit("60/minute")  # Apply rate limit to this endpoint
async def handle_classify(
    request: Request,
    classifier_type: str,
    product_description: str = Form(...),
    top_k: int = Form(5),
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

    config = CLASSIFIER_CONFIG.get(classifier_type)
    if not config:
        raise HTTPException(
            status_code=404, detail=f"Classifier '{classifier_type}' not found"
        )

    version_config = config.get("versions", {}).get(version)
    if not version_config:
        raise HTTPException(
            status_code=404,
            detail=f"Version '{version}' for classifier '{classifier_type}' not found",
        )

    if not embed_client or not qdrant_client:
        raise HTTPException(
            status_code=503,
            detail="Backend services not available. Please check server logs.",
        )

    # Validate input: Check if text is empty or only whitespace
    if not product_description or not product_description.strip():
        # Return the results partial with an empty list or specific message
        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "query": product_description,  # Pass original (potentially empty) text
                "results_for_query": [],  # Empty results
            },
        )

    # Validate input length to prevent DoS attacks
    if len(product_description) > 4000:  # Token limit is 2048 for gemini-embedding-001
        raise HTTPException(
            status_code=400,
            detail="Product description too long. Maximum 4000 characters allowed.",
        )

    # Debug: Print the raw product_description to verify newlines are preserved
    # print(f"Raw product description (repr): {repr(product_description)}")
    # print(f"Product description length: {len(product_description)}")
    # print(f"Number of newlines in description: {product_description.count(chr(10))}")
    # print(f"Product description (first 200 chars): {product_description[:200]}")

    # Start timer for total duration
    start_total_time = time.perf_counter()

    collection_name = version_config["collection_name"]
    embed_model_name = config["embed_model_name"]

    try:
        # Call the batch classification function with the specific collection name
        # IMPORTANT: product_description is passed exactly as received from the form,
        # preserving all newlines, whitespace, and formatting for accurate embedding.
        # batch_results is now List[List[Dict[str, Any]]]
        # where each inner list is the hits for a query.
        results_for_single_query: List[List[Dict[str, Any]]] = (
            await classify_string_batch(
                qdrant_client=qdrant_client,  # Pass qdrant_client
                embed_client=embed_client,  # Pass embed_client
                embed_model_name=embed_model_name,  # Use from config
                query_texts=[product_description],  # Original text with all formatting
                collection_name=collection_name,  # Pass the correct collection
                embed_dims=config.get("embed_dims"),  # Use embed_dims from config
                top_k=top_k,
            )
        )

        classification_results: List[Dict[str, Any]] = []
        if results_for_single_query:
            classification_results = results_for_single_query[0]

        result_lines = [
            f"{result['payload'].get('original_id', 'N/A')} - {result['payload'].get('class_name', 'N/A')}"
            for result in classification_results
        ]
        print(
            f"👇 Results for '{product_description}' in '{collection_name}':\n"
            + "\n".join(result_lines)
        )

    except Exception as e:
        print(f"❌ Error during '{classifier_type}' classification: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}"
        )

    end_total_time = time.perf_counter()  # End timer for total duration
    total_request_time = end_total_time - start_total_time
    print(f"Total request processing time was {total_request_time:.4f}s")

    # Render the results partial
    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "query": product_description,
            "results_for_query": classification_results,
            "base_url": version_config.get("base_url", ""),
            "tooltip": version_config.get("tooltip", ""),
            "total_request_time": total_request_time,
        },
    )


# npm install tailwindcss @tailwindcss/cli
# npx @tailwindcss/cli -i ./app/static/css/input.css -o ./app/static/css/styles.css --watch

# uvicorn app.main:app --reload --port 8001
