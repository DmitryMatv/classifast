import os
import time
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Callable, Awaitable

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
from qdrant_client import AsyncQdrantClient, QdrantClient

from .classifier import classify_string_batch

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs when the application starts
    global embed_client, embed_model_name, qdrant_client
    embed_client = None  # Initialize to None
    qdrant_client = None  # Initialize to None

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
        print(f"Connecting to Qdrant at URL: {QDRANT_URL}")
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
                print(
                    f"Qdrant client initialized. Found collections: {[col.name for col in collections_result.collections]}"
                )
            except Exception as e:
                print(f"Qdrant client initialized, but could not list collections: {e}")
                # Depending on severity, you might still want to set qdrant_client to None or raise
        else:
            print("Qdrant client could not be initialized.")

        # Verify collections exist and store their vector sizes
        for classifier_type, config in CLASSIFIER_CONFIG.items():
            for version, version_config in config.get("versions", {}).items():
                collection_name = version_config.get("collection_name")
                if not collection_name:
                    continue
                if not await qdrant_client.collection_exists(collection_name):
                    print(
                        f"Warning: Collection {collection_name} for {classifier_type} version {version} does not exist"
                    )
                    continue

                # Get collection info and check vector configuration
                collection_info = await qdrant_client.get_collection(collection_name)
                vector_params = collection_info.config.params.vectors
                embed_dims = version_config.get("embed_dims")

                if isinstance(vector_params, dict) and "size" in vector_params:
                    vector_size = vector_params["size"]
                    if vector_size != embed_dims:
                        print(
                            f"Warning: Collection {collection_name} has vector size {vector_size} but config specifies {embed_dims}"
                        )

    except Exception as e:
        print(f"Error initializing Qdrant client: {e}")

    if not embed_client or not qdrant_client:
        print(
            "Critical Error: One or more clients failed to initialize. The application might not function correctly."
        )

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


"""
# Security Headers Middleware
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        # Set security headers
        response.headers["X-Content-Type-Options"] = (
            "nosniff"  # Prevents MIME type sniffing
        )
        response.headers["X-Frame-Options"] = "DENY"  # Prevents clickjacking
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"  # HSTS
        )
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.tailwindcss.com https://umami.classifast.com https://unpkg.com; "
            "script-src-elem 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.tailwindcss.com https://umami.classifast.com https://unpkg.com; "
            "script-src-attr 'unsafe-inline'; "  # Allow inline event handlers and script attributes
            "style-src 'self' https://cdn.tailwindcss.com 'unsafe-inline'; "
            "style-src-elem 'self' https://cdn.tailwindcss.com 'unsafe-inline'; "
            "style-src-attr 'unsafe-inline'; "
            "img-src 'self' https://*.classifast.com data: /static/images/; "
            "font-src 'self' https://fonts.gstatic.com; "
            "connect-src 'self' https://umami.classifast.com; "
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
"""

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
                response.headers["Cache-Control"] = "public, max-age=604800"  # 1 week
            else:
                response.headers["Cache-Control"] = "public, max-age=86400"  # 1 day
            response.headers["ETag"] = f'"{hash(path)}"'
        return response


app.mount("/static", CachedStaticFiles(directory="app/static"), name="static")


@app.get("/favicon.ico", response_class=FileResponse, include_in_schema=False)
async def favicon():
    return FileResponse("app/static/images/favicon.ico")


@app.get("/robots.txt", response_class=FileResponse)
async def robots_txt():
    return "app/static/robots.txt"


@app.get("/sitemap.xml", response_class=FileResponse)
async def sitemap_xml():
    return "app/static/sitemap.xml"


@app.get("/ads.txt", response_class=FileResponse)
async def ads_txt():
    return "app/static/ads.txt"


# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

app.state.limiter = limiter


async def custom_rate_limit_exceeded_handler(request: Request, exc: Exception):
    if isinstance(exc, RateLimitExceeded):
        return HTMLResponse(
            content=f"<p>Too many requests. Please try again in {exc.detail}.</p>",
            status_code=429,
        )
    return HTMLResponse(content="Internal Server Error", status_code=500)


app.add_exception_handler(RateLimitExceeded, custom_rate_limit_exceeded_handler)


# Healthcheck
@app.get("/health")
async def health_check():
    """
    Health check endpoint for Docker/Kubernetes.
    """
    # Basic check: if we can reach here, the app is running.
    # More sophisticated checks could be added here (e.g., DB connectivity).
    if embed_client and qdrant_client:
        # Optionally, perform a quick check on clients
        try:
            # Test embed client
            embed_client.models.list()
            # Test qdrant client
            await qdrant_client.get_collections()
            return {"status": "healthy", "embed_client": "ok", "qdrant_client": "ok"}
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Service Unavailable: Client error - {str(e)}",
            )
    elif not embed_client:
        raise HTTPException(
            status_code=503,
            detail="Service Unavailable: embed_client not initialized",
        )
    elif not qdrant_client:
        raise HTTPException(
            status_code=503,
            detail="Service Unavailable: qdrant_client not initialized",
        )
    return {"status": "unhealthy", "detail": "One or more clients are not initialized"}


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
            "Cache-Control": "public, max-age=86400, s-maxage=86400",
            "Vary": "Accept-Encoding",
            "Content-Type": "text/html; charset=utf-8",
        }
        return Response(headers=headers)

    response = templates.TemplateResponse("index.html", {"request": request})

    # Cloudflare-friendly cache headers (same as classifier pages)
    response.headers["Cache-Control"] = "public, max-age=86400, s-maxage=86400"
    response.headers["Vary"] = "Accept-Encoding"

    return response


# Dictionary to map classifier types to their configurations
CLASSIFIER_CONFIG = {
    "etim": {
        "title": "ETIM Classifier",
        "heading": "Find relevant EC classes from the ETIM International standard",
        "description": "ETIM (ETIM Technical Information Model) is a format to share and exchange product data based on taxonomic identification. This widely used classification standard for technical products was developed to structure the information flow between B2B professionals.",
        "example": """Example:
Miniature circuit breaker
Current rating: 16A
Characteristic curve: C-curve
Number of poles: 1P+N
Breaking capacity: 6kA
Mounting: DIN rail""",
        "base_url": "https://prod.etim-international.com/Class/Details?classId=",
        "versions": {
            "ETIM version 10.0 (2024-12-10)": {
                "embed_model_name": "gemini-embedding-exp-03-07",
                "embed_dims": 3072,
                "collection_name": "ETIM_10_eng_3072_exp",
            },
        },
    },
    "unspsc": {
        "title": "UNSPSC Classifier",
        "heading": "Find the right UNSPSC codes for your products and services",
        "description": "The United Nations Standard Products and Services Code (UNSPSC), owned by the United Nations Development Programme (UNDP), is an open, global, multi-sector standard for efficient, accurate classification of products and services. It is used by organizations worldwide to facilitate procurement, in spend analysis, and in supply chain management.",
        "example": "Example: Laptop computer, 15 inch screen, 8GB RAM",
        "base_url": "https://usa.databasesets.com/unspsc/search?keywords=",  # Example, replace with actual if known
        "versions": {
            "UNSPSC UNv260801 (August 14, 2023)": {
                "embed_model_name": "text-embedding-004",
                "embed_dims": 768,
                "collection_name": "UNSPSC_eng_UNv260801-1_768",
            },
        },
    },
    "naics": {
        "title": "NAICS Classifier",
        "heading": "Find appropriate NAICS codes from the NAICS standard",
        "description": "The North American Industry Classification System (NAICS) is the standard used by Federal statistical agencies in classifying business establishments for the purpose of collecting, analyzing, and publishing statistical data related to the U.S. business economy.",
        "example": "Example: Software publishers",
        "base_url": "https://www.naics.com/naics-code-description/?v=2022&code=",
        "tooltip": "T = Canadian, Mexican, and United States industries are comparable",
        "versions": {
            "2022 NAICS": {
                "embed_model_name": "gemini-embedding-exp-03-07",
                "embed_dims": 3072,
                "collection_name": "NAICS_2022_eng_3072_exp",
            },
        },
    },
}


@app.get("/{classifier_type}", response_class=HTMLResponse)
async def show_classifier_page(request: Request, classifier_type: str):
    """Serves the specific classifier page with Cloudflare-friendly caching."""
    config = CLASSIFIER_CONFIG.get(classifier_type)
    if not config:
        raise HTTPException(
            status_code=404, detail=f"Classifier '{classifier_type}' not found"
        )

    response = templates.TemplateResponse(
        "classifier_page.html",
        {
            "classifier_type": classifier_type,
            "request": request,
            "title": config["title"],
            "heading": config["heading"],
            "description": config["description"],
            "versions": list(config["versions"].keys()),
            "example": config["example"],
        },
    )

    # Cloudflare-friendly cache headers
    response.headers["Cache-Control"] = "public, max-age=86400, s-maxage=86400"
    response.headers["Vary"] = "Accept-Encoding"

    return response


@app.post("/{classifier_type}", response_class=HTMLResponse)
@limiter.limit("300/minute")  # Apply rate limit to this endpoint
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

    if not product_description or not product_description.strip():
        # Return the results partial with an empty list or specific message
        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "query": product_description,
                "results_for_query": [],  # Empty results
            },
        )

    print(
        f"Received query for '{classifier_type}' classification with version '{version}'."
    )

    # Start timer for total duration
    start_total_time = time.perf_counter()

    collection_name = version_config["collection_name"]
    embed_model_name = version_config["embed_model_name"]

    try:
        # Call the batch classification function with the specific collection name
        # batch_results is now List[List[Dict[str, Any]]]
        # where each inner list is the hits for a query.
        results_for_single_query: List[List[Dict[str, Any]]] = (
            await classify_string_batch(
                qdrant_client=qdrant_client,  # Pass qdrant_client
                embed_client=embed_client,  # Pass embed_client
                embed_model_name=embed_model_name,  # Use from config
                query_texts=[product_description],
                collection_name=collection_name,  # Pass the correct collection
                top_k=top_k,
            )
        )

        classification_results: List[Dict[str, Any]] = []
        if results_for_single_query:
            classification_results = results_for_single_query[0]

        print(
            f"Results for '{product_description}' in '{collection_name}':\n{classification_results}"
        )

    except Exception as e:
        print(f"Error during '{classifier_type}' classification: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}"
        )

    end_total_time = time.perf_counter()  # End timer for total duration
    total_request_time = end_total_time - start_total_time
    print(f"Total request processing time: {total_request_time:.6f} seconds")

    # Render the results partial
    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "query": product_description,
            "results_for_query": classification_results,
            "base_url": config.get("base_url", ""),
            "tooltip": config.get("tooltip", ""),
            "total_request_time": total_request_time,
        },
    )


# uvicorn app.main:app --reload --port 8001
