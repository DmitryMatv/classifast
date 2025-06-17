import os
import time
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Callable, Awaitable

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.requests import Request

from google import genai
from qdrant_client import QdrantClient

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
        qdrant_client = QdrantClient(
            api_key=QDRANT_API_KEY,
            host=QDRANT_URL,
            port=443,
            https=True,
            prefer_grpc=False,
            timeout=60,
        )

        # Check if Qdrant client can list collections as a health check
        if qdrant_client:
            try:
                collections_result = qdrant_client.get_collections()
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
            if not qdrant_client.collection_exists(config["collection_name"]):
                print(
                    f"Warning: Collection {config['collection_name']} for {classifier_type} does not exist"
                )
                continue

            # Get collection info and check vector configuration
            collection_info = qdrant_client.get_collection(config["collection_name"])
            vector_params = collection_info.config.params.vectors
            embed_dims = config["embed_dims"]

            if isinstance(vector_params, dict) and "size" in vector_params:
                vector_size = vector_params["size"]
                if vector_size != embed_dims:
                    print(
                        f"Warning: Collection {config['collection_name']} has vector size {vector_size} but config specifies {embed_dims}"
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
            qdrant_client.close()
            print("Qdrant client closed.")
        except Exception as e:
            print(f"Error closing Qdrant client: {e}")


app = FastAPI(lifespan=lifespan)

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

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")


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
limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])

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
            embed_client.models.list()  # Simple check for Google client
            qdrant_client.get_collections()  # Simple check for Qdrant client
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
async def read_root(request: Request):
    """Serves the main homepage.
    This route now renders the general landing page.
    """
    return templates.TemplateResponse("index.html", {"request": request})


# Dictionary to map classifier types to their configurations
CLASSIFIER_CONFIG = {
    "etim": {
        "title": "ETIM Classifier",
        "heading": "Lookup ETIM codes of relevant categories from the ETIM International standard.",
        "description": "ETIM (ETIM Technical Information Model) is an open standard for the unambiguous grouping and specification of products in the technical sector through a uniform classification system. It is an initiative started to standardize the electronic exchange of product data for technical products, to enable the electronic trading of these products.",
        "version": "ETIM version 10.0 (2024-12-10)",
        "collection_name": "ETIM_10_eng_3072_exp",  # Specific collection for ETIM
        "example": "Example: Miniature circuit breaker, 16A, C-curve, 1P+N",
        "base_url": "https://prod.etim-international.com/Class/Details?classId=",
        "embed_model_name": "gemini-embedding-exp-03-07",
        "embed_dims": 3072,
    },
    # Add other classifiers here in the future
    "unspsc": {
        "title": "UNSPSC Classifier",
        "heading": "Find the right UNSPSC codes for your products and services.",
        "description": "UNSPSC is a global standard used to organize products and services into hierarchical categories. Accurate classification helps businesses improve spend analytics, streamline procurement, and enhance data governance-key steps toward efficiency and cost savings.",
        "version": "UNSPSC UNv260801 (August 14, 2023)",
        "collection_name": "UNSPSC_eng_UNv260801-1_768",
        "example": "Example: Laptop computer, 15 inch screen, 8GB RAM",
        "base_url": "https://usa.databasesets.com/unspsc/search?keywords=",  # Example, replace with actual if known
        "embed_model_name": "text-embedding-004",
        "embed_dims": 768,
    },
}


@app.get("/{classifier_type}", response_class=HTMLResponse)
async def show_classifier_page(request: Request, classifier_type: str):
    """Serves the specific classifier page based on the type.
    Renders the classifier_page.html template with context.
    """
    config = CLASSIFIER_CONFIG.get(classifier_type)
    if not config:
        raise HTTPException(
            status_code=404, detail=f"Classifier '{classifier_type}' not found"
        )

    return templates.TemplateResponse(
        "classifier_page.html",
        {
            "classifier_type": classifier_type,  # Pass type for form action URL
            "request": request,
            "title": config["title"],
            "heading": config["heading"],
            "description": config["description"],
            "version": config["version"],
            "example": config["example"],
        },
    )


@app.post("/{classifier_type}", response_class=HTMLResponse)
@limiter.limit("10/minute")  # Apply rate limit to this endpoint
async def handle_classify(
    request: Request,
    classifier_type: str,
    product_description: str = Form(...),
    top_k: int = Form(5),
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

    print(f"Received query for '{classifier_type}' classification.")

    # Start timer for total duration
    start_total_time = time.perf_counter()

    collection_name = config["collection_name"]

    try:
        # Call the batch classification function with the specific collection name
        # batch_results is now List[List[Dict[str, Any]]]
        # where each inner list is the hits for a query.
        results_for_single_query: List[List[Dict[str, Any]]] = classify_string_batch(
            qdrant_client=qdrant_client,  # Pass qdrant_client
            embed_client=embed_client,  # Pass embed_client
            embed_model_name=config["embed_model_name"],  # Use from config
            query_texts=[product_description],
            collection_name=collection_name,  # Pass the correct collection
            top_k=top_k,
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
            "total_request_time": total_request_time,
            "base_url": config.get("base_url"),
        },
    )


# uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
