import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from google import genai
from qdrant_client import QdrantClient

from .classifier import classify_string_batch

# from starlette.middleware.base import BaseHTTPMiddleware
# from starlette.types import CallNext

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs when the application starts
    global EMBED_CLIENT, EMBED_MODEL_NAME, QDRANT_CLIENT  # QDRANT_COLLECTION removed

    print("FastAPI application startup...")

    # Initialize Embedding Client (Google GenAI)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not found in environment variables.")
        # In a real app, you might raise an exception or handle this more gracefully
    else:
        try:
            EMBED_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
            EMBED_CLIENT.models.list()  # Test connection
            print("Google GenAI Client initialized successfully.")
        except Exception as e:
            print(f"Error initializing Google GenAI Client: {e}")
            EMBED_CLIENT = None  # Ensure it's None if init fails

    # Initialize Qdrant Client
    # We'll allow QDRANT_URL (for remote/dockerized) or QDRANT_PATH (for local)
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_db")  # Default local path
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    try:
        if QDRANT_URL:
            print(f"Connecting to Qdrant at URL: {QDRANT_URL}")
            QDRANT_CLIENT = QdrantClient(
                api_key=QDRANT_API_KEY,
                # url=QDRANT_URL,
                host=QDRANT_URL,
                port=443,  # Use HTTPS port since Traefik handles SSL
                https=True,
                prefer_grpc=False,
                timeout=60,  # Add a longer timeout just in case
            )
        else:
            print(f"Initializing Qdrant client with local path: {QDRANT_PATH}")
            QDRANT_CLIENT = QdrantClient(path=QDRANT_PATH)

        # Load model name from env or use defaults
        EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "text-embedding-004")

        # Check if Qdrant client can list collections as a health check
        if QDRANT_CLIENT:
            try:
                collections_result = QDRANT_CLIENT.get_collections()
                print(
                    f"Qdrant client initialized. Found collections: {[col.name for col in collections_result.collections]}"
                )
            except Exception as e:
                print(f"Qdrant client initialized, but could not list collections: {e}")
                # Depending on severity, you might still want to set QDRANT_CLIENT to None or raise
        else:
            print("Qdrant client could not be initialized.")

    except Exception as e:
        print(f"Error initializing Qdrant client: {e}")
        QDRANT_CLIENT = None

    if not EMBED_CLIENT or not QDRANT_CLIENT:
        print(
            "Critical Error: One or more clients failed to initialize. The application might not function correctly."
        )

    yield
    # Runs when the application is shutting down
    print("FastAPI application shutdown...")
    if QDRANT_CLIENT:
        try:
            QDRANT_CLIENT.close()
            print("Qdrant client closed.")
        except Exception as e:
            print(f"Error closing Qdrant client: {e}")


app = FastAPI(lifespan=lifespan)

"""
# Security Headers Middleware
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: CallNext) -> Response:
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
            "script-src 'self' https://cdn.tailwindcss.com https://unpkg.com https://umami.classifast.com; "
            "style-src 'self' https://cdn.tailwindcss.com 'unsafe-inline'; "
            "img-src 'self' data: /static/images/favicon-32x32.png; "
            "object-src 'none'; "
            "frame-ancestors 'none';"
        )
        return response


app.add_middleware(SecurityHeadersMiddleware)
"""

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/favicon.ico", response_class=FileResponse, include_in_schema=False)
async def favicon():
    return FileResponse("app/static/images/favicon-32x32.png")


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


async def custom_rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    return HTMLResponse(
        content=f"<p>Too many requests. Please try again in {exc.detail}.</p>",
        status_code=429,
    )


app.add_exception_handler(RateLimitExceeded, custom_rate_limit_exceeded_handler)


# Healthcheck
@app.get("/health")
async def health_check():
    """
    Health check endpoint for Docker/Kubernetes.
    """
    # Basic check: if we can reach here, the app is running.
    # More sophisticated checks could be added here (e.g., DB connectivity).
    if EMBED_CLIENT and QDRANT_CLIENT:
        # Optionally, perform a quick check on clients
        try:
            EMBED_CLIENT.models.list()  # Simple check for Google client
            QDRANT_CLIENT.get_collections()  # Simple check for Qdrant client
            return {"status": "healthy", "embed_client": "ok", "qdrant_client": "ok"}
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Service Unavailable: Client error - {str(e)}",
            )
    elif not EMBED_CLIENT:
        raise HTTPException(
            status_code=503, detail="Service Unavailable: EMBED_CLIENT not initialized"
        )
    elif not QDRANT_CLIENT:
        raise HTTPException(
            status_code=503, detail="Service Unavailable: QDRANT_CLIENT not initialized"
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
        "title": "ETIM International",
        "description": "Classify products based on the ETIM International standard.",
        "collection_name": "ETIM_10_eng_3072_exp",  # Specific collection for ETIM
        "placeholder": "Resistor, 10 Ohm, 1W",
        "base_url": "https://prod.etim-international.com/Class/Details?classId=",
    },
    # Add other classifiers here in the future
    "unspsc": {
        "title": "UNSPSC",
        "description": "Classify products based on the UNSPSC standard.",
        "collection_name": "UNSPSC_v24_google",
        "placeholder": "Computer monitor, 24 inch",
        "base_url": "https://www.unspsc.org/search-code=",  # Example, replace with actual if known
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
            "request": request,
            "title": config["title"],
            "description": config["description"],
            "placeholder": config["placeholder"],
            "classifier_type": classifier_type,  # Pass type for form action URL
        },
    )


@app.post("/{classifier_type}", response_class=HTMLResponse)
@limiter.limit("10/minute")  # Apply rate limit to this endpoint
async def handle_classify(
    request: Request, classifier_type: str, product_description: str = Form(...)
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

    if not EMBED_CLIENT or not QDRANT_CLIENT:
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
                "results_for_query": [],
                "base_url": config.get("base_url"),
            },
        )

    print(
        f"Received query for '{classifier_type}' classification: '{product_description}'"
    )

    collection_name = config["collection_name"]

    try:
        # Call the batch classification function with the specific collection name
        batch_results: List[List[Dict[str, Any]]] = classify_string_batch(
            qdrant_client=QDRANT_CLIENT,  # Pass QDRANT_CLIENT
            embed_client=EMBED_CLIENT,  # Pass EMBED_CLIENT
            embed_model_name=EMBED_MODEL_NAME,  # Pass EMBED_MODEL_NAME
            query_texts=[product_description],
            collection_name=collection_name,  # Pass the correct collection
            top_k=5,
        )

        classification_results = []
        query_time = None
        if batch_results and len(batch_results) > 0:
            # batch_results is now a list of dicts, each with 'hits' and 'time'
            # Assuming single query in batch for this endpoint
            classification_results = batch_results[0].get("hits", [])
            query_time = batch_results[0].get("time")

        print(
            f"Classification results for '{product_description}' in '{collection_name}': {classification_results}"
        )
        if query_time is not None:
            print(f"Qdrant query time: {query_time:.6f} seconds")

    except Exception as e:
        print(f"Error during '{classifier_type}' classification: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error processing your request: {str(e)}"
        )

    # Render the results partial
    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "query": product_description,
            "results_for_query": classification_results,
            "base_url": config.get("base_url"),
        },
    )


# uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
