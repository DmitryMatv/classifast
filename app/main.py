import os
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from typing import List, Dict, Any
from google import genai
from .classifier import classify_string_batch
from starlette.middleware.base import BaseHTTPMiddleware

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs when the application starts
    global EMBED_CLIENT, QDRANT_CLIENT, EMBED_MODEL_NAME, QDRANT_COLLECTION

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
    # User wants a placeholder for Qdrant connection.
    # We'll allow QDRANT_URL (for remote/dockerized) or QDRANT_PATH (for local)
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_db")  # Default local path
    QDRANT_API_KEY = os.getenv(
        "QDRANT_API_KEY"
    )  # Optional API key for Qdrant Cloud/secured instances

    try:
        from qdrant_client import QdrantClient

        if QDRANT_URL:
            print(f"Connecting to Qdrant at URL: {QDRANT_URL}")
            QDRANT_CLIENT = QdrantClient(
                host=QDRANT_URL,
                port=443,  # Use HTTPS port since Traefik handles SSL
                api_key=QDRANT_API_KEY,
                https=True,
                prefer_grpc=False,
                timeout=60,  # Add a longer timeout just in case
            )
        else:
            print(f"Initializing Qdrant client with local path: {QDRANT_PATH}")
            QDRANT_CLIENT = QdrantClient(path=QDRANT_PATH)

        # Make QDRANT_CLIENT available to classifier.py through its global 'client'
        # This is a bit of a hack due to not modifying classifier.py.
        # A better way would be to pass clients explicitly to functions.
        import app.classifier as classifier_module

        classifier_module.client = QDRANT_CLIENT
        classifier_module.EMBED_CLIENT = (
            EMBED_CLIENT  # Also set the embed client for classifier
        )

        # Load model name and collection name from env or use defaults
        EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "text-embedding-004")
        QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", "default_collection")

        classifier_module.EMBED_MODEL = EMBED_MODEL_NAME
        classifier_module.QDRANT_COLLECTION_NAME = QDRANT_COLLECTION

        # Check if collection exists (optional, but good for early feedback)
        if QDRANT_CLIENT and not QDRANT_CLIENT.collection_exists(
            collection_name=QDRANT_COLLECTION
        ):
            print(
                f"Warning: Qdrant collection '{QDRANT_COLLECTION}' does not exist yet."
            )
        else:
            print(f"Qdrant client initialized. Using collection: {QDRANT_COLLECTION}")

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


# Middleware to add Cache-Control headers for specific static files
class CacheControlMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if request.url.path == "/static/styles.css":
            # Set Cache-Control for 1 day (86400 seconds).
            # For infrequently changing files with cache-busting (e.g., style.v1.css),
            # you can use a much longer duration (e.g., 31536000 for 1 year).
            response.headers["Cache-Control"] = "public, max-age=86400"
        return response


app.add_middleware(CacheControlMiddleware)

# Mount static files (for CSS, JS)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="app/templates")


# --- Health Check Endpoint ---
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
            # For Qdrant, checking collection existence or a simple count might be too slow
            # A basic check that the client object exists is often sufficient for a healthcheck
            pass
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


# --- Routes ---

# Dictionary to map classifier types to their configurations
CLASSIFIER_CONFIG = {
    "etim": {
        "title": "ETIM International",
        "description": "Classify products based on the ETIM International standard.",
        "collection_name": "ETIM_10_eng_768",  # Specific collection for ETIM
        "placeholder": "Resistor, 10 Ohm, 1W",
    },
    # Add other classifiers here in the future
    "unspsc": {
        "title": "UNSPSC",
        "description": "Classify products based on the UNSPSC standard.",
        "collection_name": "UNSPSC_v24_google",
        "placeholder": "Computer monitor, 24 inch",
    },
}


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serves the main homepage.
    This route now renders the general landing page.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/classify/{classifier_type}", response_class=HTMLResponse)
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


@app.post("/classify/{classifier_type}", response_class=HTMLResponse)
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
            {"request": request, "results_for_query": [], "query": product_description},
        )

    print(
        f"Received query for '{classifier_type}' classification: '{product_description}'"
    )

    collection_name = config["collection_name"]

    try:
        # Ensure classifier module globals are set (redundant if lifespan works, but safe)
        import app.classifier as classifier_module

        if not hasattr(classifier_module, "client") or classifier_module.client is None:
            classifier_module.client = QDRANT_CLIENT
        if (
            not hasattr(classifier_module, "EMBED_CLIENT")
            or classifier_module.EMBED_CLIENT is None
        ):
            classifier_module.EMBED_CLIENT = EMBED_CLIENT
        if (
            not hasattr(classifier_module, "EMBED_MODEL")
            or classifier_module.EMBED_MODEL is None
        ):
            classifier_module.EMBED_MODEL = EMBED_MODEL_NAME

        # Call the batch classification function with the specific collection name
        batch_results: List[List[Dict[str, Any]]] = classify_string_batch(
            query_texts=[product_description],
            collection_name=collection_name,  # Pass the correct collection
            top_k=5,
        )

        classification_results = []
        if batch_results and len(batch_results) > 0:
            classification_results = batch_results[0]

        print(
            f"Classification results for '{product_description}' in '{collection_name}': {classification_results}"
        )

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
        },
    )


# To run this app (from the directory containing the 'app' folder):
# uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
# Ensure you have .env file with GEMINI_API_KEY, and optionally QDRANT_URL or QDRANT_PATH,
# QDRANT_COLLECTION_NAME, EMBED_MODEL_NAME.
