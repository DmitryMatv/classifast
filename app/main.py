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

# We need to make the clients and config available to classifier functions
# One way is to make them global here, or pass them explicitly.
# Since classifier.py uses them as globals, we'll define them globally here after init.

# --- Global variables to be initialized on startup ---
EMBED_CLIENT = None
QDRANT_CLIENT = None
EMBED_MODEL_NAME = "text-embedding-004"  # Default, can be overridden by env
QDRANT_COLLECTION = "ETIM10_google"  # Default, can be overridden by env
# --- End Global Variables ---

# Load environment variables from .env file
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
        EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", EMBED_MODEL_NAME)
        QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", QDRANT_COLLECTION)

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
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serves the main page with the input form."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/classify", response_class=HTMLResponse)
async def handle_classify(request: Request, product_description: str = Form(...)):
    """
    Receives product description, classifies it, and returns HTML partial with results.
    """
    if not EMBED_CLIENT or not QDRANT_CLIENT:
        raise HTTPException(
            status_code=503,
            detail="Backend services not available. Please check server logs.",
        )

    if not product_description or not product_description.strip():
        # Return an empty result or an error message partial
        return templates.TemplateResponse(
            "results.html", {"request": request, "results_for_query": []}
        )

    print(f"Received query for classification: '{product_description}'")

    # The classifier.py script's classify_string_batch expects query_texts to be a list.
    # It also relies on global variables for clients and config, which we set up in lifespan.
    try:
        # Ensure global variables in classifier module are correctly set if not already
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
        if (
            not hasattr(classifier_module, "QDRANT_COLLECTION_NAME")
            or classifier_module.QDRANT_COLLECTION_NAME is None
        ):
            classifier_module.QDRANT_COLLECTION_NAME = QDRANT_COLLECTION

        # Call the batch classification function (even for a single query)
        batch_results: List[List[Dict[str, Any]]] = classify_string_batch(
            query_texts=[product_description], top_k=3  # Or make this configurable
        )

        classification_results = []
        if batch_results and len(batch_results) > 0:
            classification_results = batch_results[
                0
            ]  # We sent one query, so we take the first list of results

        print(
            f"Classification results for '{product_description}': {classification_results}"
        )

    except Exception as e:
        print(f"Error during classification: {e}")
        # Consider how to inform the user; for now, return empty results
        # You might want to return a specific error message partial
        raise HTTPException(
            status_code=500, detail=f"Error processing your request: {str(e)}"
        )

    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "query": product_description,
            "results_for_query": classification_results,
        },
    )


# To run this app (from the directory containing the 'app' folder):
# uvicorn app.main:app --reload
# Ensure you have .env file with GEMINI_API_KEY, and optionally QDRANT_URL or QDRANT_PATH,
# QDRANT_COLLECTION_NAME, EMBED_MODEL_NAME.
