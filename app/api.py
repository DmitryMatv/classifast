import asyncio
import logging
import os
import time
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .classifier import get_classification_cache_headers
from .classifier_config import CLASSIFIER_CONFIG

logger = logging.getLogger(__name__)

router = APIRouter()

# ===== RAPIDAPI INTEGRATION =====


# Pydantic models for RapidAPI
class ClassificationResult(BaseModel):
    code: str = Field(..., description="Classification code")
    name: str = Field(..., description="Classification name/description")
    score: float = Field(..., description="Similarity score (0-1)")
    url: Optional[str] = Field(None, description="External URL for more information")


class RapidAPIResponse(BaseModel):
    query: str = Field(..., description="Original query")
    standard: str = Field(..., description="Classification standard used")
    version: str = Field(..., description="Version of the standard used")
    results: list[ClassificationResult] = Field(
        ..., description="Classification results"
    )
    processing_time: float = Field(..., description="Processing time in seconds")


# RapidAPI configuration
RAPIDAPI_SECRET = os.getenv("RAPIDAPI_SECRET")


def verify_rapidapi_auth(request: Request) -> bool:
    """
    Verify RapidAPI authentication via proxy secret.

    RapidAPI validates API keys on their infrastructure and forwards requests
    with X-RapidAPI-Proxy-Secret header. We verify this secret to ensure
    requests came through the official RapidAPI proxy.
    """
    if not RAPIDAPI_SECRET:
        logger.warning("RAPIDAPI_SECRET not configured - API authentication disabled")
        raise HTTPException(
            status_code=503,
            detail="API authentication not configured",
        )

    proxy_secret = request.headers.get("X-RapidAPI-Proxy-Secret")
    if not proxy_secret:
        raise HTTPException(
            status_code=401,
            detail="Missing authentication - use RapidAPI to access this endpoint",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if proxy_secret != RAPIDAPI_SECRET:
        logger.warning("Invalid RapidAPI proxy secret received")
        raise HTTPException(status_code=401, detail="Invalid authentication")

    return True


@router.get(
    "/classify",
    response_model=RapidAPIResponse,
    dependencies=[Depends(verify_rapidapi_auth)],
)
async def rapid_classify(
    request: Request,
    query: str = Query(..., description="Product or service description to classify"),
    standard: str = Query(
        ..., description="Classification standard (UNSPSC, ETIM, NAICS, ISIC, HS)"
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

    # Normalize inputs early to ensure cache hits and prevent unnecessary API calls
    normalized_query = query.strip()
    normalized_standard = standard.strip().upper()

    if not normalized_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if not normalized_standard:
        raise HTTPException(status_code=400, detail="Standard cannot be empty")

    logger.info(
        "RapidAPI classification request: %s <- %s",
        normalized_standard,
        normalized_query,
    )

    start_time = time.perf_counter()

    try:
        outcome = await request.app.state.classification_service.classify(
            query=normalized_query,
            classifier_type=normalized_standard,
            version=version,
            top_k=top_k or 1,
        )

        classification_results = outcome.results

        # Format results for API response
        formatted_results = []
        for r in classification_results:
            payload = r.get("payload", {})
            base_url = outcome.version_config.get("base_url", "")
            append_code_to_url = outcome.version_config.get("append_code_to_url", True)
            code = payload.get("original_id", "")

            formatted_result = ClassificationResult(
                code=code,
                name=payload.get("class_name", ""),
                score=r.get("score", 0.0),
                url=(f"{base_url}{code}" if append_code_to_url else base_url)
                if base_url and code
                else None,
            )
            formatted_results.append(formatted_result)

        processing_time = time.perf_counter() - start_time

        response_data = RapidAPIResponse(
            query=normalized_query,
            standard=normalized_standard.lower(),
            version=outcome.version_name,
            results=formatted_results,
            processing_time=processing_time,
        )

        # RapidAPI does its own metering upstream, so this surface bypasses website quota tracking.
        cache_headers = get_classification_cache_headers()
        return JSONResponse(content=response_data.model_dump(), headers=cache_headers)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("RapidAPI classification error: %s", e)
        raise HTTPException(status_code=500, detail="Classification failed")


@router.get("/standards", dependencies=[Depends(verify_rapidapi_auth)])
async def rapid_standards(request: Request):
    """List available classification standards and their versions."""
    standards_info = {}

    for standard_key, config in CLASSIFIER_CONFIG.items():
        standards_info[standard_key] = {
            "title": config["title"],
            "description": config["description"],
            "versions": list(config["versions"].keys()),
            "example": config["example"].replace("Example:", "").strip(),
        }

    cache_headers = get_classification_cache_headers()
    return JSONResponse(
        content={"standards": standards_info, "timestamp": time.time()},
        headers=cache_headers,
    )


@router.get("/ping")
async def rapid_health_public(request: Request):
    """Public health check endpoint for RapidAPI consumers."""
    health_status: dict[str, Any] = {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {},
    }

    embed_client = getattr(request.app.state, "embed_client", None)
    qdrant_client = getattr(request.app.state, "qdrant_client", None)

    # Check embedding service
    if embed_client:
        health_status["services"]["embedding"] = "configured"
    else:
        health_status["services"]["embedding"] = "unavailable"

    # Check Qdrant service
    if qdrant_client:
        try:
            await asyncio.wait_for(
                asyncio.to_thread(qdrant_client.get_collections), timeout=5
            )
            health_status["services"]["database"] = "healthy"
        except Exception:
            health_status["services"]["database"] = "unhealthy"
    else:
        health_status["services"]["database"] = "unavailable"

    # Overall health
    all_healthy = all(
        v in ("healthy", "configured") for v in health_status["services"].values()
    )
    status_code = 200 if all_healthy else 503

    return JSONResponse(content=health_status, status_code=status_code)


@router.get("/debug-headers")
def debug_headers(request: Request) -> JSONResponse:
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
