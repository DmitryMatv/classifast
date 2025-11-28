import logging
import os
import time
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .classifier import perform_classification
from .classifier_config import CLASSIFIER_CONFIG
from .dependencies import rapid_limiter

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
    results: List[ClassificationResult] = Field(
        ..., description="Classification results"
    )
    processing_time: float = Field(..., description="Processing time in seconds")


# RapidAPI configuration
RAPIDAPI_SECRET = os.getenv("RAPIDAPI_SECRET")


async def verify_rapidapi_auth(request: Request) -> bool:
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
@rapid_limiter.limit("600/minute")
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
    normalized_query = query.strip()
    logger.info("RapidAPI classification request: %s <- %s", standard, normalized_query)

    start_time = time.perf_counter()

    try:
        # Use shared classification service
        result = await perform_classification(
            embed_client=request.app.state.embed_client,
            qdrant_client=request.app.state.qdrant_client,
            query=normalized_query,
            classifier_type=standard,
            version=version,
            top_k=top_k or 1,
            quantization_cache=getattr(
                request.app.state, "collection_quantization_cache", None
            ),
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
        raise
    except Exception as e:
        logger.error("RapidAPI classification error: %s", e)
        raise HTTPException(status_code=500, detail="Classification failed")


@router.get("/standards", dependencies=[Depends(verify_rapidapi_auth)])
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


@router.get("/ping")
@rapid_limiter.limit("600/minute")
async def rapid_health_public(request: Request):
    """Public health check endpoint for RapidAPI consumers."""
    health_status = {"status": "healthy", "timestamp": time.time(), "services": {}}

    embed_client = getattr(request.app.state, "embed_client", None)
    qdrant_client = getattr(request.app.state, "qdrant_client", None)

    # Check embedding service
    if embed_client:
        try:
            embed_client.models.list()
            health_status["services"]["embedding"] = "healthy"
        except Exception:
            health_status["services"]["embedding"] = "unhealthy"
    else:
        health_status["services"]["embedding"] = "unavailable"

    # Check Qdrant service
    if qdrant_client:
        try:
            await qdrant_client.get_collections()
            health_status["services"]["database"] = "healthy"
        except Exception:
            health_status["services"]["database"] = "unhealthy"
    else:
        health_status["services"]["database"] = "unavailable"

    # Overall health
    all_healthy = all(v == "healthy" for v in health_status["services"].values())
    status_code = 200 if all_healthy else 503

    return JSONResponse(content=health_status, status_code=status_code)


@router.get("/debug-headers")
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
