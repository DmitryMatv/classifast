import asyncio
import logging
import re
import time
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from google import genai
from google.genai import types
from qdrant_client import AsyncQdrantClient, models
from tenacity import retry, stop_after_attempt, wait_exponential

from .classifier_config import CLASSIFIER_CONFIG

logger = logging.getLogger(__name__)


# ===== Input Sanitization & Validation =====


def sanitize_query_text(query: str) -> str:
    """
    Sanitize query text to prevent malicious input.

    Args:
        query: Raw query string

    Returns:
        Sanitized query string

    Raises:
        HTTPException: If query contains invalid content
    """
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Strip and normalize
    query = query.strip()

    # Length validation
    if len(query) > 4000:
        raise HTTPException(
            status_code=400, detail="Query too long (max 4000 characters)"
        )

    if len(query) < 2:
        raise HTTPException(
            status_code=400, detail="Query too short (min 2 characters)"
        )

    # Remove trailing slashes
    query = query.rstrip("/")

    # Basic character validation - allow Unicode letters, numbers, spaces, and basic punctuation
    # This prevents obvious injection attempts while allowing normal product descriptions
    allowed_pattern = r"^[\w\s\-\.\,\:\;\(\)\[\]\{\}\/\\\&\@\#\%\+\=\*\?\!\~\`\'\"\<\>\u00A0-\uFFFF]+$"
    if not re.match(allowed_pattern, query):
        raise HTTPException(
            status_code=400,
            detail="Query contains invalid characters. Please use standard text characters only.",
        )

    return query


def sanitize_text_search_query(query: str) -> str:
    """
    Additional sanitization for text search queries to prevent injection.

    Args:
        query: Query string for text search

    Returns:
        Sanitized query for text search
    """
    # For text search, be more restrictive - mainly alphanumeric and basic symbols
    query = re.sub(r"[^\w\s\-\.\,\:\;\/\\\(\)\[\]]+", " ", query)
    return re.sub(r"\s+", " ", query).strip()


# ===== Embedding Generation =====


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def get_embedding(
    embed_client: genai.Client,
    model_name: str,
    text: str,
    task_type: str = "RETRIEVAL_QUERY",
    embed_dims: Optional[int] = None,
) -> List[float]:
    """
    Generate a single embedding for text using Google GenAI.

    Args:
        embed_client: The Google GenAI client
        model_name: The embedding model name
        text: Text to embed
        task_type: Task type for embedding
        embed_dims: Expected embedding dimensions

    Returns:
        Embedding vector as list of floats

    Raises:
        HTTPException: If embedding generation fails
    """
    start_time = time.time()

    try:
        logger.debug(
            "Generating embedding: model=%s, task_type=%s, dims=%s",
            model_name,
            task_type,
            embed_dims,
        )

        config = types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=embed_dims,
        )

        api_start = time.time()
        response = await embed_client.aio.models.embed_content(
            model=model_name,
            contents=[text],
            config=config,
        )
        api_duration = time.time() - api_start
        logger.debug("Gemini API embedding call: %.3fs", api_duration)

        # Validate response
        if len(response.embeddings) != 1:
            raise RuntimeError(f"Expected 1 embedding, got {len(response.embeddings)}")

        embedding_vector = response.embeddings[0].values

        if not embedding_vector:
            raise RuntimeError("Empty embedding generated")

        if embed_dims and len(embedding_vector) != embed_dims:
            raise RuntimeError(
                f"Embedding dimension mismatch: expected {embed_dims}, got {len(embedding_vector)}"
            )

        return embedding_vector

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("Embedding generation failed: %s (%.3fs elapsed)", e, elapsed)
        raise HTTPException(
            status_code=500, detail="Failed to generate embedding for classification"
        )


# ===== Search Functions =====


async def perform_text_search(
    qdrant_client: AsyncQdrantClient,
    collection_name: str,
    query_text: str,
) -> List[Dict[str, Any]]:
    """
    Perform exact text-based search using MatchValue condition.
    Searches original_id and class_name fields for exact full matches only.

    Args:
        qdrant_client: The Qdrant client instance
        collection_name: The name of the Qdrant collection
        query_text: The search query (sanitized)

    Returns:
        List of search results with 99.9% confidence scores (max 1 result)
    """
    start_time = time.time()

    try:
        # Convert to lowercase for case-insensitive exact matching
        safe_query = sanitize_text_search_query(query_text).lower()

        # Use MatchValue for exact string matching (not token/partial matching)
        # Check if either original_id OR class_name exactly matches the query
        filter_condition = models.Filter(
            should=[
                models.FieldCondition(
                    key="original_id",
                    match=models.MatchValue(value=safe_query),
                ),
                models.FieldCondition(
                    key="class_name",
                    match=models.MatchValue(value=safe_query),
                ),
            ]
        )

        query_start = time.time()
        # Always limit to 1 result for exact matching
        scroll_result = await qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=filter_condition,
            limit=1,  # Force limit to 1 for exact matching
            with_payload=True,
            with_vectors=False,
        )

        points, _ = scroll_result

        query_duration = time.time() - query_start
        logger.debug(
            "Qdrant exact text search: %.3fs, collection=%s, found=%d results",
            query_duration,
            collection_name,
            len(points),
        )

        return [
            {"score": 0.999, "payload": point.payload, "id": point.id}
            for point in points
        ]

    except Exception as e:
        elapsed = time.time() - start_time
        logger.warning("Exact text search failed: %s (%.3fs elapsed)", e, elapsed)
        return []


async def perform_semantic_search(
    qdrant_client: AsyncQdrantClient,
    collection_name: str,
    query_embedding: List[float],
    top_k: int = 10,
    has_quantization: bool = False,
) -> List[Dict[str, Any]]:
    """
    Perform semantic search using embedding vector.

    Args:
        qdrant_client: The Qdrant client instance
        collection_name: The name of the Qdrant collection
        query_embedding: Query embedding vector
        top_k: Maximum number of results to return
        has_quantization: Whether collection has quantization enabled

    Returns:
        List of semantic search results with confidence scores
    """
    start_time = time.time()

    try:
        # Prepare search parameters
        internal_top_k = 100 if has_quantization else top_k

        search_params = models.SearchParams(
            hnsw_ef=256,
            exact=False,
        )

        # For quantized collections, add quantization search params
        if has_quantization:
            search_params.quantization = models.QuantizationSearchParams(
                ignore=False,
                rescore=True,
                oversampling=3.0,
            )

        query_start = time.time()
        search_result = await qdrant_client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            query_filter=None,
            limit=internal_top_k,
            with_payload=True,
            with_vectors=False,
            search_params=search_params,
        )
        query_duration = time.time() - query_start
        logger.debug(
            "Qdrant semantic search: %.3fs, collection=%s, top_k=%d, found=%d results",
            query_duration,
            collection_name,
            internal_top_k,
            len(search_result.points),
        )

        return [
            {"score": hit.score, "payload": hit.payload, "id": hit.id}
            for hit in search_result.points
        ]

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("Semantic search failed: %s (%.3fs elapsed)", e, elapsed)
        raise HTTPException(
            status_code=500, detail="Semantic search failed. Please try again."
        )


async def perform_hybrid_search(
    qdrant_client: AsyncQdrantClient,
    collection_name: str,
    query_text: str,
    query_embedding: List[float],
    top_k: int = 10,
    has_quantization: bool = False,
) -> List[Dict[str, Any]]:
    """
    Perform hybrid search combining exact text matches and semantic search.
    Text matches are prioritized (score=1.0), then semantic results are appended.
    Duplicates are removed.

    Args:
        qdrant_client: The Qdrant client instance
        collection_name: The name of the Qdrant collection
        query_text: Original query text
        query_embedding: Query embedding vector
        top_k: Maximum number of results to return
        has_quantization: Whether collection has quantization enabled

    Returns:
        Merged list of search results
    """
    try:
        # Run text and semantic searches in parallel with proper error handling
        text_search_task = asyncio.create_task(
            perform_text_search(qdrant_client, collection_name, query_text)
        )
        semantic_search_task = asyncio.create_task(
            perform_semantic_search(
                qdrant_client, collection_name, query_embedding, top_k, has_quantization
            )
        )

        # Wait for both with exception handling
        results = await asyncio.gather(
            text_search_task, semantic_search_task, return_exceptions=True
        )

        text_results = results[0] if not isinstance(results[0], Exception) else []
        semantic_results = results[1] if not isinstance(results[1], Exception) else []

        # Log any search failures
        if isinstance(results[0], Exception):
            logger.warning(f"Text search failed: {results[0]}")
        if isinstance(results[1], Exception):
            logger.error(f"Semantic search failed: {results[1]}")
            raise HTTPException(
                status_code=500, detail="Semantic search failed. Please try again."
            )

        # Deduplicate - text results have priority
        seen_ids = {r.get("id") for r in text_results if r.get("id") is not None}

        # Filter semantic results, excluding duplicates
        filtered_semantic = [
            {"score": r["score"], "payload": r["payload"]}
            for r in semantic_results
            if r.get("id") not in seen_ids
        ]

        # Merge: text matches first, then semantic results
        merged_results = [
            {"score": r["score"], "payload": r["payload"]} for r in text_results
        ] + filtered_semantic

        return merged_results[:top_k]

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Hybrid search failed: %s", e)
        raise HTTPException(status_code=500, detail="Search failed. Please try again.")


# ===== Classification Functions =====


async def validate_and_prepare_classification(
    classifier_type: str,
    version: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Validate classifier type and version, return configuration.

    Args:
        classifier_type: The classification standard (e.g., 'unspsc', 'etim')
        version: Optional specific version to use

    Returns:
        Dictionary with configuration details

    Raises:
        HTTPException: If classifier or version is invalid
    """
    config = CLASSIFIER_CONFIG.get(classifier_type.upper())
    if not config:
        raise HTTPException(
            status_code=404, detail=f"Classifier '{classifier_type}' not found"
        )

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

    return {
        "config": config,
        "version_name": version_name,
        "version_config": version_config,
        "collection_name": version_config["collection_name"],
        "embed_model_name": config["embed_model_name"],
    }


async def check_collection_quantization(
    qdrant_client: AsyncQdrantClient,
    collection_name: str,
) -> bool:
    """
    Check if a collection has quantization enabled.

    Args:
        qdrant_client: The Qdrant client instance
        collection_name: The name of the collection to check

    Returns:
        True if quantization is enabled, False otherwise
    """
    try:
        collection_info = await qdrant_client.get_collection(collection_name)
        return collection_info.config.quantization_config is not None
    except Exception:
        return False


async def perform_classification(
    embed_client: genai.Client,
    qdrant_client: AsyncQdrantClient,
    query: str,
    classifier_type: str,
    version: Optional[str] = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Classify a single query using hybrid search (text + semantic).

    Args:
        embed_client: The Google GenAI client
        qdrant_client: The Qdrant client
        query: The product/service description to classify
        classifier_type: The classification standard (e.g., 'unspsc', 'etim')
        version: Optional specific version to use
        top_k: Number of results to return

    Returns:
        Dict containing classification results and metadata
    """
    # Validate and prepare configuration
    try:
        validation_result = await validate_and_prepare_classification(
            classifier_type, version
        )
        config = validation_result["config"]
        version_name = validation_result["version_name"]
        version_config = validation_result["version_config"]
        collection_name = validation_result["collection_name"]
        embed_model_name = validation_result["embed_model_name"]
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Validation failed for '%s': %s", classifier_type, e)
        raise HTTPException(status_code=500, detail="Invalid classifier configuration")

    # Validate clients
    if not embed_client or not qdrant_client:
        raise HTTPException(
            status_code=503,
            detail="Backend services not available. Please check server logs.",
        )

    # Sanitize and validate query
    try:
        normalized_query = sanitize_query_text(query)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Query validation failed: %s", e)
        raise HTTPException(status_code=400, detail="Invalid query format")

    logger.info(
        "CLASSIFICATION_QUERY: classifier=%s query='%s'",
        classifier_type,
        normalized_query,
    )

    try:
        # Check collection quantization
        has_quantization = await check_collection_quantization(
            qdrant_client, collection_name
        )

        # Generate embedding for the query
        query_embedding = await get_embedding(
            embed_client=embed_client,
            model_name=embed_model_name,
            text=normalized_query,
            task_type="RETRIEVAL_QUERY",
            embed_dims=config.get("embed_dims"),
        )

        # Perform hybrid search (text + semantic)
        classification_results = await perform_hybrid_search(
            qdrant_client=qdrant_client,
            collection_name=collection_name,
            query_text=normalized_query,
            query_embedding=query_embedding,
            top_k=top_k,
            has_quantization=has_quantization,
        )

        return {
            "results": classification_results,
            "collection_name": collection_name,
            "version_name": version_name,
            "version_config": version_config,
            "config": config,
            "query": normalized_query,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Classification error for '%s': %s", classifier_type, e)
        raise HTTPException(status_code=500, detail="Error processing request")
