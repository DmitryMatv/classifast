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


# ===== Input Sanitization =====


def sanitize_query_text(query: str, for_search: bool = False) -> str:
    """
    Sanitize query text to prevent malicious input.

    Args:
        query: Raw query string
        for_search: If True, returns simplified sanitization for exact text search

    Returns:
        Sanitized query string

    Raises:
        HTTPException: If query contains invalid content
    """
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    query = query.strip().rstrip("/")

    if for_search:
        # Minimal sanitization for exact ID matching - preserve most characters
        query = re.sub(
            r"[^\w\s\-\.\,\:\;\(\)\{\}\[\]\/\'\"\&\%\#\+\=\!\@]+", " ", query
        )
        return re.sub(r"\s+", " ", query).strip()

    # Full validation for user queries
    if len(query) > 4000:
        raise HTTPException(
            status_code=400, detail="Query too long (max 4000 characters)"
        )

    if len(query) < 2:
        raise HTTPException(
            status_code=400, detail="Query too short (min 2 characters)"
        )

    # Normalize whitespace
    query = re.sub(r"\s+", " ", query)

    # Basic character validation - allow Unicode letters, numbers, spaces, basic punctuation
    allowed_pattern = r"^[\w\s\-\.\,\:\;\(\)\[\]\{\}\/\\\&\@\#\%\+\=\*\?\!\~\`\'\"\<\>\u00A0-\uFFFF]+$"
    if not re.match(allowed_pattern, query):
        raise HTTPException(
            status_code=400,
            detail="Query contains invalid characters. Please use standard text characters only.",
        )

    return query.strip()


def normalize_for_partial_match(query: str) -> str:
    """
    Normalize query for partial matching by removing dots, spaces, dashes,
    and both leading/trailing zeros.

    Args:
        query: Sanitized query string

    Returns:
        Normalized query string for partial matching
    """
    # Remove dots, spaces, and dashes
    normalized = query.replace(".", "").replace(" ", "")  # .replace("-", "")

    # Strip leading and trailing zeros
    normalized = normalized.lstrip("0").rstrip("0")

    # If empty after stripping, return original (handles case of "000")
    if not normalized:
        normalized = query.replace(".", "").replace(" ", "")  # .replace("-", "")

    return normalized


# ===== Embedding Generation =====


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=4, min=4, max=10))
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
        if response.embeddings is None or len(response.embeddings) != 1:
            raise RuntimeError(
                f"Expected 1 embedding, got {len(response.embeddings) if response.embeddings else 'None'}"
            )

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
        internal_top_k = 50 if has_quantization else top_k

        search_params = models.SearchParams(
            hnsw_ef=256,  # Default is 128, higher ef improves recall
            exact=False,
        )

        # For quantized collections, add quantization search params
        if has_quantization:
            search_params.quantization = models.QuantizationSearchParams(
                ignore=False,
                rescore=True,
                oversampling=2.0,
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


async def perform_partial_id_search(
    qdrant_client: AsyncQdrantClient,
    collection_name: str,
    normalized_query: str,
) -> List[Dict[str, Any]]:
    """
    Perform partial match search on original_id field.
    Only called when exact ID match returns no results.

    Args:
        qdrant_client: The Qdrant client instance
        collection_name: The name of the Qdrant collection
        normalized_query: Normalized query (dots, spaces, dashes, leading/trailing zeros removed)

    Returns:
        List of partial match results with score=0.95
    """
    try:
        partial_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="original_id",
                    match=models.MatchText(text=normalized_query),
                )
            ]
        )

        scroll_result = await qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=partial_filter,
            limit=100,
            with_payload=True,
            with_vectors=False,
        )

        partial_results = []
        for point in scroll_result[0]:
            if point.payload:
                original_id_value = point.payload.get("original_id", "")
                # Normalize the stored original_id for comparison
                normalized_original_id = normalize_for_partial_match(original_id_value)
                # Check if normalized original_id contains the normalized query
                if normalized_query in normalized_original_id:  # .lower() ?
                    partial_results.append(
                        {"score": 0.95, "payload": point.payload, "id": point.id}
                    )

        return partial_results

    except Exception as e:
        logger.warning("Partial ID search failed: %s", e)
        return []


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
    Text matches (exact ID) are prioritized (score=1.0), then semantic results are appended.
    If no exact matches, partial ID matching is attempted with lower score (0.95).
    Duplicates are removed.

    Args:
        qdrant_client: The Qdrant client instance
        collection_name: The name of the Qdrant collection
        query_text: Original query text (sanitized for general use)
        query_embedding: Query embedding vector
        top_k: Maximum number of results to return
        has_quantization: Whether collection has quantization enabled

    Returns:
        Merged list of search results
    """
    start_time = time.time()

    try:
        # 1. Exact ID match via text search (fast, synchronous with semantic)
        safe_query = sanitize_query_text(query_text, for_search=True)
        id_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="original_id",
                    match=models.MatchValue(value=safe_query),
                )
            ]
        )

        text_search_task = asyncio.create_task(
            qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=id_filter,
                limit=3,  # Max 3 exact ID matches
                with_payload=True,
                with_vectors=False,
            )
        )

        semantic_search_task = asyncio.create_task(
            perform_semantic_search(
                qdrant_client, collection_name, query_embedding, top_k, has_quantization
            )
        )

        # Wait for both with exception handling
        text_result, semantic_result = await asyncio.gather(
            text_search_task, semantic_search_task, return_exceptions=True
        )

        # Process text results (exact ID matches get score=1.0)
        text_results: List[Dict[str, Any]] = []
        if isinstance(text_result, Exception):
            logger.warning(f"Text search failed: {text_result}")
        elif isinstance(text_result, tuple):
            points: List[Any] = text_result[0]
            text_results = [
                {"score": 1.0, "payload": point.payload, "id": point.id}
                for point in points
            ]

        # 2. Partial ID match only if no exact matches found
        partial_results: List[Dict[str, Any]] = []
        if not text_results:
            normalized_query = normalize_for_partial_match(safe_query)
            if len(normalized_query) >= 3:
                partial_results = await perform_partial_id_search(
                    qdrant_client, collection_name, normalized_query
                )

        # Process semantic results
        semantic_results: List[Dict[str, Any]] = []
        if isinstance(semantic_result, Exception):
            logger.error(f"Semantic search failed: {semantic_result}")
            raise HTTPException(
                status_code=500, detail="Semantic search failed. Please try again."
            )
        elif isinstance(semantic_result, list):
            semantic_results = semantic_result

        # Deduplicate - text and partial results have priority
        seen_ids = set()
        for r in text_results + partial_results:
            if r.get("id") is not None:
                seen_ids.add(r.get("id"))

        # Filter semantic results, excluding duplicates
        filtered_semantic = [
            {"score": r["score"], "payload": r["payload"]}
            for r in semantic_results
            if r.get("id") not in seen_ids
        ]

        # Merge: exact matches first, then partial, then semantic results
        merged_results = (
            [{"score": r["score"], "payload": r["payload"]} for r in text_results]
            + [{"score": r["score"], "payload": r["payload"]} for r in partial_results]
            + filtered_semantic
        )

        query_duration = time.time() - start_time
        logger.debug(
            "Hybrid search completed: %.3fs, exact_id=%d, partial=%d, semantic=%d, merged=%d",
            query_duration,
            len(text_results),
            len(partial_results),
            len(semantic_results),
            len(merged_results[:top_k]),
        )

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


async def perform_classification(
    embed_client: genai.Client,
    qdrant_client: AsyncQdrantClient,
    query: str,
    classifier_type: str,
    version: Optional[str] = None,
    top_k: int = 3,
    quantization_cache: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    """
    Classify a single query using hybrid search (exact text + semantic).

    Args:
        embed_client: The Google GenAI client
        qdrant_client: The Qdrant client
        query: The product/service description to classify
        classifier_type: The classification standard (e.g., 'unspsc', 'etim')
        version: Optional specific version to use
        top_k: Number of results to return
        quantization_cache: Optional cache mapping collection names to quantization status

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
        # Check collection quantization from cache (populated at startup)
        has_quantization = False
        if quantization_cache:
            has_quantization = quantization_cache.get(collection_name, False)

        # Generate embedding for the query
        query_embedding = await get_embedding(
            embed_client=embed_client,
            model_name=embed_model_name,
            text=normalized_query,
            task_type="RETRIEVAL_QUERY",
            embed_dims=config.get("embed_dims"),
        )

        # Perform hybrid search (exact text + semantic)
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
