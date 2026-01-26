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

    # Normalize internal whitespace (collapse multiple spaces/newlines into single space)
    query = re.sub(r"\s+", " ", query)

    # Basic character validation - allow Unicode letters, numbers, spaces, and basic punctuation
    # This prevents obvious injection attempts while allowing normal product descriptions
    allowed_pattern = r"^[\w\s\-\.\,\:\;\(\)\[\]\{\}\/\\\&\@\#\%\+\=\*\?\!\~\`\'\"\<\>\u00A0-\uFFFF]+$"
    if not re.match(allowed_pattern, query):
        raise HTTPException(
            status_code=400,
            detail="Query contains invalid characters. Please use standard text characters only.",
        )

    return query.strip()


def sanitize_text_search_query(query: str) -> str:
    """
    Additional sanitization for text search queries to prevent injection.

    Args:
        query: Query string for text search

    Returns:
        Sanitized query for text search
    """
    # Preserve more characters for class name matching (alphanumeric + basic punctuation)
    query = re.sub(r"[^\w\s\-\.\,\:\;\(\)\{\}\[\]\/\'\"\&\%\#\+\=\!\@]+", " ", query)
    return re.sub(r"\s+", " ", query).strip()


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
    Perform enhanced text-based search with exact and partial matching.

    Priority order:
    1. Exact ID matches (score: 0.999)
    2. Exact NAME matches (score: 0.980)
    3. Partial ID matches (score: 0.950, requires 3+ chars)
    4. Partial ID matches with trailing zeros stripped (score: 0.900, requires 3+ chars)

    Args:
        qdrant_client: The Qdrant client instance
        collection_name: The name of the Qdrant collection
        query_text: The search query (sanitized)

    Returns:
        List of search results with confidence scores (max 3 results)
    """
    start_time = time.time()

    try:
        safe_query = sanitize_text_search_query(query_text).lower()
        dotless_query = normalize_search_query(safe_query)

        exact_id_results = []
        exact_name_results = []
        partial_ids = set()

        # Stage 1: Exact ID match (highest priority)
        id_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="original_id",
                    match=models.MatchValue(value=safe_query),
                )
            ]
        )
        scroll_result = await qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=id_filter,
            limit=3,  # Limit to 3 exact ID matches in case of duplicates
            with_payload=True,
            with_vectors=False,
        )
        exact_id_results = [
            {"score": 0.999, "payload": point.payload, "id": point.id}
            for point in scroll_result[0]
        ]
        partial_ids.update(
            r.get("id") for r in exact_id_results if r.get("id") is not None
        )

        # Stage 2: Exact NAME match (second priority, case-insensitive)
        # Use MatchText to find candidates, then filter for exact match
        name_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="class_name",
                    match=models.MatchText(text=safe_query),
                )
            ]
        )
        scroll_result = await qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=name_filter,
            limit=30,  # Fetch more to filter down to exact matches
            with_payload=True,
            with_vectors=False,
        )
        # Post-query filter for exact match (case-insensitive)
        exact_name_results = [
            {"score": 0.980, "payload": point.payload, "id": point.id}
            for point in scroll_result[0]
            if point.payload
            and point.payload.get("class_name", "").lower() == safe_query
        ]
        partial_ids.update(r.get("id") for r in exact_name_results if r.get("id"))

        partial_results = []

        # Stage 3: Partial match on dotless_query (dots/dashes removed, zeros preserved)
        # This handles queries like "73.12" -> "7312", which finds IDs starting with or ending with "7312"
        if len(dotless_query) >= 3:
            partial_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="original_id",
                        match=models.MatchText(text=dotless_query),
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
            for point in scroll_result[0]:
                point_id = point.id
                original_id_value = (
                    point.payload.get("original_id", "") if point.payload else ""
                )
                original_id_normalized = normalize_search_query(
                    original_id_value.lower()
                )
                if (
                    point_id
                    and point_id not in partial_ids
                    and (
                        original_id_normalized.startswith(dotless_query)
                        or original_id_normalized.endswith(dotless_query)
                    )
                ):
                    partial_results.append(
                        {"score": 0.950, "payload": point.payload, "id": point_id}
                    )
                    partial_ids.add(point_id)

        # Stage 4: Partial match with trailing zeros stripped
        # Only do this for pure numeric queries with many trailing zeros (e.g., "13110000" -> "1311")
        if dotless_query.isdigit() and len(dotless_query) >= 3:
            stripped_query = strip_trailing_zeros(dotless_query)
            if stripped_query != dotless_query and len(stripped_query) >= 3:
                stripped_partial_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="original_id",
                            match=models.MatchText(text=stripped_query),
                        )
                    ]
                )
                scroll_result = await qdrant_client.scroll(
                    collection_name=collection_name,
                    scroll_filter=stripped_partial_filter,
                    limit=100,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in scroll_result[0]:
                    point_id = point.id
                    original_id_value = (
                        point.payload.get("original_id", "") if point.payload else ""
                    )
                    original_id_normalized = normalize_search_query(
                        original_id_value.lower()
                    )
                    if (
                        point_id
                        and point_id not in partial_ids
                        and (
                            original_id_normalized.startswith(stripped_query)
                            or original_id_normalized.endswith(stripped_query)
                        )
                    ):
                        partial_results.append(
                            {"score": 0.900, "payload": point.payload, "id": point_id}
                        )
                        partial_ids.add(point_id)

        query_duration = time.time() - start_time
        logger.debug(
            "Qdrant enhanced text search: %.3fs, collection=%s, exact_id=%d, exact_name=%d, partial=%d",
            query_duration,
            collection_name,
            len(exact_id_results),
            len(exact_name_results),
            len(partial_results),
        )

        # Merge results in priority order and deduplicate
        seen_ids = set()
        merged_results = []

        for result in exact_id_results + exact_name_results + partial_results:
            result_id = result.get("id")
            if result_id is not None and result_id not in seen_ids:
                seen_ids.add(result_id)
                merged_results.append(result)

        return merged_results

    except Exception as e:
        elapsed = time.time() - start_time
        logger.warning("Enhanced text search failed: %s (%.3fs elapsed)", e, elapsed)
        return []


def normalize_search_query(query: str) -> str:
    """
    Normalize query for text search by removing dots and dashes.
    Does NOT strip trailing zeros.

    Args:
        query: Raw query string

    Returns:
        Normalized query string (dots and dashes removed)
    """
    # Remove dots and dashes only, preserve zeros
    normalized = query.replace(".", "").replace("-", "")
    return normalized


def strip_trailing_zeros(query: str) -> str:
    """
    Strip trailing zeros from a numeric query.
    Used for handling queries with extra trailing zeros (e.g., "13110000" -> "1311").

    Args:
        query: Numeric query string

    Returns:
        Query string with trailing zeros removed
    """
    stripped = query.rstrip("0")
    if not stripped:
        stripped = "0"
    return stripped


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
            hnsw_ef=128,
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
