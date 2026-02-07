import logging
import re
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import tenacity
from fastapi import HTTPException
from google import genai
from google.genai import types
from qdrant_client import QdrantClient, models

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


@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
    retry=tenacity.retry_if_exception_type((ConnectionError, TimeoutError)),
    reraise=True,
)
def get_embedding(
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
        response = embed_client.models.embed_content(
            model=model_name,
            contents=text,
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
        # Re-raise retryable exceptions for tenacity to handle
        if isinstance(e, (ConnectionError, TimeoutError)):
            raise
        raise HTTPException(
            status_code=500, detail="Failed to generate embedding for classification"
        )


@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
    retry=tenacity.retry_if_exception_type((ConnectionError, TimeoutError)),
    reraise=True,
)
def get_embeddings_batch(
    embed_client: genai.Client,
    model_name: str,
    texts: Union[str, List[str]],
    task_type: str = "RETRIEVAL_QUERY",
    embed_dims: Optional[int] = None,
) -> List[List[float]]:
    """
    Generate embeddings for single or multiple texts using Google GenAI.

    Args:
        embed_client: The Google GenAI client
        model_name: The embedding model name
        texts: Single text string or list of text strings to embed
        task_type: Task type for embedding (RETRIEVAL_QUERY, RETRIEVAL_DOCUMENT, etc.)
        embed_dims: Expected embedding dimensions

    Returns:
        List of embedding vectors, where each vector is a list of floats.
        For single text input, returns a list with one embedding.
        For multiple texts, returns embeddings in the same order as input.

    Raises:
        HTTPException: If embedding generation fails
    """
    start_time = time.time()

    # Normalize input to list
    is_single = isinstance(texts, str)
    contents = [texts] if is_single else list(texts)

    if not contents:
        raise HTTPException(status_code=400, detail="No texts provided for embedding")

    try:
        logger.debug(
            "Generating embeddings batch: model=%s, task_type=%s, dims=%s, count=%d",
            model_name,
            task_type,
            embed_dims,
            len(contents),
        )

        config = types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=embed_dims,
        )

        api_start = time.time()
        response = embed_client.models.embed_content(
            model=model_name,
            contents=contents,
            config=config,
        )
        api_duration = time.time() - api_start
        logger.debug(
            "Gemini API embedding batch call: %.3fs for %d texts",
            api_duration,
            len(contents),
        )

        # Validate response
        if response.embeddings is None:
            raise RuntimeError("No embeddings returned from API")

        if len(response.embeddings) != len(contents):
            raise RuntimeError(
                f"Expected {len(contents)} embeddings, got {len(response.embeddings)}"
            )

        embeddings = []
        for i, embedding in enumerate(response.embeddings):
            if not embedding.values:
                raise RuntimeError(f"Empty embedding generated for text at index {i}")

            if embed_dims and len(embedding.values) != embed_dims:
                raise RuntimeError(
                    f"Embedding dimension mismatch at index {i}: expected {embed_dims}, got {len(embedding.values)}"
                )

            embeddings.append(embedding.values)

        return embeddings

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(
            "Batch embedding generation failed: %s (%.3fs elapsed)", e, elapsed
        )
        # Re-raise retryable exceptions for tenacity to handle
        if isinstance(e, (ConnectionError, TimeoutError)):
            raise
        raise HTTPException(
            status_code=500, detail=f"Failed to generate embeddings batch: {str(e)}"
        )


# ===== Search Functions =====


def perform_semantic_search(
    qdrant_client: QdrantClient,
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
        search_result = qdrant_client.query_points(
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


def perform_semantic_search_batch(
    qdrant_client: QdrantClient,
    collection_name: str,
    query_embeddings: List[float] | List[List[float]],
    top_k: int = 10,
    has_quantization: bool = False,
) -> List[List[Dict[str, Any]]]:
    """
    Perform semantic search for single or multiple query embeddings.

    Args:
        qdrant_client: The Qdrant client instance
        collection_name: The name of the Qdrant collection
        query_embeddings: Single embedding vector or list of embedding vectors
        top_k: Maximum number of results to return per query
        has_quantization: Whether collection has quantization enabled

    Returns:
        List of search results lists. Each inner list contains search results
        for the corresponding query embedding.
        For single embedding input, returns a list with one results list.

    Raises:
        HTTPException: If search fails
    """
    start_time = time.time()

    try:
        # Normalize input to list of embeddings
        # Check if it's a single embedding (list of floats) or multiple (list of lists)
        if not query_embeddings:
            raise HTTPException(status_code=400, detail="No query embeddings provided")

        # Determine if it's a single embedding by checking if first element is a number
        # Handles list[float] and numpy arrays by checking if first element is scalar
        first_elem = query_embeddings[0]
        is_single = isinstance(first_elem, (int, float, np.floating, np.integer))

        if is_single:
            # Single embedding case: wrap in list
            single_emb: List[float] = query_embeddings  # type: ignore
            embeddings_list = [single_emb]
        else:
            # Multiple embeddings case: convert each to list
            embeddings_list = [list(emb) for emb in query_embeddings]  # type: ignore

        logger.debug(
            "Performing semantic search batch: collection=%s, queries=%d, top_k=%d",
            collection_name,
            len(embeddings_list),
            top_k,
        )

        # For single query, use regular search for simplicity
        if len(embeddings_list) == 1:
            results = perform_semantic_search(
                qdrant_client=qdrant_client,
                collection_name=collection_name,
                query_embedding=embeddings_list[0],
                top_k=top_k,
                has_quantization=has_quantization,
            )
            return [results]

        # Validate uniform embedding dimensions for batch search
        if len(embeddings_list) > 1:
            expected_dim = len(embeddings_list[0])
            for i, emb in enumerate(embeddings_list[1:], 1):
                if len(emb) != expected_dim:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Inconsistent embedding dimensions: query 0 has {expected_dim} dimensions, query {i} has {len(emb)} dimensions",
                    )

        # For multiple queries, use batch search
        internal_top_k = 50 if has_quantization else top_k

        search_params = models.SearchParams(
            hnsw_ef=256,
            exact=False,
        )

        if has_quantization:
            search_params.quantization = models.QuantizationSearchParams(
                ignore=False,
                rescore=True,
                oversampling=2.0,
            )

        # Build batch query requests
        requests = [
            models.QueryRequest(
                query=embedding,
                limit=internal_top_k,
                with_payload=True,
                with_vector=False,
                params=search_params,
            )
            for embedding in embeddings_list
        ]

        batch_start = time.time()
        batch_result = qdrant_client.query_batch_points(
            collection_name=collection_name,
            requests=requests,
        )
        batch_duration = time.time() - batch_start

        # Process results - batch_result is a list of lists (one list per query)
        all_results = []
        for i, query_results in enumerate(batch_result):
            query_results_list = [
                {"score": hit.score, "payload": hit.payload, "id": hit.id}
                for hit in query_results.points
            ]
            all_results.append(query_results_list)

        logger.debug(
            "Qdrant semantic search batch: collection=%s, %.3fs for %d queries, avg=%.3fs/query",
            collection_name,
            batch_duration,
            len(embeddings_list),
            batch_duration / len(embeddings_list),
        )

        return all_results

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("Batch semantic search failed: %s (%.3fs elapsed)", e, elapsed)
        raise HTTPException(
            status_code=500, detail=f"Batch semantic search failed: {str(e)}"
        )


def perform_partial_id_search(
    qdrant_client: QdrantClient,
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

        scroll_result = qdrant_client.scroll(
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


def perform_hybrid_search(
    qdrant_client: QdrantClient,
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

        # Execute text search and semantic search sequentially
        text_result = None
        text_results: List[Dict[str, Any]] = []
        try:
            text_result = qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=id_filter,
                limit=3,  # Max 3 exact ID matches
                with_payload=True,
                with_vectors=False,
            )
            if isinstance(text_result, tuple):
                points: List[Any] = text_result[0]
                text_results = [
                    {"score": 1.0, "payload": point.payload, "id": point.id}
                    for point in points
                ]
        except Exception as e:
            logger.warning("Text search failed: %s", e)

        # 2. Partial ID match only if no exact matches found
        partial_results: List[Dict[str, Any]] = []
        if not text_results:
            normalized_query = normalize_for_partial_match(safe_query)
            if len(normalized_query) >= 3:
                partial_results = perform_partial_id_search(
                    qdrant_client, collection_name, normalized_query
                )

        # Execute semantic search
        semantic_results: List[Dict[str, Any]] = []
        try:
            semantic_results = perform_semantic_search(
                qdrant_client, collection_name, query_embedding, top_k, has_quantization
            )
        except Exception as e:
            logger.error("Semantic search failed: %s", e)
            raise HTTPException(
                status_code=500, detail="Semantic search failed. Please try again."
            )

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


def validate_and_prepare_classification(
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


def perform_classification(
    embed_client: genai.Client,
    qdrant_client: QdrantClient,
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
        validation_result = validate_and_prepare_classification(
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
        query_embedding = get_embedding(
            embed_client=embed_client,
            model_name=embed_model_name,
            text=normalized_query,
            task_type="RETRIEVAL_QUERY",
            embed_dims=config.get("embed_dims"),
        )

        # Perform hybrid search (exact text + semantic)
        classification_results = perform_hybrid_search(
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
