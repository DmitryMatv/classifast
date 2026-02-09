import logging
import re
import time
from typing import Any, Dict, List, Optional

import tenacity
from fastapi import HTTPException
from google import genai
from google.genai import types
from qdrant_client import QdrantClient, models
from zeroentropy import ZeroEntropy

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
                oversampling=3.0,
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


def perform_exact_id_search(
    qdrant_client: QdrantClient,
    collection_name: str,
    query_text: str,
) -> List[Dict[str, Any]]:
    """
    Perform exact ID match search on original_id field.

    Args:
        qdrant_client: The Qdrant client instance
        collection_name: The name of the Qdrant collection
        query_text: Query text to match against original_id

    Returns:
        List of exact match results with score=1.0
    """
    try:
        safe_query = sanitize_query_text(query_text, for_search=True)
        id_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="original_id",
                    match=models.MatchValue(value=safe_query),
                )
            ]
        )

        scroll_result = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=id_filter,
            limit=3,  # Max 3 exact ID matches
            with_payload=True,
            with_vectors=False,
        )

        if isinstance(scroll_result, tuple):
            points = scroll_result[0]
            return [
                {"score": 1.0, "payload": point.payload, "id": point.id}
                for point in points
            ]
        return []

    except Exception as e:
        logger.warning("Exact ID search failed: %s", e)
        return []


def normalize_for_partial_match(query: str) -> str:
    """
    Normalize query for partial matching by removing dots and spaces,
    and stripping leading/trailing zeros.

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


# ===== Cache Control Helpers =====


def get_classification_cache_headers() -> Dict[str, str]:
    """Generate Cloudflare-friendly Cache-Control headers for classification responses.

    Returns:
        Dictionary with Cache-Control and Vary headers
    """
    # Uniform cache policy across all endpoints
    # Browser: 4 hours, Cloudflare CDN: 7 days
    # Cloudflare-CDN-Cache-Control is Cloudflare-specific and not proxied downstream
    return {
        "Cache-Control": "public, max-age=14400",
        "Cloudflare-CDN-Cache-Control": "max-age=604800",
        "Vary": "Accept-Encoding",
    }


def rerank_with_zeroentropy(
    zclient: ZeroEntropy,
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 5,
    rerank_top_n: int = 15,
) -> List[Dict[str, Any]]:
    """Rerank semantic search results using ZeroEntropy rerank API.

    Args:
        zclient: Initialized ZeroEntropy client
        query: The search query text
        candidates: List of candidate matches from semantic search
        top_k: Number of top results to return after reranking
        rerank_top_n: Number of candidates to send to ZeroEntropy

    Returns:
        List of reranked candidates with zeroentropy_relevance_score field.
        Falls back to original candidates if API fails.
    """
    if not candidates or not zclient:
        return candidates

    # Limit candidates to rerank
    candidates_to_rerank = candidates[: min(rerank_top_n, len(candidates))]
    remaining_candidates = candidates[len(candidates_to_rerank) :]

    # Prepare documents for ZeroEntropy - use class_name and definition
    documents = []
    for candidate in candidates_to_rerank:
        payload = candidate.get("payload", {})
        class_name = payload.get("class_name", "")
        definition = payload.get("definition", "")

        # Build document text with class_name and definition if available
        if definition:
            doc = f"{class_name}\n\nDefinition: {definition}"
        else:
            doc = class_name
        documents.append(doc)

    try:
        logger.info(
            "RERANK: ZeroEntropy reranking %d candidates for query='%s'",
            len(documents),
            query[:50],
        )

        # Call ZeroEntropy rerank API
        response = zclient.models.rerank(
            model="zerank-2",
            query=query,
            documents=documents,
        )

        # Map rerank results back to original candidates
        reranked_candidates = []
        seen_indices = set()

        # Process results from ZeroEntropy
        for result in response.results:
            original_index = result.index
            seen_indices.add(original_index)
            original_candidate = candidates_to_rerank[original_index].copy()

            # Add ZeroEntropy relevance score (normalize to 0-100 scale)
            relevance_score = result.relevance_score * 100
            original_candidate["zeroentropy_relevance_score"] = round(
                relevance_score, 4
            )

            reranked_candidates.append(original_candidate)

        # Add any candidates that weren't returned with 0 score
        for i, candidate in enumerate(candidates_to_rerank):
            if i not in seen_indices:
                candidate_copy = candidate.copy()
                candidate_copy["zeroentropy_relevance_score"] = 0.0
                reranked_candidates.append(candidate_copy)

        if reranked_candidates:
            logger.info(
                "RERANK_COMPLETE: Top result=%s score=%.2f (reranked %d docs)",
                reranked_candidates[0].get("payload", {}).get("original_id", "N/A"),
                reranked_candidates[0].get("zeroentropy_relevance_score", 0),
                len(documents),
            )

    except Exception as e:
        logger.warning(
            "RERANK_FAILED: ZeroEntropy reranking failed: %s, using semantic search scores",
            e,
        )
        # Return original candidates with zeroentropy_relevance_score = 0
        reranked_candidates = []
        for c in candidates_to_rerank:
            c_copy = c.copy()
            c_copy["zeroentropy_relevance_score"] = 0.0
            reranked_candidates.append(c_copy)

    # Add remaining candidates (not reranked) with 0 score
    for c in remaining_candidates:
        c_copy = c.copy()
        c_copy["zeroentropy_relevance_score"] = 0.0
        reranked_candidates.append(c_copy)

    return reranked_candidates[:top_k]


# ===== Classification Function =====


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
    zclient: Optional[ZeroEntropy] = None,
) -> Dict[str, Any]:
    """
    Classify a single query using hybrid search (exact text + semantic) with optional ZeroEntropy reranking.

    Args:
        embed_client: The Google GenAI client
        qdrant_client: The Qdrant client
        query: The product/service description to classify
        classifier_type: The classification standard (e.g., 'unspsc', 'etim')
        version: Optional specific version to use
        top_k: Number of results to return
        quantization_cache: Optional cache mapping collection names to quantization status
        zclient: Optional ZeroEntropy client for reranking results

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

        # Step 1: Perform exact ID search (scroll) - these always stay on top
        scroll_results = perform_exact_id_search(
            qdrant_client=qdrant_client,
            collection_name=collection_name,
            query_text=normalized_query,
        )

        # Step 2: If no exact matches, try partial ID search
        if not scroll_results:
            normalized_id_query = normalize_for_partial_match(normalized_query)
            if len(normalized_id_query) >= 3:
                scroll_results = perform_partial_id_search(
                    qdrant_client=qdrant_client,
                    collection_name=collection_name,
                    normalized_query=normalized_id_query,
                )

        # Step 3: Generate embedding for semantic search
        query_embedding = get_embedding(
            embed_client=embed_client,
            model_name=embed_model_name,
            text=normalized_query,
            task_type="RETRIEVAL_QUERY",
            embed_dims=config.get("embed_dims"),
        )

        # Step 4: Perform semantic search with more candidates if reranking
        semantic_top_k = 50 if zclient else top_k
        semantic_results = perform_semantic_search(
            qdrant_client=qdrant_client,
            collection_name=collection_name,
            query_embedding=query_embedding,
            top_k=semantic_top_k,
            has_quantization=has_quantization,
        )

        # Step 5: Get IDs from scroll results to deduplicate semantic results
        scroll_ids = {r.get("id") for r in scroll_results if r.get("id") is not None}

        # Step 6: Filter out semantic results that match scroll results
        filtered_semantic = [
            r for r in semantic_results if r.get("id") not in scroll_ids
        ]

        # Step 7: Rerank only semantic results if ZeroEntropy is available
        if zclient and filtered_semantic:
            logger.info(
                "RERANK_STATUS: Using ZeroEntropy for %d semantic candidates",
                len(filtered_semantic),
            )
            reranked_semantic = rerank_with_zeroentropy(
                zclient=zclient,
                query=normalized_query,
                candidates=filtered_semantic,
                top_k=top_k,  # Will be limited later with scroll results
                rerank_top_n=30,
            )
        else:
            # No ZeroEntropy, just limit results and add zero score
            if not zclient:
                logger.info(
                    "RERANK_STATUS: ZeroEntropy not available, skipping reranking"
                )
            elif not filtered_semantic:
                logger.debug("RERANK_STATUS: No semantic candidates to rerank")
            reranked_semantic = []
            for result in filtered_semantic[:top_k]:
                result_copy = result.copy()
                result_copy["zeroentropy_relevance_score"] = 0.0
                reranked_semantic.append(result_copy)

        # Step 8: Merge results - scroll first, then reranked semantic
        # For reranked semantic results, update score to use normalized ZeroEntropy score
        # This ensures proper sorting - ZeroEntropy scores are 0-100, normalize to 0-1
        for result in reranked_semantic:
            ze_score = result.get("zeroentropy_relevance_score", 0)
            if ze_score > 0:
                result["score"] = ze_score / 100.0
            # else: keep original semantic search score

        classification_results = scroll_results + reranked_semantic

        # Step 9: Sort all results by score descending (top scores first)
        classification_results.sort(key=lambda x: x.get("score", 0), reverse=True)

        # Step 10: Limit to top_k (scroll + semantic count toward limit)
        classification_results = classification_results[:top_k]

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
