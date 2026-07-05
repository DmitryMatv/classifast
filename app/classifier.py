import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import httpx
import tenacity
from fastapi import HTTPException
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError, InferenceTimeoutError
from qdrant_client import QdrantClient, models
from zeroentropy import ZeroEntropy

from .cache_profiles import CLASSIFICATION_RESULT, add_vary, build_cache_headers
from .classifier_config import CLASSIFIER_CONFIG
from .id_lookup import (
    ORIGINAL_ID_FIELD,
    ORIGINAL_ID_NORMALIZED_FIELD,
    ORIGINAL_ID_NORMALIZED_REVERSED_FIELD,
    normalize_original_id_for_lookup,
    reverse_normalized_id,
)

logger = logging.getLogger(__name__)

SEARCH_SANITIZE_PATTERN = re.compile(
    r"[^\w\s\-\.\,\:\;\(\)\{\}\[\]\/\'\"\&\%\#\+\=\!\@]+"
)
WHITESPACE_PATTERN = re.compile(r"\s+")
PURE_NUMERIC_CODE_PATTERN = re.compile(r"^[\d\s\.\-]+$")
REPEATED_URL_ENCODING_PATTERN = re.compile(r"(?:25){2,}")
HEX_SEQUENCE_PATTERN = re.compile(r"[0-9A-Fa-f]{4,}")
HEX_LETTER_PATTERN = re.compile(r"[A-Fa-f]")
DIGIT_PATTERN = re.compile(r"\d")
ALLOWED_QUERY_PATTERN = re.compile(
    r"^[\w\s\-\.\,\:\;\(\)\[\]\{\}\/\\\&\@\#\%\+\=\*\?\!\~\`\'\"\<\>\u00A0-\uFFFF]+$"
)
TRANSIENT_HF_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
DEFAULT_RERANK_CANDIDATE_LIMIT = 100


# ===== Input Sanitization =====


def _sanitize_search_query(query: str) -> str:
    query = SEARCH_SANITIZE_PATTERN.sub(" ", query)
    return WHITESPACE_PATTERN.sub(" ", query).strip()


def _validate_query_length(query: str) -> None:
    if len(query) > 4000:
        raise HTTPException(
            status_code=400, detail="Query too long (max 4000 characters)"
        )

    if len(query) < 2:
        raise HTTPException(
            status_code=400, detail="Query too short (min 2 characters)"
        )


def _is_pure_numeric_code(query: str) -> bool:
    return bool(PURE_NUMERIC_CODE_PATTERN.match(query))


def _reject_suspicious_encoding(query: str, is_pure_numeric: bool) -> None:
    if REPEATED_URL_ENCODING_PATTERN.search(query):
        raise HTTPException(
            status_code=400,
            detail="Query contains suspicious URL encoding patterns",
        )

    if is_pure_numeric:
        return

    hex_sequences = HEX_SEQUENCE_PATTERN.findall(query)
    if len(hex_sequences) >= 10:
        raise HTTPException(
            status_code=400,
            detail="Query contains suspicious hex encoding patterns",
        )

    hex_letters = len(HEX_LETTER_PATTERN.findall(query))
    has_digits = bool(DIGIT_PATTERN.search(query))
    non_space_chars = len(query.replace(" ", ""))
    if has_digits and non_space_chars > 0 and hex_letters / non_space_chars > 0.7:
        raise HTTPException(
            status_code=400,
            detail="Query appears to be encoded garbage",
        )


def _collapse_query_whitespace(query: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", query)


def _validate_allowed_query_characters(query: str) -> None:
    if not ALLOWED_QUERY_PATTERN.match(query):
        raise HTTPException(
            status_code=400,
            detail="Query contains invalid characters. Please use standard text characters only.",
        )


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
        return _sanitize_search_query(query)

    _validate_query_length(query)
    is_pure_numeric = _is_pure_numeric_code(query)
    _reject_suspicious_encoding(query, is_pure_numeric)
    query = _collapse_query_whitespace(query)
    _validate_allowed_query_characters(query)

    return query.strip()


def _is_transient_hf_error(error: BaseException) -> bool:
    if isinstance(error, (InferenceTimeoutError, httpx.TransportError)):
        return True
    if isinstance(error, HfHubHTTPError):
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
        return status_code in TRANSIENT_HF_STATUS_CODES
    return False


def _normalize_embedding_response(response: Any) -> List[float]:
    if hasattr(response, "tolist"):
        response = response.tolist()

    if not isinstance(response, list):
        raise RuntimeError("Embedding response is not a list")

    if len(response) == 1 and isinstance(response[0], list):
        response = response[0]
    elif response and isinstance(response[0], list):
        raise RuntimeError(
            "Embedding response is token-level; expected a pooled sentence vector"
        )

    if not response:
        raise RuntimeError("Empty embedding generated")

    try:
        return [float(value) for value in response]
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Embedding response contains non-numeric values") from exc


def _embedding_dimension_mismatch(
    embedding_vector: List[float],
    embed_dims: Optional[int],
) -> bool:
    return embed_dims is not None and len(embedding_vector) != embed_dims


def build_query_embedding_text(query: str, instruction: Optional[str]) -> str:
    """Format query-side text for instruction-aware embedding models."""
    if not instruction:
        return query
    return f"Instruct: {instruction.strip()}\nQuery: {query}"


def build_rerank_query_text(query: str, instruction: Optional[str]) -> str:
    """Format reranker query text with ZeroEntropy instruction context."""
    if not instruction:
        return query
    return f"Query: {query}\nInstructions: {instruction.strip()}"


def _build_embedding_retry() -> tenacity.Retrying:
    return tenacity.Retrying(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
        retry=tenacity.retry_if_exception(_is_transient_hf_error),
        reraise=True,
    )


def get_embedding(
    embed_client: InferenceClient,
    model_name: str,
    text: str,
    embed_dims: Optional[int] = None,
) -> List[float]:
    """
    Generate a single embedding for text using Hugging Face Inference.

    Args:
        embed_client: The Hugging Face Inference client
        model_name: The embedding model name
        text: Text to embed
        embed_dims: Expected embedding dimensions

    Returns:
        Embedding vector as list of floats

    Raises:
        HTTPException: If embedding generation fails
    """
    start_time = time.time()

    try:
        logger.debug(
            "Generating embedding: model=%s, dims=%s",
            model_name,
            embed_dims,
        )

        api_start = time.time()
        retry = _build_embedding_retry()
        if embed_dims is None:
            response = retry(
                embed_client.feature_extraction,
                text,
                model=model_name,
            )
        else:
            response = retry(
                embed_client.feature_extraction,
                text,
                model=model_name,
                dimensions=embed_dims,
            )
        api_duration = time.time() - api_start
        logger.debug("Hugging Face embedding call: %.3fs", api_duration)

        embedding_vector = _normalize_embedding_response(response)

        if _embedding_dimension_mismatch(embedding_vector, embed_dims):
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


def _point_result(point: Any, score: float) -> Dict[str, Any]:
    return {"score": score, "payload": point.payload, "id": point.id}


def _scroll_points(scroll_result: Any) -> List[Any]:
    if isinstance(scroll_result, tuple):
        return list(scroll_result[0])
    return []


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
            limit=top_k,
            with_payload=True,
            with_vectors=False,
            search_params=search_params,
        )
        query_duration = time.time() - query_start
        logger.debug(
            "Qdrant semantic search: %.3fs, collection=%s, top_k=%d, found=%d results",
            query_duration,
            collection_name,
            top_k,
            len(search_result.points),
        )

        return [_point_result(hit, hit.score) for hit in search_result.points]

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
                    key=ORIGINAL_ID_FIELD,
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

        return [_point_result(point, 1.0) for point in _scroll_points(scroll_result)]

    except Exception as e:
        logger.warning("Exact ID search failed: %s", e)
        return []


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
        List of partial match results with score=0.90
    """
    try:
        partial_filter = models.Filter(
            should=[
                models.FieldCondition(
                    key=ORIGINAL_ID_NORMALIZED_FIELD,
                    match=models.MatchText(text=normalized_query),
                ),
                models.FieldCondition(
                    key=ORIGINAL_ID_NORMALIZED_REVERSED_FIELD,
                    match=models.MatchText(
                        text=reverse_normalized_id(normalized_query)
                    ),
                ),
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
        seen_ids = set()
        for point in _scroll_points(scroll_result):
            if point.payload:
                if point.id in seen_ids:
                    continue

                original_id_value = point.payload.get(ORIGINAL_ID_FIELD, "")
                raw_normalized_id = point.payload.get(ORIGINAL_ID_NORMALIZED_FIELD)
                normalized_original_id = (
                    str(raw_normalized_id)
                    if raw_normalized_id is not None
                    else normalize_original_id_for_lookup(original_id_value)
                )
                if normalized_original_id.startswith(
                    normalized_query
                ) or normalized_original_id.endswith(normalized_query):
                    partial_results.append(_point_result(point, 0.90))
                    seen_ids.add(point.id)

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
    headers = build_cache_headers(CLASSIFICATION_RESULT)
    add_vary(headers, "Accept-Encoding")
    return headers


def _split_rerank_candidates(
    candidates: List[Dict[str, Any]],
    rerank_top_n: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    candidates_to_rerank = candidates[: min(rerank_top_n, len(candidates))]
    remaining_candidates = candidates[len(candidates_to_rerank) :]
    return candidates_to_rerank, remaining_candidates


def _default_zeroentropy_document(candidate: Dict[str, Any]) -> str:
    payload = candidate.get("payload", {})
    class_name = payload.get("class_name", "")
    definition = payload.get("definition", "")

    if class_name and definition:
        return f"{class_name} - Definition: {definition}"
    if definition:
        return definition
    return class_name or ""


def _build_zeroentropy_documents(
    candidates: List[Dict[str, Any]],
    document_builder: Optional[Callable[[Dict[str, Any]], str]],
) -> List[str]:
    builder = document_builder or _default_zeroentropy_document
    return [builder(candidate) for candidate in candidates]


def _copy_with_zeroentropy_score(
    candidate: Dict[str, Any],
    score: float,
) -> Dict[str, Any]:
    candidate_copy = candidate.copy()
    candidate_copy["zeroentropy_relevance_score"] = score
    return candidate_copy


def _zero_score_candidates(
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return [_copy_with_zeroentropy_score(candidate, 0.0) for candidate in candidates]


def _copy_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [candidate.copy() for candidate in candidates]


def _sort_by_score_desc(
    results: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    return sorted(results, key=lambda x: x.get("score", 0), reverse=True)[:top_k]


def _apply_zeroentropy_response(
    candidates: List[Dict[str, Any]],
    response: Any,
) -> List[Dict[str, Any]]:
    reranked_candidates = []
    seen_indices = set()

    for result in response.results:
        original_index = result.index
        seen_indices.add(original_index)
        reranked_candidates.append(
            _copy_with_zeroentropy_score(
                candidates[original_index],
                round(result.relevance_score, 4),
            )
        )

    missing_candidates = [
        candidate for i, candidate in enumerate(candidates) if i not in seen_indices
    ]
    reranked_candidates.extend(_zero_score_candidates(missing_candidates))
    return reranked_candidates


def _log_rerank_complete(
    reranked_candidates: List[Dict[str, Any]],
    document_count: int,
) -> None:
    if not reranked_candidates:
        return

    logger.info(
        "RERANK_COMPLETE: Top result=%s score=%.2f (reranked %d docs)",
        reranked_candidates[0].get("payload", {}).get("original_id", "N/A"),
        reranked_candidates[0].get("zeroentropy_relevance_score", 0) * 100,
        document_count,
    )


def rerank_with_zeroentropy(
    zclient: ZeroEntropy,
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 5,
    rerank_top_n: int = 15,
    document_builder: Optional[Callable[[Dict[str, Any]], str]] = None,
    query_instruction: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Rerank semantic search results using ZeroEntropy rerank API.

    Args:
        zclient: Initialized ZeroEntropy client
        query: The search query text
        candidates: List of candidate matches from semantic search
        top_k: Number of top results to return after reranking
        rerank_top_n: Number of candidates to send to ZeroEntropy
        document_builder: Optional callback to build document text from a candidate.
            Receives the full candidate dict and returns a string.
            If None, uses class_name + definition from payload.
        query_instruction: Optional reranker instruction to wrap around the query.

    Returns:
        List of reranked candidates with zeroentropy_relevance_score field.
        Falls back to original candidates if API fails.
    """
    if not candidates or not zclient:
        return candidates

    candidates_to_rerank, remaining_candidates = _split_rerank_candidates(
        candidates, rerank_top_n
    )
    documents = _build_zeroentropy_documents(candidates_to_rerank, document_builder)
    rerank_query = build_rerank_query_text(query, query_instruction)

    try:
        logger.info(
            "RERANK: ZeroEntropy reranking %d candidates for query='%s'",
            len(documents),
            query[:50],
        )

        # Call ZeroEntropy rerank API
        response = zclient.models.rerank(
            model="zerank-2",
            query=rerank_query,
            documents=documents,
            top_n=top_k,
        )

        reranked_candidates = _apply_zeroentropy_response(
            candidates_to_rerank, response
        )
        _log_rerank_complete(reranked_candidates, len(documents))

    except Exception as e:
        logger.warning(
            "RERANK_FAILED: ZeroEntropy reranking failed: %s, using semantic search scores",
            e,
        )
        return _copy_candidates(candidates[:top_k])

    reranked_candidates.extend(_zero_score_candidates(remaining_candidates))

    return reranked_candidates[:top_k]


# ===== Classification Function =====


@dataclass(frozen=True)
class _ClassificationContext:
    config: Dict[str, Any]
    version_name: str
    version_config: Dict[str, Any]
    collection_name: str
    embed_model_name: str
    normalized_query: str
    classifier_type: str


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

    versions = config["versions"]
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


def _prepare_classification_context(
    embed_client: InferenceClient,
    qdrant_client: QdrantClient,
    query: str,
    classifier_type: str,
    version: Optional[str],
) -> _ClassificationContext:
    try:
        validation_result = validate_and_prepare_classification(
            classifier_type, version
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Validation failed for '%s': %s", classifier_type, e)
        raise HTTPException(status_code=500, detail="Invalid classifier configuration")

    if not embed_client or not qdrant_client:
        raise HTTPException(
            status_code=503,
            detail="Backend services not available. Please check server logs.",
        )

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

    return _ClassificationContext(
        config=validation_result["config"],
        version_name=validation_result["version_name"],
        version_config=validation_result["version_config"],
        collection_name=validation_result["collection_name"],
        embed_model_name=validation_result["embed_model_name"],
        normalized_query=normalized_query,
        classifier_type=classifier_type,
    )


def _classification_response(
    context: _ClassificationContext,
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "results": results,
        "collection_name": context.collection_name,
        "version_name": context.version_name,
        "version_config": context.version_config,
        "config": context.config,
        "query": context.normalized_query,
    }


def _collection_has_quantization(
    collection_name: str,
    quantization_cache: Optional[Dict[str, bool]],
) -> bool:
    if not quantization_cache:
        return False
    return quantization_cache.get(collection_name, False)


def _timed_exact_id_search(
    qdrant_client: QdrantClient,
    collection_name: str,
    normalized_query: str,
) -> tuple[List[Dict[str, Any]], float]:
    exact_start = time.perf_counter()
    exact_results = perform_exact_id_search(
        qdrant_client=qdrant_client,
        collection_name=collection_name,
        query_text=normalized_query,
    )
    exact_ms = (time.perf_counter() - exact_start) * 1000
    return exact_results, exact_ms


def _timed_partial_id_search(
    qdrant_client: QdrantClient,
    collection_name: str,
    normalized_query: str,
) -> tuple[List[Dict[str, Any]], float]:
    normalized_id_query = normalize_original_id_for_lookup(normalized_query)
    if len(normalized_id_query) < 3:
        return [], 0.0

    partial_start = time.perf_counter()
    partial_results = perform_partial_id_search(
        qdrant_client=qdrant_client,
        collection_name=collection_name,
        normalized_query=normalized_id_query,
    )
    partial_ms = (time.perf_counter() - partial_start) * 1000
    return partial_results, partial_ms


def _prepare_exact_id_shortcut_results(
    exact_results: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    classification_results = _sort_by_score_desc(exact_results, top_k)
    for result in classification_results:
        result["zeroentropy_relevance_score"] = 0.0
    return classification_results


def _run_semantic_classification_search(
    embed_client: InferenceClient,
    qdrant_client: QdrantClient,
    context: _ClassificationContext,
    top_k: int,
    has_quantization: bool,
) -> List[Dict[str, Any]]:
    embedding_text = build_query_embedding_text(
        context.normalized_query,
        context.config.get("query_instruction"),
    )
    query_embedding = get_embedding(
        embed_client=embed_client,
        model_name=context.embed_model_name,
        text=embedding_text,
        embed_dims=context.config.get("embed_dims"),
    )

    return perform_semantic_search(
        qdrant_client=qdrant_client,
        collection_name=context.collection_name,
        query_embedding=query_embedding,
        top_k=top_k,
        has_quantization=has_quantization,
    )


def _exclude_id_match_results(
    semantic_results: List[Dict[str, Any]],
    id_match_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    match_ids = {r.get("id") for r in id_match_results if r.get("id") is not None}
    return [r for r in semantic_results if r.get("id") not in match_ids]


def _semantic_retrieve_limit(top_k: int, reranking_enabled: bool) -> int:
    if not reranking_enabled:
        return top_k
    return max(top_k, DEFAULT_RERANK_CANDIDATE_LIMIT)


def _rank_semantic_results(
    zclient: Optional[ZeroEntropy],
    normalized_query: str,
    filtered_semantic: List[Dict[str, Any]],
    id_match_results: List[Dict[str, Any]],
    top_k: int,
    rerank_top_n: int,
    query_instruction: Optional[str],
) -> List[Dict[str, Any]]:
    if zclient is not None and not id_match_results and filtered_semantic:
        logger.info(
            "RERANK_STATUS: Using ZeroEntropy for %d semantic candidates",
            len(filtered_semantic),
        )
        reranked_semantic = rerank_with_zeroentropy(
            zclient=zclient,
            query=normalized_query,
            candidates=filtered_semantic,
            top_k=top_k,
            rerank_top_n=rerank_top_n,
            query_instruction=query_instruction,
        )
        for result in reranked_semantic:
            if "zeroentropy_relevance_score" in result:
                result["score"] = result["zeroentropy_relevance_score"]
        return reranked_semantic

    if id_match_results:
        logger.info("RERANK_STATUS: Skipped - ID matches present")
    elif not zclient:
        logger.info("RERANK_STATUS: Skipped - ZeroEntropy not available")

    return _zero_score_candidates(filtered_semantic[:top_k])


def _merge_classification_results(
    id_match_results: List[Dict[str, Any]],
    semantic_results: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    return _sort_by_score_desc(id_match_results + semantic_results, top_k)


def perform_classification(
    embed_client: InferenceClient,
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
        embed_client: The Hugging Face Inference client
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
    context = _prepare_classification_context(
        embed_client=embed_client,
        qdrant_client=qdrant_client,
        query=query,
        classifier_type=classifier_type,
        version=version,
    )

    try:
        has_quantization = _collection_has_quantization(
            context.collection_name, quantization_cache
        )
        exact_results, exact_ms = _timed_exact_id_search(
            qdrant_client, context.collection_name, context.normalized_query
        )

        if exact_results:
            logger.info(
                "ID_SEARCH: exact=%d partial=%d exact_ms=%.2f partial_ms=%.2f",
                len(exact_results),
                0,
                exact_ms,
                0.0,
            )
            logger.info(
                "ID_SEARCH_SHORTCUT: classifier=%s query='%s' matches=%d",
                context.classifier_type,
                context.normalized_query,
                len(exact_results),
            )
            return _classification_response(
                context,
                _prepare_exact_id_shortcut_results(exact_results, top_k),
            )

        partial_results, partial_ms = _timed_partial_id_search(
            qdrant_client, context.collection_name, context.normalized_query
        )
        logger.info(
            "ID_SEARCH: exact=%d partial=%d exact_ms=%.2f partial_ms=%.2f",
            0,
            len(partial_results),
            exact_ms,
            partial_ms,
        )

        reranking_enabled = zclient is not None and not partial_results
        semantic_retrieve_limit = _semantic_retrieve_limit(top_k, reranking_enabled)

        logger.info(
            "SEMANTIC_SEARCH: Fetching top %d candidates (reranking=%s, id_matches=%d)",
            semantic_retrieve_limit,
            "enabled" if reranking_enabled else "disabled",
            len(partial_results),
        )
        semantic_results = _run_semantic_classification_search(
            embed_client=embed_client,
            qdrant_client=qdrant_client,
            context=context,
            top_k=semantic_retrieve_limit,
            has_quantization=has_quantization,
        )
        filtered_semantic = _exclude_id_match_results(semantic_results, partial_results)
        ranked_semantic = _rank_semantic_results(
            zclient,
            context.normalized_query,
            filtered_semantic,
            partial_results,
            top_k,
            semantic_retrieve_limit,
            context.config.get("query_instruction"),
        )
        classification_results = _merge_classification_results(
            partial_results, ranked_semantic, top_k
        )

        return _classification_response(context, classification_results)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Classification error for '%s': %s", context.classifier_type, e)
        raise HTTPException(status_code=500, detail="Error processing request")
