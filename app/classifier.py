import logging
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from google import genai
from google.genai import types
from qdrant_client import AsyncQdrantClient, models
from tenacity import retry, stop_after_attempt, wait_exponential

from .classifier_config import CLASSIFIER_CONFIG

logger = logging.getLogger(__name__)


def validate_embedding_correspondence(
    texts: List[str], embeddings: List[List[float]], context: str = ""
) -> bool:
    """
    Validates that each embedding corresponds to its expected input text.

    Args:
        texts: Original input texts
        embeddings: Generated embeddings
        context: Context for error messages (e.g., "query", "document")

    Returns:
        True if validation passes, raises RuntimeError if validation fails
    """
    context_str = f" for {context}" if context else ""

    if len(texts) != len(embeddings):
        raise RuntimeError(
            f"Embedding count mismatch{context_str}: "
            f"expected {len(texts)} embeddings, got {len(embeddings)}"
        )

    if not embeddings:
        raise RuntimeError(f"No embeddings generated{context_str}")

    # Validate each embedding is not empty and has consistent dimensions
    expected_dim = None
    for i, embedding in enumerate(embeddings):
        if not embedding:
            raise RuntimeError(f"Empty embedding{context_str} at index {i}")

        if expected_dim is None:
            expected_dim = len(embedding)
        elif len(embedding) != expected_dim:
            raise RuntimeError(
                f"Inconsistent embedding dimensions{context_str}: "
                f"index {i} has {len(embedding)} dimensions, expected {expected_dim}"
            )

    return True


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def get_embeddings_batch(
    embed_client,
    model_name: str,
    texts: List[str],
    task_type: str,
    titles: Optional[List[str]] = None,
    embed_dims: Optional[int] = None,
) -> List[List[float]]:
    """
    Generates embeddings for a batch of texts using the embedding API.

    Args:
        embed_client: The Google GenAI client.
        model_name: The name of the embedding model to use.
        task_type: The task type for the embedding (e.g., "RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY").
        texts: A list of strings to embed.
        titles: Optional list of titles for documents (only used with task_type="RETRIEVAL_DOCUMENT").
        embed_dims: The expected dimension size for the embedding model.

    Returns:
        A list of embedding vectors (each a list of floats).
        Returns an empty list if an error occurs.
    """
    if not texts:
        return []

    try:
        logger.debug(
            "Embedding generation: %d texts, model=%s, task_type=%s, dims=%s",
            len(texts),
            model_name,
            task_type,
            embed_dims,
        )

        # Create config with title support for RETRIEVAL_DOCUMENT task type
        config = types.EmbedContentConfig(
            task_type=task_type, output_dimensionality=embed_dims
        )

        # If titles are provided and task_type is RETRIEVAL_DOCUMENT, process each text with its title
        if titles and task_type == "RETRIEVAL_DOCUMENT":
            if len(titles) != len(texts):
                raise ValueError(
                    f"Number of titles ({len(titles)}) must match number of texts ({len(texts)})"
                )

            logger.debug("Using individual API calls with title context")
            # Process each text with its corresponding title while preserving order
            embeddings_by_index = {}  # Use dict to preserve index ordering
            successful_indices = []
            failed_indices = []

            for index, (text, title) in enumerate(zip(texts, titles)):
                try:
                    config_with_title = types.EmbedContentConfig(
                        task_type=task_type,
                        title=title,
                        output_dimensionality=embed_dims,
                    )
                    response = await embed_client.aio.models.embed_content(
                        model=model_name,
                        contents=[text],
                        config=config_with_title,
                    )

                    # Validate response contains exactly one embedding
                    if len(response.embeddings) != 1:
                        failed_indices.append(index)
                        logger.warning(
                            "Embedding API returned %d embeddings for text at index %d, expected 1",
                            len(response.embeddings),
                            index,
                        )
                        continue

                    embedding_vector = response.embeddings[0].values
                    embeddings_by_index[index] = embedding_vector
                    successful_indices.append(index)

                except Exception as e:
                    failed_indices.append(index)
                    logger.error(
                        "Failed to generate embedding for text at index %d: %s",
                        index,
                        e,
                    )
                    continue

            # Validate order preservation and handle partial failures
            if failed_indices:
                logger.warning(
                    "Embedding generation completed with %d failures out of %d total texts",
                    len(failed_indices),
                    len(texts),
                )
                logger.warning("Failed indices: %s", failed_indices)

                # For critical applications, we might want to raise an error or implement retry logic
                # For now, we'll proceed with successful embeddings but document the issue

            # Build ordered embeddings list preserving original text order
            ordered_embeddings = []
            for i in range(len(texts)):
                if i in embeddings_by_index:
                    ordered_embeddings.append(embeddings_by_index[i])
                else:
                    # For failed embeddings, we could:
                    # 1. Skip (causes length mismatch)
                    # 2. Add zero vector
                    # 3. Raise an exception
                    # For now, we'll raise an exception to maintain consistency
                    raise RuntimeError(
                        f"Failed to generate embedding for text at index {i}: {texts[i][:50]}..."
                    )

            logger.debug("Generated %d individual embeddings", len(ordered_embeddings))
            return ordered_embeddings
        else:
            # For queries or when no titles provided, use batch processing
            response = await embed_client.aio.models.embed_content(
                model=model_name,
                contents=texts,
                config=config,
            )

            # Validate batch response preserves order
            if len(response.embeddings) != len(texts):
                raise RuntimeError(
                    f"Embedding API returned {len(response.embeddings)} embeddings for {len(texts)} texts. "
                    f"Expected 1:1 correspondence."
                )

            return [embedding.values for embedding in response.embeddings]
    except Exception as e:
        logger.error("An unexpected error occurred during embedding generation: %s", e)
        return []  # Return empty list on error


async def classify_string_batch(
    qdrant_client: AsyncQdrantClient,
    embed_client: genai.Client,
    embed_model_name: str,
    query_texts: List[str],
    collection_name: str,
    embed_dims: Optional[int] = None,
    top_k: int = 3,
    quantization_cache: Optional[Dict[str, bool]] = None,
) -> List[List[Dict[str, Any]]]:
    """
    Takes a list of string inputs, gets their embeddings in a batch,
    and queries the specified Qdrant collection using batch search to find the most
    semantically similar entries for each query.

    Args:
        qdrant_client: The Qdrant client instance.
        embed_client: The Google GenAI client instance.
        embed_model_name: The name of the embedding model to use.
        query_texts: A list of input query_texts to classify/find similar items for.
        collection_name: The name of the Qdrant collection to query.
        embed_dims: The expected dimension size for the embedding model.
        top_k: The number of top similar results to return for each query.
        quantization_cache: Optional cache of collection quantization configs.

    Returns:
        A list of lists of search results (dictionaries with score and payload).
        Each inner list corresponds to an input query.
        Returns an empty list if a major error occurs.
    """

    if not query_texts:
        return []

    # 1. Get Embeddings for the Query Texts
    query_embeddings = await get_embeddings_batch(
        embed_client,
        embed_model_name,
        task_type="RETRIEVAL_QUERY",
        texts=query_texts,
        embed_dims=embed_dims,
    )

    if not query_embeddings:
        logger.error("Could not generate any embeddings for the query batch.")
        return []

    if len(query_embeddings) != len(query_texts):
        logger.error(
            "Embedding count mismatch. Got %d embeddings for %d queries.",
            len(query_embeddings),
            len(query_texts),
        )
        logger.error(
            "This indicates a serious issue with embedding ordering or API response handling."
        )
        # Instead of returning empty list, we could return partial results or handle more gracefully
        return []

    # Validate embeddings before proceeding
    validate_embedding_correspondence(query_texts, query_embeddings, "queries")

    # 2. Check if collection has quantization enabled (use cache if available)
    if quantization_cache and collection_name in quantization_cache:
        has_quantization = quantization_cache[collection_name]
    else:
        try:
            collection_info = await qdrant_client.get_collection(collection_name)
            has_quantization = collection_info.config.quantization_config is not None
        except Exception:
            has_quantization = False

    # 3. Prepare Search Parameters
    search_params = models.SearchParams(
        hnsw_ef=256,
        exact=False,
        quantization=(
            models.QuantizationSearchParams(
                ignore=False,
                rescore=True,  # Enable rescoring for better accuracy
                oversampling=3.0,
            )
            if has_quantization
            else None
        ),
    )

    # 4. Execute Search Queries
    internal_top_k = 100 if has_quantization else top_k

    # Use individual searches with conditional parameters
    batch_results = []
    for embedding in query_embeddings:
        if isinstance(embedding, list) and len(embedding) > 0:
            search_kwargs = {
                "collection_name": collection_name,
                "query": embedding,
                "query_filter": None,  # No additional filters
                "limit": internal_top_k,
                "with_payload": True,
                "with_vectors": False,  # Usually not needed in the response
            }

            search_kwargs["search_params"] = search_params

            search_result = await qdrant_client.query_points(**search_kwargs)
            batch_results.append(search_result.points)
        else:
            # Handle edge case: empty or invalid embedding
            batch_results.append([])

    # 4. Process and Format Search Results from individual searches
    all_formatted_results = []
    # Iterate through the list of search results (each is a list of ScoredPoint)
    for search_result in batch_results:
        formatted_hits = [
            {
                "score": hit.score,
                "payload": hit.payload,  # Return the whole payload
            }
            for hit in search_result  # search_result is already a list of ScoredPoint
        ]

        # Filter to user-requested top_k when quantization is enabled
        if has_quantization:
            formatted_hits = formatted_hits[:top_k]

        all_formatted_results.append(formatted_hits)

    return all_formatted_results


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
    Shared classification service that handles all common logic between web form and API endpoints.

    Args:
        embed_client: The Google GenAI client.
        qdrant_client: The Qdrant client.
        classifier_type: The classification standard (e.g., 'unspsc', 'etim', etc.)
        query: The product/service description to classify
        version: Optional specific version to use
        top_k: Number of results to return
        quantization_cache: Optional cache of collection quantization configs.

    Returns:
        Dict containing classification results and metadata
    """
    # Validate classifier type
    config = CLASSIFIER_CONFIG.get(classifier_type.upper())
    if not config:
        raise HTTPException(
            status_code=404, detail=f"Classifier '{classifier_type}' not found"
        )

    # Validate version or use default
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
    collection_name = version_config["collection_name"]
    embed_model_name = config["embed_model_name"]

    # Validate clients
    if not embed_client or not qdrant_client:
        raise HTTPException(
            status_code=503,
            detail="Backend services not available. Please check server logs.",
        )

    # Validate and normalize query - remove trailing slashes
    normalized_query = query.strip()

    logger.info(
        "CLASSIFICATION_QUERY: classifier=%s query='%s'",
        classifier_type,
        normalized_query,
    )

    if not normalized_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if len(normalized_query) > 4000:
        raise HTTPException(
            status_code=400, detail="Query too long (max 4000 characters)"
        )

    try:
        # Perform classification with normalized query
        results_for_single_query = await classify_string_batch(
            qdrant_client=qdrant_client,
            embed_client=embed_client,
            embed_model_name=embed_model_name,
            query_texts=[normalized_query],
            collection_name=collection_name,
            embed_dims=config.get("embed_dims"),
            top_k=top_k,
            quantization_cache=quantization_cache,
        )

        classification_results = (
            results_for_single_query[0] if results_for_single_query else []
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
