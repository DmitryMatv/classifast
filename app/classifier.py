import numpy as np
from google import genai
from google.genai import types
from qdrant_client import AsyncQdrantClient, models
from typing import List, Dict, Any, Optional

from tenacity import retry, stop_after_attempt, wait_exponential


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

    print(
        f"[OK] Embedding validation passed{context_str}: "
        f"{len(embeddings)} embeddings, {expected_dim} dimensions each"
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
        # Enhanced logging for embedding generation
        print(
            f"[START] Embedding generation: {len(texts)} texts, model={model_name}, task_type={task_type}"
        )
        if embed_dims:
            print(f"   Target dimensions: {embed_dims}")
        if titles:
            print(f"   Processing with titles: {len(titles)} titles provided")

        # Debug: Log the texts being sent to embedding API to verify newlines
        # print(f"Embedding API input texts (repr): {[repr(text) for text in texts]}")
        # print(f"Total texts to embed: {len(texts)}")
        # for i, text in enumerate(texts):
        #     print(f"Text {i+1} length: {len(text)}, newlines: {text.count(chr(10))}")

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

            print(f"   Using individual API calls to preserve order with title context")
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
                        print(
                            f"[WARN] Embedding API returned {len(response.embeddings)} embeddings for text at index {index}, expected 1"
                        )
                        continue

                    embedding_vector = response.embeddings[0].values
                    embeddings_by_index[index] = embedding_vector
                    successful_indices.append(index)

                    # Log successful embedding generation
                    if index < 3:  # Only log first few to avoid spam
                        print(
                            f"   [OK] Embedding {index+1}/{len(texts)}: {len(embedding_vector)} dimensions, title='{title[:30]}...'"
                        )
                    elif index == 3:
                        print(
                            f"   ... (processing remaining {len(texts) - 3} embeddings)"
                        )

                except Exception as e:
                    failed_indices.append(index)
                    print(
                        f"[WARN] Failed to generate embedding for text at index {index}: {e}"
                    )
                    continue

            # Validate order preservation and handle partial failures
            if failed_indices:
                print(
                    f"[WARN] Embedding generation completed with {len(failed_indices)} failures out of {len(texts)} total texts"
                )
                print(f"Failed indices: {failed_indices}")

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

            # Validate embedding-text correspondence for ordered embeddings
            validate_embedding_correspondence(
                texts, ordered_embeddings, "document with titles"
            )

            print(
                f"   [OK] Completed individual embeddings: {len(ordered_embeddings)} embeddings generated"
            )
            return ordered_embeddings
        else:
            # For queries or when no titles provided, use batch processing without titles
            print(f"   Using batch API call for {len(texts)} texts without titles")
            response = await embed_client.aio.models.embed_content(
                model=model_name,
                contents=texts,
                config=config,
            )

            print(f"   [OK] Batch API returned {len(response.embeddings)} embeddings")

            # Validate batch response preserves order
            if len(response.embeddings) != len(texts):
                raise RuntimeError(
                    f"Embedding API returned {len(response.embeddings)} embeddings for {len(texts)} texts. "
                    f"Expected 1:1 correspondence."
                )

            raw_embeddings = [embedding.values for embedding in response.embeddings]

            # Validate embedding-text correspondence
            validate_embedding_correspondence(texts, raw_embeddings, "batch query")

            print(
                f"   [OK] Completed batch embedding: {len(raw_embeddings)} embeddings generated"
            )
            return raw_embeddings
    except Exception as e:
        print(f"An unexpected error occurred during embedding generation: {e}")
        return []  # Return empty list on error


async def classify_string_batch(
    qdrant_client: AsyncQdrantClient,  # Add qdrant_client parameter
    embed_client: genai.Client,  # Add embed_client parameter
    embed_model_name: str,  # Add embed_model_name parameter
    query_texts: List[str],
    collection_name: str,
    embed_dims: Optional[int] = None,
    top_k: int = 3,
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
        embed_dims: The expected dimension size for the embedding model. Used to control
                   normalization behavior based on vector dimensionality.
        top_k: The number of top similar results to return for each query.

    Returns:
        A list of lists of search results (dictionaries with score and payload).
        Each inner list corresponds to an input query.
        Returns an empty list if a major error occurs.
    """

    if not query_texts:
        print("Input query list is empty.")
        return []

    try:

        # 1. Get Embeddings for the Query Texts in a Single Batch Call
        print(
            f"Generating embeddings for {len(query_texts)} queries using model {embed_model_name}..."
        )
        query_embeddings = await get_embeddings_batch(
            embed_client,
            embed_model_name,
            task_type="RETRIEVAL_QUERY",
            texts=query_texts,
            embed_dims=embed_dims,
        )

        if not query_embeddings:
            print("Error: Could not generate any embeddings for the query batch.")
            return []

        if len(query_embeddings) != len(query_texts):
            print(
                f"Error: Embedding count mismatch. Got {len(query_embeddings)} embeddings for {len(query_texts)} queries."
            )
            print(
                "This indicates a serious issue with embedding ordering or API response handling."
            )
            # Instead of returning empty list, we could return partial results or handle more gracefully
            return []

        # Validate embeddings before proceeding
        validate_embedding_correspondence(query_texts, query_embeddings, "queries")

        query_embeddings_np = np.array(query_embeddings)
        query_embeddings = query_embeddings_np.tolist()

        # 2. Check if collection has quantization enabled
        try:
            collection_info = await qdrant_client.get_collection(collection_name)
            has_quantization = collection_info.config.quantization_config is not None
            print(
                f"Collection '{collection_name}' quantization enabled: {has_quantization}"
            )
        except Exception as e:
            print(
                f"Warning: Could not check collection configuration for '{collection_name}': {e}"
            )
            has_quantization = False

        # 3. Prepare Search Parameters with consistent hnsw_ef, adding quantization settings when available
        search_params = models.SearchParams(
            hnsw_ef=256,
            exact=False,  # Ensure ANN index is used
            quantization=(
                models.QuantizationSearchParams(
                    ignore=False,
                    rescore=False,
                    oversampling=2.0,
                )
                if has_quantization
                else None
            ),
        )

        if has_quantization:
            print(
                f"Using quantization search parameters (hnsw_ef=256, rescore=True, oversampling=2.0)"
            )
        else:
            print(f"Using default search parameters (hnsw_ef=256, no quantization)")

        # 4. Execute Individual Search Queries with appropriate parameters
        search_type = "quantized" if has_quantization else "standard"

        # Calculate internal top_k for better rescore/oversampling accuracy
        if has_quantization:
            internal_top_k = 100  # Ensure sufficient candidates for rescore
            print(
                f"Querying collection '{collection_name}' with {len(query_embeddings)} individual searches ({search_type})..."
                f" Internal top_k: {internal_top_k}, User top_k: {top_k}"
            )
        else:
            internal_top_k = top_k
            print(
                f"Querying collection '{collection_name}' with {len(query_embeddings)} individual searches ({search_type})..."
            )

        # Use individual searches with conditional parameters
        batch_results = []
        for embedding in query_embeddings:
            if isinstance(embedding, list) and len(embedding) > 0:
                search_kwargs = {
                    "collection_name": collection_name,
                    "query_vector": embedding,
                    "query_filter": None,  # No additional filters
                    "limit": internal_top_k,
                    "with_payload": True,
                    "with_vectors": False,  # Usually not needed in the response
                }

                search_kwargs["search_params"] = search_params

                search_result = await qdrant_client.search(**search_kwargs)
                batch_results.append(search_result)
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
            # print(f"Query '{query_texts[i][:50]}...': Found {len(formatted_hits)} results.")

        if all_formatted_results and any(results for results in all_formatted_results):
            print(
                f"[OK] Batch query finished successfully. Returning {len(all_formatted_results)} sets of results."
            )
        else:
            print(
                f"[WARN] Batch query finished but returned empty results for all queries."
            )
        return all_formatted_results

    except Exception as e:
        print(f"[ERROR] Error during batch classification: {e}")
        # Depending on the desired error handling, you might want to raise
        # the exception or return an empty list.
        return []
