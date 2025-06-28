import os
from google import genai
from google.genai import types
from qdrant_client import AsyncQdrantClient, models
from dotenv import load_dotenv
from typing import List, Dict, Any

from tenacity import retry, stop_after_attempt, wait_exponential


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def get_embeddings_batch(
    embed_client, model_name: str, task_type: str, texts: List[str]
) -> List[List[float]]:
    """
    Generates embeddings for a batch of texts using the embedding API.

    Args:
        embed_client: The Google GenAI client.
        model_name: The name of the embedding model to use.
        task_type: The task type for the embedding (e.g., "CLASSIFICATION").
        texts: A list of strings to embed.

    Returns:
        A list of embedding vectors (each a list of floats).
        Returns an empty list if an error occurs.
    """
    if not texts:
        return []

    try:
        # Debug: Log the texts being sent to embedding API to verify newlines
        # print(f"Embedding API input texts (repr): {[repr(text) for text in texts]}")
        # print(f"Total texts to embed: {len(texts)}")
        # for i, text in enumerate(texts):
        #     print(f"Text {i+1} length: {len(text)}, newlines: {text.count(chr(10))}")

        response = await embed_client.aio.models.embed_content(
            model=model_name,
            contents=texts,
            config=types.EmbedContentConfig(task_type=task_type),
            # "CLASSIFICATION" is made for use with ML algorithms, train/test sets, etc.
        )
        return [embedding.values for embedding in response.embeddings]
    except Exception as e:
        print(f"An unexpected error occurred during embedding generation: {e}")
        return []  # Return empty list on error


async def classify_string_batch(
    qdrant_client: AsyncQdrantClient,  # Add qdrant_client parameter
    embed_client: genai.Client,  # Add embed_client parameter
    embed_model_name: str,  # Add embed_model_name parameter
    query_texts: List[str],
    collection_name: str,
    top_k: int = 5,
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
        )

        if not query_embeddings or len(query_embeddings) != len(query_texts):
            print(
                "Error: Could not generate embeddings accurately for the query batch."
            )
            # Return a list of empty lists matching the input size for partial failure?
            # Or return [] for complete failure? Let's return [] for simplicity here.
            return []

        # 2. Prepare Batch Query Requests for Qdrant using models.QueryRequest
        query_requests = [
            models.QueryRequest(
                query=embedding,  # Pass the embedding vector as the 'query'
                limit=top_k,
                with_payload=True,  # To get the original data back
                # with_vectors=False,  # Usually not needed in the response
            )
            for embedding in query_embeddings
        ]

        # 3. Execute the Batch Search Query
        print(
            f"Querying collection '{collection_name}' with a batch of {len(query_requests)} requests..."
        )
        batch_results = await qdrant_client.query_batch_points(
            collection_name=collection_name,
            requests=query_requests,
            # consistency=models.ReadConsistencyType.MAJORITY, # Optional: Adjust consistency
        )

        # 4. Process and Format Batch Results from QueryResponse objects
        all_formatted_results = []
        # Iterate through the list of QueryResponse objects
        for i, response in enumerate(batch_results):
            # Access the list of ScoredPoint objects via the .points attribute
            formatted_hits = [
                {
                    "score": hit.score,
                    "payload": hit.payload,  # Return the whole payload
                }
                # Iterate through the points within *this* response object
                for hit in response.points
            ]
            all_formatted_results.append(formatted_hits)
            # print(f"Query '{query_texts[i][:50]}...': Found {len(formatted_hits)} results.")

        print(
            f"Batch query finished. Returning {len(all_formatted_results)} sets of results."
        )
        return all_formatted_results

    except Exception as e:
        print(f"An error occurred during batch classification: {e}")
        # Depending on the desired error handling, you might want to raise
        # the exception or return an empty list.
        return []


# Example usage of the classify_string_batch function
async def main():
    EMBED_MODEL = "text-embedding-004"

    QDRANT_DB_PATH = "./qdrant_db"  # Local path to store Qdrant data
    QDRANT_COLLECTION_NAME = "ETIM10_google"  # Name for the Qdrant collection

    # 1. Load Environment Variables
    print("Loading environment variables...")
    load_dotenv()

    # 2. Initialize embedding API client
    print("Initializing embedding API client...")
    embed_client_instance = None  # Initialize to None
    try:
        embed_client_instance = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        # Test connection (optional, but good practice)
        embed_client_instance.models.list()
        print("Embedding client initialized successfully.")
    except Exception as e:
        print(f"Error initializing embedding client: {e}")
        return

    # 3. Initialize Qdrant Client
    print("Initializing Qdrant client...")
    qdrant_client_instance = AsyncQdrantClient(
        path=QDRANT_DB_PATH
    )  # Initialize qdrant_client_instance

    # Check if collection exists before querying
    try:
        collection_info = await qdrant_client_instance.get_collection(
            collection_name=QDRANT_COLLECTION_NAME
        )
        print(f"Collection '{QDRANT_COLLECTION_NAME}' found.")
    except Exception:
        print(
            f"Error: Collection '{QDRANT_COLLECTION_NAME}' does not exist in Qdrant at {QDRANT_DB_PATH}."
        )
        return

    # --- Example Classification ---

    test_queries = [
        "Miniature circuit breaker (MCB), 10 A, 1p, characteristic: B ",
        "Double 2-way switch 10AX beige Sedna Design",
        "Combiner Box (Photovoltaik), 1100 V, 2 MPP's, 2 Inputs / 1 Output per",
        "LEDtube 1200mm 12,5W/830 HO 2000Lm 50tH MASTER",
        "UK 6-FSI/C - Fuse modular terminal block, 6 mm², 1-pole, 6 A",
    ]

    batch_search_results = await classify_string_batch(
        qdrant_client=qdrant_client_instance,  # Pass qdrant_client_instance
        embed_client=embed_client_instance,  # Pass embed_client_instance
        embed_model_name=EMBED_MODEL,  # Pass embed_model_name
        query_texts=test_queries,
        collection_name=QDRANT_COLLECTION_NAME,
        top_k=3,
    )

    print("\n--- Batch Classification Results ---")
    if batch_search_results:
        for i, results_for_query in enumerate(batch_search_results):
            print(f"\nResults for Query: '{test_queries[i]}'")
            if results_for_query:  # Check if the list of hits is not empty
                for result in results_for_query:  # Iterate directly over hits
                    print(f"  Original ID: {result['payload']['original_id']}")
                    print(f"  Class name: {result['payload']['class_name']}")
                    print(f"  Similarity score: {result['score']:.3f}")
                    print("-" * 10)
            else:
                print("  No similar items found for this query.")
    else:
        print("Batch classification failed or returned no results.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
