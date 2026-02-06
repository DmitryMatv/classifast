import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()
QDRANT_REMOTE_URL = os.getenv("QDRANT_URL")
QDRANT_REMOTE_API_KEY = os.getenv("QDRANT_API_KEY")

# Your existing client setup
client = QdrantClient(
    host=QDRANT_REMOTE_URL,
    port=443,
    api_key=QDRANT_REMOTE_API_KEY,
    https=True,
    prefer_grpc=False,
    timeout=60,
)

# --- Add the code below ---

# Define your collection name
COLLECTION_NAME = (
    "collection_name"  # <--- IMPORTANT: Change this!
)

# 1. & 2. Update Quantization and HNSW Config
# This operation tells Qdrant to update the collection's configuration.
# It will start a background process to re-index all the vectors.
# This might take a while depending on the size of your collection.
print(f"Updating configuration for collection: {COLLECTION_NAME}")

client.update_collection(
    collection_name=COLLECTION_NAME,
    # Vector configuration: Update on_disk parameter
    vectors_config={
        # To put vector data on disk for a collection that does not have named vectors, use "" as name:
        "": models.VectorParamsDiff(
            hnsw_config=models.HnswConfigDiff(
                m=32,
                ef_construct=256,
                max_indexing_threads=2,
                on_disk=True,
            ),
            on_disk=True,  # Change to False to enable RAM storage
        )
    },
    # Global HNSW config
    hnsw_config=models.HnswConfigDiff(
        m=32,
        ef_construct=256,
        max_indexing_threads=2,
        on_disk=True,
        inline_storage=True,
    ),
    # Quantization config
    quantization_config=models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8, quantile=0.99, always_ram=True
        ),
    ),
    optimizer_config=models.OptimizersConfigDiff(
        memmap_threshold=100000,
        indexing_threshold=10000,
        max_optimization_threads=2,
    ),
)

print("Collection configuration update initiated successfully.")
print("Please monitor the collection status. Re-indexing may take some time.")

"""
# 3. Set hnsw_ef at Search Time
# hnsw_ef is NOT a collection parameter. It's a search parameter.
# You provide it with each search request to balance speed and accuracy.

print("\n--- Example of how to set hnsw_ef during a search ---")

# A placeholder for your 3072-dimensional query vector
query_vector = [0.1] * 3072

# A reasonable starting value for hnsw_ef is 128.
# You can increase it for higher accuracy or decrease it for faster searches.
search_params = models.SearchParams(
    hnsw_ef=128, exact=False  # Ensure ANN index is used
)

# Perform the search with the specified search parameters
search_result = client.search(
    collection_name=COLLECTION_NAME,
    query_vector=query_vector,
    search_params=search_params,
    limit=10,  # How many results to return
)

print("\nExample search results:")
print(search_result)
"""
