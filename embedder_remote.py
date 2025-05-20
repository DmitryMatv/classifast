import os
import time
import pandas as pd
from google import genai
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
from typing import Optional
from tqdm import tqdm


from app.classifier import get_embeddings_batch


def load_and_prepare_data(
    csv_path: str,
    id_column: str,
    class_column: str,
    sample_n: Optional[int] = None,
) -> pd.DataFrame:
    """
    Reads data from a CSV, optionally samples it, and creates a combinedtext column for embedding.

    Args:
        csv_path: Path to the input CSV file.
        id_column: Name of the column containing unique IDs for Qdrant points.
        class_column: Name of the column whose value will be stored as payload.
        sample_n: If not None, randomly sample this many rows from the dataframe.

    Returns:
        A pandas DataFrame with 'id', 'class_column', and 'embedding_text' columns.
        Returns None if file not found or required columns are missing.
    """

    print(f"Loading data from {csv_path}")
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return None

    try:
        df = pd.read_csv(csv_path, delimiter="|")  # Adjust delimiter if needed

        # --- Select specific row ---
        # df = df.iloc[[2200]].copy()
        # print(f"Selected row at index 2200.")

        # --- Sampling ---
        if sample_n is not None and sample_n > 0 and sample_n <= len(df):
            print(f"Sampling {sample_n} rows from the dataframe.")
            df = df.sample(frac=sample_n / len(df))
        elif sample_n is not None:
            print(
                f"Warning: sample_n={sample_n} is invalid for df of size {len(df)}. Using full df."
            )

        df.info()  # Print DataFrame info for debugging

        # --- Column Validation ---
        required_columns = [id_column, class_column]  # + columns_to_embed
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns in CSV: {missing_cols}")
            return None

        # --- Prepare ID and Payload ---
        # Store original IDs and Class name in payload for reference
        df["original_id"] = df[id_column]
        df["class_name"] = df[class_column]

        # Convert IDs by removing 'EC' prefix and leading zeros, then convert to integers
        # df["id"] = df[id_column].astype(str).str.replace("EC", "").str.lstrip("0").astype(int))
        df["id"] = df[id_column].astype(str).str.replace("EC", "").astype(int)

        # Use UUIDs for unique IDs
        # df["uuid"] = df[id_column].apply(lambda x: str(uuid.uuid5(uuid.NAMESPACE_DNS, str(x))))

        # --- Create Embedding Text ---
        print(f"Creating embedding texts...")
        # Convert selected columns to string and join with spaces
        # df["embedding_text"] = df[columns_to_embed].astype(str).agg(" ".join, axis=1)

        # Replace _NEWLINE_ in "Properties" column with actual newlines
        df["Properties_lines"] = df["Properties"].str.replace("_NEWLINE_", "\n")

        df["embedding_text"] = (
            "NAME: "
            + df["Name"]
            + "\n"  # This should be \n if a real newline is desired for embedding
            + "SYNONYMS: "
            + df["Synonyms"]
            + "\n"  # This should be \n
            + "DEFINITION: "
            + df["Definition"]
            + "\n"  # This should be \n
            + "PARENT: "
            + df["Parent"]
            + "\n"  # This should be \n
            + "FEATURES: "
            + "\n"  # This should be \n
            + df["Properties_lines"]  # \n from the .str.replace()
            + "\n"  # This should be \n
            + "PROPERTY SETS: "
            + df["Property_sets"]
        ).str.strip()

        # Preview the first embedding text
        if not df.empty:
            print(
                f"Preview of the first embedding text:\n{df['embedding_text'].iloc[0]}"
            )

        print(f"Texts prepared. {len(df)} string(s) are ready for to be embedded.")

        # Select and rename columns for clarity
        return df[["id", "original_id", "class_name", "embedding_text"]].copy()

    except Exception as e:
        print(f"Error processing CSV file: {e}")
        return None


def create_and_populate_qdrant(
    data: pd.DataFrame,
    collection_name: str,
    vector_size: int,
    distance_metric: models.Distance,
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    qdrant_path: Optional[str] = None,  # Keep for fallback or local-only mode
    embedding_batch_size: int = 1,  # Process N texts per API call (128 for VoyagerAI API)
) -> bool:
    """
    Connects to Qdrant, creates a collection if needed, generates embeddings,
    and upserts the data.

    Args:
        collection_name: Name of the collection to create/use.
        vector_size: The dimensionality of the vectors.
        distance_metric: The distance metric for similarity search.
        data: DataFrame containing 'id', 'original_id', 'class_name', and 'embedding_text'.
        qdrant_url: URL for the remote Qdrant instance.
        qdrant_api_key: API key for the remote Qdrant instance.
        qdrant_path: Path for the local Qdrant database (used if qdrant_url is not provided).
        embedding_batch_size: How many embeddings to generate per API call.

    Returns:
        True if successful, False otherwise.
    """
    if qdrant_url:
        print(f"Initializing Qdrant client with remote URL: {qdrant_url}")
        client = QdrantClient(
            host=qdrant_url,
            port=443,  # Use HTTPS port since Traefik handles SSL
            api_key=qdrant_api_key,
            https=True,
            prefer_grpc=False,
            timeout=60,  # Add a longer timeout just in case
        )
    elif qdrant_path:
        print(f"Initializing Qdrant client with local path: {qdrant_path}")
        client = QdrantClient(path=qdrant_path)
    else:
        print("Error: Qdrant connection details (URL or path) not provided.")
        return False

    # --- Create Collection (if it doesn't exist) ---
    try:

        """
        # Delete collection if it exists
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)
            print(f"Deleted existing collection: {collection_name}")

        # Create a new collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=distance_metric,  # Using DOT product as recommended by OpenAI
            ),
        )
        print(f"Created Qdrant collection: '{collection_name}' with {vector_size} dimensions and {distance_metric} distance.")
        """

        if not client.collection_exists(collection_name):
            print(
                f"Creating Qdrant collection: '{collection_name}' with {vector_size} dimensions and {distance_metric} distance."
            )
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size, distance=distance_metric
                ),
                on_disk_payload=True,
                hnsw_config=models.HnswConfigDiff(
                    m=32, ef_construct=200, max_indexing_threads=2, on_disk=True
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    flush_interval_sec=60,
                    max_optimization_threads=1,
                ),
            )
        else:
            # Get collection info and check vector configuration
            coll_info = client.get_collection(collection_name)
            vector_config = coll_info.config.params
            if (
                vector_config.vectors.size != vector_size
                or vector_config.vectors.distance != distance_metric
            ):
                print(
                    f"Warning: Collection '{collection_name}' exists but has different parameters."
                )
                print(
                    f"Existing: size={vector_config.vectors.size}, distance={vector_config.vectors.distance}"
                )
                print(f"Required: size={vector_size}, distance={distance_metric}")
                print(
                    "Consider deleting the collection manually or using a different name if parameters need to change."
                )
            else:
                print(
                    f"Collection '{collection_name}' already exists. Upserting data..."
                )

    except Exception as e:
        print(f"Error interacting with Qdrant collection: {e}")
        return False

    # --- Fetch existing original_ids from the collection ---
    print(f"Fetching existing original_ids from collection '{collection_name}'...")
    existing_original_ids = set()
    try:
        next_offset = None
        processed_count = 0
        # Scroll through all points in the collection to get existing original_ids
        while True:
            records_batch, next_offset = client.scroll(
                collection_name=collection_name,
                limit=250,  # Adjust batch size as needed
                offset=next_offset,
                with_payload=["original_id"],  # Only fetch the original_id from payload
                with_vectors=False,  # We don't need vectors for this
            )
            if not records_batch:
                break  # No more points

            for record in records_batch:
                # Ensure payload is not None and original_id is present
                if record.payload and "original_id" in record.payload:
                    existing_original_ids.add(record.payload["original_id"])

            processed_count += len(records_batch)
            if processed_count > 0 and processed_count % 1000 == 0:  # Log progress
                print(
                    f"Scanned {processed_count} existing points in '{collection_name}'..."
                )

            if next_offset is None:  # No more pages
                break
        print(
            f"Found {len(existing_original_ids)} unique existing original_ids in '{collection_name}'."
        )
    except Exception as e:
        print(f"Error fetching existing points from Qdrant: {e}")
        print(
            "Proceeding to upsert all data. Duplicates may occur if points already exist."
        )
        existing_original_ids = (
            set()
        )  # Reset to empty set on error to attempt upserting all

    # --- Filter data to exclude existing original_ids ---
    if existing_original_ids:  # Only filter if we successfully fetched some IDs
        print(f"Original data size before filtering: {len(data)}")
        # Create a boolean series for rows where 'original_id' is NOT in existing_original_ids
        data_to_upsert = data[~data["original_id"].isin(existing_original_ids)].copy()
        print(f"Number of new points to upsert after filtering: {len(data_to_upsert)}")
    else:
        # If no existing IDs were found or an error occurred during fetch, use all data
        data_to_upsert = data.copy()
        print(
            f"No existing points found or check skipped. Number of points to upsert: {len(data_to_upsert)}"
        )

    if data_to_upsert.empty:
        print(
            "No new data to upsert. All points (based on original_id) may already exist in the collection."
        )
        return True  # Successfully did nothing, which is the correct outcome

    # --- Generate Embeddings and Upsert in Batches ---
    num_rows = len(data_to_upsert)  # Use the filtered data for upsertion
    print(f"Generating embeddings and upserting {num_rows} new points...")

    calls_this_minute = 0
    minute_start_time = time.time()

    try:
        for i in tqdm(
            range(0, num_rows, embedding_batch_size),
            desc="Upserting new points",  # Updated description
        ):
            batch_df = data_to_upsert.iloc[
                i : i + embedding_batch_size
            ]  # Use the filtered data
            texts_to_embed = batch_df["embedding_text"].tolist()

            # Rate limiting
            current_time = time.time()
            if current_time - minute_start_time >= 60:
                minute_start_time = current_time
                calls_this_minute = 0

            if calls_this_minute >= 10:
                sleep_duration = 60 - (current_time - minute_start_time)
                if sleep_duration > 0:
                    print(
                        f"Rate limit reached. Sleeping for {sleep_duration:.2f} seconds."
                    )
                    time.sleep(sleep_duration)
                minute_start_time = time.time()  # Reset timer after sleeping
                calls_this_minute = 0

            # Get embeddings for the batch
            embeddings = get_embeddings_batch(
                EMBED_CLIENT,
                EMBED_MODEL,
                task_type="RETRIEVAL_DOCUMENT",
                texts=texts_to_embed,
            )
            calls_this_minute += 1

            if not embeddings or len(embeddings) != len(batch_df):
                print(f"Error embedding batch at index {i}. Skipping batch.")
                # Decide on error handling: continue, retry, or stop? Here we skip.
                continue

            # Prepare points for Qdrant upsert
            points_to_upsert = [
                models.PointStruct(
                    id=row["id"],
                    vector=embeddings[j],
                    payload={
                        "original_id": row["original_id"],
                        "class_name": row["class_name"],
                    },
                )
                for j, (_, row) in enumerate(batch_df.iterrows())
                # Use enumerate to match embeddings
            ]

            if points_to_upsert:
                client.upsert(
                    collection_name=collection_name,
                    points=points_to_upsert,
                    wait=True,  # Wait for operation to complete
                )

        print("Data upserted successfully into Qdrant.")
        return True

    except Exception as e:
        print(f"Error during embedding generation or Qdrant upsert: {e}")
        return False


if __name__ == "__main__":

    EMBED_MODEL_DEFAULT = "text-embedding-004"
    EMBED_DIMS_DEFAULT = 768

    # QDRANT_DB_PATH will be used as a fallback if QDRANT_URL is not in .env
    QDRANT_DB_PATH_FALLBACK = "./qdrant_db"
    QDRANT_COLLECTION_NAME = "ETIM_10_eng_3072_exp"  # Not using .env for Embedder
    QDRANT_DISTANCE_METRIC = models.Distance.DOT

    # 1. Load Environment Variables
    print("Loading environment variables...")
    load_dotenv()

    # Get Qdrant connection details from .env
    # These will be preferred if present
    QDRANT_REMOTE_URL = os.getenv("QDRANT_URL")
    QDRANT_REMOTE_API_KEY = os.getenv("QDRANT_API_KEY")

    # Get other configurations from .env, with fallbacks to defaults defined above
    EMBED_MODEL = os.getenv("EMBED_MODEL_NAME", EMBED_MODEL_DEFAULT)
    EMBED_DIMS = int(os.getenv("EMBED_DIMS", EMBED_DIMS_DEFAULT))

    # 2. Initialize embedding API client
    print("Initializing embedding API client...")
    try:
        EMBED_CLIENT = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        # Test connection (optional, but good practice)
        EMBED_CLIENT.models.list()
        print("Embedding client initialized successfully.")
    except Exception as e:
        print(f"Error initializing embedding client: {e}")
        exit()

    # 3. Load and Prepare Data
    prepared_data = load_and_prepare_data(
        csv_path="etim_classes_data.csv",  # Your specific CSV
        id_column="Code",  # Column with unique IDs
        class_column="Name",  # Column to store as payload
        # sample_n=69,  # For testing
    )

    # 4. Create/Populate Qdrant DB (The "Training" Step)
    if prepared_data is not None:
        success = create_and_populate_qdrant(
            data=prepared_data,
            qdrant_url=QDRANT_REMOTE_URL,
            qdrant_api_key=QDRANT_REMOTE_API_KEY,
            qdrant_path=(
                QDRANT_DB_PATH_FALLBACK if not QDRANT_REMOTE_URL else None
            ),  # Pass path only if URL is not set
            collection_name=QDRANT_COLLECTION_NAME,
            vector_size=EMBED_DIMS,
            distance_metric=QDRANT_DISTANCE_METRIC,
        )
        if not success:
            print("Failed to create/populate Qdrant database.")
            exit()  # Stop if training failed

    else:
        print("Failed to load or prepare data. Exiting.")
        exit()  # Stop if data loading failed
