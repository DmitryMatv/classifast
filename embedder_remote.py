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
    definition_column: str,  # Added definition_column
    sample_n: Optional[int] = None,
) -> pd.DataFrame:
    """
    Reads data from a CSV, optionally samples it, and creates a combinedtext column for embedding.

    Args:
        csv_path: Path to the input CSV file.
        id_column: Name of the column containing unique IDs for Qdrant points.
        class_column: Name of the column whose value will be stored as payload.
        definition_column: Name of the column containing the definition. # Added
        sample_n: If not None, randomly sample this many rows from the dataframe.

    Returns:
        A pandas DataFrame with 'id', 'original_id', 'class_name', 'definition', and 'embedding_text' columns.
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
        required_columns = [
            id_column,
            class_column,
            definition_column,
        ]  # Added definition_column
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns in CSV: {missing_cols}")
            return None

        # --- Prepare ID and Payload ---
        # Store original IDs and Class name in payload for reference
        df["original_id"] = df[id_column]
        df["class_name"] = df[class_column]
        df["definition"] = df[definition_column]  # Added definition

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
        # print(f"Preview of the first embedding text:\n{df['embedding_text'].iloc[0]}")

        print(f"Texts prepared. {len(df)} string(s) are ready for to be embedded.")

        # Select and rename columns for clarity
        return df[
            ["id", "original_id", "class_name", "definition", "embedding_text"]
        ].copy()  # Added "definition"

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
                    f"Collection '{collection_name}' already exists. Preparing to update and upsert data..."
                )

    except Exception as e:
        print(f"Error interacting with Qdrant collection: {e}")
        return False

    # --- Fetch existing Qdrant point IDs and their original_ids ---
    print(
        f"Fetching existing point IDs and original_ids from collection '{collection_name}'..."
    )
    existing_qdrant_points_map = {}  # Maps original_id to Qdrant point_id
    try:
        next_offset = None
        processed_count = 0
        while True:
            records_batch, next_offset = client.scroll(
                collection_name=collection_name,
                limit=250,
                offset=next_offset,
                with_payload=["original_id"],  # Only fetch original_id
                with_vectors=False,
            )
            if not records_batch:
                break
            for record in records_batch:
                if record.payload and "original_id" in record.payload:
                    # Store mapping from original_id to the Qdrant point's actual ID (which could be int or UUID)
                    existing_qdrant_points_map[record.payload["original_id"]] = (
                        record.id
                    )
            processed_count += len(records_batch)
            if processed_count > 0 and processed_count % 1000 == 0:
                print(
                    f"Scanned {processed_count} existing points in '{collection_name}'..."
                )
            if next_offset is None:
                break
        print(
            f"Found {len(existing_qdrant_points_map)} existing points with 'original_id' in '{collection_name}'."
        )
    except Exception as e:
        print(f"Error fetching existing points from Qdrant: {e}")
        # If we can't fetch, we can't reliably update. Depending on requirements,
        # one might choose to only upsert new items or stop.
        # For this implementation, we'll proceed cautiously, prioritizing updates if possible.
        # If this map is empty, all data will be treated as new if not for the next filtering step.

    # --- Separate data for payload update and new upserts ---
    points_to_update_payload = []
    data_to_upsert_new_df_list = []

    print("Separating data for payload updates and new point upserts...")
    for _, row in data.iterrows():
        original_id = row["original_id"]
        if original_id in existing_qdrant_points_map:
            qdrant_point_id = existing_qdrant_points_map[original_id]
            points_to_update_payload.append(
                {
                    "id": qdrant_point_id,  # The actual ID in Qdrant
                    "payload": {
                        "original_id": row["original_id"],
                        "class_name": row["class_name"],
                        "definition": row["definition"],
                    },
                }
            )
        else:
            data_to_upsert_new_df_list.append(row.to_dict())

    data_to_upsert_new = pd.DataFrame(data_to_upsert_new_df_list)

    print(f"Found {len(points_to_update_payload)} points for payload update.")
    print(f"Found {len(data_to_upsert_new)} new points to embed and upsert.")

    # --- Update Payload for Existing Points ---
    if points_to_update_payload:
        print(
            f"Updating payload for {len(points_to_update_payload)} existing points using client.set_payload iteratively..."
        )
        try:
            for point_data in tqdm(
                points_to_update_payload,  # List of {'id': qdrant_id, 'payload': complete_new_payload}
                desc="Updating payloads",
            ):
                client.set_payload(
                    collection_name=collection_name,
                    payload=point_data[
                        "payload"
                    ],  # This is the complete payload for the point
                    points=[point_data["id"]],  # Apply to the specific point ID
                    wait=True,
                )
            print(
                f"Payload updated for {len(points_to_update_payload)} existing points."
            )
        except Exception as e:
            print(f"Error updating payloads for existing points: {e}")
            # Potentially return False or log and continue with new points

    # --- Generate Embeddings and Upsert New Points in Batches ---
    if data_to_upsert_new.empty:
        if not points_to_update_payload:  # No updates were made either
            print(
                "No new data to upsert and no existing points found for payload update."
            )
        else:
            print(
                "No new data to upsert. Payload updates for existing points (if any) are complete."
            )
        return True  # Successfully did what was needed

    num_rows = len(data_to_upsert_new)
    print(f"Generating embeddings and upserting {num_rows} new points...")

    calls_this_minute = 0
    minute_start_time = time.time()

    try:
        for i in tqdm(
            range(0, num_rows, embedding_batch_size),
            desc="Upserting new points",
        ):
            batch_df = data_to_upsert_new.iloc[i : i + embedding_batch_size]
            texts_to_embed = batch_df["embedding_text"].tolist()

            current_time = time.time()
            if current_time - minute_start_time >= 60:
                minute_start_time = current_time
                calls_this_minute = 0

            if calls_this_minute >= 10:  # Assuming a rate limit (adjust as needed)
                sleep_duration = 60 - (current_time - minute_start_time)
                if sleep_duration > 0:
                    print(
                        f"Rate limit reached. Sleeping for {sleep_duration:.2f} seconds."
                    )
                    time.sleep(sleep_duration)
                minute_start_time = time.time()
                calls_this_minute = 0

            embeddings = get_embeddings_batch(
                EMBED_CLIENT,  # Make sure EMBED_CLIENT is passed or defined globally
                EMBED_MODEL,  # Make sure EMBED_MODEL is passed or defined globally
                task_type="RETRIEVAL_DOCUMENT",
                texts=texts_to_embed,
            )
            calls_this_minute += 1

            if not embeddings or len(embeddings) != len(batch_df):
                print(f"Error embedding batch at index {i}. Skipping batch.")
                continue

            points_to_upsert = [
                models.PointStruct(
                    id=row["id"],  # This is the 'id' from the CSV, converted earlier
                    vector=embeddings[j],
                    payload={
                        "original_id": row["original_id"],
                        "class_name": row["class_name"],
                        "definition": row["definition"],
                    },
                )
                for j, (_, row) in enumerate(batch_df.iterrows())
            ]

            if points_to_upsert:
                client.upsert(
                    collection_name=collection_name,
                    points=points_to_upsert,
                    wait=True,
                )

        print("New data upserted successfully into Qdrant.")
        return True

    except Exception as e:
        print(f"Error during embedding generation or Qdrant upsert of new points: {e}")
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
        definition_column="Definition",  # Added definition column
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
