"""
One-time script to create full-text indexes on payload fields for text_any search.

This enables hybrid search: exact text matches (code lookups) + semantic search.

Usage:
    python utilities/create_text_indexes.py

The script will:
1. Connect to Qdrant
2. Iterate through all collections defined in CLASSIFIER_CONFIG
3. Create text indexes on 'original_id' and 'class_name' fields
"""

import os
import sys

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

# Add parent directory to path so we can import from app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.classifier_config import CLASSIFIER_CONFIG

load_dotenv()
QDRANT_REMOTE_URL = os.getenv("QDRANT_URL")
QDRANT_REMOTE_API_KEY = os.getenv("QDRANT_API_KEY")

if not QDRANT_REMOTE_URL:
    raise ValueError("QDRANT_URL environment variable is required")
if not QDRANT_REMOTE_API_KEY:
    raise ValueError("QDRANT_API_KEY environment variable is required")

client = QdrantClient(
    host=QDRANT_REMOTE_URL,
    port=443,
    api_key=QDRANT_REMOTE_API_KEY,
    https=True,
    prefer_grpc=False,
    timeout=120,
)


def get_all_collection_names() -> list[str]:
    """Extract all unique collection names from CLASSIFIER_CONFIG."""
    collection_names = set()
    for config in CLASSIFIER_CONFIG.values():
        versions = config.get("versions", {})
        for version_config in versions.values():
            collection_name = version_config.get("collection_name")
            if collection_name:
                collection_names.add(collection_name)
    return sorted(collection_names)


def create_text_index(collection_name: str, field_name: str) -> bool:
    """
    Create a full-text index on a payload field.

    Returns True if successful, False otherwise.
    """
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=models.TextIndexParams(
                type="text",
                tokenizer=models.TokenizerType.WORD,
                min_token_len=1,
                max_token_len=30,
                lowercase=True,
            ),
        )
        print(f"  + Created text index on '{field_name}'")
        return True
    except Exception as e:
        error_msg = str(e)
        if "already exists" in error_msg.lower():
            print(f"  - Index on '{field_name}' already exists, skipping")
            return True
        print(f"  ! Error creating index on '{field_name}': {e}")
        return False


def main():
    print("=" * 60)
    print("Creating Full-Text Indexes for Hybrid Search")
    print("=" * 60)

    collection_names = get_all_collection_names()
    print(f"\nFound {len(collection_names)} collections to process:\n")

    fields_to_index = ["original_id", "class_name"]
    success_count = 0
    error_count = 0

    for collection_name in collection_names:
        print(f"\nProcessing: {collection_name}")

        # Verify collection exists
        try:
            client.get_collection(collection_name)
        except Exception as e:
            print(f"  ! Collection not found or error: {e}")
            error_count += 1
            continue

        # Create indexes on both fields
        collection_success = True
        for field_name in fields_to_index:
            if not create_text_index(collection_name, field_name):
                collection_success = False

        if collection_success:
            success_count += 1
        else:
            error_count += 1

    print("\n" + "=" * 60)
    print(f"Completed: {success_count} collections indexed successfully")
    if error_count:
        print(f"Errors: {error_count} collections had issues")
    print("=" * 60)


if __name__ == "__main__":
    main()
