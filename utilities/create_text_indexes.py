"""
Manual remediation utility for Qdrant collections created with the old keyword-index contract.

Run this after deploying the text-index startup fix to recreate payload indexes for
fields used by MatchText search.

Usage:
    python utilities/create_text_indexes.py

The script will:
1. Connect to Qdrant
2. Iterate through all collections defined in CLASSIFIER_CONFIG
3. Delete any existing payload index on 'original_id' and 'class_name'
4. Recreate each field as a text index with the expected tokenizer settings
"""

import os
import sys

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

# Add parent directory to path so we can import from app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.classifier_config import CLASSIFIER_CONFIG

load_dotenv()

TEXT_SEARCH_FIELDS = ["original_id", "class_name"]


def build_text_index_params() -> models.TextIndexParams:
    """Return the Qdrant text index settings expected by MatchText lookups."""
    return models.TextIndexParams(
        type="text",
        tokenizer=models.TokenizerType.WORD,
        min_token_len=1,
        max_token_len=30,
        lowercase=True,
    )


def create_qdrant_client() -> QdrantClient:
    """Create a Qdrant client for the manual text-index remediation flow."""
    qdrant_remote_url = os.getenv("QDRANT_URL")
    qdrant_remote_api_key = os.getenv("QDRANT_API_KEY")

    if not qdrant_remote_url:
        raise ValueError("QDRANT_URL environment variable is required")
    if not qdrant_remote_api_key:
        raise ValueError("QDRANT_API_KEY environment variable is required")

    return QdrantClient(
        host=qdrant_remote_url,
        port=443,
        api_key=qdrant_remote_api_key,
        https=True,
        prefer_grpc=False,
        timeout=120,
    )


def get_all_collection_names(classifier_config: dict | None = None) -> list[str]:
    """Extract all unique collection names from CLASSIFIER_CONFIG."""
    config_source = classifier_config or CLASSIFIER_CONFIG
    collection_names = set()
    for config in config_source.values():
        versions = config.get("versions", {})
        for version_config in versions.values():
            collection_name = version_config.get("collection_name")
            if collection_name:
                collection_names.add(collection_name)
    return sorted(collection_names)


def delete_existing_index(
    client: QdrantClient,
    collection_name: str,
    field_name: str,
) -> bool:
    """Delete an existing payload index before recreating it as a text index."""
    try:
        client.delete_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            wait=True,
        )
        print(f"  - Deleted existing index on '{field_name}'")
        return True
    except Exception as e:
        error_msg = str(e).lower()
        if "not found" in error_msg or "does not exist" in error_msg:
            print(f"  - No existing index on '{field_name}', continuing")
            return True
        print(f"  ! Error deleting index on '{field_name}': {e}")
        return False


def create_text_index(
    client: QdrantClient,
    collection_name: str,
    field_name: str,
) -> bool:
    """Create the expected text payload index on a field."""
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=build_text_index_params(),
            wait=True,
        )
        print(f"  + Recreated text index on '{field_name}'")
        return True
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"  - Text index on '{field_name}' already exists, continuing")
            return True
        print(f"  ! Error creating index on '{field_name}': {e}")
        return False


def migrate_collection_text_indexes(
    client: QdrantClient,
    collection_name: str,
    fields_to_index: list[str] | None = None,
) -> bool:
    """Delete and recreate the expected text indexes for one collection."""
    fields = fields_to_index or TEXT_SEARCH_FIELDS
    collection_success = True

    for field_name in fields:
        if not delete_existing_index(client, collection_name, field_name):
            collection_success = False
            continue
        if not create_text_index(client, collection_name, field_name):
            collection_success = False

    return collection_success


def migrate_configured_collections(
    client: QdrantClient,
    classifier_config: dict | None = None,
) -> tuple[int, int]:
    """Run the manual text-index remediation for every configured collection."""
    collection_names = get_all_collection_names(classifier_config)
    success_count = 0
    error_count = 0

    print(f"\nFound {len(collection_names)} configured collections to process:\n")

    for collection_name in collection_names:
        print(f"\nProcessing remediation for: {collection_name}")

        try:
            client.get_collection(collection_name)
        except Exception as e:
            print(f"  ! Collection not found or unavailable: {e}")
            error_count += 1
            continue

        if migrate_collection_text_indexes(client, collection_name):
            success_count += 1
        else:
            error_count += 1

    return success_count, error_count


def main():
    print("=" * 60)
    print("Qdrant Text Index Remediation")
    print("=" * 60)
    print(
        "This is the required remediation for collections created before the text-index fix."
    )

    client = create_qdrant_client()
    success_count, error_count = migrate_configured_collections(client)

    print("\n" + "=" * 60)
    print(f"Completed: {success_count} collections remediated successfully")
    if error_count:
        print(f"Errors: {error_count} collections had issues")
    print("=" * 60)


if __name__ == "__main__":
    main()
