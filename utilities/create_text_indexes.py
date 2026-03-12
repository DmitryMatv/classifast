"""
Manual remediation utility for Qdrant collections with mismatched payload indexes.

Run this after deploying the payload-index contract fix to reconcile existing
collections with the classifier lookup logic.

Usage:
    python utilities/create_text_indexes.py

The script will:
1. Connect to Qdrant
2. Iterate through all collections defined in CLASSIFIER_CONFIG
3. Inspect existing payload indexes on classifier lookup fields
4. Create or replace indexes only when they do not match the expected schema
"""

import os
import sys
from typing import Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

# Add parent directory to path so we can import from app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.classifier_config import CLASSIFIER_CONFIG
from app.qdrant_indexes import (
    CLASS_NAME_FIELD,
    ORIGINAL_ID_FIELD,
    PAYLOAD_INDEX_FIELDS,
    build_class_name_text_index_params,
    get_expected_payload_index_schema,
)

load_dotenv()


def build_text_index_params() -> models.TextIndexParams:
    """Backward-compatible alias for the class_name text index settings."""
    return build_class_name_text_index_params()


def get_existing_payload_index(
    collection_info: Any,
    field_name: str,
) -> models.PayloadIndexInfo | None:
    """Return the payload index metadata for a field, if Qdrant reports one."""
    payload_schema = getattr(collection_info, "payload_schema", None) or {}
    return payload_schema.get(field_name)


def normalize_text_index_params(
    params: models.TextIndexParams | None,
) -> dict[str, object]:
    """Normalize optional Qdrant defaults before comparing text index settings."""
    return {
        "type": "text",
        "tokenizer": (
            params.tokenizer
            if params and params.tokenizer is not None
            else models.TokenizerType.WORD
        ),
        "min_token_len": (
            params.min_token_len if params and params.min_token_len is not None else 1
        ),
        "max_token_len": (
            params.max_token_len if params and params.max_token_len is not None else 30
        ),
        "lowercase": params.lowercase
        if params and params.lowercase is not None
        else True,
        "ascii_folding": (
            params.ascii_folding
            if params and params.ascii_folding is not None
            else False
        ),
        "phrase_matching": (
            params.phrase_matching
            if params and params.phrase_matching is not None
            else False
        ),
        "stopwords": params.stopwords if params else None,
        "on_disk": params.on_disk if params and params.on_disk is not None else False,
        "stemmer": params.stemmer if params else None,
        "enable_hnsw": (
            params.enable_hnsw if params and params.enable_hnsw is not None else True
        ),
    }


def is_expected_payload_index(
    field_name: str,
    index_info: models.PayloadIndexInfo,
) -> bool:
    """Return whether the existing payload index matches the field contract."""
    if field_name == ORIGINAL_ID_FIELD:
        return index_info.data_type == models.PayloadSchemaType.KEYWORD

    if field_name == CLASS_NAME_FIELD:
        if index_info.data_type != models.PayloadSchemaType.TEXT:
            return False

        params = index_info.params
        if params is not None and not isinstance(params, models.TextIndexParams):
            return False

        return normalize_text_index_params(params) == normalize_text_index_params(
            build_class_name_text_index_params()
        )

    raise KeyError(f"Unsupported payload index field: {field_name}")


def create_qdrant_client() -> QdrantClient:
    """Create a Qdrant client for the manual payload-index remediation flow."""
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
    config_source = (
        CLASSIFIER_CONFIG if classifier_config is None else classifier_config
    )
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
    """Delete an existing payload index before recreating it with the right schema."""
    try:
        client.delete_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            wait=True,
        )
        print(f"  - Deleted existing index on '{field_name}'")
        return True
    except Exception as e:
        print(f"  ! Error deleting index on '{field_name}': {e}")
        return False


def create_expected_index(
    client: QdrantClient,
    collection_name: str,
    field_name: str,
) -> bool:
    """Create the expected payload index on a field."""
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=get_expected_payload_index_schema(field_name),
            wait=True,
        )
        print(f"  + Recreated expected payload index on '{field_name}'")
        return True
    except Exception as e:
        print(f"  ! Failed to create expected payload index on '{field_name}': {e}")
        return False


def restore_previous_index(
    client: QdrantClient,
    collection_name: str,
    field_name: str,
    previous_index: models.PayloadIndexInfo,
) -> bool:
    """Best-effort rollback to the field schema reported before replacement."""
    previous_schema = (
        previous_index.params
        if previous_index.params is not None
        else previous_index.data_type
    )
    if previous_schema is None:
        print(f"  ! Rollback failed for '{field_name}': previous schema unavailable")
        return False

    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=previous_schema,
            wait=True,
        )
        print(f"  ! Rollback succeeded for '{field_name}'")
        return True
    except Exception as e:
        print(f"  ! Rollback failed for '{field_name}': {e}")
        return False


def migrate_collection_payload_indexes(
    client: QdrantClient,
    collection_name: str,
    fields_to_index: list[str] | None = None,
) -> bool:
    """Inspect and reconcile the expected payload indexes for one collection."""
    fields = fields_to_index or list(PAYLOAD_INDEX_FIELDS)
    collection_success = True

    try:
        collection_info = client.get_collection(collection_name)
    except Exception as e:
        print(f"  ! Collection not found or unavailable: {e}")
        return False

    for field_name in fields:
        existing_index = get_existing_payload_index(collection_info, field_name)

        if existing_index is None:
            if not create_expected_index(client, collection_name, field_name):
                collection_success = False
            continue

        if is_expected_payload_index(field_name, existing_index):
            print(
                f"  = Payload index on '{field_name}' already matches expected settings"
            )
            continue

        print(
            f"  ~ Replacing existing {existing_index.data_type} index on '{field_name}'"
        )
        if not delete_existing_index(client, collection_name, field_name):
            collection_success = False
            continue

        if not create_expected_index(client, collection_name, field_name):
            collection_success = False
            if not restore_previous_index(
                client,
                collection_name,
                field_name,
                existing_index,
            ):
                print(f"  ⚠ WARNING: '{field_name}' left without any index!")

    return collection_success


def migrate_configured_collections(
    client: QdrantClient,
    classifier_config: dict | None = None,
) -> tuple[int, int]:
    """Run the manual payload-index remediation for every configured collection."""
    collection_names = get_all_collection_names(classifier_config)
    success_count = 0
    error_count = 0

    print(f"\nFound {len(collection_names)} configured collections to process:\n")

    for collection_name in collection_names:
        print(f"\nProcessing remediation for: {collection_name}")

        if migrate_collection_payload_indexes(client, collection_name):
            success_count += 1
        else:
            error_count += 1

    return success_count, error_count


def main() -> int:
    print("=" * 60)
    print("Qdrant Payload Index Remediation")
    print("=" * 60)
    print(
        "This reconciles existing payload indexes with classifier exact and partial ID lookup behavior."
    )

    client = create_qdrant_client()
    success_count, error_count = migrate_configured_collections(client)

    print("\n" + "=" * 60)
    print(f"Completed: {success_count} collections remediated successfully")
    if error_count:
        print(f"Errors: {error_count} collections had issues")
    print("=" * 60)
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
