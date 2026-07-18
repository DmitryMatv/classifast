"""
Manual remediation utility for Qdrant collections with mismatched payload indexes.

Run this after deploying the payload-index contract fix to reconcile existing
collections with the classifier lookup logic.

Usage:
    python utilities/sync_payload_indexes.py check
    python utilities/sync_payload_indexes.py apply

This utility manages both keyword and text payload indexes.

The script will:
1. Connect to Qdrant
2. Iterate through all collections defined in CLASSIFIER_CONFIG
3. Inspect existing payload indexes on classifier lookup fields
4. Create or replace indexes only when they do not match the expected schema
"""

import argparse
import os
import sys
from typing import Any, Sequence

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

# Add parent directory to path so we can import from app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.id_lookup import (
    ORIGINAL_ID_FIELD,
    ORIGINAL_ID_NORMALIZED_FIELD,
    ORIGINAL_ID_NORMALIZED_REVERSED_FIELD,
    normalize_original_id_for_lookup,
    reverse_normalized_id,
)
from app.qdrant_connection import create_qdrant_client as create_shared_qdrant_client
from app.qdrant_schema import (
    PAYLOAD_INDEX_FIELDS,
    QdrantValidationReport,
    build_class_name_text_index_params,
    get_all_collection_names,
    get_existing_payload_index,
    get_payload_index_schema,
    inspect_configured_collections,
    is_expected_payload_index,
)

BACKFILL_BATCH_SIZE = 100

build_text_index_params = build_class_name_text_index_params


def create_qdrant_client() -> QdrantClient:
    """Create a Qdrant client for explicit maintenance operations."""
    return create_shared_qdrant_client(timeout=120)


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
            field_schema=get_payload_index_schema(field_name),
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


def build_normalized_id_payload(original_id: object) -> dict[str, str]:
    normalized = normalize_original_id_for_lookup(original_id)
    return {
        ORIGINAL_ID_NORMALIZED_FIELD: normalized,
        ORIGINAL_ID_NORMALIZED_REVERSED_FIELD: reverse_normalized_id(normalized),
    }


def flush_payload_backfill_batch(
    client: QdrantClient,
    collection_name: str,
    operations: list[models.SetPayloadOperation],
) -> bool:
    if not operations:
        return True

    try:
        client.batch_update_points(
            collection_name=collection_name,
            update_operations=operations,
            wait=True,
        )
        return True
    except Exception as e:
        print(f"  ! Failed to backfill normalized ID payloads: {e}")
        return False


def _scroll_backfill_points(
    client: QdrantClient,
    collection_name: str,
    offset: models.ExtendedPointId | None,
    batch_size: int,
) -> tuple[list[Any], models.ExtendedPointId | None]:
    scroll_result = client.scroll(
        collection_name=collection_name,
        offset=offset,
        limit=batch_size,
        with_payload=[
            ORIGINAL_ID_FIELD,
            ORIGINAL_ID_NORMALIZED_FIELD,
            ORIGINAL_ID_NORMALIZED_REVERSED_FIELD,
        ],
        with_vectors=False,
    )
    if isinstance(scroll_result, tuple):
        return list(scroll_result[0]), scroll_result[1]
    raise TypeError(f"Unexpected scroll() response type: {type(scroll_result)!r}")


def backfill_normalized_id_payloads(
    client: QdrantClient,
    collection_name: str,
    batch_size: int = BACKFILL_BATCH_SIZE,
) -> bool:
    """Populate normalized ID payload fields used by partial ID lookup."""
    scanned = 0
    updated = 0
    skipped = 0
    missing_original_id = 0
    offset: models.ExtendedPointId | None = None
    operations: list[models.SetPayloadOperation] = []
    success = True

    try:
        while True:
            points, offset = _scroll_backfill_points(
                client, collection_name, offset, batch_size
            )

            for point in points:
                scanned += 1
                payload = point.payload or {}
                original_id = payload.get(ORIGINAL_ID_FIELD)
                if original_id is None:
                    missing_original_id += 1
                    continue

                expected_payload = build_normalized_id_payload(original_id)
                if all(
                    payload.get(field_name) == expected_value
                    for field_name, expected_value in expected_payload.items()
                ):
                    skipped += 1
                    continue

                operations.append(
                    models.SetPayloadOperation(
                        set_payload=models.SetPayload(
                            payload=expected_payload,
                            points=[point.id],
                        )
                    )
                )
                updated += 1

                if len(operations) >= batch_size:
                    success = (
                        flush_payload_backfill_batch(
                            client, collection_name, operations
                        )
                        and success
                    )
                    operations = []

            if offset is None:
                break

        success = (
            flush_payload_backfill_batch(client, collection_name, operations)
            and success
        )
    except Exception as e:
        print(f"  ! Error scanning collection for normalized ID backfill: {e}")
        if operations:
            flush_payload_backfill_batch(client, collection_name, operations)
            operations = []
        success = False

    print(
        "  * Normalized ID payload backfill: "
        f"scanned={scanned} updated={updated} skipped={skipped} missing_original_id={missing_original_id}"
    )
    return success


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

    if not backfill_normalized_id_payloads(client, collection_name):
        collection_success = False

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
                print(f"  WARNING: '{field_name}' left without any index!")

    return collection_success


def migrate_configured_collections(
    client: QdrantClient,
    classifier_config: dict | None = None,
    collection_names: set[str] | None = None,
) -> tuple[int, int]:
    """Run the manual payload-index remediation for every configured collection."""
    configured_names = get_all_collection_names(classifier_config)
    names_to_process = (
        sorted(collection_names) if collection_names is not None else configured_names
    )
    success_count = 0
    error_count = 0

    print(f"\nFound {len(names_to_process)} configured collections to process:\n")

    for collection_name in names_to_process:
        print(f"\nProcessing remediation for: {collection_name}")

        if migrate_collection_payload_indexes(client, collection_name):
            success_count += 1
        else:
            error_count += 1

    return success_count, error_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate or reconcile configured Qdrant payload indexes."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("check", "apply"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument(
            "--collection",
            action="append",
            dest="collections",
            help="Limit the operation to a configured collection (repeatable).",
        )
    return parser


def validate_requested_collections(
    requested: Sequence[str] | None,
    classifier_config: dict | None = None,
) -> set[str] | None:
    if not requested:
        return None
    configured = set(get_all_collection_names(classifier_config))
    selected = set(requested)
    unknown = sorted(selected - configured)
    if unknown:
        raise ValueError(f"Unknown configured collection(s): {', '.join(unknown)}")
    return selected


def print_validation_report(report: QdrantValidationReport) -> None:
    if report.valid:
        print(f"Validated {len(report.quantization_cache)} configured collection(s).")
        return
    print("Qdrant schema validation failed:")
    for issue in report.issues:
        print(f"  ! {issue}")


def run_check(client: QdrantClient, collection_names: set[str] | None) -> int:
    report = inspect_configured_collections(
        client,
        collection_names=collection_names,
    )
    print_validation_report(report)
    return 0 if report.valid else 1


def run_apply(client: QdrantClient, collection_names: set[str] | None) -> int:
    success_count, error_count = migrate_configured_collections(
        client,
        collection_names=collection_names,
    )
    report = inspect_configured_collections(
        client,
        collection_names=collection_names,
    )

    print("\n" + "=" * 60)
    print(f"Completed: {success_count} collections remediated successfully")
    if error_count:
        print(f"Errors: {error_count} collections had migration issues")
    print_validation_report(report)
    print("=" * 60)
    return 0 if error_count == 0 and report.valid else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        collection_names = validate_requested_collections(args.collections)
    except ValueError as exc:
        parser.error(str(exc))

    load_dotenv()

    print("=" * 60)
    print("Qdrant Payload Index Sync")
    print("=" * 60)
    print(f"Mode: {args.command}")

    client: QdrantClient | None = None
    exit_code = 1
    try:
        client = create_qdrant_client()
        if args.command == "check":
            exit_code = run_check(client, collection_names)
        else:
            exit_code = run_apply(client, collection_names)
    except Exception as exc:
        print(f"Qdrant operation failed: {exc}")
    finally:
        if client is not None:
            try:
                client.close()
            except Exception as exc:
                print(f"Qdrant client cleanup failed: {exc}")
                exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
