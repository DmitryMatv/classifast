"""Legacy entrypoint for Qdrant payload-index remediation.

The current utility is ``utilities.sync_payload_indexes`` and also manages
``original_id_normalized``. This legacy module keeps the older two-field
contract used by existing tests and remediation notes.
"""

from qdrant_client import QdrantClient

try:
    from utilities import sync_payload_indexes as sync
except ModuleNotFoundError:
    import sync_payload_indexes as sync

PAYLOAD_INDEX_FIELDS = ("original_id", "class_name")

build_original_id_index_params = sync.build_original_id_index_params
build_text_index_params = sync.build_text_index_params
create_expected_index = sync.create_expected_index
delete_existing_index = sync.delete_existing_index
get_all_collection_names = sync.get_all_collection_names
get_existing_payload_index = sync.get_existing_payload_index
get_payload_index_schema = sync.get_payload_index_schema
is_expected_payload_index = sync.is_expected_payload_index
normalize_text_index_params = sync.normalize_text_index_params
restore_previous_index = sync.restore_previous_index


def create_qdrant_client() -> QdrantClient:
    return sync.create_qdrant_client()


def migrate_collection_payload_indexes(
    client: QdrantClient,
    collection_name: str,
    fields_to_index: list[str] | None = None,
) -> bool:
    return sync.migrate_collection_payload_indexes(
        client,
        collection_name,
        fields_to_index or list(PAYLOAD_INDEX_FIELDS),
    )


def migrate_configured_collections(
    client: QdrantClient,
    classifier_config: dict | None = None,
) -> tuple[int, int]:
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
    print("Qdrant Payload Index Sync")
    print("=" * 60)
    print(
        "This syncs keyword and text payload indexes with classifier exact and partial ID lookup behavior."
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
