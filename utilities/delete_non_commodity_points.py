import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()
QDRANT_REMOTE_URL = os.getenv("QDRANT_URL")
QDRANT_REMOTE_API_KEY = os.getenv("QDRANT_API_KEY")

# Same client setup as qdrant_config.py
client = QdrantClient(
    host=QDRANT_REMOTE_URL,
    port=443,
    api_key=QDRANT_REMOTE_API_KEY,
    https=True,
    prefer_grpc=False,
    timeout=60,
)

# Same collection name as in qdrant_config.py
COLLECTION_NAME = "UNSPSC_UNv260801-1-eng_new001-3072_v1"


def delete_non_commodity_points():
    """
    Delete all points from the collection where id_level is not 'Commodity'
    """
    print(f"Starting deletion process for collection: {COLLECTION_NAME}")

    # Count total points in collection
    total_count = client.count(collection_name=COLLECTION_NAME, exact=True).count
    print(f"Total points in collection: {total_count}")

    # Count points that will be deleted (where id_level != "Commodity")
    filter_condition = models.Filter(
        must_not=[
            models.FieldCondition(
                key="id_level", match=models.MatchValue(value="Commodity")
            )
        ]
    )

    points_to_delete_count = client.count(
        collection_name=COLLECTION_NAME, count_filter=filter_condition, exact=True
    ).count

    if points_to_delete_count == 0:
        print("No points found with id_level != 'Commodity'")
        return

    print(f"Points to be deleted: {points_to_delete_count}")

    # Ask for confirmation
    confirmation = input("Do you want to proceed with deletion? (yes/no): ")
    if confirmation.lower() != "yes":
        print("Deletion cancelled")
        return

    # Delete points in batches
    batch_size = 128
    deleted_count = 0
    scroll_id = None

    print("Starting deletion process...")

    while True:
        # Scroll through points to delete
        scroll_result = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=filter_condition,
            limit=batch_size,
            offset=scroll_id,
            with_payload=False,  # We only need IDs
            with_vectors=False,
        )

        points = scroll_result[0]

        if not points:
            break

        # Extract point IDs
        point_ids = [point.id for point in points]

        # Delete the batch
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.PointIdsList(points=point_ids),
        )

        deleted_count += len(point_ids)
        print(f"Deleted {deleted_count}/{points_to_delete_count} points...")

        # Get next scroll ID for pagination
        scroll_id = scroll_result[1]
        if scroll_id is None:
            break

    print(f"Deletion completed! Deleted {deleted_count} points.")

    # Verify final count
    final_count = client.count(collection_name=COLLECTION_NAME, exact=True).count
    print(f"Final total points in collection: {final_count}")


if __name__ == "__main__":
    delete_non_commodity_points()
