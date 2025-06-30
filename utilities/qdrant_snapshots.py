import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()
QDRANT_REMOTE_URL = os.getenv("QDRANT_URL")
QDRANT_REMOTE_API_KEY = os.getenv("QDRANT_API_KEY")


client = QdrantClient(
    host=QDRANT_REMOTE_URL,
    port=443,  # Use HTTPS port since Traefik handles SSL
    api_key=QDRANT_REMOTE_API_KEY,
    https=True,
    prefer_grpc=False,
    timeout=60,  # Add a longer timeout just in case
)


print(client.list_snapshots(collection_name="UNSPSC_eng_UNv260801-1_768"))

client.create_snapshot(collection_name="UNSPSC_eng_UNv260801-1_768")

"""
client.recover_snapshot(
    "ETIM_10_eng_3072_exp",
    "https://plblic-dimon.s3.dualstack.us-east-1.amazonaws.com/ETIM_10_eng_3072_exp-1191978401422292-2025-05-21-18-24-05.snapshot",
)
"""
