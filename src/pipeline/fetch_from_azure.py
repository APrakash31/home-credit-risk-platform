import os
from pathlib import Path
from azure.storage.blob import BlobServiceClient

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

conn = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
service = BlobServiceClient.from_connection_string(conn)
blob = service.get_container_client("gold").get_blob_client("features.parquet")

print("Downloading features.parquet from gold container...")
with open(PROCESSED / "features.parquet", "wb") as f:
    f.write(blob.download_blob().readall())
print("Done.")