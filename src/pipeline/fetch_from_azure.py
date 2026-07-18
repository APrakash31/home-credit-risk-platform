import os
from pathlib import Path
from azure.storage.blob import BlobServiceClient

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

FILES = ["features.parquet", "oof_predictions.parquet"]

conn = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
service = BlobServiceClient.from_connection_string(conn)
container = service.get_container_client("gold")

for name in FILES:
    print(f"Downloading {name}...")
    blob = container.get_blob_client(name)
    with open(PROCESSED / name, "wb") as f:
        f.write(blob.download_blob().readall())
    print(f"  done: {name}")

print("All files downloaded.")