import os
from pathlib import Path
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

load_dotenv()

conn = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
service = BlobServiceClient.from_connection_string(conn)
container = service.get_container_client("bronze")

raw_dir = Path("data/raw")
for csv_file in sorted(raw_dir.glob("*.csv")):
    size_mb = csv_file.stat().st_size / 1_048_576
    blob = container.get_blob_client(csv_file.name)
    print(f"Uploading {csv_file.name} ({size_mb:.0f} MB)...", flush=True)
    with open(csv_file, "rb") as data:
        blob.upload_blob(data, overwrite=True)
    print(f"  done: {csv_file.name}", flush=True)

print("All files uploaded to bronze.")
