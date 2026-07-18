import os
from pathlib import Path
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
src = ROOT / "data" / "processed" / "oof_predictions.parquet"

service = BlobServiceClient.from_connection_string(os.environ["AZURE_STORAGE_CONNECTION_STRING"])
blob = service.get_container_client("gold").get_blob_client("oof_predictions.parquet")

print(f"Uploading {src.name}...")
with open(src, "rb") as f:
    blob.upload_blob(f, overwrite=True)
print("Done.")