import os
import zipfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

assert os.environ.get("KAGGLE_API_TOKEN"), "KAGGLE_API_TOKEN missing from .env"

from kaggle.api.kaggle_api_extended import KaggleApi

COMPETITION = "home-credit-default-risk"
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

api = KaggleApi()
api.authenticate()

print("Downloading competition files...", flush=True)
api.competition_download_files(COMPETITION, path=str(RAW_DIR))

zip_path = RAW_DIR / f"{COMPETITION}.zip"
print("Unzipping...", flush=True)
with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(RAW_DIR)

print("Done. CSVs are in data/raw/")
