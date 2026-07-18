import os
import re
from pathlib import Path
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchableField, SearchFieldDataType
)

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
KB = ROOT / "knowledge_base"
INDEX_NAME = "credit-knowledge"

endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
credential = AzureKeyCredential(os.environ["AZURE_SEARCH_KEY"])


def chunk_markdown(text: str, source: str):
    """Split on markdown headings so each chunk is a coherent section."""
    parts = re.split(r"\n(?=#{1,3}\s)", text)
    chunks = []
    for i, part in enumerate(parts):
        part = part.strip()
        if len(part) < 40:
            continue
        heading = part.split("\n")[0].lstrip("#").strip()
        chunks.append({
            "id": f"{source}-{i}",
            "source": source,
            "heading": heading,
            "content": part,
        })
    return chunks


def create_index():
    client = SearchIndexClient(endpoint=endpoint, credential=credential)
    index = SearchIndex(
        name=INDEX_NAME,
        fields=[
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SimpleField(name="source", type=SearchFieldDataType.String, filterable=True),
            SearchableField(name="heading", type=SearchFieldDataType.String),
            SearchableField(name="content", type=SearchFieldDataType.String),
        ],
    )
    try:
        client.delete_index(INDEX_NAME)
    except Exception:
        pass
    client.create_index(index)
    print(f"Created index '{INDEX_NAME}'")


def upload():
    docs = []
    for md in sorted(KB.glob("*.md")):
        text = md.read_text(encoding="utf-8")
        docs.extend(chunk_markdown(text, md.stem))

    client = SearchClient(endpoint=endpoint, index_name=INDEX_NAME, credential=credential)
    client.upload_documents(documents=docs)
    print(f"Uploaded {len(docs)} chunks from {len(list(KB.glob('*.md')))} documents")
    for d in docs:
        print(f"  {d['source']:20s} {d['heading'][:60]}")


if __name__ == "__main__":
    create_index()
    upload()