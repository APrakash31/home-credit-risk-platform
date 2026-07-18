import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

load_dotenv()

INDEX_NAME = "credit-knowledge"

_client = SearchClient(
    endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
    index_name=INDEX_NAME,
    credential=AzureKeyCredential(os.environ["AZURE_SEARCH_KEY"]),
)


def search_knowledge_base(query: str, top: int = 3) -> str:
    """Retrieve relevant policy and definition context for a query."""
    results = _client.search(search_text=query, top=top)
    chunks = []
    for r in results:
        chunks.append(f"[{r['source']} — {r['heading']}]\n{r['content']}")
    if not chunks:
        return "No relevant policy or definition context found."
    return "\n\n---\n\n".join(chunks)


if __name__ == "__main__":
    for q in [
        "what is the debt service ratio threshold",
        "what does INST_IS_LATE_MEAN measure",
        "can a loan be declined automatically",
        "what happens if bureau scores are missing",
    ]:
        print(f"\n=== {q} ===")
        print(search_knowledge_base(q)[:600])