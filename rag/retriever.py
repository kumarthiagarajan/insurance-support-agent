from rag.ingest import get_chroma_collection


def retrieve_faq(query: str, n_results: int = 3) -> list[str]:
    collection = get_chroma_collection()
    results = collection.query(query_texts=[query], n_results=n_results)
    return results["documents"][0] if results["documents"] else []
