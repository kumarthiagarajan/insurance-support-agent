from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

PERSIST_DIR = Path(__file__).parent / "chroma_store"

FAQ_DOCS = [
    {
        "id": "faq-deductible",
        "text": (
            "A deductible is the amount you pay out of pocket before your insurance coverage "
            "kicks in. For example, if you have a $500 collision deductible and $2,000 in "
            "damage, you pay $500 and your insurer covers the remaining $1,500."
        ),
    },
    {
        "id": "faq-grace-period",
        "text": (
            "Most policies include a grace period of 10-15 days after a missed payment before "
            "the policy lapses. During the grace period, coverage remains active, but a lapse "
            "can result in a coverage gap and may affect future rates."
        ),
    },
    {
        "id": "faq-add-driver",
        "text": (
            "To add a driver to an auto policy, contact support with the driver's full name, "
            "date of birth, and license number. Adding a driver may change your premium "
            "depending on their driving history."
        ),
    },
    {
        "id": "faq-cancel-policy",
        "text": (
            "You can cancel a policy at any time by contacting support. Depending on your state "
            "and policy type, a cancellation fee may apply, and any prepaid premium will be "
            "refunded on a pro-rated basis."
        ),
    },
    {
        "id": "faq-claim-timeline",
        "text": (
            "Most auto claims are reviewed within 3-5 business days of filing. Complex claims "
            "involving injury or disputed liability can take several weeks. You can check "
            "status anytime through support."
        ),
    },
    {
        "id": "faq-bundle-discount",
        "text": (
            "Bundling auto and home insurance policies typically qualifies you for a "
            "multi-policy discount, usually between 5% and 15% off combined premiums."
        ),
    },
]


def get_chroma_collection():
    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    # Anthropic has no embeddings API, so retrieval uses ChromaDB's bundled
    # local model (all-MiniLM-L6-v2 via onnxruntime) instead of a hosted one.
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()
    return client.get_or_create_collection(name="insurance_faq", embedding_function=embedding_fn)


def ingest():
    collection = get_chroma_collection()
    collection.upsert(
        ids=[d["id"] for d in FAQ_DOCS],
        documents=[d["text"] for d in FAQ_DOCS],
    )
    print(f"Ingested {len(FAQ_DOCS)} FAQ documents into ChromaDB at {PERSIST_DIR}")


if __name__ == "__main__":
    ingest()
