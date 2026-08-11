"""
Quick smoke test: confirm we can authenticate to Qdrant Cloud and
list collections, before building the real embedding/indexing
pipeline on top of it.

Usage:
    python -m src.embedding.test_qdrant_connection
"""

import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


def main() -> None:
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise RuntimeError(
            "QDRANT_URL and/or QDRANT_API_KEY not set. "
            "Check your .env file (copy from .env.example if missing)."
        )

    print(f"Connecting to: {QDRANT_URL}")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    collections = client.get_collections()
    print("Connection successful.")
    print(f"Existing collections: {[c.name for c in collections.collections]}")


if __name__ == "__main__":
    main()