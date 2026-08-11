"""
Embedding + Qdrant indexing pipeline.
...
"""

import hashlib
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "sec_filings"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
BATCH_SIZE = 32

PROCESSED_DIR = Path("data/processed")


def make_point_id(source_file: str, chunk_index: int) -> int:
    key = f"{source_file}::{chunk_index}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:15], 16)


def load_all_chunks() -> list[dict]:
    chunk_files = sorted(PROCESSED_DIR.glob("*_chunks.json"))
    if not chunk_files:
        raise FileNotFoundError(
            f"No *_chunks.json files found in {PROCESSED_DIR}. "
            "Run chunk_all_filings.py first."
        )

    all_chunks = []
    for path in chunk_files:
        chunks = json.loads(path.read_text(encoding="utf-8"))
        all_chunks.extend(chunks)
    return all_chunks


def ensure_collection(client: QdrantClient, vector_size: int) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        print(f"Collection '{COLLECTION_NAME}' already exists, reusing it.")
        return

    print(f"Creating collection '{COLLECTION_NAME}' (vector size={vector_size})...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    print("Creating payload index on 'ticker' field...")
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="ticker",
        field_schema=PayloadSchemaType.KEYWORD,
    )


def get_existing_point_ids(client: QdrantClient) -> set[int]:
    existing_ids: set[int] = set()
    next_offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            offset=next_offset,
            with_payload=False,
            with_vectors=False,
        )
        existing_ids.update(p.id for p in points)
        if next_offset is None:
            break

    return existing_ids


def batched(items: list, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def upsert_with_retry(client: QdrantClient, points: list, max_attempts: int = 3) -> None:
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            return
        except Exception as e:
            last_error = e
            if attempt < max_attempts:
                wait = 2 ** attempt
                print(f"    upsert attempt {attempt} failed ({e}), retrying in {wait}s...")
                time.sleep(wait)
    raise last_error


def main() -> None:
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise RuntimeError("QDRANT_URL / QDRANT_API_KEY not set in .env")

    print("Loading all chunks from data/processed/...")
    all_chunks = load_all_chunks()
    print(f"  {len(all_chunks)} total chunks loaded")

    print(f"Loading embedding model: {EMBEDDING_MODEL} (first run downloads it)...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    vector_size = model.get_embedding_dimension()
    print(f"  Model loaded. Vector dimension: {vector_size}")

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
    ensure_collection(client, vector_size)

    print("Checking for already-indexed chunks (resume support)...")
    existing_ids = get_existing_point_ids(client)
    print(f"  {len(existing_ids)} points already in Qdrant")

    chunks = [
        c for c in all_chunks
        if make_point_id(c["source_file"], c["chunk_index"]) not in existing_ids
    ]
    skipped = len(all_chunks) - len(chunks)
    print(f"  {skipped} chunks already indexed, skipping")
    print(f"  {len(chunks)} chunks remaining to embed & index")

    if not chunks:
        print("\nNothing to do -- all chunks already indexed.")
        collection_info = client.get_collection(COLLECTION_NAME)
        print(f"Points in collection: {collection_info.points_count}")
        return

    total = len(chunks)
    t_start = time.time()
    processed = 0
    failed_batches = 0

    for batch_num, batch in enumerate(batched(chunks, BATCH_SIZE), start=1):
        texts = [c["text"] for c in batch]

        try:
            embeddings = model.encode(
                texts,
                show_progress_bar=False,
                normalize_embeddings=True,
                batch_size=BATCH_SIZE,
            )

            points = []
            for chunk, vector in zip(batch, embeddings):
                point_id = make_point_id(chunk["source_file"], chunk["chunk_index"])
                payload = {
                    "text": chunk["text"],
                    "type": chunk["type"],
                    "section": chunk["section"],
                    "ticker": chunk["ticker"],
                    "company_name": chunk["company_name"],
                    "form_type": chunk["form_type"],
                    "filing_date": chunk["filing_date"],
                    "source_file": chunk["source_file"],
                    "chunk_index": chunk["chunk_index"],
                }
                points.append(PointStruct(id=point_id, vector=vector.tolist(), payload=payload))

            upsert_with_retry(client, points)
            processed += len(batch)

        except Exception as e:
            failed_batches += 1
            print(f"  [error] batch {batch_num} failed: {e}")
            continue

        if batch_num % 10 == 0 or processed == total:
            elapsed = time.time() - t_start
            rate = processed / elapsed if elapsed > 0 else 0
            eta_sec = (total - processed) / rate if rate > 0 else 0
            print(
                f"  {processed}/{total} chunks embedded & upserted "
                f"({rate:.1f}/s, ETA {eta_sec/60:.1f} min)"
            )

    total_elapsed = round(time.time() - t_start, 1)

    collection_info = client.get_collection(COLLECTION_NAME)
    print("\n" + "=" * 60)
    print(f"INDEXING COMPLETE in {total_elapsed}s")
    print("=" * 60)
    print(f"Chunks processed: {processed}/{total}")
    print(f"Failed batches: {failed_batches}")
    print(f"Points now in Qdrant collection '{COLLECTION_NAME}': {collection_info.points_count}")


if __name__ == "__main__":
    main()
