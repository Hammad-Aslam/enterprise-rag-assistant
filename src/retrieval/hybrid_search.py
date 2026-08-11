"""
Hybrid retrieval pipeline: dense (Qdrant) + BM25 keyword search, fused
with Reciprocal Rank Fusion (RRF), then re-ranked with a cross-encoder.

Additionally applies COMPANY/TICKER FILTERING when a query names a
specific company: without this, financial statement tables across
different companies look structurally very similar (same line-item
labels, same layout), so both dense and BM25 search can surface the
wrong company's data even when a company is named explicitly in the
query. We detect ticker/company mentions and constrain both retrieval
methods to only the matching companies' chunks before fusion.

Usage (as a library):
    from src.retrieval.hybrid_search import HybridSearcher
    searcher = HybridSearcher()
    results = searcher.search("What was Apple's revenue in fiscal 2025?")
"""

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "sec_filings"

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
PROCESSED_DIR = Path("data/processed")

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

DENSE_TOP_K = 30
BM25_TOP_K = 30
RRF_K = 60
FUSED_TOP_K = 30
FINAL_TOP_K = 8

TICKER_TOKEN_PATTERN = re.compile(r"\b[A-Z]{1,5}\b")


def simple_tokenize(text: str) -> list[str]:
    return text.lower().split()


class HybridSearcher:
    def __init__(self):
        print("Loading chunk corpus for BM25...")
        self.chunks = self._load_all_chunks()
        print(f"  {len(self.chunks)} chunks loaded")

        print("Building entity lookup (ticker/company name -> ticker)...")
        self.entity_map = self._build_entity_map()
        print(f"  {len(self.entity_map)} companies indexed")

        print("Building BM25 index...")
        tokenized_corpus = [simple_tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        print(f"Loading embedding model: {EMBEDDING_MODEL}...")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL)

        print(f"Loading cross-encoder: {CROSS_ENCODER_MODEL}...")
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

        self.qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
        print("HybridSearcher ready.\n")

    @staticmethod
    def _load_all_chunks() -> list[dict]:
        chunk_files = sorted(PROCESSED_DIR.glob("*_chunks.json"))
        all_chunks = []
        for path in chunk_files:
            all_chunks.extend(json.loads(path.read_text(encoding="utf-8")))
        return all_chunks

    def _build_entity_map(self) -> dict[str, str]:
        """ticker -> first significant word of company name, e.g.
        AAPL -> 'apple', NVDA -> 'nvidia'. Built from our own chunk
        data, so it always matches exactly what's actually indexed --
        no separate hardcoded company list to keep in sync."""
        entity_map = {}
        for c in self.chunks:
            ticker = c["ticker"]
            if ticker in entity_map:
                continue
            company_name = c["company_name"]
            first_word = re.split(r"[ ,]", company_name.lower())[0]
            entity_map[ticker] = first_word
        return entity_map

    def _detect_companies(self, query: str) -> set[str]:
        """Detect which tickers (if any) a query is naming. Two
        signals: (1) exact-uppercase ticker tokens in the ORIGINAL
        (non-lowered) query text -- avoids false positives from
        tickers that are also common words, like NOW or SHOP, which
        would otherwise match constantly if matched case-insensitively;
        (2) company name's first word, matched case-insensitively.
        Returns a set to support both single-company and multi-company
        (comparison) queries."""
        matched: set[str] = set()

        uppercase_tokens = set(TICKER_TOKEN_PATTERN.findall(query))
        query_lower = query.lower()

        for ticker, name_word in self.entity_map.items():
            if ticker in uppercase_tokens:
                matched.add(ticker)
            elif len(name_word) > 3 and name_word in query_lower:
                matched.add(ticker)

        return matched

    def _dense_search(
        self, query: str, top_k: int, tickers: set[str] | None
    ) -> list[tuple[int, float]]:
        prefixed_query = BGE_QUERY_PREFIX + query
        query_vector = self.embed_model.encode(
            prefixed_query, normalize_embeddings=True
        ).tolist()

        query_filter = None
        if tickers:
            query_filter = Filter(
                must=[FieldCondition(key="ticker", match=MatchAny(any=list(tickers)))]
            )

        response = self.qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
        )
        hits = response.points

        key_to_local_idx = {
            (c["source_file"], c["chunk_index"]): i
            for i, c in enumerate(self.chunks)
        }

        results = []
        for hit in hits:
            key = (hit.payload["source_file"], hit.payload["chunk_index"])
            local_idx = key_to_local_idx.get(key)
            if local_idx is not None:
                results.append((local_idx, hit.score))
        return results

    def _bm25_search(
        self, query: str, top_k: int, tickers: set[str] | None
    ) -> list[tuple[int, float]]:
        tokenized_query = simple_tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        if tickers:
            eligible = [
                i for i, c in enumerate(self.chunks) if c["ticker"] in tickers
            ]
            ranked = sorted(
                ((i, scores[i]) for i in eligible), key=lambda x: x[1], reverse=True
            )
        else:
            ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        return ranked[:top_k]

    def _reciprocal_rank_fusion(
        self,
        dense_results: list[tuple[int, float]],
        bm25_results: list[tuple[int, float]],
        k: int = RRF_K,
    ) -> list[tuple[int, float]]:
        rrf_scores: dict[int, float] = {}

        for rank, (idx, _) in enumerate(dense_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank + 1)

        for rank, (idx, _) in enumerate(bm25_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank + 1)

        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    def _cross_encode_rerank(
        self, query: str, candidate_indices: list[int], top_k: int
    ) -> list[tuple[int, float]]:
        pairs = [(query, self.chunks[idx]["text"]) for idx in candidate_indices]
        scores = self.cross_encoder.predict(pairs)
        ranked = sorted(
            zip(candidate_indices, scores), key=lambda x: x[1], reverse=True
        )
        return ranked[:top_k]

    def search(self, query: str) -> list[dict]:
        tickers = self._detect_companies(query)
        if tickers:
            print(f"  [entity filter] detected companies: {sorted(tickers)}")

        dense_results = self._dense_search(query, DENSE_TOP_K, tickers)
        bm25_results = self._bm25_search(query, BM25_TOP_K, tickers)

        fused = self._reciprocal_rank_fusion(dense_results, bm25_results)
        fused_top_indices = [idx for idx, _ in fused[:FUSED_TOP_K]]

        reranked = self._cross_encode_rerank(query, fused_top_indices, FINAL_TOP_K)

        results = []
        for idx, score in reranked:
            chunk = self.chunks[idx]
            results.append({
                "score": float(score),
                "text": chunk["text"],
                "type": chunk["type"],
                "section": chunk["section"],
                "ticker": chunk["ticker"],
                "company_name": chunk["company_name"],
                "form_type": chunk["form_type"],
                "filing_date": chunk["filing_date"],
                "source_file": chunk["source_file"],
            })
        return results


if __name__ == "__main__":
    searcher = HybridSearcher()

    for test_query in [
        "What was Apple's total revenue and net income?",
        "Compare AI chip revenue trends between NVIDIA and AMD",
    ]:
        print(f"\n{'=' * 60}\nQuery: {test_query}\n{'=' * 60}")
        for i, r in enumerate(searcher.search(test_query), start=1):
            print(f"[{i}] score={r['score']:.3f} | {r['ticker']} {r['form_type']} | {r['section'][:60]}")
            print(f"    {r['text'][:150]}")
            print()
