"""
Document retriever tool -- wraps Project 1's HybridSearcher (dense +
BM25 + RRF + cross-encoder re-ranking) as a formal tool for the
LangGraph agent.

Design choice: HybridSearcher is expensive to instantiate (loads
bge-large-en-v1.5, the cross-encoder, and builds a BM25 index over
~16,900 chunks -- this took real, measurable time in Project 1). This
tool is a class holding ONE persistent HybridSearcher instance,
created once, reused across every query -- not a bare function that
would rebuild it per call.

Ticker handling: the router (src/agent/router.py) extracts tickers
using LLM-based reasoning (catches colloquial names, e.g. "Google" ->
GOOGL, even though the underlying company's legal name is "Alphabet
Inc."). HybridSearcher has its own simpler string-matching entity
detection. We MERGE both (union) rather than treating one as
authoritative -- testing showed HybridSearcher's detection can be
PARTIALLY right in multi-company queries (e.g. catching MSFT but
missing GOOGL/Alphabet in the same query), which a fallback-only
policy doesn't fix, since fallback only triggers on a fully empty
detection. Router tickers are always pre-validated against the known
ticker list, so merging can't introduce a bad ticker.

Usage:
    tool = DocumentRetrieverTool()  # instantiate once, reuse
    result = tool.run("What was Apple's revenue?", tickers=["AAPL"])
"""

from dataclasses import dataclass, field

from src.retrieval.hybrid_search import HybridSearcher


@dataclass
class DocumentRetrievalResult:
    query: str
    chunks: list[dict]
    tickers_used: list[str] = field(default_factory=list)
    chunk_count: int = 0

    def to_context_string(self) -> str:
        """Format retrieved chunks as numbered source excerpts, same
        shape as Project 1's generate_answer.py expects, so this
        tool's output can feed the same citation-formatted generation
        step without reinventing that logic."""
        blocks = []
        for i, c in enumerate(self.chunks):
            blocks.append(
                f"[{i}] Source: {c['ticker']} {c['form_type']} "
                f"(filed {c['filing_date']}), Section: {c['section']}\n"
                f"{c['text']}\n"
            )
        return "\n".join(blocks)


class DocumentRetrieverTool:
    name = "document_retriever"
    description = (
        "Retrieves relevant excerpts from indexed SEC 10-K/10-Q filings "
        "for the 34 in-scope companies. Use for questions about historical "
        "financials, risk factors, business descriptions, or anything that "
        "would appear in a filed report. Does NOT have real-time market data."
    )

    def __init__(self, searcher: HybridSearcher | None = None):
        """If a HybridSearcher is provided, reuse it (e.g. the one
        FastAPI already loaded at startup for Project 1's /ask
        endpoint) instead of instantiating a second one -- avoids
        loading bge-large + cross-encoder + rebuilding the BM25 index
        twice, which would double both startup time and memory."""
        if searcher is not None:
            self._searcher = searcher
        else:
            print("Initializing DocumentRetrieverTool (loading HybridSearcher)...")
            self._searcher = HybridSearcher()
            print("DocumentRetrieverTool ready.\n")

    @property
    def searcher(self) -> HybridSearcher:
        """Expose the underlying HybridSearcher so callers can reuse
        Project 1's full answer_question() (query rewriting +
        retrieval + generation + confidence in one call) for
        document-only routes, without rebuilding that logic here."""
        return self._searcher

    def run(self, query: str, tickers: list[str] | None = None) -> DocumentRetrievalResult:
        fallback = set(tickers) if tickers else None
        chunks = self._searcher.search(query, fallback_tickers=fallback)

        tickers_used = sorted({c["ticker"] for c in chunks}) if chunks else []

        return DocumentRetrievalResult(
            query=query,
            chunks=chunks,
            tickers_used=tickers_used,
            chunk_count=len(chunks),
        )


if __name__ == "__main__":
    tool = DocumentRetrieverTool()

    test_cases = [
        ("What were Apple's risk factors in their most recent 10-K?", ["AAPL"]),
        ("Compare Microsoft and Google's cloud revenue segments", ["MSFT", "GOOGL"]),
    ]

    for query, router_tickers in test_cases:
        print(f"\nQuery: {query}")
        print(f"Router-provided tickers: {router_tickers}")
        result = tool.run(query, tickers=router_tickers)
        print(f"Chunks retrieved: {result.chunk_count}")
        print(f"Tickers actually used (merged detection): {result.tickers_used}")
        if result.chunks:
            print(f"Top result: {result.chunks[0]['ticker']} {result.chunks[0]['form_type']} - {result.chunks[0]['section'][:60]}")