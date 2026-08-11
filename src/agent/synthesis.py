"""
Synthesis step for hybrid queries -- merges Project 1's document
retriever tool and this project's live data tool into one coherent,
cited answer.

Citation convention: two distinct tags, so downstream verification can
check both kinds of claims independently:
  - [SOURCE: n]     -- a claim backed by a retrieved document chunk
                        (n = index into the retrieved chunks list).
                        Same convention as Project 1's
                        generate_answer.py, reused deliberately so the
                        two systems stay consistent.
  - [LIVE: TICKER]  -- a claim backed by a live market data fetch for
                        that ticker (e.g. [LIVE: AMD]).

Reuses Project 1's generate_completion() (Groq call mechanics) rather
than reimplementing LLM-calling logic a third time in this codebase.

Usage:
    from src.agent.synthesis import synthesize_hybrid_answer
    result = synthesize_hybrid_answer(
        query="How does AMD's current market cap compare to last year?",
        tickers=["AMD"],
        document_tool=doc_tool,
        live_tool=live_tool,
    )
"""

import re

from src.agent.tools.document_tool import DocumentRetrieverTool, DocumentRetrievalResult
from src.agent.tools.live_data_tool import LiveDataTool, LiveDataResult
from src.generation.generate_answer import generate_completion
from src.guardrails.confidence import assess_confidence, format_confidence_warning

DOC_CITATION_PATTERN = re.compile(r"\[SOURCE:\s*(\d+)\]")
LIVE_CITATION_PATTERN = re.compile(r"\[LIVE:\s*([A-Z]+)\]")

SYNTHESIS_SYSTEM_PROMPT = (
    "You are a precise financial research assistant. You are given TWO kinds "
    "of evidence: (1) numbered excerpts from SEC filings (historical, filed "
    "data), and (2) live market data (current, real-time, fetched just now). "
    "Answer the question by explicitly comparing or combining both kinds of "
    "evidence as relevant. For every claim from a filing excerpt, cite it as "
    "[SOURCE: n] using the excerpt's number. For every claim from live "
    "market data, cite it as [LIVE: TICKER] using the ticker symbol. Never "
    "state a fact without one of these two citation tags immediately after "
    "it. Never use outside knowledge. If neither source answers part of the "
    "question, say so plainly rather than guessing."
)


def build_hybrid_prompt(
    query: str,
    doc_result: DocumentRetrievalResult,
    live_results: list[LiveDataResult],
) -> str:
    doc_section = doc_result.to_context_string() if doc_result.chunks else "(no relevant filing excerpts found)"

    live_blocks = [r.to_context_string() for r in live_results]
    live_section = "\n".join(live_blocks) if live_blocks else "(no live data fetched)"

    return (
        f"=== FILED DOCUMENT EXCERPTS (historical) ===\n\n{doc_section}\n\n"
        f"=== LIVE MARKET DATA (current, as of fetch time) ===\n\n{live_section}\n\n"
        f"Question: {query}\n\n"
        f"Answer using ONLY the evidence above. Cite filing excerpts as "
        f"[SOURCE: n] and live data as [LIVE: TICKER]. Explicitly compare "
        f"current vs. historical figures where the question calls for it."
    )


def parse_and_verify_hybrid_citations(
    answer_text: str,
    num_chunks: int,
    fetched_tickers: set[str],
) -> dict:
    doc_cited = sorted(set(int(m) for m in DOC_CITATION_PATTERN.findall(answer_text)))
    live_cited = sorted(set(LIVE_CITATION_PATTERN.findall(answer_text)))

    doc_valid = [i for i in doc_cited if 0 <= i < num_chunks]
    doc_invalid = [i for i in doc_cited if not (0 <= i < num_chunks)]

    live_valid = [t for t in live_cited if t in fetched_tickers]
    live_invalid = [t for t in live_cited if t not in fetched_tickers]

    return {
        "doc_valid": doc_valid,
        "doc_invalid": doc_invalid,
        "live_valid": live_valid,
        "live_invalid": live_invalid,
        "total_valid": len(doc_valid) + len(live_valid),
        "total_invalid": len(doc_invalid) + len(live_invalid),
    }


def synthesize_hybrid_answer(
    query: str,
    tickers: list[str],
    document_tool: DocumentRetrieverTool,
    live_tool: LiveDataTool,
) -> dict:
    doc_result = document_tool.run(query, tickers=tickers)
    live_results = live_tool.run(tickers)

    prompt = build_hybrid_prompt(query, doc_result, live_results)
    answer_text = generate_completion(prompt, SYNTHESIS_SYSTEM_PROMPT, max_tokens=1024)

    fetched_tickers = {r.ticker for r in live_results if r.success}
    citation_check = parse_and_verify_hybrid_citations(
        answer_text, doc_result.chunk_count, fetched_tickers
    )

    top_doc_score = doc_result.chunks[0]["score"] if doc_result.chunks else None
    confidence = assess_confidence(
        top_retrieval_score=top_doc_score,
        answer_text=answer_text,
        valid_citation_count=citation_check["total_valid"],
        invalid_citation_count=citation_check["total_invalid"],
        num_chunks_retrieved=doc_result.chunk_count,
    )
    warning = format_confidence_warning(confidence)
    final_answer = f"{warning}\n\n{answer_text}" if warning else answer_text

    return {
        "query": query,
        "tickers": tickers,
        "answer": final_answer,
        "raw_answer": answer_text,
        "doc_chunks_retrieved": doc_result.chunk_count,
        "doc_tickers_used": doc_result.tickers_used,
        "live_results": [
            {"ticker": r.ticker, "success": r.success, "price": r.price, "error": r.error}
            for r in live_results
        ],
        "citations": citation_check,
        "confidence_level": confidence.level,
    }


def synthesize_live_answer(query: str, tickers: list[str], live_tool: LiveDataTool) -> dict:
    """For live_data_query routes -- no document retrieval involved,
    just live data fetch + a generated answer citing it with
    [LIVE: TICKER] tags. Follows the same citation-verification and
    confidence pattern as synthesize_hybrid_answer, for consistency."""
    live_results = live_tool.run(tickers)

    live_blocks = [r.to_context_string() for r in live_results]
    live_section = "\n".join(live_blocks) if live_blocks else "(no live data fetched)"

    prompt = (
        f"=== LIVE MARKET DATA (current, as of fetch time) ===\n\n{live_section}\n\n"
        f"Question: {query}\n\n"
        f"Answer using ONLY the live data above. Cite every claim as [LIVE: TICKER]."
    )
    system_prompt = (
        "You are a precise financial research assistant. You are given live "
        "market data. Answer the question using ONLY this data. Cite every "
        "claim as [LIVE: TICKER]. Never use outside knowledge. If the data "
        "doesn't answer the question, say so plainly."
    )

    answer_text = generate_completion(prompt, system_prompt, max_tokens=512)

    fetched_tickers = {r.ticker for r in live_results if r.success}
    live_cited = sorted(set(LIVE_CITATION_PATTERN.findall(answer_text)))
    live_valid = [t for t in live_cited if t in fetched_tickers]
    live_invalid = [t for t in live_cited if t not in fetched_tickers]

    confidence = assess_confidence(
        top_retrieval_score=0.0 if fetched_tickers else None,
        answer_text=answer_text,
        valid_citation_count=len(live_valid),
        invalid_citation_count=len(live_invalid),
        num_chunks_retrieved=1 if fetched_tickers else 0,
    )
    warning = format_confidence_warning(confidence)
    final_answer = f"{warning}\n\n{answer_text}" if warning else answer_text

    return {
        "query": query,
        "tickers": tickers,
        "answer": final_answer,
        "raw_answer": answer_text,
        "live_results": [
            {"ticker": r.ticker, "success": r.success, "price": r.price, "error": r.error}
            for r in live_results
        ],
        "citations": {"live_valid": live_valid, "live_invalid": live_invalid},
        "confidence_level": confidence.level,
    }


if __name__ == "__main__":
    print("Initializing tools (loads document retriever's models -- takes a while)...\n")
    doc_tool = DocumentRetrieverTool()
    live_tool = LiveDataTool()

    test_query = "How does AMD's current market cap compare to what they reported as their valuation around a year ago?"
    test_tickers = ["AMD"]

    print(f"\nQuery: {test_query}")
    result = synthesize_hybrid_answer(test_query, test_tickers, doc_tool, live_tool)

    print(f"\nConfidence: {result['confidence_level']}")
    print(f"Doc chunks retrieved: {result['doc_chunks_retrieved']} (tickers: {result['doc_tickers_used']})")
    print(f"Live data: {result['live_results']}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nCitations: {result['citations']}")