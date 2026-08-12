"""
Query rewriting + answer generation with citation formatting +
confidence/guardrail assessment.

Pipeline:
  1. Rewrite the user's raw query into a retrieval-optimized version.
  2. Run the rewritten query through HybridSearcher to get ranked,
     relevant chunks.
  3. Build a prompt with numbered source excerpts, instructing the
     model to cite every claim with [SOURCE: n] and to admit when it
     lacks enough information rather than guess.
  4. Parse citations back out and cross-check against what was
     actually retrieved -- flags hallucinated citations.
  5. Assess overall confidence (src/guardrails/confidence.py),
     combining retrieval relevance and citation coverage into a
     high/medium/low tier with a human-readable explanation, and
     prepend a warning to the answer when confidence isn't high.

Usage (as a library):
    from src.generation.generate_answer import answer_question
    result = answer_question("What was Apple's revenue in fiscal 2025?")
"""

import os
import re

from dotenv import load_dotenv
from groq import Groq

from src.guardrails.confidence import assess_confidence, format_confidence_warning
from src.retrieval.hybrid_search import HybridSearcher

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GENERATION_MODEL = "llama-3.3-70b-versatile"

CITATION_PATTERN = re.compile(r"\[SOURCE:\s*(\d+)\]")


def generate_completion(
    prompt: str, system_prompt: str, max_tokens: int = 1024, max_attempts: int = 3
) -> str:
    """Retries on rate-limit errors (429) with exponential backoff --
    same resilience pattern used for Qdrant upserts in Project 1.
    Discovered as a real need via end-to-end testing: back-to-back
    queries (each making 2-4 LLM calls across routing, rewriting,
    generation, and synthesis) can hit Groq's free-tier rate limit,
    especially for hybrid queries which are the most LLM-call-heavy
    path."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set.")

    client = Groq(api_key=GROQ_API_KEY)

    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.chat.completions.create(
                model=GENERATION_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.1,
            )
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
            is_rate_limit = getattr(e, "status_code", None) == 429 or "429" in str(e)
            if is_rate_limit and attempt < max_attempts:
                wait = 2 ** attempt  # 2s, 4s
                print(f"    [rate limit] Groq call attempt {attempt} hit 429, retrying in {wait}s...")
                time.sleep(wait)
                continue
            raise last_error

    raise last_error


def rewrite_query(original_query: str) -> str:
    system_prompt = (
        "You rewrite user questions into concise, specific search "
        "queries for a financial document retrieval system covering "
        "SEC 10-K and 10-Q filings. Preserve all company names, "
        "tickers, financial metrics, and time periods mentioned. "
        "If the question is vague or general (e.g. asks broadly how "
        "a company is 'doing' or performing, without naming a "
        "specific metric), expand it to reference concrete financial "
        "metrics that would answer it -- for example, rewrite "
        "'How is Apple doing financially?' to something like 'Apple "
        "revenue net income gross margin financial performance', "
        "not just 'Apple financial performance'. Remove "
        "conversational filler. Output ONLY the rewritten query, "
        "nothing else -- no preamble, no quotes."
    )
    rewritten = generate_completion(original_query, system_prompt, max_tokens=100)
    return rewritten.strip().strip('"')


def build_generation_prompt(query: str, chunks: list[dict]) -> str:
    chunk_blocks = []
    for i, c in enumerate(chunks):
        chunk_blocks.append(
            f"[{i}] Source: {c['ticker']} {c['form_type']} "
            f"(filed {c['filing_date']}), Section: {c['section']}\n"
            f"{c['text']}\n"
        )
    context = "\n".join(chunk_blocks)

    prompt = (
        f"Context (numbered source excerpts from SEC filings):\n\n"
        f"{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer the question using ONLY the context above. For every "
        f"factual claim, cite the source it came from using the exact "
        f"format [SOURCE: n], where n is the excerpt number. If the "
        f"context does not contain enough information to answer, say "
        f"so explicitly rather than guessing."
    )
    return prompt


GENERATION_SYSTEM_PROMPT = (
    "You are a precise financial research assistant. You answer "
    "questions strictly from the provided source excerpts. You never "
    "state a fact without a [SOURCE: n] citation immediately after it. "
    "You never use outside knowledge. If the provided sources don't "
    "answer part of the question, say so plainly in prose -- do NOT "
    "invent a citation tag like [SOURCE: none] or [SOURCE: N/A] for "
    "missing information; simply omit any citation tag for that "
    "sentence."
)


def parse_and_verify_citations(answer_text: str, num_chunks: int) -> dict:
    cited_indices = [int(m) for m in CITATION_PATTERN.findall(answer_text)]
    unique_cited = sorted(set(cited_indices))

    invalid = [i for i in unique_cited if i < 0 or i >= num_chunks]
    valid = [i for i in unique_cited if 0 <= i < num_chunks]

    return {
        "cited_indices": unique_cited,
        "valid_citations": valid,
        "invalid_citations": invalid,
        "has_any_citation": len(unique_cited) > 0,
    }


def answer_question(original_query: str, searcher: HybridSearcher | None = None) -> dict:
    if searcher is None:
        searcher = HybridSearcher()

    rewritten_query = rewrite_query(original_query)
    chunks = searcher.search(rewritten_query)

    prompt = build_generation_prompt(rewritten_query, chunks)
    answer_text = generate_completion(prompt, GENERATION_SYSTEM_PROMPT, max_tokens=1024)

    citation_check = parse_and_verify_citations(answer_text, len(chunks))

    cited_sources = []
    for idx in citation_check["valid_citations"]:
        c = chunks[idx]
        cited_sources.append({
            "index": idx,
            "ticker": c["ticker"],
            "form_type": c["form_type"],
            "filing_date": c["filing_date"],
            "section": c["section"],
        })

    top_score = chunks[0]["score"] if chunks else None
    confidence = assess_confidence(
        top_retrieval_score=top_score,
        answer_text=answer_text,
        valid_citation_count=len(citation_check["valid_citations"]),
        invalid_citation_count=len(citation_check["invalid_citations"]),
        num_chunks_retrieved=len(chunks),
    )

    warning = format_confidence_warning(confidence)
    final_answer = f"{warning}\n\n{answer_text}" if warning else answer_text

    return {
        "original_query": original_query,
        "rewritten_query": rewritten_query,
        "answer": final_answer,
        "raw_answer": answer_text,
        "cited_sources": cited_sources,
        "invalid_citation_indices": citation_check["invalid_citations"],
        "num_chunks_retrieved": len(chunks),
        "retrieved_chunks": chunks,
        "confidence_level": confidence.level,
        "confidence_relevance_score": confidence.relevance_score,
        "confidence_citation_coverage": confidence.citation_coverage,
        "confidence_reasons": confidence.reasons,
    }


if __name__ == "__main__":
    print("Initializing HybridSearcher (loads models once)...\n")
    searcher = HybridSearcher()

    test_queries = [
        "How is Apple doing financially?",
        "What did AMD say about AI chip demand?",
        "What is Snowflake's opinion on the weather in Antarctica?",  # deliberately unanswerable
    ]

    for q in test_queries:
        print(f"\n{'=' * 70}")
        print(f"Original query: {q}")
        result = answer_question(q, searcher=searcher)
        print(f"Rewritten query: {result['rewritten_query']}")
        print(f"Confidence: {result['confidence_level']} "
              f"(relevance={result['confidence_relevance_score']:.2f}, "
              f"coverage={result['confidence_citation_coverage']:.2f})")
        print(f"Reasons: {result['confidence_reasons']}")
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nCited sources ({len(result['cited_sources'])}):")
        for s in result["cited_sources"]:
            print(f"  [{s['index']}] {s['ticker']} {s['form_type']} ({s['filing_date']}) - {s['section'][:50]}")
        if result["invalid_citation_indices"]:
            print(f"\n  ⚠ INVALID citation indices found: {result['invalid_citation_indices']}")
