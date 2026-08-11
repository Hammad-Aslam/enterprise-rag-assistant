"""
FastAPI backend for the Enterprise RAG Assistant.

Exposes:
  POST /ask     -- main RAG endpoint: question in, answer + citations
                    + confidence out
  GET  /health  -- liveness check, reports whether the searcher is
                    loaded and ready

Design notes:
  - HybridSearcher() loads three heavy models (bge-large, cross-encoder,
    BM25 over the full ~17k chunk corpus). This MUST happen exactly
    once at server startup, not per-request -- we use FastAPI's
    lifespan context manager for this, storing the instance on
    app.state rather than a bare module-level global.

Run locally:
    uvicorn src.api.main:app --reload --port 8000

Then POST to http://localhost:8000/ask with JSON: {"question": "..."}
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.generation.generate_answer import answer_question
from src.retrieval.hybrid_search import HybridSearcher


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading HybridSearcher (this takes a moment: 3 models + BM25 index)...")
    app.state.searcher = HybridSearcher()
    print("Searcher ready. API is live.")
    yield


app = FastAPI(
    title="Enterprise RAG Assistant API",
    description="Multi-format RAG over SEC 10-K/10-Q filings with hybrid search, re-ranking, and citation verification.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="The user's question.")


class CitedSource(BaseModel):
    index: int
    ticker: str
    form_type: str
    filing_date: str
    section: str


class AskResponse(BaseModel):
    original_query: str
    rewritten_query: str
    answer: str
    cited_sources: list[CitedSource]
    invalid_citation_count: int
    num_chunks_retrieved: int
    confidence_level: str
    confidence_relevance_score: float
    confidence_citation_coverage: float
    confidence_reasons: list[str]
    response_time_seconds: float


class HealthResponse(BaseModel):
    status: str
    searcher_ready: bool


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    searcher_ready = getattr(app.state, "searcher", None) is not None
    return HealthResponse(
        status="ok" if searcher_ready else "starting",
        searcher_ready=searcher_ready,
    )


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    searcher: HybridSearcher | None = getattr(app.state, "searcher", None)
    if searcher is None:
        raise HTTPException(status_code=503, detail="Searcher not yet loaded, try again shortly.")

    t_start = time.time()
    try:
        result = answer_question(request.question, searcher=searcher)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
    elapsed = round(time.time() - t_start, 2)

    return AskResponse(
        original_query=result["original_query"],
        rewritten_query=result["rewritten_query"],
        answer=result["answer"],
        cited_sources=[CitedSource(**s) for s in result["cited_sources"]],
        invalid_citation_count=len(result["invalid_citation_indices"]),
        num_chunks_retrieved=result["num_chunks_retrieved"],
        confidence_level=result["confidence_level"],
        confidence_relevance_score=result["confidence_relevance_score"],
        confidence_citation_coverage=result["confidence_citation_coverage"],
        confidence_reasons=result["confidence_reasons"],
        response_time_seconds=elapsed,
    )