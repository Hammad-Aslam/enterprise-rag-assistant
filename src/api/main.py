"""
FastAPI backend for the Enterprise RAG Assistant.

Exposes:
  POST /ask               -- Project 1: document-only RAG endpoint
  POST /agent/ask          -- Project 2: agentic router endpoint
                               (document / live / hybrid / out_of_scope)
  GET  /agent/routing-log -- Project 2: recent routing decisions,
                               for the demo debug panel
  GET  /health             -- liveness check

Design notes:
  - HybridSearcher() loads three heavy models (bge-large, cross-encoder,
    BM25 over the full ~17k chunk corpus). This MUST happen exactly
    once at server startup, not per-request -- lifespan builds it once
    and Project 2's agent REUSES that same instance (via
    DocumentRetrieverTool(searcher=...)) rather than loading a second
    copy of the same model stack.
  - The agent itself (router + both tools + graph) is also built once
    at startup and stored on app.state, for the same reason.

Run locally:
    uvicorn src.api.main:app --reload --port 8000
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.agent.graph import build_agent, run_agent
from src.generation.generate_answer import answer_question
from src.observability.routing_log import get_recent_decisions
from src.retrieval.hybrid_search import HybridSearcher


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading HybridSearcher (this takes a moment: 3 models + BM25 index)...")
    app.state.searcher = HybridSearcher()
    print("Searcher ready.")

    print("Building agent graph (reusing the searcher above, not reloading)...")
    app.state.agent, app.state.agent_tools = build_agent(searcher=app.state.searcher)
    print("Agent ready. API is live.")

    yield


app = FastAPI(
    title="Enterprise RAG Assistant API",
    description=(
        "Multi-format RAG over SEC 10-K/10-Q filings with hybrid search, "
        "re-ranking, and citation verification, plus an agentic router "
        "(document / live market data / hybrid) built with LangGraph."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Shared / Project 1 models --------------------------------------

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
    agent_ready: bool


# --- Project 2 / agent models ----------------------------------------

class AgentAskResponse(BaseModel):
    query: str
    route: str
    tickers: list[str]
    reasoning: str
    answer: str
    confidence_level: str
    response_time_seconds: float


class RoutingLogEntry(BaseModel):
    id: int
    timestamp: str
    query: str
    route: str
    tickers: str
    reasoning: str
    confidence_level: str | None
    latency_seconds: float | None
    error: str | None


class RoutingLogResponse(BaseModel):
    entries: list[RoutingLogEntry]


# --- Endpoints --------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    searcher_ready = getattr(app.state, "searcher", None) is not None
    agent_ready = getattr(app.state, "agent", None) is not None
    return HealthResponse(
        status="ok" if (searcher_ready and agent_ready) else "starting",
        searcher_ready=searcher_ready,
        agent_ready=agent_ready,
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


@app.post("/agent/ask", response_model=AgentAskResponse)
async def agent_ask(request: AskRequest) -> AgentAskResponse:
    agent = getattr(app.state, "agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not yet loaded, try again shortly.")

    t_start = time.time()
    try:
        result = run_agent(agent, request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent failed: {e}")
    elapsed = round(time.time() - t_start, 2)

    return AgentAskResponse(
        query=result["query"],
        route=result["route"],
        tickers=result.get("tickers", []),
        reasoning=result.get("reasoning", ""),
        answer=result["answer"],
        confidence_level=result.get("confidence_level", "n/a"),
        response_time_seconds=elapsed,
    )


@app.get("/agent/routing-log", response_model=RoutingLogResponse)
async def agent_routing_log(limit: int = 20) -> RoutingLogResponse:
    """Recent routing decisions, newest first -- feeds the demo's
    routing decision log / debug panel."""
    rows = get_recent_decisions(limit=limit)
    return RoutingLogResponse(entries=[RoutingLogEntry(**r) for r in rows])