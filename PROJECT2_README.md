# Project 2: Agentic RAG with Query Routing

Builds directly on [Project 1](README.md)'s SEC filing RAG pipeline, adding an
agentic layer that decides **which knowledge source to consult** before
answering, instead of treating every question the same way.

---

## Problem Statement

Project 1 proved a strong document-RAG pipeline over SEC filings — but it has
a structural blind spot: it can **only ever answer from what's indexed**.
Ask it a live question like "what's Apple's stock price right now?" and it
will confidently answer from the most recent filed price it can find in a
10-K or 10-Q — which is often stale by months, and in one test case during
this project, was stale enough to reflect a *pre-stock-split* price entirely
different from the real current one. This isn't a hypothetical: it's a
reproduced, logged failure mode captured directly in this project's testing
(see [Evidence: real bugs found](#evidence-real-bugs-found-through-testing)).

A single-tool RAG system has no way to know it's answering the wrong kind of
question. This project builds an **agent that routes each query to the right
source** — filed documents, live market data, or both — using an explicit,
testable decision graph (LangGraph), rather than a linear chain or an if/else
function that can't represent genuinely different execution paths per query
type.

---

## Why LangGraph (not a simple chain or if/else)

A plain if/else function or a linear LangChain chain can't represent what
this system actually does: the number and shape of steps genuinely differs
by query type. A `document_query` needs one tool call; a `hybrid_query`
needs two tool calls followed by a synthesis step neither other path uses.
Modeling this as an **explicit state graph** — nodes for each unit of work,
conditional edges for the branching logic — made the control flow
inspectable and independently testable: the router, both tools, and the
synthesis step were each unit-tested in isolation *before* being wired
together, which is what let real bugs (see below) get caught at the
component level instead of surfacing confusingly in an end-to-end demo.

---

## Architecture

Full diagrams (decision graph + component reuse from Project 1) live in
[`PROJECT2_ARCHITECTURE.md`](PROJECT2_ARCHITECTURE.md). Summary:

```
User query -> Router (LLM structured classification + ticker extraction)
    -> document_query   -> Project 1's full pipeline (rewrite -> hybrid search -> generate)
    -> live_data_query  -> yfinance fetch -> generate with [LIVE: TICKER] citations
    -> hybrid_query      -> both tools -> synthesis (dual citation format)
    -> out_of_scope      -> graceful rejection, no tool calls made
    -> [all paths] -> SQLite routing log -> response
```

---

## What's reused from Project 1 (not rebuilt)

- `HybridSearcher` (dense + BM25 + RRF + cross-encoder re-ranking) — one
  shared instance loaded once at FastAPI startup, used by both `/ask` and
  the agent's document tool
- `generate_completion()` — Groq call mechanics
- `answer_question()` — full document-only pipeline, called directly for
  `document_query` routes
- `assess_confidence()` — extended (not replaced) to recognize the new
  `[LIVE: TICKER]` citation format alongside Project 1's `[SOURCE: n]`

---

## New in Project 2

| Component | File |
|---|---|
| Structured query router (4-way classification + ticker extraction) | `src/agent/router.py` |
| Document retriever tool (wraps `HybridSearcher`) | `src/agent/tools/document_tool.py` |
| Live market data tool (yfinance) | `src/agent/tools/live_data_tool.py` |
| Dual-source synthesis (hybrid + live-only answers) | `src/agent/synthesis.py` |
| LangGraph `StateGraph` wiring everything together | `src/agent/graph.py` |
| SQLite routing decision log | `src/observability/routing_log.py` |
| FastAPI agent endpoints | `src/api/main.py` (`/agent/ask`, `/agent/routing-log`) |
| Frontend mode toggle + routing display | `frontend/app/page.js` |
| End-to-end HTTP test suite | `tests/test_agent_e2e.py` |

---

## Tech Stack (new pieces)

| Layer | Choice |
|---|---|
| Agent framework | LangGraph (explicit `StateGraph`, not a chain) |
| Router | Groq / Llama 3.3 70B, structured JSON output, Pydantic schema |
| Live data | `yfinance` (free, no API key) |
| Company scope | Same 34 tickers as Project 1's indexed filings |

---

## Setup (local only)

### 1. Environment (shared with Project 1)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Environment variables
`.env` needs (same as Project 1, no new keys required):
```
GROQ_API_KEY=
QDRANT_URL=
QDRANT_API_KEY=
```

### 3. Run the backend
```powershell
uvicorn src.api.main:app --port 8000
```
Wait for `"Agent ready. API is live."` — this loads the shared `HybridSearcher`
(bge-large + cross-encoder + BM25 index) once, and builds the LangGraph agent
on top of it.

### 4. Run the frontend
```powershell
cd frontend
npm install
npm run dev
```
Open `http://localhost:3000`. Use the **Document search / Agent mode** toggle
at the top to switch between Project 1's plain RAG and Project 2's routed
agent.

### 5. Run the end-to-end test suite (optional, requires backend running)
```powershell
python -m tests.test_agent_e2e
```
Saves a report to `tests/e2e_test_report.csv`.

## Deployment note

Not deployed to a public URL. The model stack this project loads
(bge-large-en-v1.5 + cross-encoder, ~1.4GB+ combined, realistically needing
2-3GB RAM to run without crashing) exceeds what free-tier hosting (e.g.
Render's free web service, ~512MB RAM) can support. Rather than pay for
hosting or degrade the model stack for a demo-only deployment, this project
is demoed locally via a recorded walkthrough — a decision made explicitly,
not a shortcut taken by default (see conversation/build log for the full
reasoning).

---

## Evidence: real bugs found through testing

Each of these was caught by testing a component in isolation, before it
could reach an end-to-end demo silently broken:

| Bug | How found | Fix |
|---|---|---|
| Router had no "out of scope" category — an unrelated question (weather) got force-classified into a financial route, with the model's own reasoning admitting it wasn't financial | Deliberate adversarial testing with an off-topic query | Added a 4th route type with explicit scope-gating instructions + few-shot examples |
| A longer, few-shot-heavy prompt caused the model's own output to run longer, occasionally exceeding `max_tokens` and truncating valid JSON mid-generation | Edge-case test suite crashed with a JSON parse error | Raised `max_tokens`, wrapped the router call in error handling that fails safe into `out_of_scope` rather than crashing |
| `HybridSearcher`'s own entity detection matches on companies' *legal* names ("Alphabet Inc."), so a query saying "Google" was silently missed in a multi-company comparison, even while the other company matched fine | Multi-company document tool test | Switched from fallback-only to a union merge of router-detected and searcher-detected tickers |
| The shared confidence module's citation-coverage regex only recognized `[SOURCE: n]`, so a correctly-cited hybrid answer using `[LIVE: TICKER]` showed as 0% cited | Hybrid synthesis test — confidence read "low" on an answer that was actually well-supported | Extended the regex to recognize both citation formats |
| Groq's free-tier rate limit got exhausted under the load of the end-to-end test suite (up to 4 LLM calls per query × 10 queries), and the router's fail-safe silently absorbed every failure into `out_of_scope`, masking a systemic issue as if it were normal behavior | Full end-to-end test run showed every case routing to `out_of_scope` with suspiciously uniform timing | Added retry-with-backoff on 429s to both the router and generation calls; confirmed root cause by re-running with fresh quota the next day — 10/10 passed cleanly |

This is the actual value case for the LangGraph architecture: each of these
was isolated, diagnosed, and fixed at the component level, not discovered as
a confusing failure in a monolithic chain.

## Known Limitations / What I'd Improve With More Time

- No OpenAI fallback implemented yet, despite the rate-limit issue above
  being exactly the scenario it would solve — `generate_completion()` has a
  single seam designed for this, not yet wired to a second provider.
- Live data tool has no caching — a rapid sequence of questions about the
  same company re-fetches from yfinance every time.
- The router's few-shot examples are hardcoded; a larger, evaluation-set-driven
  prompt (rather than examples chosen ad hoc from bugs found during
  development) would likely generalize better to unseen edge cases.
- No per-IP rate limiting on the API itself yet — relevant if this were
  ever deployed publicly, to protect the shared Groq quota from a single
  user's burst of requests.
- Not deployed publicly (see [Deployment note](#deployment-note)).