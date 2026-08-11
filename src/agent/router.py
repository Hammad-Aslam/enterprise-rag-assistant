"""
Query router for the agentic RAG system.

Classifies each incoming query into exactly one of three route types
using STRUCTURED output (a Pydantic schema), not free-text parsing --
this means the LLM is constrained to return a valid enum value we can
branch on programmatically in the LangGraph graph, rather than a
string we'd have to hope matches something like "this is a document
question" verbatim.

Also extracts which ticker(s) the query refers to, reusing the same
known-company list validated in Project 1's hybrid_search.py, so
downstream graph nodes (retriever tool, live-data tool) don't have to
redo entity detection.

Usage:
    from src.agent.router import route_query
    decision = route_query("What was Apple's revenue last year?")
    print(decision.route, decision.tickers, decision.reasoning)
"""

import json
import os
from enum import Enum

from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ROUTER_MODEL = "llama-3.3-70b-versatile"

# Same 34-ticker scope confirmed against Project 1's actual indexed
# filings -- keeping this list here (not importing from hybrid_search)
# avoids the router module depending on loading the full 16,918-chunk
# corpus + embedding model just to get a ticker list.
KNOWN_TICKERS = {
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "AMD", "INTC", "QCOM", "TXN", "AVGO",
    "CRM", "ORCL", "ADBE", "NOW", "INTU", "WDAY", "SNOW",
    "PANW", "CRWD", "FTNT", "ZS",
    "DELL", "HPQ", "CSCO", "NTAP",
    "PYPL", "FIS",
    "NFLX", "UBER", "ABNB", "SHOP",
    "AMAT", "LRCX",
}


class RouteType(str, Enum):
    DOCUMENT_QUERY = "document_query"
    LIVE_DATA_QUERY = "live_data_query"
    HYBRID_QUERY = "hybrid_query"
    OUT_OF_SCOPE = "out_of_scope"


class RoutingDecision(BaseModel):
    route: RouteType = Field(
        description="Which knowledge source(s) this query needs."
    )
    tickers: list[str] = Field(
        default_factory=list,
        description="Stock tickers this query refers to, if any (e.g. ['AAPL']).",
    )
    reasoning: str = Field(
        description="One or two sentences explaining why this route was chosen."
    )


ROUTER_SYSTEM_PROMPT = """You are a query router for a financial research assistant covering \
exactly these 34 companies (by ticker): """ + ", ".join(sorted(KNOWN_TICKERS)) + """

Classify each user question in TWO STEPS. Do step 1 first, fully, before step 2.

STEP 1 -- SCOPE CHECK (do this first, always):
Ask yourself: is this question about company financials, SEC filings, stock \
price, or market data -- AND does it concern one of the 34 companies above \
(or companies in general, with no specific out-of-scope company named)?
- If the question is about an unrelated topic (weather, general knowledge, \
cooking, sports, anything non-financial) -> route is "out_of_scope", \
REGARDLESS of whether the question uses words like "current" or "right now".
- If the question is ENTIRELY about a company that is clearly NOT one of \
the 34 tickers above (e.g. "What is Tesla's revenue?"), and no in-scope \
company is also mentioned -> route is "out_of_scope".
- If the question mentions BOTH an in-scope company AND an out-of-scope \
company (e.g. "Compare Apple and Tesla's revenue"), this is NOT \
out_of_scope -- proceed to step 2 normally, and only include the in-scope \
ticker(s) in your answer.
- Otherwise, continue to step 2.

STEP 2 -- ROUTE TYPE (only if step 1 passed):
- "document_query": needs SEC 10-K/10-Q filing content (historical \
financials, risk factors, business descriptions, MD&A). Filings do NOT \
contain today's stock price.
- "live_data_query": needs real-time/current market data (current stock \
price, today's market cap, current P/E) not found in any filed document.
- "hybrid_query": explicitly compares/combines current data with \
historical filed data.

Also extract any in-scope ticker(s) mentioned or implied by company name. \
Never include a ticker outside the 34 above. If the question is general or \
broad and does not name any specific company (e.g. "risks facing tech \
companies in general"), leave tickers EMPTY -- do not list all 34 \
companies just because the topic could apply to any of them. An empty \
ticker list means "search broadly, no specific company filter."

EXAMPLES:
Q: "What's the weather like in San Francisco?"
A: {"route": "out_of_scope", "tickers": [], "reasoning": "Not a financial question at all -- unrelated to any company or market data."}

Q: "What is Tesla's revenue?"
A: {"route": "out_of_scope", "tickers": [], "reasoning": "Tesla is not one of the 34 in-scope companies."}

Q: "Compare Apple and Tesla's revenue"
A: {"route": "document_query", "tickers": ["AAPL"], "reasoning": "Apple is in scope; Tesla is not, so only Apple's ticker is included. This is not out_of_scope since one in-scope company is present."}

Q: "What are the biggest risks facing tech companies right now?"
A: {"route": "document_query", "tickers": [], "reasoning": "General question about risk factors, no specific company named -- search broadly rather than listing every in-scope company."}

Q: "What were Apple's risk factors in their 10-K?"
A: {"route": "document_query", "tickers": ["AAPL"], "reasoning": "Risk factors come from filed documents."}

Q: "What's NVIDIA's stock price right now?"
A: {"route": "live_data_query", "tickers": ["NVDA"], "reasoning": "Current price is live market data, not in filings."}

Q: "How does AMD's current market cap compare to what they reported last year?"
A: {"route": "hybrid_query", "tickers": ["AMD"], "reasoning": "Combines current live data with historical filed data."}

Respond ONLY with a JSON object matching this exact schema, nothing else:
{"route": "document_query" | "live_data_query" | "hybrid_query" | "out_of_scope", \
"tickers": ["TICKER", ...], "reasoning": "short explanation"}
"""


def route_query(query: str) -> RoutingDecision:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set.")

    client = Groq(api_key=GROQ_API_KEY)

    try:
        response = client.chat.completions.create(
            model=ROUTER_MODEL,
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            max_tokens=350,  # raised from 200 -- few-shot examples in the
            # prompt made the model's own reasoning field longer on
            # average, and 200 was occasionally too tight, truncating
            # valid JSON mid-generation (observed directly: a
            # multi-ticker hybrid query hit this exact failure)
            temperature=0.0,  # deterministic classification, not creative
            response_format={"type": "json_object"},  # Groq's JSON-mode constraint
        )
        raw = response.choices[0].message.content
        data = json.loads(raw)
    except Exception as e:
        # If the router itself fails (truncated JSON, API error, etc.),
        # fail safe into out_of_scope rather than crashing the whole
        # agent -- a routing failure should degrade to "I can't answer
        # that", not take down the request.
        return RoutingDecision(
            route=RouteType.OUT_OF_SCOPE,
            tickers=[],
            reasoning=f"Router failed to produce a valid classification ({type(e).__name__}); failing safe.",
        )

    # Validate tickers against our known list -- if the model
    # hallucinated one outside scope, drop it rather than let it flow
    # downstream into a tool call that will fail confusingly later.
    valid_tickers = [t for t in data.get("tickers", []) if t in KNOWN_TICKERS]
    data["tickers"] = valid_tickers

    return RoutingDecision(**data)


OUT_OF_SCOPE_MESSAGE = (
    "I can only answer questions about the financials, filings, and stock "
    "data of these {n} companies: {tickers}. That question falls outside "
    "what I'm set up to answer."
)


def format_out_of_scope_response() -> str:
    return OUT_OF_SCOPE_MESSAGE.format(
        n=len(KNOWN_TICKERS), tickers=", ".join(sorted(KNOWN_TICKERS))
    )


if __name__ == "__main__":
    test_queries = [
        "What were the risk factors in Apple's 2025 10-K?",
        "What's NVIDIA's current stock price?",
        "How does AMD's current market cap compare to what they reported as their valuation last year?",
        "What is Tesla's revenue?",  # deliberately out-of-scope ticker
        "What's the weather like in San Francisco?",  # deliberately unrelated
    ]

    for q in test_queries:
        decision = route_query(q)
        print(f"\nQuery: {q}")
        print(f"  Route: {decision.route.value}")
        print(f"  Tickers: {decision.tickers}")
        print(f"  Reasoning: {decision.reasoning}")
        if decision.route == RouteType.OUT_OF_SCOPE:
            print(f"  Response: {format_out_of_scope_response()}")