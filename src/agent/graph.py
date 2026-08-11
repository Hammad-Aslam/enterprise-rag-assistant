"""
The agent's LangGraph decision graph: router -> tool selection ->
execution -> synthesis.

This is deliberately an EXPLICIT graph (nodes + conditional edges),
not a simple if/else function or a linear LangChain chain -- per the
project spec's requirement to demonstrate real agentic routing. Each
node is a distinct, independently-testable unit (we already unit
tested the router and both tools in isolation); this module's only
job is wiring them together and logging the decision made at each run.

Graph shape:

    START -> router -> [conditional branch on route] -> {
        document_query  -> document_node  -> END
        live_data_query -> live_node      -> END
        hybrid_query     -> hybrid_node    -> END
        out_of_scope     -> out_of_scope_node -> END
    }

Every path logs its routing decision (query, route, tickers,
reasoning, confidence, latency) via src/observability/routing_log.py
before returning.

Usage:
    from src.agent.graph import build_agent, run_agent

    agent = build_agent()  # instantiate once -- loads document tool's models
    result = run_agent(agent, "What was Apple's revenue last year?")
"""

import time
from typing import TypedDict

from langgraph.graph import StateGraph, END

from src.agent.router import route_query, RouteType, format_out_of_scope_response
from src.agent.synthesis import synthesize_hybrid_answer, synthesize_live_answer
from src.agent.tools.document_tool import DocumentRetrieverTool
from src.agent.tools.live_data_tool import LiveDataTool
from src.generation.generate_answer import answer_question
from src.observability.routing_log import init_db, log_decision


class AgentState(TypedDict, total=False):
    query: str
    route: str
    tickers: list[str]
    reasoning: str
    answer: str
    confidence_level: str
    start_time: float


class AgentTools:
    """Holds the persistent, expensive-to-instantiate tool instances.
    Built once via build_agent(), passed into every graph node
    invocation via closures -- not re-instantiated per query."""
    def __init__(self, searcher=None):
        self.document_tool = DocumentRetrieverTool(searcher=searcher)
        self.live_tool = LiveDataTool()


def build_agent(searcher=None) -> tuple[object, AgentTools]:
    """Builds and compiles the LangGraph agent. Returns (compiled_graph,
    tools). Pass an existing HybridSearcher (e.g. one FastAPI already
    loaded) to avoid loading the model stack a second time."""
    init_db()
    tools = AgentTools(searcher=searcher)

    graph = StateGraph(AgentState)

    def router_node(state: AgentState) -> AgentState:
        decision = route_query(state["query"])
        return {
            **state,
            "route": decision.route.value,
            "tickers": decision.tickers,
            "reasoning": decision.reasoning,
        }

    def document_node(state: AgentState) -> AgentState:
        result = answer_question(state["query"], searcher=tools.document_tool.searcher)
        return {
            **state,
            "answer": result["answer"],
            "confidence_level": result["confidence_level"],
        }

    def live_node(state: AgentState) -> AgentState:
        result = synthesize_live_answer(state["query"], state["tickers"], tools.live_tool)
        return {
            **state,
            "answer": result["answer"],
            "confidence_level": result["confidence_level"],
        }

    def hybrid_node(state: AgentState) -> AgentState:
        result = synthesize_hybrid_answer(
            state["query"], state["tickers"], tools.document_tool, tools.live_tool
        )
        return {
            **state,
            "answer": result["answer"],
            "confidence_level": result["confidence_level"],
        }

    def out_of_scope_node(state: AgentState) -> AgentState:
        return {
            **state,
            "answer": format_out_of_scope_response(),
            "confidence_level": "n/a",
        }

    def log_node(state: AgentState) -> AgentState:
        elapsed = time.time() - state.get("start_time", time.time())
        log_decision(
            query=state["query"],
            route=state["route"],
            tickers=state.get("tickers", []),
            reasoning=state.get("reasoning", ""),
            confidence_level=state.get("confidence_level"),
            latency_seconds=round(elapsed, 2),
        )
        return state

    graph.add_node("router", router_node)
    graph.add_node("document", document_node)
    graph.add_node("live", live_node)
    graph.add_node("hybrid", hybrid_node)
    graph.add_node("out_of_scope", out_of_scope_node)
    graph.add_node("log", log_node)

    graph.set_entry_point("router")

    def route_selector(state: AgentState) -> str:
        return {
            RouteType.DOCUMENT_QUERY.value: "document",
            RouteType.LIVE_DATA_QUERY.value: "live",
            RouteType.HYBRID_QUERY.value: "hybrid",
            RouteType.OUT_OF_SCOPE.value: "out_of_scope",
        }[state["route"]]

    graph.add_conditional_edges(
        "router",
        route_selector,
        {
            "document": "document",
            "live": "live",
            "hybrid": "hybrid",
            "out_of_scope": "out_of_scope",
        },
    )

    graph.add_edge("document", "log")
    graph.add_edge("live", "log")
    graph.add_edge("hybrid", "log")
    graph.add_edge("out_of_scope", "log")
    graph.add_edge("log", END)

    compiled = graph.compile()
    return compiled, tools


def run_agent(compiled_graph, query: str) -> AgentState:
    initial_state: AgentState = {"query": query, "start_time": time.time()}
    return compiled_graph.invoke(initial_state)


if __name__ == "__main__":
    print("Building agent (loading tools -- takes a while on first run)...\n")
    agent, tools = build_agent()

    test_queries = [
        "What were the risk factors in Apple's most recent 10-K?",       # document
        "What's NVIDIA's current stock price?",                          # live
        "How does AMD's current market cap compare to what they reported around a year ago?",  # hybrid
        "What's the weather like in San Francisco?",                     # out_of_scope
    ]

    for q in test_queries:
        print(f"\n{'=' * 70}")
        print(f"Query: {q}")
        result = run_agent(agent, q)
        print(f"Route: {result['route']}")
        print(f"Tickers: {result.get('tickers')}")
        print(f"Reasoning: {result['reasoning']}")
        print(f"Confidence: {result.get('confidence_level')}")
        print(f"\nAnswer:\n{result['answer']}")

    print(f"\n{'=' * 70}")
    print("Routing log (most recent 10):")
    from src.observability.routing_log import get_recent_decisions
    for row in get_recent_decisions(limit=10):
        print(f"  [{row['route']}] {row['query'][:60]}... ({row['latency_seconds']}s, tickers={row['tickers']})")