"""
Edge-case stress test for the router. Not part of the normal module --
run standalone before trusting the router as the agent's entry point.

Usage:
    python -m src.agent.test_router_edge_cases
"""

from src.agent.router import route_query

EDGE_CASES = [
    # Multi-ticker comparison, no live/current language -> should be document_query
    ("Compare Microsoft's and Google's cloud revenue segments in their latest 10-Ks", "document_query", 2),

    # Multi-ticker, explicit current + historical -> hybrid
    ("How do Nvidia and AMD's current valuations compare to what they reported a year ago?", "hybrid_query", 2),

    # No ticker at all, general/vague -> ambiguous, worth seeing what happens
    ("What are the biggest risks facing tech companies right now?", None, 0),

    # Ticker symbol typed directly in caps, no company name
    ("What's PANW trading at right now?", "live_data_query", 1),

    # Implicit "current" via casual phrasing, no explicit word "current"
    ("How much is Snowflake worth today?", "live_data_query", 1),

    # Historical-sounding but actually needs live data (contains "recent" trap)
    ("What was Apple's most recent quarterly earnings report?", "document_query", 1),

    # Genuinely hybrid without the word "compare"
    ("Is Uber's stock price today higher or lower than what they projected in their last annual report?", "hybrid_query", 1),

    # Off-topic / not finance at all -- should now be out_of_scope
    ("What's the weather like in San Francisco?", "out_of_scope", 0),

    # Company name only, casual, ticker must be inferred
    ("Tell me about Adobe's business segments", "document_query", 1),

    # Two companies, one in-scope one not
    ("Compare Apple and Tesla's revenue", "document_query", 1),  # only AAPL should appear
]


def main() -> None:
    mismatches = []

    for query, expected_route, expected_ticker_count in EDGE_CASES:
        decision = route_query(query)
        print(f"\nQuery: {query}")
        print(f"  Route: {decision.route.value}")
        print(f"  Tickers: {decision.tickers}")
        print(f"  Reasoning: {decision.reasoning}")

        flags = []
        if expected_route and decision.route.value != expected_route:
            flags.append(f"expected route={expected_route}, got {decision.route.value}")
        if len(decision.tickers) != expected_ticker_count:
            flags.append(f"expected {expected_ticker_count} ticker(s), got {len(decision.tickers)}")

        if flags:
            print(f"  ⚠ {'; '.join(flags)}")
            mismatches.append((query, flags))

    print(f"\n{'=' * 60}")
    print(f"{len(EDGE_CASES) - len(mismatches)}/{len(EDGE_CASES)} matched expectations")
    if mismatches:
        print("Review these:")
        for q, flags in mismatches:
            print(f"  - {q}: {flags}")


if __name__ == "__main__":
    main()