"""
End-to-end test suite for the agentic RAG system -- hits the ACTUAL
running FastAPI server's /agent/ask endpoint (not the in-process graph
directly, which Step 6's testing already covered). This exercises the
real path a client (frontend, external caller) would take: HTTP ->
FastAPI -> lifespan-loaded agent -> LangGraph -> tools -> response.

Requires the server to already be running:
    uvicorn src.api.main:app --port 8000

Usage:
    python -m tests.test_agent_e2e
"""

import csv
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import requests

API_URL = "http://localhost:8000"
REPORT_PATH = Path("tests/e2e_test_report.csv")


@dataclass
class TestCase:
    query: str
    expected_route: str
    # Soft content checks -- substrings we expect SOMEWHERE in the
    # answer if the route/tool worked correctly. Not exhaustive
    # correctness checks (we can't assert exact numbers, live data
    # changes), just sanity signals that the right kind of evidence
    # was actually used.
    expect_dollar_sign: bool = False
    expect_citation_tag: bool = False


TEST_CASES = [
    # document_query
    TestCase("What was Apple's total revenue and net income?", "document_query", expect_dollar_sign=True, expect_citation_tag=True),
    TestCase("What did AMD say about AI chip demand?", "document_query", expect_citation_tag=True),
    TestCase("Tell me about Adobe's business segments", "document_query", expect_citation_tag=True),

    # live_data_query
    TestCase("What's NVIDIA's current stock price?", "live_data_query", expect_dollar_sign=True, expect_citation_tag=True),
    TestCase("What's Microsoft's market cap right now?", "live_data_query", expect_dollar_sign=True, expect_citation_tag=True),
    TestCase("What's PANW trading at?", "live_data_query", expect_dollar_sign=True, expect_citation_tag=True),

    # hybrid_query
    TestCase("How does AMD's current market cap compare to what they reported around a year ago?", "hybrid_query", expect_citation_tag=True),
    TestCase("Is Uber's stock price today higher than what they projected in their last annual report?", "hybrid_query", expect_citation_tag=True),

    # out_of_scope
    TestCase("What's the weather like in San Francisco?", "out_of_scope"),
    TestCase("What is Tesla's revenue?", "out_of_scope"),
]


@dataclass
class TestResult:
    query: str
    expected_route: str
    actual_route: str
    route_match: bool
    http_status: int
    latency_seconds: float
    confidence_level: str
    answer_length: int
    dollar_check: str  # "pass" | "fail" | "n/a"
    citation_check: str  # "pass" | "fail" | "n/a"
    error: str


def run_test_case(tc: TestCase) -> TestResult:
    t0 = time.time()
    try:
        resp = requests.post(
            f"{API_URL}/agent/ask",
            json={"question": tc.query},
            timeout=60,
        )
    except requests.RequestException as e:
        return TestResult(
            query=tc.query, expected_route=tc.expected_route, actual_route="",
            route_match=False, http_status=0, latency_seconds=round(time.time() - t0, 2),
            confidence_level="", answer_length=0, dollar_check="n/a", citation_check="n/a",
            error=f"Request failed: {e}",
        )

    latency = round(time.time() - t0, 2)

    if resp.status_code != 200:
        return TestResult(
            query=tc.query, expected_route=tc.expected_route, actual_route="",
            route_match=False, http_status=resp.status_code, latency_seconds=latency,
            confidence_level="", answer_length=0, dollar_check="n/a", citation_check="n/a",
            error=f"Non-200 status: {resp.text}",
        )

    data = resp.json()
    answer = data.get("answer", "")

    dollar_check = "n/a"
    if tc.expect_dollar_sign:
        dollar_check = "pass" if "$" in answer else "fail"

    citation_check = "n/a"
    if tc.expect_citation_tag:
        citation_check = "pass" if ("[SOURCE:" in answer or "[LIVE:" in answer) else "fail"

    return TestResult(
        query=tc.query,
        expected_route=tc.expected_route,
        actual_route=data.get("route", ""),
        route_match=data.get("route") == tc.expected_route,
        http_status=resp.status_code,
        latency_seconds=latency,
        confidence_level=data.get("confidence_level", ""),
        answer_length=len(answer),
        dollar_check=dollar_check,
        citation_check=citation_check,
        error="",
    )


def main() -> None:
    print(f"Checking server health at {API_URL}/health ...")
    try:
        health = requests.get(f"{API_URL}/health", timeout=10).json()
    except requests.RequestException as e:
        print(f"FATAL: could not reach server ({e}). Is uvicorn running?")
        return

    if not health.get("agent_ready"):
        print(f"FATAL: agent not ready ({health}). Wait for startup to finish.")
        return

    print("Server healthy. Running end-to-end test cases...\n")

    results = []
    for i, tc in enumerate(TEST_CASES, start=1):
        print(f"[{i}/{len(TEST_CASES)}] {tc.query[:60]}...", end=" ")
        result = run_test_case(tc)
        time.sleep(15)  # give Groq's rate limit window real breathing room
        results.append(result)

        if result.error:
            print(f"ERROR: {result.error}")
        else:
            flags = []
            if not result.route_match:
                flags.append(f"route mismatch (expected {tc.expected_route}, got {result.actual_route})")
            if result.dollar_check == "fail":
                flags.append("missing expected $ figure")
            if result.citation_check == "fail":
                flags.append("missing expected citation tag")
            flag_str = f" ⚠ {'; '.join(flags)}" if flags else " ✓"
            print(f"{result.actual_route} ({result.latency_seconds}s){flag_str}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    total = len(results)
    route_ok = sum(1 for r in results if r.route_match)
    no_errors = sum(1 for r in results if not r.error)
    dollar_fails = sum(1 for r in results if r.dollar_check == "fail")
    citation_fails = sum(1 for r in results if r.citation_check == "fail")
    avg_latency = round(sum(r.latency_seconds for r in results) / total, 2)

    print("\n" + "=" * 60)
    print("END-TO-END TEST SUMMARY")
    print("=" * 60)
    print(f"Total cases: {total}")
    print(f"No request errors: {no_errors}/{total}")
    print(f"Route classification correct: {route_ok}/{total}")
    print(f"Content checks failed: {dollar_fails} dollar-sign, {citation_fails} citation")
    print(f"Average latency: {avg_latency}s")
    print(f"Full report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()