"""
Routing decision log -- SQLite-backed, so every query the agent
handles gets a permanent, queryable record of: what was asked, which
route was chosen, which tickers were involved, WHY that route was
chosen (the router's own reasoning), and how long it took.

This is the "routing decision log" deliverable from the project spec
-- visible in the demo as a small table, and useful for debugging
routing behavior after the fact without needing to re-run queries.

Usage:
    from src.observability.routing_log import init_db, log_decision, get_recent_decisions

    init_db()  # call once at startup
    log_decision(query=..., route=..., tickers=[...], reasoning=..., ...)
    rows = get_recent_decisions(limit=20)
"""

import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("data/routing_log.db")


@contextmanager
def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS routing_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                query TEXT NOT NULL,
                route TEXT NOT NULL,
                tickers TEXT,
                reasoning TEXT,
                confidence_level TEXT,
                latency_seconds REAL,
                error TEXT
            )
        """)


def log_decision(
    query: str,
    route: str,
    tickers: list[str],
    reasoning: str,
    confidence_level: str | None,
    latency_seconds: float,
    error: str | None = None,
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO routing_decisions
                (timestamp, query, route, tickers, reasoning, confidence_level, latency_seconds, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                query,
                route,
                ", ".join(tickers) if tickers else "",
                reasoning,
                confidence_level,
                latency_seconds,
                error,
            ),
        )


def get_recent_decisions(limit: int = 20) -> list[dict]:
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM routing_decisions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


class timed_decision:
    """Small context manager to measure latency around a routing +
    execution block, so callers don't have to hand-roll time.time()
    bookkeeping at every call site."""
    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.time() - self._start


if __name__ == "__main__":
    init_db()
    log_decision(
        query="What was Apple's revenue?",
        route="document_query",
        tickers=["AAPL"],
        reasoning="Test entry",
        confidence_level="high",
        latency_seconds=1.23,
    )
    for row in get_recent_decisions(limit=5):
        print(row)