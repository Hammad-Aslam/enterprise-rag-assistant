"""
Live financial data tool -- wraps yfinance to fetch current stock
price, market cap, P/E ratio, and related live market data for the
same 34 companies indexed in Project 1's document corpus.

Design notes:
  - Ticker validation reuses the exact same known-company list as the
    router (imported, not duplicated) -- an out-of-scope ticker is
    rejected explicitly with a clear error, rather than silently
    passed through to yfinance and producing a confusing result.
  - yfinance hits an unofficial, no-contract endpoint (no real API
    key/SLA) -- it can occasionally return incomplete data or fail
    outright. Every fetch is wrapped so a failure degrades to a clear
    per-ticker error, not a crash that would take down an entire
    hybrid-query request that also needed document results.
  - Every result is timestamped at fetch time. Live data is
    time-sensitive in a way filed documents aren't -- a price quote
    needs an "as of" moment attached to be honestly citable, especially
    when a synthesis step (Step 5) compares it against a fixed filing
    date.
  - Supports multiple tickers per call, since hybrid/comparison
    queries commonly need more than one company's live data at once.

Usage:
    tool = LiveDataTool()
    results = tool.run(["AAPL", "MSFT"])
"""

from dataclasses import dataclass
from datetime import datetime, timezone

import yfinance as yf

from src.agent.router import KNOWN_TICKERS


@dataclass
class LiveDataResult:
    ticker: str
    success: bool
    fetched_at: str  # ISO 8601 UTC timestamp
    price: float | None = None
    previous_close: float | None = None
    market_cap: int | None = None
    pe_ratio: float | None = None
    week_52_high: float | None = None
    week_52_low: float | None = None
    error: str | None = None

    def to_context_string(self) -> str:
        if not self.success:
            return f"Live data for {self.ticker}: unavailable ({self.error})"

        def fmt_money(v):
            return f"${v:,.2f}" if v is not None else "N/A"

        def fmt_large(v):
            if v is None:
                return "N/A"
            if v >= 1e9:
                return f"${v / 1e9:,.2f}B"
            return f"${v:,.0f}"

        return (
            f"Live market data for {self.ticker} (as of {self.fetched_at}):\n"
            f"  Current price: {fmt_money(self.price)}\n"
            f"  Previous close: {fmt_money(self.previous_close)}\n"
            f"  Market cap: {fmt_large(self.market_cap)}\n"
            f"  P/E ratio: {self.pe_ratio if self.pe_ratio is not None else 'N/A'}\n"
            f"  52-week range: {fmt_money(self.week_52_low)} - {fmt_money(self.week_52_high)}\n"
        )


class LiveDataTool:
    name = "live_market_data"
    description = (
        "Fetches current/real-time stock price, market cap, P/E ratio, "
        "and 52-week range for the 34 in-scope companies. Use for "
        "questions about current or today's market data. Does NOT have "
        "historical filing content."
    )

    def _fetch_one(self, ticker: str) -> LiveDataResult:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        if ticker not in KNOWN_TICKERS:
            return LiveDataResult(
                ticker=ticker,
                success=False,
                fetched_at=now,
                error=f"{ticker} is not one of the 34 in-scope companies.",
            )

        try:
            t = yf.Ticker(ticker)
            info = t.info

            price = info.get("currentPrice") or info.get("regularMarketPrice")
            if price is None:
                return LiveDataResult(
                    ticker=ticker,
                    success=False,
                    fetched_at=now,
                    error="yfinance returned no price data for this ticker.",
                )

            return LiveDataResult(
                ticker=ticker,
                success=True,
                fetched_at=now,
                price=price,
                previous_close=info.get("previousClose"),
                market_cap=info.get("marketCap"),
                pe_ratio=info.get("trailingPE"),
                week_52_high=info.get("fiftyTwoWeekHigh"),
                week_52_low=info.get("fiftyTwoWeekLow"),
            )
        except Exception as e:
            return LiveDataResult(
                ticker=ticker,
                success=False,
                fetched_at=now,
                error=f"{type(e).__name__}: {e}",
            )

    def run(self, tickers: list[str]) -> list[LiveDataResult]:
        return [self._fetch_one(t) for t in tickers]


if __name__ == "__main__":
    tool = LiveDataTool()

    test_cases = [
        ["AAPL"],
        ["NVDA", "AMD"],
        ["TSLA"],  # deliberately out-of-scope
    ]

    for tickers in test_cases:
        print(f"\nFetching: {tickers}")
        results = tool.run(tickers)
        for r in results:
            print(r.to_context_string())