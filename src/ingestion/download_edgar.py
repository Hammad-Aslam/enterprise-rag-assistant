"""
SEC EDGAR bulk filing downloader.

Downloads recent 10-K and 10-Q filings for a fixed list of tech-sector
tickers, using SEC's official submissions API (JSON, no scraping).

SEC EDGAR fair-access rules:
- Must set a descriptive User-Agent: "Company/App Name your-email@example.com"
- Max ~10 requests/second recommended; we throttle to ~4/sec to be safe.
- Docs: https://www.sec.gov/os/webmaster-faq#developers

Usage:
    python -m src.ingestion.download_edgar
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict

import requests

# --- CONFIG ------------------------------------------------------------

USER_AGENT = "Saad Ahmed saadahmedofficial44@gmail.com"

HEADERS = {"User-Agent": USER_AGENT}

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "AMD", "INTC", "QCOM", "TXN", "AVGO",
    "CRM", "ORCL", "ADBE", "NOW", "INTU", "WDAY", "SNOW",
    "PANW", "CRWD", "FTNT", "ZS",
    "DELL", "HPQ", "CSCO", "NTAP",
    "PYPL", "SQ", "FIS",
    "NFLX", "UBER", "ABNB", "SHOP", "SPOT",
    "AMAT", "LRCX",
]

FILING_TYPES = ("10-K", "10-Q")
FILINGS_PER_TYPE = 1

RAW_DIR = Path("data/raw")
MANIFEST_PATH = RAW_DIR / "manifest.json"

TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

REQUEST_DELAY_SEC = 0.25


@dataclass
class FilingRecord:
    ticker: str
    company_name: str
    cik: int
    form_type: str
    filing_date: str
    accession_number: str
    primary_document: str
    file_path: str


def load_ticker_cik_map() -> dict[str, dict]:
    resp = requests.get(TICKER_CIK_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return {v["ticker"]: v for v in data.values()}


def get_submissions(cik: int) -> dict:
    url = SUBMISSIONS_URL.format(cik=cik)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def find_recent_filings(submissions: dict, form_type: str, n: int) -> list[dict]:
    recent = submissions["filings"]["recent"]
    matches = []
    for i, form in enumerate(recent["form"]):
        if form == form_type:
            matches.append(
                {
                    "accessionNumber": recent["accessionNumber"][i],
                    "filingDate": recent["filingDate"][i],
                    "primaryDocument": recent["primaryDocument"][i],
                }
            )
        if len(matches) >= n:
            break
    return matches


def download_filing(cik: int, accession_number: str, primary_document: str, dest: Path) -> None:
    acc_nodash = accession_number.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{primary_document}"
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching ticker -> CIK map...")
    ticker_map = load_ticker_cik_map()

    manifest: list[FilingRecord] = []

    for ticker in TICKERS:
        info = ticker_map.get(ticker)
        if not info:
            print(f"  [skip] {ticker}: not found in SEC ticker map")
            continue

        cik = info["cik_str"]
        company_name = info["title"]
        print(f"Processing {ticker} ({company_name}, CIK {cik})...")

        try:
            submissions = get_submissions(cik)
        except requests.HTTPError as e:
            print(f"  [error] failed to fetch submissions for {ticker}: {e}")
            time.sleep(REQUEST_DELAY_SEC)
            continue

        for form_type in FILING_TYPES:
            filings = find_recent_filings(submissions, form_type, FILINGS_PER_TYPE)
            for f in filings:
                ext = Path(f["primaryDocument"]).suffix or ".htm"
                out_name = f"{ticker}_{form_type}_{f['filingDate']}{ext}"
                out_path = RAW_DIR / ticker / out_name

                try:
                    download_filing(cik, f["accessionNumber"], f["primaryDocument"], out_path)
                    print(f"  [ok] {form_type} {f['filingDate']} -> {out_path}")
                    manifest.append(
                        FilingRecord(
                            ticker=ticker,
                            company_name=company_name,
                            cik=cik,
                            form_type=form_type,
                            filing_date=f["filingDate"],
                            accession_number=f["accessionNumber"],
                            primary_document=f["primaryDocument"],
                            file_path=str(out_path),
                        )
                    )
                except requests.HTTPError as e:
                    print(f"  [error] failed to download {form_type} for {ticker}: {e}")

                time.sleep(REQUEST_DELAY_SEC)

        time.sleep(REQUEST_DELAY_SEC)

    MANIFEST_PATH.write_text(
        json.dumps([asdict(m) for m in manifest], indent=2), encoding="utf-8"
    )
    print(f"\nDone. {len(manifest)} filings downloaded. Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()