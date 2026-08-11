"""
Rebuilds an audit report by scanning data/processed/*_chunks.json
directly, independent of whether the batch run that created them
finished cleanly. This means a crash, low battery, or manual
interrupt during chunk_all_filings.py never costs you your audit
trail -- you can regenerate it any time from what's already on disk.

Usage:
    python -m src.chunking.generate_report
"""

import csv
import json
from pathlib import Path

from src.chunking.chunking_core import MAX_CHUNK_CHARS_HARD

PROCESSED_DIR = Path("data/processed")
REPORT_PATH = PROCESSED_DIR / "chunking_report.csv"
MANIFEST_PATH = Path("data/raw/manifest.json")


def audit_file(path: Path) -> dict:
    chunks = json.loads(path.read_text(encoding="utf-8"))

    warnings = []
    for c in chunks:
        if not c["text"].strip():
            warnings.append(f"empty chunk text at index {c['chunk_index']}")
        if not c.get("section"):
            warnings.append(f"missing section label at index {c['chunk_index']}")
        if c["type"] == "text" and len(c["text"]) > MAX_CHUNK_CHARS_HARD:
            warnings.append(
                f"oversized TEXT chunk at index {c['chunk_index']} ({len(c['text'])} chars)"
            )

    table_count = sum(1 for c in chunks if c["type"] == "table")
    text_count = sum(1 for c in chunks if c["type"] == "text")
    char_lens = [len(c["text"]) for c in chunks] or [0]

    meta = chunks[0] if chunks else {}
    return {
        "ticker": meta.get("ticker", ""),
        "form_type": meta.get("form_type", ""),
        "filing_date": meta.get("filing_date", ""),
        "output_file": str(path),
        "total_chunks": len(chunks),
        "table_chunks": table_count,
        "text_chunks": text_count,
        "min_chunk_chars": min(char_lens),
        "max_chunk_chars": max(char_lens),
        "avg_chunk_chars": sum(char_lens) // len(char_lens),
        "warning_count": len(warnings),
        "warnings": " | ".join(warnings) if warnings else "",
    }


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    expected = {
        f"{r['ticker']}_{r['form_type']}_{r['filing_date']}" for r in manifest
    }

    chunk_files = sorted(PROCESSED_DIR.glob("*_chunks.json"))
    rows = [audit_file(p) for p in chunk_files]

    done_labels = {
        f"{r['ticker']}_{r['form_type']}_{r['filing_date']}" for r in rows
    }
    missing = expected - done_labels

    if rows:
        fieldnames = list(rows[0].keys())
        with REPORT_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"Audited {len(rows)}/{len(expected)} filings.")
    print(f"Report saved to: {REPORT_PATH}")

    if missing:
        print(f"\nStill missing ({len(missing)}):")
        for m in sorted(missing):
            print(f"  - {m}")

    warned = [r for r in rows if r["warning_count"] > 0]
    if warned:
        print(f"\nFilings with warnings ({len(warned)}):")
        for r in warned:
            print(f"  - {r['ticker']} {r['form_type']} {r['filing_date']}: {r['warnings']}")


if __name__ == "__main__":
    main()