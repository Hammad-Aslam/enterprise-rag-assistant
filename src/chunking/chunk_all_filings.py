"""
Batch chunking pipeline for all filings in manifest.json.

Reuses the exact chunking logic validated in chunk_filing.py (via
chunking_core.py) and applies it across the full dataset. This is
deliberately NOT a copy-paste of chunk_filing.py's logic -- both
scripts import from the same module, so there is one chunking
implementation, not two that can drift apart.

For each filing, this script:
  1. Parses the HTML with unstructured
  2. Runs the shared chunking pipeline (tables + semantic text chunks)
  3. VALIDATES the output before trusting it:
       - no "text" chunk should exceed the hard size cap (only
         "table" chunks are allowed to, by design)
       - every chunk must have non-empty text and a real section label
  4. Saves one JSON file per filing to data/processed/
  5. Records per-file stats to build an aggregate CSV report at the end

Failures on individual files are caught and logged -- one bad filing
must not abort the whole batch, matching the same resilience pattern
used in download_edgar.py and scan_all_filings.py.

Usage:
    python -m src.chunking.chunk_all_filings
"""

import csv
import json
import time
import traceback
from pathlib import Path

from unstructured.partition.html import partition_html

from src.chunking.chunking_core import (
    Chunk,
    chunk_elements,
    chunks_to_dicts,
    load_similarity_model,
    MAX_CHUNK_CHARS_HARD,
)

MANIFEST_PATH = Path("data/raw/manifest.json")
PROCESSED_DIR = Path("data/processed")
REPORT_PATH = PROCESSED_DIR / "chunking_report.csv"


def validate_chunks(chunks: list[Chunk], file_label: str) -> list[str]:
    """Sanity-check chunk output before we trust it downstream.
    Returns a list of warning strings (empty list = all clear).
    We warn rather than hard-fail here, since a single bad chunk
    shouldn't discard an otherwise-good filing -- but we DO want a
    visible record of anything unexpected for later review."""
    warnings = []

    if not chunks:
        warnings.append("zero chunks produced")
        return warnings

    for c in chunks:
        if not c.text.strip():
            warnings.append(f"empty chunk text at index {c.chunk_index}")
        if not c.section or c.section == "":
            warnings.append(f"missing section label at index {c.chunk_index}")
        if c.type == "text" and len(c.text) > MAX_CHUNK_CHARS_HARD:
            warnings.append(
                f"oversized TEXT chunk at index {c.chunk_index} "
                f"({len(c.text)} chars, cap is {MAX_CHUNK_CHARS_HARD}) "
                f"in {file_label}"
            )

    return warnings


def process_one_filing(record: dict, model, skip_existing: bool = True) -> dict | None:
    """Process a single filing end-to-end. Returns a stats dict, or
    None if skipped because output already exists (resume support --
    lets us restart an interrupted batch without repeating expensive
    CPU work on filings that already finished successfully)."""
    file_path = Path(record["file_path"])
    if not file_path.exists():
        raise FileNotFoundError(f"Source file missing: {file_path}")

    meta = {
        "ticker": record["ticker"],
        "company_name": record["company_name"],
        "form_type": record["form_type"],
        "filing_date": record["filing_date"],
    }

    file_label = f"{meta['ticker']}_{meta['form_type']}_{meta['filing_date']}"
    out_path = PROCESSED_DIR / f"{file_label}_chunks.json"

    if skip_existing and out_path.exists():
        return None  # already done, nothing to do

    elements = partition_html(filename=str(file_path))
    chunks = chunk_elements(elements, meta, str(file_path), model)

    warnings = validate_chunks(chunks, file_label)

    out_path.write_text(
        json.dumps(chunks_to_dicts(chunks), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    table_count = sum(1 for c in chunks if c.type == "table")
    text_count = sum(1 for c in chunks if c.type == "text")
    char_lens = [len(c.text) for c in chunks] or [0]

    return {
        "ticker": meta["ticker"],
        "form_type": meta["form_type"],
        "filing_date": meta["filing_date"],
        "source_file": str(file_path),
        "output_file": str(out_path),
        "total_elements": len(elements),
        "total_chunks": len(chunks),
        "table_chunks": table_count,
        "text_chunks": text_count,
        "min_chunk_chars": min(char_lens),
        "max_chunk_chars": max(char_lens),
        "avg_chunk_chars": sum(char_lens) // len(char_lens),
        "warning_count": len(warnings),
        "warnings": " | ".join(warnings) if warnings else "",
        "error": "",
    }

def main() -> None:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found at {MANIFEST_PATH}. "
            "Run download_edgar.py first."
        )

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading similarity model (once, for the whole batch)...")
    model = load_similarity_model()

    rows = []
    t_start = time.time()

    for i, record in enumerate(manifest, start=1):
        label = f"{record['ticker']}_{record['form_type']}_{record['filing_date']}"

        t0 = time.time()
        try:
            stats = process_one_filing(record, model)
            if stats is None:
                print(f"[{i}/{len(manifest)}] {label}... already done, skipping")
                continue
            elapsed = round(time.time() - t0, 1)
            stats["seconds"] = elapsed
            flag = f" ⚠ {stats['warning_count']} warning(s)" if stats["warning_count"] else ""
            print(f"[{i}/{len(manifest)}] {label}... {stats['total_chunks']} chunks ({elapsed}s){flag}")
        except Exception as e:
            stats = {
                "ticker": record.get("ticker", ""),
                "form_type": record.get("form_type", ""),
                "filing_date": record.get("filing_date", ""),
                "source_file": record.get("file_path", ""),
                "output_file": "",
                "total_elements": 0,
                "total_chunks": 0,
                "table_chunks": 0,
                "text_chunks": 0,
                "min_chunk_chars": 0,
                "max_chunk_chars": 0,
                "avg_chunk_chars": 0,
                "warning_count": 0,
                "warnings": "",
                "error": f"{type(e).__name__}: {e}",
                "seconds": round(time.time() - t0, 1),
            }
            print(f"[{i}/{len(manifest)}] {label}... FAILED: {e}")
            traceback.print_exc(limit=2)

        rows.append(stats)

    total_elapsed = round(time.time() - t_start, 1)

    fieldnames = list(rows[0].keys())
    with REPORT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    succeeded = [r for r in rows if not r["error"]]
    failed = [r for r in rows if r["error"]]
    with_warnings = [r for r in succeeded if r["warning_count"] > 0]

    total_chunks = sum(r["total_chunks"] for r in succeeded)
    total_table_chunks = sum(r["table_chunks"] for r in succeeded)
    total_text_chunks = sum(r["text_chunks"] for r in succeeded)

    print("\n" + "=" * 60)
    print(f"BATCH COMPLETE in {total_elapsed}s")
    print("=" * 60)
    print(f"Succeeded: {len(succeeded)}/{len(rows)}")
    print(f"Failed:    {len(failed)}/{len(rows)}")
    print(f"Filings with validation warnings: {len(with_warnings)}")
    print(f"Total chunks produced: {total_chunks} "
          f"({total_table_chunks} table, {total_text_chunks} text)")
    print(f"Report saved to: {REPORT_PATH}")

    if failed:
        print("\nFailed filings:")
        for r in failed:
            print(f"  - {r['ticker']} {r['form_type']} {r['filing_date']}: {r['error']}")

    if with_warnings:
        print("\nFilings with warnings (review recommended):")
        for r in with_warnings:
            print(f"  - {r['ticker']} {r['form_type']} {r['filing_date']}: {r['warnings']}")


if __name__ == "__main__":
    main()