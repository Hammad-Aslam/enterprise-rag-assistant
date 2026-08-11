"""
Batch scan all filings in manifest.json and collect element-type
statistics using unstructured's HTML partitioner. This is a diagnostic
step -- it does NOT chunk or save parsed content. Its purpose is to
reveal patterns (how much UncategorizedText, how many real vs
layout-only tables, doc size variance) across the full dataset before
we commit to chunking rules in the next step.

Usage:
    python -m src.ingestion.scan_all_filings
"""

import csv
import json
import time
from collections import Counter
from pathlib import Path

from unstructured.partition.html import partition_html

MANIFEST_PATH = Path("data/raw/manifest.json")
OUTPUT_CSV = Path("data/raw/scan_report.csv")


def scan_file(file_path: Path) -> dict:
    elements = partition_html(filename=str(file_path))

    type_counts = Counter(el.category for el in elements)
    total_chars = sum(len(str(el)) for el in elements)

    table_sizes = [len(str(el)) for el in elements if el.category == "Table"]
    largest_table_chars = max(table_sizes) if table_sizes else 0
    # A rough heuristic: layout tables (addresses, headers) tend to be
    # short. Real financial tables (balance sheets, income statements)
    # tend to be long, since they pack in many numeric rows.
    likely_real_tables = sum(1 for s in table_sizes if s > 1000)

    return {
        "total_elements": len(elements),
        "total_chars": total_chars,
        "narrative_text": type_counts.get("NarrativeText", 0),
        "uncategorized_text": type_counts.get("UncategorizedText", 0),
        "table_count": type_counts.get("Table", 0),
        "likely_real_tables": likely_real_tables,
        "largest_table_chars": largest_table_chars,
        "list_items": type_counts.get("ListItem", 0),
    }


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    rows = []
    for i, record in enumerate(manifest, start=1):
        file_path = Path(record["file_path"])
        print(f"[{i}/{len(manifest)}] Scanning {file_path.name}...")

        start = time.time()
        try:
            stats = scan_file(file_path)
            stats["parse_seconds"] = round(time.time() - start, 1)
            stats["error"] = ""
        except Exception as e:
            stats = {
                "total_elements": 0, "total_chars": 0, "narrative_text": 0,
                "uncategorized_text": 0, "table_count": 0,
                "likely_real_tables": 0, "largest_table_chars": 0,
                "list_items": 0, "parse_seconds": 0,
                "error": str(e),
            }
            print(f"  [error] {e}")

        row = {
            "ticker": record["ticker"],
            "form_type": record["form_type"],
            "file_name": file_path.name,
            **stats,
        }
        rows.append(row)

    fieldnames = list(rows[0].keys())
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Report saved to {OUTPUT_CSV}")

    # Print a quick aggregate summary to console too
    errors = [r for r in rows if r["error"]]
    print(f"\n{len(rows) - len(errors)}/{len(rows)} parsed successfully")
    if errors:
        print(f"{len(errors)} failed: {[r['file_name'] for r in errors]}")

    avg_uncategorized_pct = sum(
        r["uncategorized_text"] / r["total_elements"] * 100
        for r in rows if r["total_elements"] > 0
    ) / len([r for r in rows if r["total_elements"] > 0])
    print(f"Average % of elements classified as UncategorizedText: {avg_uncategorized_pct:.1f}%")


if __name__ == "__main__":
    main()