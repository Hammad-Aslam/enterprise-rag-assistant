"""
Quick smoke test: parse a single SEC filing and inspect the element
types unstructured extracts. Run this before building the full batch
pipeline, to confirm HTML partitioning is behaving sensibly on a real
10-K/10-Q before we commit to a chunking strategy around it.

Usage:
    python -m src.ingestion.test_parse_single
"""

from collections import Counter
from pathlib import Path

from unstructured.partition.html import partition_html

# Pick one filing to test on. Apple's 10-K is a good first test:
# large, well-structured, heavy on tables.
TEST_FILE = Path("data/raw/AAPL/AAPL_10-K_2025-10-31.htm")


def main() -> None:
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Test file not found: {TEST_FILE}")

    print(f"Parsing: {TEST_FILE}")
    elements = partition_html(filename=str(TEST_FILE))

    print(f"\nTotal elements extracted: {len(elements)}")

    type_counts = Counter(el.category for el in elements)
    print("\nElement type breakdown:")
    for category, count in type_counts.most_common():
        print(f"  {category}: {count}")

    print("\n--- First 5 non-empty elements (preview) ---")
    shown = 0
    for el in elements:
        text = str(el).strip()
        if not text:
            continue
        print(f"\n[{el.category}] {text[:200]}")
        shown += 1
        if shown >= 5:
            break

    print("\n--- First Table element found (if any) ---")
    for el in elements:
        if el.category == "Table":
            print(str(el)[:500])
            break
    else:
        print("No Table elements found.")


if __name__ == "__main__":
    main()