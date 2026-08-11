"""
Semantic chunking pipeline for parsed SEC filings.

Takes a filing's `unstructured` elements and produces retrieval-ready
chunks:
  - Real financial tables (>1000 chars) become their own atomic chunk,
    tagged type="table". Never split mid-table.
  - Small "layout" tables get folded into the surrounding narrative
    text instead of becoming their own noisy chunk.
  - NarrativeText / ListItem / substantive UncategorizedText get
    grouped into semantic chunks: we walk through them in document
    order and start a new chunk whenever topic similarity to the
    current chunk drops below a threshold, a size cap is hit, or the
    section changes.
  - Any single raw text unit that is itself larger than our max chunk
    size (e.g. one very long Risk Factors paragraph) gets pre-split
    on sentence boundaries before chunking, so no single element can
    blow past our size caps on its own.
  - Every chunk is tagged with its section. SEC filings almost never
    use real <h1>-<h6> tags (they're plain divs/spans styled with
    CSS), so `unstructured` rarely emits Title elements for them. We
    detect sections ourselves by pattern-matching the standardized
    "Item N." headings that every 10-K/10-Q legally must use --
    this is our citation anchor instead of a page number, since these
    are HTML filings with no fixed pages.

This is a first pass: it processes ONE file end-to-end so we can
inspect output quality before running it across all 68 filings.

Usage:
    python -m src.chunking.chunk_filing
"""

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from unstructured.partition.html import partition_html

# --- CONFIG --------------------------------------------------------

TEST_FILE = Path("data/raw/AAPL/AAPL_10-K_2025-10-31.htm")
TEST_META = {
    "ticker": "AAPL",
    "company_name": "Apple Inc.",
    "form_type": "10-K",
    "filing_date": "2025-10-31",
}

OUTPUT_PATH = Path("data/processed/AAPL_10-K_2025-10-31_chunks.json")

# Model used just for measuring semantic similarity while chunking.
# Separate concern from the final embedding model we'll use for
# retrieval in Step 5 (bge-large-en-v1.5) -- here we just need
# something fast and decent to detect topic shifts between elements.
CHUNK_SIMILARITY_MODEL = "sentence-transformers/all-mpnet-base-v2"

SIMILARITY_THRESHOLD = 0.45   # below this cosine similarity -> new chunk
MAX_CHUNK_CHARS = 1500        # soft cap while building a chunk
MAX_CHUNK_CHARS_HARD = int(MAX_CHUNK_CHARS * 1.5)  # hard cap incl. merges
MIN_CHUNK_CHARS = 100         # avoid tiny leftover chunks where possible

REAL_TABLE_MIN_CHARS = 1000   # matches the heuristic from our scan step
MIN_TEXT_UNIT_CHARS = 15      # below this, treat as noise/label, drop

# SEC filings follow a legally standardized "Item N." heading format
# (Item 1., Item 1A., Item 7A., etc.) -- we detect sections by
# matching this pattern directly, since unstructured's generic Title
# detection often finds zero real headings in this kind of HTML.
ITEM_HEADING_PATTERN = re.compile(
    r"^item\s+\d+[a-z]?\.?\s+.{3,100}$", re.IGNORECASE
)


@dataclass
class Chunk:
    text: str
    type: str            # "table" or "text"
    section: str
    ticker: str
    company_name: str
    form_type: str
    filing_date: str
    source_file: str
    chunk_index: int


def get_section_title(text: str, max_len: int = 120) -> str:
    text = " ".join(text.split())
    return text[:max_len]


def looks_like_section_heading(text: str) -> bool:
    """Detect SEC 'Item N.' style section headings by pattern."""
    text = text.strip()
    if len(text) > 150:
        return False
    return bool(ITEM_HEADING_PATTERN.match(text))


def is_section_heading(category: str, text: str) -> bool:
    """A section boundary is either an unstructured Title element,
    OR text matching the SEC 'Item N.' pattern -- covers both
    cleanly-tagged HTML and the more common untagged case."""
    return category == "Title" or looks_like_section_heading(text)


def split_long_text(text: str, max_len: int) -> list[str]:
    """Split a single oversized text unit into smaller pieces on
    sentence boundaries, so no individual unit can single-handedly
    blow past our chunk size cap. Falls back to hard character
    splitting if no sentence boundaries are found (e.g. one giant
    run-on block with no punctuation)."""
    if len(text) <= max_len:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    pieces = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) + 1 <= max_len:
            current = (current + " " + sent).strip()
        else:
            if current:
                pieces.append(current)
            current = sent
    if current:
        pieces.append(current)

    # fallback: if a single "sentence" is itself still too long
    # (e.g. no punctuation at all), hard-split it
    final_pieces = []
    for p in pieces:
        if len(p) <= max_len:
            final_pieces.append(p)
        else:
            for i in range(0, len(p), max_len):
                final_pieces.append(p[i:i + max_len])
    return final_pieces


def build_text_units(elements) -> list[dict]:
    """
    Walk through elements in order, tracking current section, and
    produce a flat list of "text units" -- one per usable text-bearing
    element (pre-split if oversized) -- each carrying its section
    label. Small tables get folded in as plain text units instead of
    separate table chunks. Very short / noisy fragments are dropped.
    """
    units = []
    current_section = "Preamble"

    for el in elements:
        category = getattr(el, "category", "")
        text = str(el).strip()
        if not text:
            continue

        if is_section_heading(category, text):
            current_section = get_section_title(text)
            continue  # headings become section labels, not their own chunk

        if category == "Table" and len(text) >= REAL_TABLE_MIN_CHARS:
            continue  # handled separately as an atomic table chunk

        if len(text) < MIN_TEXT_UNIT_CHARS:
            continue  # drop short boilerplate/labels

        for piece in split_long_text(text, MAX_CHUNK_CHARS):
            units.append({"text": piece, "section": current_section})

    return units


def build_table_chunks(elements, meta) -> list[Chunk]:
    chunks = []
    idx = 0
    current_section = "Preamble"
    for el in elements:
        category = getattr(el, "category", "")
        text = str(el).strip()

        if is_section_heading(category, text):
            current_section = get_section_title(text)
            continue

        if category == "Table" and len(text) >= REAL_TABLE_MIN_CHARS:
            chunks.append(Chunk(
                text=text,
                type="table",
                section=current_section,
                ticker=meta["ticker"],
                company_name=meta["company_name"],
                form_type=meta["form_type"],
                filing_date=meta["filing_date"],
                source_file=str(TEST_FILE),
                chunk_index=idx,
            ))
            idx += 1
    return chunks


def semantic_chunk_units(units: list[dict], model: SentenceTransformer, meta) -> list[Chunk]:
    if not units:
        return []

    texts = [u["text"] for u in units]
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    chunks = []
    current_texts = [texts[0]]
    current_section = units[0]["section"]
    current_len = len(texts[0])
    idx = 0

    def flush():
        nonlocal idx
        combined = "\n".join(current_texts).strip()
        if len(combined) >= MIN_CHUNK_CHARS or not chunks:
            chunks.append(Chunk(
                text=combined,
                type="text",
                section=current_section,
                ticker=meta["ticker"],
                company_name=meta["company_name"],
                form_type=meta["form_type"],
                filing_date=meta["filing_date"],
                source_file=str(TEST_FILE),
                chunk_index=idx,
            ))
            idx += 1
        else:
            # too small on its own -> merge into previous chunk, but
            # only if that won't blow the chunk way past our size cap
            # (otherwise keep it as a small standalone chunk rather
            # than creating an oversized one)
            if chunks and len(chunks[-1].text) + len(combined) <= MAX_CHUNK_CHARS_HARD:
                chunks[-1].text = chunks[-1].text + "\n" + combined
            else:
                chunks.append(Chunk(
                    text=combined,
                    type="text",
                    section=current_section,
                    ticker=meta["ticker"],
                    company_name=meta["company_name"],
                    form_type=meta["form_type"],
                    filing_date=meta["filing_date"],
                    source_file=str(TEST_FILE),
                    chunk_index=idx,
                ))
                idx += 1

    for i in range(1, len(texts)):
        sim = float(np.dot(embeddings[i - 1], embeddings[i]))  # normalized -> cosine sim
        section_changed = units[i]["section"] != current_section
        would_overflow = current_len + len(texts[i]) > MAX_CHUNK_CHARS

        if sim < SIMILARITY_THRESHOLD or would_overflow or section_changed:
            flush()
            current_texts = [texts[i]]
            current_section = units[i]["section"]
            current_len = len(texts[i])
        else:
            current_texts.append(texts[i])
            current_len += len(texts[i])

    flush()
    return chunks


def main() -> None:
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Test file not found: {TEST_FILE}")

    print(f"Parsing: {TEST_FILE}")
    elements = partition_html(filename=str(TEST_FILE))
    print(f"  {len(elements)} elements extracted")

    print("Building table chunks...")
    table_chunks = build_table_chunks(elements, TEST_META)
    print(f"  {len(table_chunks)} atomic table chunks")

    print("Building text units...")
    text_units = build_text_units(elements)
    print(f"  {len(text_units)} text units after filtering/splitting")

    unique_sections = sorted(set(u["section"] for u in text_units))
    print(f"  {len(unique_sections)} unique sections detected")

    print(f"Loading similarity model: {CHUNK_SIMILARITY_MODEL}...")
    model = SentenceTransformer(CHUNK_SIMILARITY_MODEL)

    print("Running semantic chunking...")
    text_chunks = semantic_chunk_units(text_units, model, TEST_META)
    print(f"  {len(text_chunks)} semantic text chunks")

    all_chunks = table_chunks + text_chunks
    for i, c in enumerate(all_chunks):
        c.chunk_index = i

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps([asdict(c) for c in all_chunks], indent=2),
        encoding="utf-8",
    )

    print(f"\nDone. {len(all_chunks)} total chunks saved to {OUTPUT_PATH}")

    char_lens = [len(c.text) for c in all_chunks]
    print(f"Chunk char length: min={min(char_lens)}, max={max(char_lens)}, avg={sum(char_lens)//len(char_lens)}")
    print(f"Table chunks: {len(table_chunks)}, Text chunks: {len(text_chunks)}")

    oversized = [c for c in all_chunks if len(c.text) > MAX_CHUNK_CHARS_HARD]
    print(f"Chunks exceeding hard cap ({MAX_CHUNK_CHARS_HARD} chars): {len(oversized)}")

    print("\nSample detected sections:")
    for s in unique_sections[:15]:
        print(f"  - {s}")


if __name__ == "__main__":
    main()