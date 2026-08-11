"""
Shared chunking logic for SEC filings.

This module contains everything that turns a list of `unstructured`
elements into retrieval-ready Chunks. It is imported by both:
  - chunk_filing.py       (single-file test / inspection tool)
  - chunk_all_filings.py  (batch processor for all filings)

Keeping this logic in one place means both callers always run the
exact same chunking rules -- no risk of the single-file test and the
batch run silently drifting apart over time.
"""

import re
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# --- CONFIG --------------------------------------------------------

SIMILARITY_THRESHOLD = 0.45
MAX_CHUNK_CHARS = 1500
MAX_CHUNK_CHARS_HARD = int(MAX_CHUNK_CHARS * 1.5)
MIN_CHUNK_CHARS = 100

REAL_TABLE_MIN_CHARS = 1000
MIN_TEXT_UNIT_CHARS = 15

CHUNK_SIMILARITY_MODEL = "sentence-transformers/all-mpnet-base-v2"

ITEM_HEADING_PATTERN = re.compile(
    r"^item\s+\d+[a-z]?\.?\s+.{3,100}$", re.IGNORECASE
)


@dataclass
class Chunk:
    text: str
    type: str
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
    text = text.strip()
    if len(text) > 150:
        return False
    return bool(ITEM_HEADING_PATTERN.match(text))


def is_section_heading(category: str, text: str) -> bool:
    return category == "Title" or looks_like_section_heading(text)


def split_long_text(text: str, max_len: int) -> list[str]:
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

    final_pieces = []
    for p in pieces:
        if len(p) <= max_len:
            final_pieces.append(p)
        else:
            for i in range(0, len(p), max_len):
                final_pieces.append(p[i:i + max_len])
    return final_pieces


def build_text_units(elements) -> list[dict]:
    units = []
    current_section = "Preamble"

    for el in elements:
        category = getattr(el, "category", "")
        text = str(el).strip()
        if not text:
            continue

        if is_section_heading(category, text):
            current_section = get_section_title(text)
            continue

        if category == "Table" and len(text) >= REAL_TABLE_MIN_CHARS:
            continue

        if len(text) < MIN_TEXT_UNIT_CHARS:
            continue

        for piece in split_long_text(text, MAX_CHUNK_CHARS):
            units.append({"text": piece, "section": current_section})

    return units


def build_table_chunks(elements, meta: dict, source_file: str) -> list[Chunk]:
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
                source_file=source_file,
                chunk_index=idx,
            ))
            idx += 1
    return chunks


def semantic_chunk_units(
    units: list[dict], model: SentenceTransformer, meta: dict, source_file: str
) -> list[Chunk]:
    if not units:
        return []

    texts = [u["text"] for u in units]
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    chunks: list[Chunk] = []
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
                source_file=source_file,
                chunk_index=idx,
            ))
            idx += 1
        else:
            if chunks and len(chunks[-1].text) + 1 + len(combined) <= MAX_CHUNK_CHARS_HARD:
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
                    source_file=source_file,
                    chunk_index=idx,
                ))
                idx += 1

    for i in range(1, len(texts)):
        sim = float(np.dot(embeddings[i - 1], embeddings[i]))
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


def chunk_elements(elements, meta: dict, source_file: str, model: SentenceTransformer) -> list[Chunk]:
    table_chunks = build_table_chunks(elements, meta, source_file)
    text_units = build_text_units(elements)
    text_chunks = semantic_chunk_units(text_units, model, meta, source_file)

    all_chunks = table_chunks + text_chunks
    for i, c in enumerate(all_chunks):
        c.chunk_index = i
    return all_chunks


def load_similarity_model() -> SentenceTransformer:
    return SentenceTransformer(CHUNK_SIMILARITY_MODEL)


def chunks_to_dicts(chunks: list[Chunk]) -> list[dict]:
    return [asdict(c) for c in chunks]