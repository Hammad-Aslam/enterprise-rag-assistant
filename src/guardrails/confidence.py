"""
Confidence / guardrail scoring for generated answers.

Combines two independent signals we already compute elsewhere in the
pipeline into a single confidence tier, so a thin, poorly-supported
answer doesn't look equally authoritative as a well-grounded one:

  1. RETRIEVAL RELEVANCE -- the top cross-encoder re-rank score from
     hybrid_search.py. The cross-encoder (ms-marco-MiniLM-L-6-v2)
     outputs raw, unbounded logits. We apply a sigmoid to compress
     the logit into a 0-1 pseudo-probability, which is standard
     practice for MS MARCO cross-encoders -- but the specific
     thresholds below are calibrated against ACTUAL scores observed
     from our own test queries against this corpus, not generic
     textbook values. Our single best real match (AAPL's income
     statement table) scored -0.272, which sigmoid-maps to only 0.43
     -- a naive 0.6 "high" threshold would be nearly unreachable.

  2. CITATION COVERAGE -- what fraction of the answer's sentences
     carry a valid citation tag ([SOURCE: n] for document chunks, or
     [LIVE: TICKER] for live market data, used by Project 2's hybrid
     synthesis step).

  Zero retrieved chunks, or zero valid citations in a non-empty
  answer, are hard floors to "low" -- no formula needed for a
  clear-cut failure case.
"""

import math
import re
from dataclasses import dataclass

SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
# Matches both Project 1's document-only citation format [SOURCE: n]
# and Project 2's dual-source hybrid format [LIVE: TICKER]. Safe,
# additive change -- Project 1's own answers never produce [LIVE: ...]
# tags, so this doesn't alter Project 1's existing behavior, but
# without it, this module undercounted citation coverage on hybrid
# answers (a genuine [LIVE: AMD] citation was invisible to the
# coverage calculation, making a correctly-cited answer look 0% cited).
CITATION_PATTERN = re.compile(r"\[SOURCE:\s*\d+\]|\[LIVE:\s*[A-Z]+\]")

RELEVANCE_HIGH = 0.35
RELEVANCE_MEDIUM = 0.10

COVERAGE_HIGH = 0.6
COVERAGE_MEDIUM = 0.3


@dataclass
class ConfidenceAssessment:
    level: str
    relevance_score: float
    citation_coverage: float
    reasons: list[str]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _citation_coverage(answer_text: str) -> float:
    sentences = [s for s in SENTENCE_SPLIT_PATTERN.split(answer_text.strip()) if s.strip()]
    if not sentences:
        return 0.0
    cited = sum(1 for s in sentences if CITATION_PATTERN.search(s))
    return cited / len(sentences)


def assess_confidence(
    top_retrieval_score: float | None,
    answer_text: str,
    valid_citation_count: int,
    invalid_citation_count: int,
    num_chunks_retrieved: int,
) -> ConfidenceAssessment:
    if num_chunks_retrieved == 0:
        return ConfidenceAssessment(
            level="low",
            relevance_score=0.0,
            citation_coverage=0.0,
            reasons=["No chunks were retrieved for this query."],
        )

    if valid_citation_count == 0 and answer_text.strip():
        return ConfidenceAssessment(
            level="low",
            relevance_score=_sigmoid(top_retrieval_score) if top_retrieval_score is not None else 0.0,
            citation_coverage=0.0,
            reasons=["The answer contains no valid citations to retrieved sources."],
        )

    relevance = _sigmoid(top_retrieval_score) if top_retrieval_score is not None else 0.0
    coverage = _citation_coverage(answer_text)
    reasons = []

    if relevance >= RELEVANCE_HIGH:
        reasons.append(f"Top retrieved source is strongly relevant (score={relevance:.2f}).")
    elif relevance >= RELEVANCE_MEDIUM:
        reasons.append(f"Top retrieved source is moderately relevant (score={relevance:.2f}).")
    else:
        reasons.append(f"Top retrieved source has low relevance (score={relevance:.2f}).")

    if coverage >= COVERAGE_HIGH:
        reasons.append(f"Most of the answer is directly cited ({coverage:.0%} of sentences).")
    elif coverage >= COVERAGE_MEDIUM:
        reasons.append(f"Some of the answer is cited ({coverage:.0%} of sentences).")
    else:
        reasons.append(f"Little of the answer is cited ({coverage:.0%} of sentences).")

    if invalid_citation_count > 0:
        reasons.append(
            f"{invalid_citation_count} citation(s) referenced a source that wasn't "
            f"actually retrieved -- treat with extra caution."
        )

    if relevance >= RELEVANCE_HIGH and coverage >= COVERAGE_HIGH and invalid_citation_count == 0:
        level = "high"
    elif relevance >= RELEVANCE_MEDIUM and coverage >= COVERAGE_MEDIUM:
        level = "medium"
    else:
        level = "low"

    return ConfidenceAssessment(
        level=level,
        relevance_score=relevance,
        citation_coverage=coverage,
        reasons=reasons,
    )


def format_confidence_warning(assessment: ConfidenceAssessment) -> str | None:
    if assessment.level == "high":
        return None

    label = "⚠ Low confidence" if assessment.level == "low" else "⚠ Moderate confidence"
    reason_text = " ".join(assessment.reasons)
    return f"{label}: {reason_text}"
