"""
Evaluation metrics for the grounded RAG pipeline.

All metrics are programmatic (no LLM-judge required):

  Retrieval quality
    • recall_at_k(retrieved, gold, k) : fraction of gold items present in top-k retrieved
    • mrr(retrieved, gold)            : mean reciprocal rank of first gold hit

  Grounding / faithfulness
    • parse_citation_tags(text)       : extract [DS-N]/[P-N]/[C-N] from any text
    • compute_grounding_rate(output, evidence_ids) : fraction of emitted tags resolvable

  Abstention (for out-of-scope queries)
    • is_abstention_correct(output, expected_empty)
"""
import re
from typing import Iterable


_TAG_RE = re.compile(r"\[(DS|P|C)-(\d+)\]")


# ── Retrieval metrics ────────────────────────────────────────────────────────

def recall_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int = 5) -> float | None:
    """
    Fraction of gold items appearing in the top-k retrieved ordered list.
    Returns None if gold is empty (signal that metric is not applicable).
    """
    if not gold_ids:
        return None
    top_k = set(retrieved_ids[:k])
    gold_set = set(gold_ids)
    hits = len(top_k & gold_set)
    return hits / len(gold_set)


def mrr(retrieved_ids: list[str], gold_ids: list[str]) -> float | None:
    """Mean reciprocal rank of first gold hit in the retrieved list (1-indexed)."""
    if not gold_ids:
        return None
    gold_set = set(gold_ids)
    for i, rid in enumerate(retrieved_ids, 1):
        if rid in gold_set:
            return 1.0 / i
    return 0.0


# ── Grounding metrics ────────────────────────────────────────────────────────

def parse_citation_tags(text: str) -> list[str]:
    """Extract short tags like 'DS-3', 'P-1', 'C-7' from any text blob."""
    if not text:
        return []
    return [f"{kind}-{n}" for kind, n in _TAG_RE.findall(text)]


def compute_grounding_rate(
    emitted_tags: Iterable[str],
    evidence_ids: set[str],
) -> dict:
    """
    emitted_tags: every tag the LLM emitted anywhere (text + structured fields)
    evidence_ids: the set of valid tags in the evidence block e.g. {'DS-1','P-3','C-7'}
    Returns dict with tags_total, tags_found, grounding_rate, unknown_tags.
    """
    tags = list(emitted_tags)
    total = len(tags)
    if total == 0:
        return {"tags_total": 0, "tags_found": 0, "grounding_rate": 1.0, "unknown_tags": []}
    unknown = [t for t in tags if t not in evidence_ids]
    found = total - len(unknown)
    return {
        "tags_total": total,
        "tags_found": found,
        "grounding_rate": round(found / total, 4),
        "unknown_tags": sorted(set(unknown)),
    }


def collect_emitted_tags(final_answer_dict: dict) -> list[str]:
    """Collect every citation tag the LLM emitted across all output fields."""
    tags: list[str] = []

    direct = final_answer_dict.get("direct_answer") or final_answer_dict.get("final_text") or ""
    tags += parse_citation_tags(direct)

    for item in final_answer_dict.get("recommended_datasets") or []:
        if isinstance(item, dict):
            tags += [c for c in (item.get("citations") or [])]
            # Also recommended_datasets emit a dataset_id-like field; we match the
            # item's dataset_id against the evidence block externally.
    for item in final_answer_dict.get("recommended_papers") or []:
        if isinstance(item, dict):
            tags += [c for c in (item.get("citations") or [])]
    for item in final_answer_dict.get("methodology_hints") or []:
        if isinstance(item, dict):
            tags += [c for c in (item.get("citations") or [])]
            tags += parse_citation_tags(item.get("hint", ""))

    return tags


# ── Abstention metric ────────────────────────────────────────────────────────

def is_abstention_correct(
    recommended_datasets: list,
    recommended_papers: list,
    uncertainty_notes: list[str],
) -> bool:
    """
    For out-of-scope queries, a system correctly abstains iff:
      - recommended_* are empty
      - uncertainty_notes mentions abstention / no evidence
    """
    if recommended_datasets or recommended_papers:
        return False
    notes_text = " ".join(uncertainty_notes or []).lower()
    abstention_keywords = [
        "no corpus evidence",
        "no evidence",
        "insufficient evidence",
        "out of scope",
        "cannot answer",
        "not available",
    ]
    return any(kw in notes_text for kw in abstention_keywords)


# ── Methodology fidelity ─────────────────────────────────────────────────────

def methodology_cites_chunks_rate(methodology_hints: list) -> float | None:
    """
    Of the non-empty methodology_hints, fraction that cite at least one [C-N] chunk.
    Returns None if no methodology_hints present.
    """
    if not methodology_hints:
        return None
    total = len(methodology_hints)
    ok = 0
    for h in methodology_hints:
        cits = h.get("citations") if isinstance(h, dict) else getattr(h, "citations", [])
        if any(str(c).startswith("C-") for c in (cits or [])):
            ok += 1
    return round(ok / total, 4) if total else None
