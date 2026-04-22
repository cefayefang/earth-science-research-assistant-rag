"""
Evaluation metrics for the grounded RAG pipeline.

All metrics are programmatic (no LLM-judge required):

  Retrieval quality
    • recall_at_k(retrieved, gold, k)          : fraction of gold items in top-k ordered list
    • recall_from_pool(pool, gold)              : fraction of gold items in an unordered candidate pool
    • precision_at_k(retrieved, gold, k)        : fraction of top-k items that are gold
    • f1_at_k(retrieved, gold, k)               : harmonic mean of precision@k and recall@k
    • mrr(retrieved, gold)                      : mean reciprocal rank of first gold hit

  Grounding / faithfulness
    • parse_citation_tags(text)                : extract [DS-N]/[P-N]/[C-N] from any text
    • compute_grounding_rate(emitted, evidence) : fraction of emitted tags resolvable
    • collect_emitted_tags(answer_dict)         : collect all citation tags across all output fields
    • citation_coverage(emitted_tags)           : whether any citations were emitted at all

  Abstention (for out-of-scope queries)
    • is_abstention_correct(datasets, papers, notes)

  Methodology fidelity
    • methodology_cites_chunks_rate(hints)      : fraction of hints citing at least one [C-N] chunk

  Answer quality (direct_answer category)
    • rouge_l_score(hypothesis, reference)      : token-level ROUGE-L F1, no external library
"""
import re
from typing import Iterable


_TAG_RE = re.compile(r"\[(DS|P|C)-(\d+)\]")


# ── Retrieval metrics ────────────────────────────────────────────────────────

def recall_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int = 5) -> float | None:
    """
    Fraction of gold items appearing in the top-k retrieved ordered list.
    Returns None if gold is empty (metric not applicable).
    """
    if not gold_ids:
        return None
    top_k = set(retrieved_ids[:k])
    gold_set = set(gold_ids)
    hits = len(top_k & gold_set)
    return round(hits / len(gold_set), 4)


def recall_from_pool(pool: set[str], gold_ids: list[str]) -> float | None:
    """
    Fraction of gold items present in an unordered candidate pool.
    Use this when merging retrieval channels where a single ordering is not preserved
    (e.g. union of top-k from two independent rankers).
    Returns None if gold is empty.
    """
    if not gold_ids:
        return None
    gold_set = set(gold_ids)
    return round(len(pool & gold_set) / len(gold_set), 4)


def precision_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int = 5) -> float | None:
    """
    Fraction of the top-k retrieved items that are gold.
    Returns None if gold is empty (metric not applicable).
    Returns 0.0 if the retrieved list is empty.
    """
    if not gold_ids:
        return None
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    gold_set = set(gold_ids)
    hits = sum(1 for rid in top_k if rid in gold_set)
    return round(hits / len(top_k), 4)


def f1_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int = 5) -> float | None:
    """Harmonic mean of precision@k and recall@k on the same ordered list."""
    p = precision_at_k(retrieved_ids, gold_ids, k)
    r = recall_at_k(retrieved_ids, gold_ids, k)
    if p is None or r is None:
        return None
    if p + r == 0:
        return 0.0
    return round(2 * p * r / (p + r), 4)


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

    grounding_rate is None when no tags are emitted — the metric is not applicable.
    Returning 1.0 for the zero-tag case is vacuously true and inflates category averages.
    """
    tags = list(emitted_tags)
    total = len(tags)
    if total == 0:
        return {"tags_total": 0, "tags_found": 0, "grounding_rate": None, "unknown_tags": []}
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
            # Parse inline citation tags from all text fields in the item.
            for field in ("reason", "summary", "description"):
                tags += parse_citation_tags(item.get(field) or "")
    for item in final_answer_dict.get("recommended_papers") or []:
        if isinstance(item, dict):
            tags += [c for c in (item.get("citations") or [])]
            for field in ("reason", "summary"):
                tags += parse_citation_tags(item.get(field) or "")
    for item in final_answer_dict.get("methodology_hints") or []:
        if isinstance(item, dict):
            tags += [c for c in (item.get("citations") or [])]
            tags += parse_citation_tags(item.get("hint", ""))

    return tags


def citation_coverage(emitted_tags: list[str]) -> dict:
    """
    Measures whether the system cited anything, independent of citation validity.
    Complements grounding_rate:
      grounding_rate     — of what was cited, how much resolves to evidence
      citation_coverage  — was anything cited at all (catches silent non-citation)
    """
    unique = set(emitted_tags)
    return {
        "has_any_citation": len(unique) > 0,
        "unique_sources_cited": len(unique),
    }


# ── Abstention metric ────────────────────────────────────────────────────────

def is_abstention_correct(
    recommended_datasets: list,
    recommended_papers: list,
    uncertainty_notes: list[str],
) -> bool:
    """
    For out-of-scope queries, a system correctly abstains iff:
      - recommended_* are empty
      - uncertainty_notes mentions abstention / no evidence (broad keyword set)
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
        "outside the scope",
        "not in the corpus",
        "no relevant",
        "no data",
        "not covered",
        "beyond the scope",
        "no information",
        "unable to find",
        "not found in",
        "outside my",
        "scope of",
        "cannot provide",
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
    return round(ok / total, 4)


# ── Answer quality (direct_answer category) ──────────────────────────────────

def _lcs_length(a: list, b: list) -> int:
    """Length of longest common subsequence (space-optimized DP)."""
    n = len(b)
    prev = [0] * (n + 1)
    for token_a in a:
        curr = [0] * (n + 1)
        for j, token_b in enumerate(b, 1):
            if token_a == token_b:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr
    return prev[n]


def rouge_l_score(hypothesis: str, reference: str) -> float | None:
    """
    Token-level ROUGE-L F1 score (no external library required).
    Returns None if either input is empty.
    """
    if not hypothesis or not reference:
        return None
    h_tokens = hypothesis.lower().split()
    r_tokens = reference.lower().split()
    if not h_tokens or not r_tokens:
        return None
    lcs = _lcs_length(h_tokens, r_tokens)
    precision = lcs / len(h_tokens)
    rec = lcs / len(r_tokens)
    if precision + rec == 0:
        return 0.0
    return round(2 * precision * rec / (precision + rec), 4)
