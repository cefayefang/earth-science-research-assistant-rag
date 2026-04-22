"""
Paper and dataset reranking.

Design (post-refactor):

Paper scoring uses provenance-specific feature sets. The two feature sets are
mutually exclusive, so there is no double-counting and "local preferred" is a
natural consequence of the score ceilings, not a hand-tuned tier bonus.

    Local paper (fulltext_supported):
        score = 0.40 * chunk_relevance
              + 0.25 * fulltext_bonus       (1.0 — flat)
              + 0.10 * recency_score
              + 0.10 * impact_score
        ceiling ≈ 0.85

    External paper (OpenAlex, metadata-only):
        score = 0.40 * semantic_similarity
              + 0.10 * recency_score
              + 0.10 * impact_score
        ceiling ≈ 0.60

Selection uses a fixed quota (local: 7, external: 3) with overflow. This lives
in the selection layer, not the scoring layer.

Dataset scoring is a single weighted sum over features that were extracted in
dataset_retriever + (optionally) upgraded by linker.build_links:
    score = 0.35 * metadata_similarity
          + 0.20 * variable_match
          + 0.25 * literature_support
          + 0.10 * spatial_match
          + 0.10 * temporal_match

Cross-source dataset deduplication (e.g. canonical vs subset) is handled by the
LLM at answer-generation time via a prompt rule, not in this module.
"""
import json
import math
import datetime
from ..core.config import get_settings, ROOT
from ..core.schemas import (
    OpenAlexPaper, PaperMatch, PaperCandidate,
    DatasetCandidate, ChunkCandidate, PaperRecord,
)
from ..ingestion.embedder import query_embedding, get_embedding_model
from ..ingestion.paper_registry import load_paper_registry
import numpy as np

CURRENT_YEAR = datetime.date.today().year
MIN_YEAR = 2010


def _recency_score(year: int | None) -> float:
    if not year:
        return 0.3
    return max(0.0, min(1.0, (year - MIN_YEAR) / (CURRENT_YEAR - MIN_YEAR)))


def _impact_score(cited_by_count: int) -> float:
    if cited_by_count <= 0:
        return 0.0
    return min(1.0, math.log10(cited_by_count + 1) / math.log10(5001))


def rerank_papers(
    openalex_papers: list[OpenAlexPaper],
    paper_matches: list[PaperMatch],
    chunk_candidates: list[ChunkCandidate],
    local_query: str,
    exclude_paper_ids: list[str] | None = None,
) -> list[PaperCandidate]:
    """Fuse chunk-based local hits with OpenAlex results, score each side with
    its own formula, select via the local/external quota, and return a flat
    ranked list.

    When `exclude_paper_ids` is provided (by the caller — in practice only on
    `re_recommend` turns, where main._run_pipeline pulls the list out of the
    session state), papers whose local_id or openalex_id appears in the set are
    dropped at the selection step so the user sees genuinely new recommendations.
    """
    exclude_set = set(exclude_paper_ids or [])
    cfg = get_settings()
    w = cfg["reranking"]["paper_weights"]
    quota = cfg["reranking"].get("paper_tier_quota", {"local": 7, "external": 3})

    # ── 1) Aggregate chunk evidence per local paper ───────────────────────────
    # A local paper surfaces only if at least one of its chunks was retrieved.
    chunk_scores: dict[str, float] = {}
    for c in chunk_candidates:
        if c.local_id:
            chunk_scores[c.local_id] = max(chunk_scores.get(c.local_id, 0.0), c.chunk_score)

    openalex_local_id_map = {m.openalex_id: m.local_id for m in paper_matches}

    registry: list[PaperRecord] = load_paper_registry()
    registry_by_local: dict[str, PaperRecord] = {r.local_id: r for r in registry}

    # ── 2) Seed candidate pool with chunk-backed local papers ─────────────────
    # key scheme: "local::{local_id}" for local, "ext::{openalex_id}" for external
    candidates: dict[str, dict] = {}

    for local_id, chunk_rel in chunk_scores.items():
        rec = registry_by_local.get(local_id)
        if not rec:
            continue
        # Seed from registry: year / abstract / doi / cited_by_count are
        # pre-enriched by build_paper_registry. If this query's OpenAlex
        # search also returns the same paper, the merge step below will
        # overwrite with fresher values.
        candidates[f"local::{local_id}"] = {
            "provenance": "local",
            "local_id": local_id,
            "openalex_id": rec.openalex_id,
            "title": rec.original_title,
            "abstract": rec.abstract,
            "year": rec.year,
            "doi": rec.doi,
            "authors": [],
            "cited_by_count": rec.cited_by_count or 0,
            "semantic_similarity": 0.0,  # unused for local
            "chunk_relevance": chunk_rel,
        }

    # ── 3) Merge OpenAlex into local (if matched) or add as external ──────────
    if openalex_papers:
        model = get_embedding_model()
        q_vec = np.array(query_embedding(local_query))
        oa_texts = [f"{p.title}. {p.abstract or ''}" for p in openalex_papers]
        oa_vecs = model.encode(oa_texts, normalize_embeddings=True)
        oa_sims = (oa_vecs @ q_vec).tolist()
    else:
        oa_sims = []

    for paper, sim in zip(openalex_papers, oa_sims):
        matched_local = openalex_local_id_map.get(paper.openalex_id)
        if matched_local:
            key = f"local::{matched_local}"
            if key in candidates:
                # Enrich the existing local candidate with OpenAlex metadata
                candidates[key]["openalex_id"] = paper.openalex_id
                candidates[key]["abstract"] = paper.abstract
                candidates[key]["year"] = paper.year
                candidates[key]["doi"] = paper.doi
                candidates[key]["authors"] = paper.authors
                candidates[key]["cited_by_count"] = paper.cited_by_count
                continue
            # OpenAlex matched a local paper that had no chunk hit. Since we
            # dropped the paper-level semantic retrieval, this paper has no
            # local evidence and is treated as external (abstract-only).
            # (This preserves full-text awareness via evidence_level below.)
            # Fall through to the external branch.

        # Pure external (no local match, or matched but no chunk hit)
        candidates[f"ext::{paper.openalex_id}"] = {
            "provenance": "external",
            "local_id": matched_local,  # kept for evidence_level tagging
            "openalex_id": paper.openalex_id,
            "title": paper.title,
            "abstract": paper.abstract,
            "year": paper.year,
            "doi": paper.doi,
            "authors": paper.authors,
            "cited_by_count": paper.cited_by_count,
            "semantic_similarity": sim,
            "chunk_relevance": 0.0,  # unused for external
        }

    # ── 4) Score each candidate using the provenance-specific formula ─────────
    ranked: list[PaperCandidate] = []
    for cand in candidates.values():
        is_local = cand["provenance"] == "local"
        has_fulltext = is_local or bool(cand["local_id"])
        fulltext_bonus = 1.0 if is_local else 0.0

        if is_local:
            paper_score = (
                w["chunk_relevance"] * cand["chunk_relevance"]
                + w["fulltext_bonus"] * fulltext_bonus
                + w["recency_score"] * _recency_score(cand["year"])
                + w["impact_score"] * _impact_score(cand["cited_by_count"])
            )
        else:
            paper_score = (
                w["semantic_similarity"] * cand["semantic_similarity"]
                + w["recency_score"] * _recency_score(cand["year"])
                + w["impact_score"] * _impact_score(cand["cited_by_count"])
            )

        ranked.append(PaperCandidate(
            openalex_id=cand["openalex_id"] or f"local_only_{cand['local_id']}",
            local_id=cand["local_id"],
            title=cand["title"] or "",
            abstract=cand["abstract"],
            year=cand["year"],
            doi=cand["doi"],
            authors=cand["authors"],
            cited_by_count=cand["cited_by_count"],
            evidence_level="fulltext_supported" if has_fulltext else "metadata_only",
            semantic_similarity=round(cand["semantic_similarity"], 4),
            chunk_relevance=round(cand["chunk_relevance"], 4),
            recency_score=round(_recency_score(cand["year"]), 4),
            impact_score=round(_impact_score(cand["cited_by_count"]), 4),
            fulltext_bonus=fulltext_bonus,
            paper_score=round(paper_score, 4),
        ))

    # ── 5) Selection: local/external quota with overflow ──────────────────────
    local_quota = quota.get("local", 7)
    external_quota = quota.get("external", 3)
    target_size = local_quota + external_quota

    # Use the candidate dict to recover provenance for each PaperCandidate
    locals_list: list[PaperCandidate] = []
    externals_list: list[PaperCandidate] = []
    for cand, pc in zip(candidates.values(), ranked):
        # Apply exclude filter (only meaningful if caller passed a non-empty set)
        if exclude_set and (
            (pc.local_id and pc.local_id in exclude_set)
            or (pc.openalex_id and pc.openalex_id in exclude_set)
        ):
            continue
        if cand["provenance"] == "local":
            locals_list.append(pc)
        else:
            externals_list.append(pc)

    locals_list.sort(key=lambda p: p.paper_score, reverse=True)
    externals_list.sort(key=lambda p: p.paper_score, reverse=True)

    selected = locals_list[:local_quota] + externals_list[:external_quota]
    leftovers = locals_list[local_quota:] + externals_list[external_quota:]
    leftovers.sort(key=lambda p: p.paper_score, reverse=True)
    while len(selected) < target_size and leftovers:
        selected.append(leftovers.pop(0))
    # Append remaining leftovers at the tail for full observability
    selected.extend(leftovers)

    # ── Debug ─────────────────────────────────────────────────────────────────
    debug_dir = ROOT / "generated" / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    with open(debug_dir / "last_reranked_papers.json", "w") as f:
        json.dump([p.model_dump() for p in selected], f, indent=2)
    with open(debug_dir / "last_paper_tiers.json", "w") as f:
        json.dump({
            "local_n":    len(locals_list),
            "external_n": len(externals_list),
            "quota":      {"local": local_quota, "external": external_quota},
            "final_top_provenance": [
                "local" if p in locals_list else "external"
                for p in selected[:target_size]
            ],
        }, f, indent=2)

    return selected


def rerank_datasets(
    dataset_candidates: list[DatasetCandidate],
    exclude_dataset_ids: list[str] | None = None,
) -> list[DatasetCandidate]:
    """Single source of truth for dataset scoring. Applies the weighted-sum
    formula to the pre-extracted features and sorts descending.

    When `exclude_dataset_ids` is non-empty, the corresponding candidates are
    dropped before sorting — used for expansion follow-ups ("more datasets").
    """
    cfg = get_settings()
    w = cfg["reranking"]["dataset_weights"]

    exclude_set = set(exclude_dataset_ids or [])
    if exclude_set:
        dataset_candidates = [
            d for d in dataset_candidates if d.dataset_id not in exclude_set
        ]

    for cand in dataset_candidates:
        cand.dataset_score = round(
            w["semantic_similarity"] * cand.metadata_similarity
            + w["variable_match"] * cand.variable_match
            + w["literature_support"] * cand.literature_support
            + w["spatial_match"] * cand.spatial_match
            + w["temporal_match"] * cand.temporal_match,
            4,
        )

    dataset_candidates.sort(key=lambda d: d.dataset_score, reverse=True)

    debug_dir = ROOT / "generated" / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    with open(debug_dir / "last_reranked_datasets.json", "w") as f:
        json.dump([d.model_dump() for d in dataset_candidates], f, indent=2)

    return dataset_candidates
