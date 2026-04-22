"""
Dataset retrieval & feature extraction (scoring lives in reranker.rerank_datasets).

For each candidate we compute the following features:
    metadata_similarity  — query vs dataset description via embedding cosine
    variable_match       — overlap between query variables and dataset keywords/variables
    spatial_match        — bbox overlap (uses `spatial_temporal_default` when unknown)
    temporal_match       — time-range overlap
    literature_support   — baseline: has_doi (0.7) if DOI present else semantic_only (0.5)
                           (linker.build_links later upgrades this if the dataset is
                            explicitly mentioned in retrieved chunks/abstracts)

Hard filtering (Phase "refactor"): when the user's query marks region or timescale
as a must-have constraint, candidates that fall below the configured match threshold
are dropped here — before they reach the scoring/LLM-context stages.

This module does NOT compute the final `dataset_score`. That is done in
`reranker.rerank_datasets` so that scoring lives in exactly one place.
"""
import json
import numpy as np
from ..core.config import get_settings, ROOT
from ..core.schemas import ParsedQuery, NormalizedDataset, DatasetCandidate
from ..ingestion.embedder import get_dataset_collection, query_embedding, embed_datasets, get_embedding_model
from ..ingestion.dataset_normalizer import load_normalized_datasets
from ..clients.zenodo_client import fetch_zenodo_datasets
from ..core.spatial_temporal_match import (
    parse_dataset_bbox, parse_dataset_temporal,
    bbox_overlap_score, temporal_overlap_score,
)


def _variable_match(variables: list[str], keywords: list[str], variables_field: list[str]) -> float:
    if not variables:
        return 0.5
    all_dataset_terms = set(k.lower() for k in keywords + variables_field)
    matched = sum(1 for v in variables if any(v.lower() in t or t in v.lower() for t in all_dataset_terms))
    return min(matched / len(variables), 1.0)


def _baseline_literature_support(has_doi: bool, lit_scores: dict) -> float:
    """has_doi → 0.7, else → 0.5. build_links may upgrade to 0.85 / 1.0."""
    return lit_scores["has_doi"] if has_doi else lit_scores["semantic_only"]


def _apply_must_have_filter(
    candidates: dict[str, DatasetCandidate],
    parsed_query: ParsedQuery,
    cfg: dict,
) -> dict[str, DatasetCandidate]:
    """Drop candidates that fail the user's hard region/timescale constraints."""
    gating = cfg["reranking"].get("must_have_gating", {})
    region_thr = gating.get("region_threshold", 0.3)
    temporal_thr = gating.get("temporal_threshold", 0.3)

    must = parsed_query.must_have_constraints
    kept: dict[str, DatasetCandidate] = {}
    for cid, cand in candidates.items():
        if must.region and cand.spatial_match < region_thr:
            continue
        if must.timescale and cand.temporal_match < temporal_thr:
            continue
        kept[cid] = cand
    return kept


def retrieve_datasets(
    parsed_query: ParsedQuery,
    openalex_dois: set[str],
    top_k: int | None = None,
) -> tuple[list[DatasetCandidate], list[NormalizedDataset]]:
    """Returns (dataset_candidates, zenodo_records_retrieved_this_query).

    The returned `dataset_candidates` carry raw features; final scoring happens
    in reranker.rerank_datasets. Order here is not meaningful — the list may be
    up to `top_k * 2` long to give the reranker some headroom before it trims.
    """
    cfg = get_settings()
    if top_k is None:
        top_k = cfg["retrieval"]["dataset_top_k"]

    lit_scores = cfg["reranking"]["literature_support_scores"]
    default_st = cfg["reranking"]["spatial_temporal_default"]

    # make sure local datasets are embedded
    datasets = load_normalized_datasets()
    collection = get_dataset_collection()
    if collection.count() == 0:
        embed_datasets(datasets)

    # ── Local retrieval ───────────────────────────────────────────────────────
    query_vec = query_embedding(parsed_query.local_query)
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=min(top_k * 2, collection.count()),
        include=["metadatas", "distances", "documents"],
    )

    dataset_lookup = {d.dataset_id: d for d in datasets}
    candidates: dict[str, DatasetCandidate] = {}

    q_bbox = parsed_query.region_bbox
    q_temp = parsed_query.parsed_timescale

    for i, did in enumerate(results["ids"][0]):
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        sim = max(0.0, 1.0 - distance)

        ds = dataset_lookup.get(did)
        keywords = meta.get("keywords", "").split(",")
        variables_field = meta.get("variables", "").split(",")
        var_match = _variable_match(parsed_query.variables, keywords, variables_field)

        ds_bbox = parse_dataset_bbox(ds.spatial_info) if ds else None
        ds_temp = parse_dataset_temporal(ds.temporal_info) if ds else None
        spatial_score = bbox_overlap_score(q_bbox, ds_bbox, default=default_st)
        temporal_score = temporal_overlap_score(q_temp, ds_temp, default=default_st)

        has_doi = bool(meta.get("doi"))

        candidates[did] = DatasetCandidate(
            dataset_id=did,
            source=meta.get("source", "unknown"),
            title=meta.get("title", ""),
            doi=meta.get("doi") or None,
            metadata_similarity=round(sim, 4),
            variable_match=round(var_match, 4),
            spatial_match=round(spatial_score, 4),
            temporal_match=round(temporal_score, 4),
            literature_support=_baseline_literature_support(has_doi, lit_scores),
        )

    # ── Zenodo retrieval ──────────────────────────────────────────────────────
    # Note: `openalex_dois` is kept in the signature for backwards compatibility
    # but we no longer use it for the dead `zenodo_doi_matches_openalex` branch.
    # Zenodo records get the has_doi baseline like any other DOI-bearing source.
    zenodo_records: list[NormalizedDataset] = []
    if parsed_query.zenodo_query:
        zenodo_results = fetch_zenodo_datasets(parsed_query.zenodo_query, openalex_dois)
        if zenodo_results:
            model = get_embedding_model()
            rec_texts = [r.retrieval_text for r, _ in zenodo_results]
            rec_vecs = model.encode(rec_texts, normalize_embeddings=True)
            qvec = np.array(query_vec)

            for (record, _has_paper_link), rec_vec in zip(zenodo_results, rec_vecs):
                sim = float(np.dot(qvec, rec_vec))
                var_match = _variable_match(parsed_query.variables, record.keywords, record.variables)

                z_bbox = parse_dataset_bbox(record.spatial_info)
                z_temp = parse_dataset_temporal(record.temporal_info)
                z_spatial = bbox_overlap_score(q_bbox, z_bbox, default=default_st)
                z_temporal = temporal_overlap_score(q_temp, z_temp, default=default_st)

                has_doi = bool(record.doi)

                candidates[record.dataset_id] = DatasetCandidate(
                    dataset_id=record.dataset_id,
                    source="zenodo",
                    title=record.display_name,
                    doi=record.doi,
                    metadata_similarity=round(max(0.0, sim), 4),
                    variable_match=round(var_match, 4),
                    spatial_match=round(z_spatial, 4),
                    temporal_match=round(z_temporal, 4),
                    literature_support=_baseline_literature_support(has_doi, lit_scores),
                )
                zenodo_records.append(record)

    # ── Hard filter on must-have constraints ──────────────────────────────────
    candidates = _apply_must_have_filter(candidates, parsed_query, cfg)

    ranked = list(candidates.values())  # order-agnostic; reranker will re-sort by score

    debug_dir = ROOT / "generated" / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    with open(debug_dir / "last_dataset_candidates.json", "w") as f:
        json.dump([c.model_dump() for c in ranked], f, indent=2)

    return ranked, zenodo_records
