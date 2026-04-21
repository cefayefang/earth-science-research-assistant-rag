import json
import numpy as np
from .config import get_settings, ROOT
from .schemas import ParsedQuery, NormalizedDataset, DatasetCandidate
from .embedder import get_dataset_collection, query_embedding, embed_datasets, get_embedding_model
from .dataset_normalizer import load_normalized_datasets
from .zenodo_client import fetch_zenodo_datasets


def _variable_match(variables: list[str], keywords: list[str], variables_field: list[str]) -> float:
    if not variables:
        return 0.5
    all_dataset_terms = set(k.lower() for k in keywords + variables_field)
    matched = sum(1 for v in variables if any(v.lower() in t or t in v.lower() for t in all_dataset_terms))
    return min(matched / len(variables), 1.0)


def retrieve_datasets(
    parsed_query: ParsedQuery,
    openalex_dois: set[str],
    top_k: int | None = None,
) -> list[DatasetCandidate]:
    cfg = get_settings()
    if top_k is None:
        top_k = cfg["retrieval"]["dataset_top_k"]

    lit_scores = cfg["reranking"]["literature_support_scores"]
    default_st = cfg["reranking"]["spatial_temporal_default"]
    w = cfg["reranking"]["dataset_weights"]

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

    for i, did in enumerate(results["ids"][0]):
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        sim = max(0.0, 1.0 - distance)

        ds = dataset_lookup.get(did)
        keywords = meta.get("keywords", "").split(",")
        variables_field = meta.get("variables", "").split(",")
        var_match = _variable_match(parsed_query.variables, keywords, variables_field)

        candidates[did] = DatasetCandidate(
            dataset_id=did,
            source=meta.get("source", "unknown"),
            title=meta.get("title", ""),
            doi=meta.get("doi") or None,
            metadata_similarity=round(sim, 4),
            variable_match=round(var_match, 4),
            spatial_match=default_st,
            temporal_match=default_st,
            literature_support=lit_scores["semantic_only"],
        )

    # ── Zenodo retrieval ──────────────────────────────────────────────────────
    if parsed_query.zenodo_query:
        zenodo_results = fetch_zenodo_datasets(parsed_query.zenodo_query, openalex_dois)
        if zenodo_results:
            model = get_embedding_model()
            rec_texts = [r.retrieval_text for r, _ in zenodo_results]
            rec_vecs = model.encode(rec_texts, normalize_embeddings=True)
            qvec = np.array(query_vec)

            for (record, has_paper_link), rec_vec in zip(zenodo_results, rec_vecs):
                lit = lit_scores["zenodo_doi_matches_openalex"] if has_paper_link else lit_scores["semantic_only"]
                sim = float(np.dot(qvec, rec_vec))
                var_match = _variable_match(parsed_query.variables, record.keywords, record.variables)

                candidates[record.dataset_id] = DatasetCandidate(
                    dataset_id=record.dataset_id,
                    source="zenodo",
                    title=record.display_name,
                    doi=record.doi,
                    metadata_similarity=round(max(0.0, sim), 4),
                    variable_match=round(var_match, 4),
                    spatial_match=default_st,
                    temporal_match=default_st,
                    literature_support=lit,
                )

    # ── Score and rank ────────────────────────────────────────────────────────
    ranked = []
    for cand in candidates.values():
        score = (
            w["semantic_similarity"] * cand.metadata_similarity
            + w["variable_match"] * cand.variable_match
            + w["literature_support"] * cand.literature_support
            + w["spatial_match"] * cand.spatial_match
            + w["temporal_match"] * cand.temporal_match
        )
        cand.dataset_score = round(score, 4)
        ranked.append(cand)

    ranked.sort(key=lambda c: c.dataset_score, reverse=True)
    top = ranked[:top_k]

    debug_dir = ROOT / "generated" / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    with open(debug_dir / "last_dataset_candidates.json", "w") as f:
        json.dump([c.model_dump() for c in top], f, indent=2)

    return top
