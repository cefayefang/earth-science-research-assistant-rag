import json
import math
import datetime
from .config import get_settings, ROOT
from .schemas import (
    OpenAlexPaper, PaperMatch, PaperCandidate,
    DatasetCandidate, ChunkCandidate,
)
from .embedder import query_embedding, get_embedding_model
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
) -> list[PaperCandidate]:
    cfg = get_settings()
    w = cfg["reranking"]["paper_weights"]

    fulltext_ids = {m.local_id for m in paper_matches if m.evidence_level == "fulltext_supported" and m.local_id}
    local_id_map = {m.openalex_id: m.local_id for m in paper_matches}

    # best chunk score per local_id
    chunk_scores: dict[str, float] = {}
    for chunk in chunk_candidates:
        lid = chunk.local_id
        chunk_scores[lid] = max(chunk_scores.get(lid, 0.0), chunk.chunk_score)

    query_vec = np.array(query_embedding(local_query))

    # Batch encode all paper texts in one model call
    paper_texts = [f"{p.title}. {p.abstract or ''}" for p in openalex_papers]
    if paper_texts:
        paper_vecs = get_embedding_model().encode(paper_texts, normalize_embeddings=True)
        sims = (paper_vecs @ query_vec).tolist()
    else:
        sims = []

    ranked: list[PaperCandidate] = []

    for paper, sim in zip(openalex_papers, sims):
        local_id = local_id_map.get(paper.openalex_id)
        has_fulltext = local_id in fulltext_ids if local_id else False
        chunk_rel = chunk_scores.get(local_id, 0.0) if local_id else 0.0
        fulltext_bonus = 0.1 if has_fulltext else 0.0

        paper_score = (
            w["semantic_similarity"] * sim
            + w["chunk_relevance"] * chunk_rel
            + w["recency_score"] * _recency_score(paper.year)
            + w["impact_score"] * _impact_score(paper.cited_by_count)
            + w["fulltext_bonus"] * fulltext_bonus
        )

        ranked.append(PaperCandidate(
            openalex_id=paper.openalex_id,
            local_id=local_id,
            title=paper.title,
            abstract=paper.abstract,
            year=paper.year,
            doi=paper.doi,
            authors=paper.authors,
            cited_by_count=paper.cited_by_count,
            evidence_level="fulltext_supported" if has_fulltext else "metadata_only",
            semantic_similarity=round(sim, 4),
            chunk_relevance=round(chunk_rel, 4),
            recency_score=round(_recency_score(paper.year), 4),
            impact_score=round(_impact_score(paper.cited_by_count), 4),
            fulltext_bonus=fulltext_bonus,
            paper_score=round(paper_score, 4),
        ))

    ranked.sort(key=lambda p: p.paper_score, reverse=True)

    debug_dir = ROOT / "generated" / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    with open(debug_dir / "last_reranked_papers.json", "w") as f:
        json.dump([p.model_dump() for p in ranked], f, indent=2)

    return ranked


def rerank_datasets(dataset_candidates: list[DatasetCandidate]) -> list[DatasetCandidate]:
    cfg = get_settings()
    w = cfg["reranking"]["dataset_weights"]

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
