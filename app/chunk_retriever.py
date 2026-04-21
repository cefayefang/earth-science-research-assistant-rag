import json
from .config import get_settings, ROOT
from .schemas import ParsedQuery, ChunkCandidate
from .embedder import get_chunk_collection, query_embedding


def retrieve_chunks(parsed_query: ParsedQuery, top_k: int | None = None) -> list[ChunkCandidate]:
    cfg = get_settings()
    if top_k is None:
        top_k = cfg["retrieval"]["chunk_top_k"]

    collection = get_chunk_collection()
    if collection.count() == 0:
        return []

    query_vec = query_embedding(parsed_query.local_query)
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=min(top_k, collection.count()),
        include=["metadatas", "distances", "documents"],
    )

    candidates = []
    for i, cid in enumerate(results["ids"][0]):
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        sim = max(0.0, 1.0 - distance)
        candidates.append(ChunkCandidate(
            chunk_id=cid,
            local_id=meta.get("local_id", ""),
            openalex_id=meta.get("openalex_id") or None,
            section_guess=meta.get("section_guess") or None,
            chunk_score=round(sim, 4),
            text=results["documents"][0][i],
        ))

    candidates.sort(key=lambda c: c.chunk_score, reverse=True)

    debug_dir = ROOT / "generated" / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    with open(debug_dir / "last_chunk_candidates.json", "w") as f:
        json.dump([c.model_dump() for c in candidates], f, indent=2)

    return candidates
