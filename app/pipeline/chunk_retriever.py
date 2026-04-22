import json
from ..core.config import get_settings, ROOT
from ..core.schemas import ParsedQuery, ChunkCandidate
from ..ingestion.embedder import get_chunk_collection, query_embedding


def retrieve_chunks_for_paper(
    local_id: str,
    question: str,
    top_k: int = 5,
) -> list[ChunkCandidate]:
    """Semantic search WITHIN a single paper's chunks.

    Used by detail_followup so that when the user asks about a specific local
    paper, we can draw on the whole paper (not just whichever chunks happened
    to surface in the last turn's top-10). Chroma's `where={"local_id": ...}`
    filter restricts the nearest-neighbor search to that paper's chunks.
    """
    if not local_id:
        return []
    collection = get_chunk_collection()
    if collection.count() == 0:
        return []

    query_vec = query_embedding(question)
    try:
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            where={"local_id": local_id},
            include=["metadatas", "distances", "documents"],
        )
    except Exception as e:
        # If the filter query fails (e.g. no chunks exist for this paper),
        # return empty and let caller fall back to other strategies.
        print(f"  [chunk_retriever warn] per-paper query failed for {local_id}: {e}")
        return []

    ids_outer = results.get("ids") or [[]]
    ids = ids_outer[0] if ids_outer else []
    if not ids:
        return []

    candidates: list[ChunkCandidate] = []
    for i, cid in enumerate(ids):
        meta = results["metadatas"][0][i] or {}
        distance = results["distances"][0][i]
        sim = max(0.0, 1.0 - distance)
        candidates.append(ChunkCandidate(
            chunk_id=cid,
            local_id=meta.get("local_id", "") or local_id,
            openalex_id=meta.get("openalex_id") or None,
            section_guess=meta.get("section_guess") or None,
            chunk_score=round(sim, 4),
            text=results["documents"][0][i],
        ))
    candidates.sort(key=lambda c: c.chunk_score, reverse=True)
    return candidates


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
