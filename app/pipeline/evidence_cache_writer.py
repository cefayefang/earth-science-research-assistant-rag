"""
Per-query evidence cache.

After each query, we snapshot the full retrieval state to disk so that:
  (1) The 'local evidence cache' requirement is literally satisfied — at the
      moment of LLM invocation, every retrieved record is on disk.
  (2) Evaluations are fully reproducible (same query_id = same evidence).
  (3) Failure analysis can inspect exactly what the system considered.

Output directory:
  generated/evidence_cache/<timestamp>_<query_hash>/
    ├── parsed_query.json
    ├── openalex.jsonl            (runtime-retrieved papers)
    ├── zenodo.jsonl              (runtime-retrieved datasets)
    ├── local_datasets.jsonl      (candidate datasets from local ChromaDB)
    ├── chunks.jsonl              (candidate evidence chunks)
    ├── evidence_block.txt        (exact text passed into the LLM prompt)
    ├── final_answer.json
    └── grounding_report.json
"""
import json
import hashlib
import datetime
from pathlib import Path
from ..core.config import get_settings, ROOT
from ..core.schemas import (
    ParsedQuery, OpenAlexPaper, NormalizedDataset,
    DatasetCandidate, ChunkCandidate, FinalAnswer,
)


def _make_query_id(query: str) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    qhash = hashlib.sha1(query.encode()).hexdigest()[:8]
    return f"{ts}_{qhash}"


def _write_jsonl(path: Path, items: list) -> None:
    with open(path, "w") as f:
        for it in items:
            if hasattr(it, "model_dump_json"):
                f.write(it.model_dump_json() + "\n")
            else:
                f.write(json.dumps(it, default=str) + "\n")


def write_evidence_cache(
    *,
    query: str,
    parsed: ParsedQuery,
    openalex_papers: list[OpenAlexPaper],
    zenodo_records: list[NormalizedDataset] | None,
    local_dataset_candidates: list[DatasetCandidate],
    chunk_candidates: list[ChunkCandidate],
    evidence_block_text: str,
    final_answer: FinalAnswer,
) -> Path:
    """Returns the cache directory path."""
    cfg = get_settings()
    cache_root = ROOT / cfg["paths"].get("evidence_cache_dir", "generated/evidence_cache")
    cache_root.mkdir(parents=True, exist_ok=True)

    query_id = _make_query_id(query)
    out_dir = cache_root / query_id
    out_dir.mkdir(exist_ok=True)

    # 1. parsed query
    with open(out_dir / "parsed_query.json", "w") as f:
        f.write(parsed.model_dump_json(indent=2))

    # 2. OpenAlex candidates (runtime)
    _write_jsonl(out_dir / "openalex.jsonl", openalex_papers)

    # 3. Zenodo candidates (runtime)
    _write_jsonl(out_dir / "zenodo.jsonl", zenodo_records or [])

    # 4. Local dataset candidates (ChromaDB-retrieved from persistent cache)
    _write_jsonl(out_dir / "local_datasets.jsonl", local_dataset_candidates)

    # 5. Chunk candidates
    _write_jsonl(out_dir / "chunks.jsonl", chunk_candidates)

    # 6. Evidence block as a single string (exactly what was sent to LLM)
    with open(out_dir / "evidence_block.txt", "w") as f:
        f.write(evidence_block_text)

    # 7. Final answer
    with open(out_dir / "final_answer.json", "w") as f:
        f.write(final_answer.model_dump_json(indent=2))

    # 8. Grounding report (extracted from final_answer for easy inspection)
    if final_answer.grounding_report is not None:
        with open(out_dir / "grounding_report.json", "w") as f:
            f.write(final_answer.grounding_report.model_dump_json(indent=2))

    # Manifest listing everything for quick inspection
    manifest = {
        "query": query,
        "query_id": query_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "counts": {
            "openalex_papers": len(openalex_papers),
            "zenodo_records": len(zenodo_records or []),
            "local_dataset_candidates": len(local_dataset_candidates),
            "chunk_candidates": len(chunk_candidates),
        },
        "answer_mode": final_answer.answer_mode,
        "grounding_report": (
            final_answer.grounding_report.model_dump()
            if final_answer.grounding_report else None
        ),
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return out_dir
