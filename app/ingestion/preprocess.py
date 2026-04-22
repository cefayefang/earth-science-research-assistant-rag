"""
Preprocessing pipeline — run once before serving queries.

Usage:
    python -m app.ingestion.preprocess
"""

import json
from pathlib import Path
from tqdm import tqdm

from ..core.config import get_settings, ROOT
from .paper_registry import build_paper_registry
from .dataset_normalizer import normalize_all_datasets
from .pdf_extractor import extract_pdf
from .chunker import chunk_paper
from .embedder import embed_datasets, embed_chunks
from ..core.schemas import Chunk, ParsedPaper


def run_preprocessing():
    cfg = get_settings()

    print("\n=== Step 1: Build paper registry ===")
    papers = build_paper_registry()

    print("\n=== Step 2: Normalize dataset metadata ===")
    datasets = normalize_all_datasets()

    print("\n=== Step 3: Extract and clean PDFs ===")
    parsed_dir = ROOT / cfg["paths"]["parsed_papers_dir"]
    parsed_dir.mkdir(parents=True, exist_ok=True)

    paper_lookup = {p.local_id: p for p in papers}
    all_chunks: list[Chunk] = []
    chunks_path = ROOT / cfg["paths"]["chunks_path"]

    for paper_record in tqdm(papers, desc="Extracting PDFs"):
        pdf_path = Path(paper_record.pdf_path)
        cache_path = parsed_dir / f"{paper_record.local_id}.json"

        if cache_path.exists():
            with open(cache_path) as f:
                parsed_data = json.load(f)
            parsed = ParsedPaper.model_validate(parsed_data)
        else:
            parsed = extract_pdf(pdf_path, paper_record.local_id, paper_record.openalex_id)
            if parsed is None:
                print(f"  [skip] {pdf_path.name} not found or failed")
                continue
            with open(cache_path, "w") as f:
                f.write(parsed.model_dump_json(indent=2))

        chunks = chunk_paper(
            parsed,
            chunk_size=cfg["pdf_processing"]["chunk_size_tokens"],
            overlap=cfg["pdf_processing"]["chunk_overlap_tokens"],
        )
        all_chunks.extend(chunks)

    print(f"Total chunks: {len(all_chunks)}")

    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    with open(chunks_path, "w") as f:
        for c in all_chunks:
            f.write(c.model_dump_json() + "\n")

    print("\n=== Step 4: Build vector indexes ===")
    print("Embedding datasets...")
    embed_datasets(datasets, batch_size=cfg["embeddings"]["batch_size"])

    print("Embedding chunks...")
    embed_chunks(all_chunks, batch_size=cfg["embeddings"]["batch_size"])

    print("\n✓ Preprocessing complete.")


if __name__ == "__main__":
    run_preprocessing()
