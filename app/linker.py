import json
import re
from pathlib import Path
from rapidfuzz import fuzz
from .config import get_settings, ROOT
from .schemas import DatasetCandidate, ChunkCandidate, OpenAlexPaper, DatasetLink
from .dataset_normalizer import load_normalized_datasets


def _load_aliases() -> dict[str, str]:
    cfg = get_settings()
    path = ROOT / cfg["paths"]["dataset_aliases_path"]
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", " ", name.lower()).strip()


def _mentions_dataset(text: str, dataset_title: str, aliases: dict[str, str]) -> bool:
    text_lower = text.lower()

    if _normalize_name(dataset_title) in text_lower:
        return True

    # check aliases
    for alias, canonical in aliases.items():
        if alias.lower() in text_lower and _normalize_name(canonical) in _normalize_name(dataset_title):
            return True

    # fuzzy fallback for short names
    words = dataset_title.split()
    if len(words) <= 4:
        ratio = fuzz.partial_ratio(dataset_title.lower(), text_lower)
        if ratio > 85:
            return True

    return False


def build_links(
    dataset_candidates: list[DatasetCandidate],
    chunk_candidates: list[ChunkCandidate],
    openalex_papers: list[OpenAlexPaper],
) -> list[DatasetLink]:
    aliases = _load_aliases()
    datasets = {d.dataset_id: d for d in load_normalized_datasets()}
    links: dict[str, DatasetLink] = {}

    lit_scores = get_settings()["reranking"]["literature_support_scores"]

    for dc in dataset_candidates:
        ds = datasets.get(dc.dataset_id)
        title = ds.display_name if ds else dc.title

        best_link: DatasetLink | None = None

        # Level 1: chunk mention
        for chunk in chunk_candidates:
            if _mentions_dataset(chunk.text, title, aliases):
                best_link = DatasetLink(
                    dataset_id=dc.dataset_id,
                    local_id=chunk.local_id,
                    openalex_id=chunk.openalex_id,
                    evidence_source="chunk",
                    confidence="high",
                    evidence_text=chunk.text[:200],
                )
                dc.literature_support = lit_scores["chunk_explicit_mention"]
                break

        if best_link is None:
            # Level 2: abstract mention
            for paper in openalex_papers:
                if paper.abstract and _mentions_dataset(paper.abstract, title, aliases):
                    best_link = DatasetLink(
                        dataset_id=dc.dataset_id,
                        local_id=None,
                        openalex_id=paper.openalex_id,
                        evidence_source="abstract",
                        confidence="medium",
                        evidence_text=paper.abstract[:200],
                    )
                    dc.literature_support = lit_scores["abstract_mention"]
                    break

        if best_link is None:
            # Level 3: Zenodo DOI already handled in dataset_retriever
            if dc.source == "zenodo" and dc.literature_support == lit_scores["zenodo_doi_matches_openalex"]:
                best_link = DatasetLink(
                    dataset_id=dc.dataset_id,
                    local_id=None,
                    openalex_id=None,
                    evidence_source="zenodo_doi",
                    confidence="high",
                    evidence_text=None,
                )
            else:
                best_link = DatasetLink(
                    dataset_id=dc.dataset_id,
                    local_id=None,
                    openalex_id=None,
                    evidence_source="semantic",
                    confidence="low",
                    evidence_text=None,
                )

        links[dc.dataset_id] = best_link

    result = list(links.values())

    debug_dir = ROOT / "generated" / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    with open(debug_dir / "last_links.json", "w") as f:
        json.dump([lk.model_dump() for lk in result], f, indent=2)

    return result
