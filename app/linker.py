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
            data = json.load(f)
        # strip meta-keys like "__comment__"
        return {k: v for k, v in data.items() if not k.startswith("__")}
    return {}


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", " ", name.lower()).strip()


def _collapse(s: str) -> str:
    """Collapse multiple whitespace to single space after normalization."""
    return re.sub(r"\s+", " ", _normalize_name(s)).strip()


def _mentions_dataset(text: str, dataset_title: str, aliases: dict[str, str]) -> bool:
    text_lower = text.lower()
    text_normalized = _collapse(text)
    dataset_normalized = _collapse(dataset_title)

    # 1. Full-title substring match (strong signal when it happens)
    if dataset_normalized and dataset_normalized in text_normalized:
        return True

    # 2. Alias match: chunk text contains alias key AND dataset title contains canonical fragment
    for alias, canonical in aliases.items():
        alias_lower = alias.lower()
        canonical_norm = _collapse(canonical)
        if alias_lower in text_lower and canonical_norm in dataset_normalized:
            return True

    # 3. Fuzzy matching for moderate-length titles (Phase 8: relaxed from <=4 to <=10 words)
    words = dataset_title.split()
    if len(words) <= 10:
        ratio = fuzz.partial_ratio(dataset_title.lower(), text_lower)
        if ratio > 90:
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

        # Level 1: chunk mention — strongest evidence (named in a retrieved local chunk)
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
                # Use max() so we never downgrade an already-strong baseline
                dc.literature_support = max(
                    dc.literature_support, lit_scores["chunk_explicit_mention"]
                )
                break

        if best_link is None:
            # Level 2: abstract mention (named in a retrieved OpenAlex abstract)
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
                    dc.literature_support = max(
                        dc.literature_support, lit_scores["abstract_mention"]
                    )
                    break

        if best_link is None:
            # No mention found. literature_support keeps its baseline (has_doi or
            # semantic_only), set in dataset_retriever. Record provenance.
            best_link = DatasetLink(
                dataset_id=dc.dataset_id,
                local_id=None,
                openalex_id=None,
                evidence_source="has_doi" if dc.doi else "semantic",
                confidence="medium" if dc.doi else "low",
                evidence_text=None,
            )

        links[dc.dataset_id] = best_link

    result = list(links.values())

    debug_dir = ROOT / "generated" / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    with open(debug_dir / "last_links.json", "w") as f:
        json.dump([lk.model_dump() for lk in result], f, indent=2)

    return result
