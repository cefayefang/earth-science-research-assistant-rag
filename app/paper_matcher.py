import json
from rapidfuzz import fuzz
from .config import ROOT
from .schemas import OpenAlexPaper, PaperMatch
from .paper_registry import load_paper_registry


def _normalize(s: str) -> str:
    return s.lower().strip().replace("  ", " ")


def match_papers(openalex_papers: list[OpenAlexPaper]) -> list[PaperMatch]:
    registry = load_paper_registry()

    registry_by_openalex = {r.openalex_id.lower(): r for r in registry if r.openalex_id}
    registry_by_title = {_normalize(r.original_title): r for r in registry}

    matches: list[PaperMatch] = []

    for paper in openalex_papers:
        local_id = None

        # 1. OpenAlex ID match
        pid = paper.openalex_id.lower()
        if pid in registry_by_openalex:
            local_id = registry_by_openalex[pid].local_id

        # 2. DOI match
        if local_id is None and paper.doi:
            for r in registry:
                pass  # DOI not stored in registry currently — skip

        # 3. Normalized title exact match
        if local_id is None:
            norm_title = _normalize(paper.title)
            if norm_title in registry_by_title:
                local_id = registry_by_title[norm_title].local_id

        # 4. Fuzzy title match
        if local_id is None:
            norm_title = _normalize(paper.title)
            best_ratio = 0
            best_local_id = None
            for reg_title, reg_record in registry_by_title.items():
                ratio = fuzz.ratio(norm_title, reg_title)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_local_id = reg_record.local_id
            if best_ratio >= 90:
                local_id = best_local_id

        evidence = "fulltext_supported" if local_id else "metadata_only"
        matches.append(PaperMatch(
            openalex_id=paper.openalex_id,
            local_id=local_id,
            evidence_level=evidence,
        ))

    debug_dir = ROOT / "generated" / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    with open(debug_dir / "last_paper_matches.json", "w") as f:
        json.dump([m.model_dump() for m in matches], f, indent=2)

    return matches
