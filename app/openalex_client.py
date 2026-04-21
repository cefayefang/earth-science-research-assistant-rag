import json
import time
import requests
from .config import get_settings, ROOT
from .schemas import OpenAlexPaper

BASE = "https://api.openalex.org"


def _get(params: dict, cfg: dict) -> list[dict]:
    params["mailto"] = cfg["openalex"]["email"]
    try:
        r = requests.get(
            f"{BASE}/works",
            params=params,
            timeout=cfg["openalex"]["timeout_seconds"],
        )
        r.raise_for_status()
        return r.json().get("results", [])
    except Exception as e:
        print(f"  [openalex warn] {e}")
        return []


def _parse(work: dict, bucket: str) -> OpenAlexPaper:
    wid = work.get("id", "").replace("https://openalex.org/", "").lower()
    abstract = work.get("abstract_inverted_index")
    if abstract:
        words = {}
        for word, positions in abstract.items():
            for pos in positions:
                words[pos] = word
        abstract = " ".join(words[i] for i in sorted(words))
    else:
        abstract = None

    authors = [
        a.get("author", {}).get("display_name", "")
        for a in work.get("authorships", [])
    ]

    oa = work.get("open_access", {})
    oa_url = oa.get("oa_url")

    return OpenAlexPaper(
        openalex_id=wid,
        title=work.get("title") or "",
        abstract=abstract,
        year=work.get("publication_year"),
        doi=(work.get("doi") or "").replace("https://doi.org/", "") or None,
        authors=[a for a in authors if a],
        cited_by_count=work.get("cited_by_count") or 0,
        bucket=bucket,
        oa_url=oa_url,
    )


_UNPUBLISHED_TYPES = {"preprint", "dataset", "paratext", "other", "reference-entry", "supplementary-materials"}


def _is_published(work: dict) -> bool:
    work_type = work.get("type", "")
    if work_type in _UNPUBLISHED_TYPES:
        return False
    if not work.get("doi") and not work.get("primary_location", {}).get("source"):
        return False
    return True


def fetch_openalex_papers(query: str) -> list[OpenAlexPaper]:
    cfg = get_settings()
    top_k_recent = cfg["openalex"]["recent_top_k"]
    top_k_impact = cfg["openalex"]["impactful_top_k"]

    recent_raw = _get({"search": query, "sort": "publication_date:desc", "per_page": top_k_recent}, cfg)
    time.sleep(0.3)
    impact_raw = _get({"search": query, "sort": "cited_by_count:desc", "per_page": top_k_impact}, cfg)

    seen: dict[str, OpenAlexPaper] = {}
    for work in recent_raw:
        if not _is_published(work):
            continue
        p = _parse(work, "recent")
        if p.openalex_id not in seen:
            seen[p.openalex_id] = p

    for work in impact_raw:
        if not _is_published(work):
            continue
        p = _parse(work, "impactful")
        if p.openalex_id not in seen:
            seen[p.openalex_id] = p

    results = list(seen.values())

    debug_dir = ROOT / "generated" / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    with open(debug_dir / "last_openalex_candidates.json", "w") as f:
        json.dump([p.model_dump() for p in results], f, indent=2)

    return results
