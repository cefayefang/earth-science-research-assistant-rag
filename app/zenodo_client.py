import json
import requests
from .config import get_settings, ROOT
from .schemas import NormalizedDataset


def _slug(raw_id: str) -> str:
    return f"zenodo_{raw_id}"


def fetch_zenodo_datasets(query: str, openalex_dois: set[str]) -> list[NormalizedDataset]:
    cfg = get_settings()
    z_cfg = cfg["zenodo"]

    try:
        r = requests.get(
            f"{z_cfg['base_url']}/records",
            params={
                "q": query,
                "size": z_cfg["top_k"],
                "type": z_cfg["type"],
                "access_right": z_cfg["access_right"],
            },
            timeout=z_cfg["timeout_seconds"],
        )
        r.raise_for_status()
        hits = r.json().get("hits", {}).get("hits", [])
    except Exception as e:
        print(f"  [zenodo warn] {e}")
        return []

    records = []
    for hit in hits:
        meta = hit.get("metadata", {})
        raw_id = str(hit.get("id", ""))
        doi = hit.get("doi") or meta.get("doi") or ""
        title = meta.get("title", "")
        desc = meta.get("description", "") or ""
        # strip HTML tags from description
        desc = __import__("re").sub(r"<[^>]+>", " ", desc).strip()
        keywords_raw = meta.get("keywords", [])
        keywords = [k if isinstance(k, str) else k.get("tag", "") for k in keywords_raw]

        retrieval_text = f"{title}. {desc[:500]}. Keywords: {', '.join(keywords[:10])}.".strip()

        # check if this Zenodo record is linked to an OpenAlex paper
        related_dois = {doi.lower().strip("/")} if doi else set()
        has_paper_link = bool(related_dois & {d.lower().strip("/") for d in openalex_dois})

        record = NormalizedDataset(
            dataset_id=_slug(raw_id),
            source="zenodo",
            source_raw_id=raw_id,
            source_title=title,
            display_name=title,
            description=desc[:1000],
            keywords=keywords,
            variables=[],
            provider="Zenodo",
            spatial_info=None,
            temporal_info=None,
            doi=doi or None,
            retrieval_text=retrieval_text,
            raw_metadata=hit,
        )
        records.append((record, has_paper_link))

    return records  # list of (NormalizedDataset, has_paper_link: bool)
