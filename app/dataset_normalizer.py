import json
import re
from functools import lru_cache
from pathlib import Path
from .config import get_settings, ROOT
from .schemas import NormalizedDataset


def _slug(source: str, raw_id: str) -> str:
    clean = re.sub(r"[^a-z0-9]", "_", raw_id.lower())[:60]
    return f"{source}_{clean}"


def _join(items: list) -> str:
    return ", ".join(str(i) for i in items if i)


# ── Per-source normalizers ────────────────────────────────────────────────────

def _from_nasa_cmr(entry: dict) -> NormalizedDataset:
    raw_id = entry.get("entry_id") or entry.get("id") or ""
    title = entry.get("title", "")
    desc = entry.get("summary", "")

    boxes = entry.get("boxes", [])
    spatial = boxes[0] if boxes else None

    t_start = (entry.get("time_start") or "")[:10]
    t_end = (entry.get("time_end") or "")[:10]
    temporal = f"{t_start} to {t_end}" if t_start else None

    raw_platforms = entry.get("platforms", [])
    platforms = [
        p if isinstance(p, str) else p.get("short_name", "")
        for p in raw_platforms
        if p
    ]
    keywords = [p for p in platforms if p]

    retrieval_text = f"{title}. {desc}. Keywords: {_join(keywords)}. Region: {spatial}. Time: {temporal}.".strip()

    return NormalizedDataset(
        dataset_id=_slug("nasa_cmr", raw_id),
        source="nasa_cmr",
        source_raw_id=raw_id,
        source_title=title,
        display_name=title,
        description=desc,
        keywords=keywords,
        variables=[],
        provider=entry.get("data_center"),
        spatial_info=spatial,
        temporal_info=temporal,
        retrieval_text=retrieval_text,
        raw_metadata=entry,
    )


def _from_stac(col: dict) -> NormalizedDataset:
    raw_id = col.get("id", "")
    title = col.get("title", raw_id)
    desc = col.get("description", "")
    keywords = col.get("keywords", [])

    extent = col.get("extent", {})
    bbox = extent.get("spatial", {}).get("bbox", [[]])[0]
    spatial = str(bbox) if bbox else None

    interval = extent.get("temporal", {}).get("interval", [[None, None]])[0]
    t_start = (interval[0] or "")[:10]
    t_end = (interval[1] or "")[:10]
    temporal = f"{t_start} to {t_end}" if t_start else None

    cube_vars = col.get("cube:variables", {})
    variables = list(cube_vars.keys()) if isinstance(cube_vars, dict) else []

    retrieval_text = f"{title}. {desc}. Variables: {_join(variables)}. Keywords: {_join(keywords)}. Region: {spatial}. Time: {temporal}.".strip()

    return NormalizedDataset(
        dataset_id=_slug("stac", raw_id),
        source="stac",
        source_raw_id=raw_id,
        source_title=title,
        display_name=title,
        description=desc,
        keywords=keywords,
        variables=variables,
        provider=None,
        spatial_info=spatial,
        temporal_info=temporal,
        retrieval_text=retrieval_text,
        raw_metadata=col,
    )


def _from_copernicus_cds(col: dict) -> NormalizedDataset:
    raw_id = col.get("id", "")
    title = col.get("title", raw_id)
    desc = col.get("description", "") or col.get("abstract", "")
    keywords = col.get("keywords", [])

    retrieval_text = f"{title}. {desc}. Keywords: {_join(keywords)}.".strip()

    return NormalizedDataset(
        dataset_id=_slug("copernicus_cds", raw_id),
        source="copernicus_cds",
        source_raw_id=raw_id,
        source_title=title,
        display_name=title,
        description=desc,
        keywords=keywords,
        variables=[],
        provider="Copernicus Climate Data Store",
        spatial_info=None,
        temporal_info=None,
        retrieval_text=retrieval_text,
        raw_metadata=col,
    )


def _from_cdse(col: dict) -> NormalizedDataset:
    raw_id = col.get("id", "")
    title = col.get("title", raw_id)
    desc = col.get("description", "")
    keywords = col.get("keywords", [])

    extent = col.get("extent", {})
    bbox = extent.get("spatial", {}).get("bbox", [[]])[0]
    spatial = str(bbox) if bbox else None

    interval = extent.get("temporal", {}).get("interval", [[None, None]])[0]
    t_start = (interval[0] or "")[:10]
    t_end = (interval[1] or "")[:10]
    temporal = f"{t_start} to {t_end}" if t_start else None

    retrieval_text = f"{title}. {desc}. Keywords: {_join(keywords)}. Region: {spatial}. Time: {temporal}.".strip()

    return NormalizedDataset(
        dataset_id=_slug("cdse", raw_id),
        source="cdse",
        source_raw_id=raw_id,
        source_title=title,
        display_name=title,
        description=desc,
        keywords=keywords,
        variables=[],
        provider="Copernicus Data Space / ESA",
        spatial_info=spatial,
        temporal_info=temporal,
        retrieval_text=retrieval_text,
        raw_metadata=col,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

_LOADERS = {
    "nasa_cmr_expanded_metadata.json": ("feed.entry", _from_nasa_cmr),
    "stac_metadata.json": ("collections", _from_stac),
    "copernicus_cds_metadata.json": ("list", _from_copernicus_cds),
    "CDSE_collections.json": ("list", _from_cdse),
}


def _extract_entries(data: dict | list, path: str) -> list:
    if path == "list":
        return data if isinstance(data, list) else []
    parts = path.split(".")
    val = data
    for p in parts:
        val = val[p]
    return val


def normalize_all_datasets() -> list[NormalizedDataset]:
    cfg = get_settings()
    meta_dir = ROOT / cfg["paths"]["dataset_metadata_dir"]
    all_records: list[NormalizedDataset] = []
    seen_ids: set[str] = set()

    for filename, (entry_path, normalizer) in _LOADERS.items():
        fpath = meta_dir / filename
        if not fpath.exists():
            print(f"  [skip] {filename} not found")
            continue

        with open(fpath) as f:
            data = json.load(f)

        entries = _extract_entries(data, entry_path)
        count = 0
        for entry in entries:
            try:
                record = normalizer(entry)
                if record.dataset_id not in seen_ids:
                    seen_ids.add(record.dataset_id)
                    all_records.append(record)
                    count += 1
            except Exception as e:
                pass

        print(f"  {filename}: {count} records normalized")

    out_path = ROOT / cfg["paths"]["normalized_datasets_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in all_records:
            f.write(r.model_dump_json() + "\n")

    print(f"Total: {len(all_records)} datasets → {out_path}")
    return all_records


@lru_cache(maxsize=1)
def load_normalized_datasets() -> list[NormalizedDataset]:
    cfg = get_settings()
    path = ROOT / cfg["paths"]["normalized_datasets_path"]
    if not path.exists():
        return normalize_all_datasets()
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(NormalizedDataset.model_validate_json(line))
    return records
