"""
Paper registry: local_id ↔ openalex_id ↔ title + OpenAlex-enriched metadata.

Built once at preprocessing time. For each local paper with a known
`openalex_id` we call OpenAlex to cache `year`, `abstract`, `doi`, and
`cited_by_count` directly in the registry file. This ensures that local papers
always carry these fields at retrieval time, even when the current query's
OpenAlex search does not happen to return them.

The enrichment call is rate-limited (~6 req/sec) and typically takes 1–2
minutes for an 82-paper corpus. It's skipped for papers without an
openalex_id. If OpenAlex is unreachable, enrichment silently falls back to
None for that paper (no crash).
"""
import json
import time
from pathlib import Path
import pandas as pd
from .config import get_settings, ROOT
from .schemas import PaperRecord
from .openalex_client import fetch_work_by_openalex_id


# Rate limit under OpenAlex's polite-pool limit (10 req/sec)
_OPENALEX_RATE_LIMIT_SLEEP = 0.15
# Cap stored abstract length to keep the registry JSON compact
_MAX_ABSTRACT_CHARS = 2000


def _enrich_from_openalex(record: PaperRecord) -> PaperRecord:
    """If record.openalex_id is set, fetch year/abstract/doi/cited_by_count.
    Returns the same record with fields populated (in-place + returned)."""
    if not record.openalex_id:
        return record
    oa = fetch_work_by_openalex_id(record.openalex_id)
    time.sleep(_OPENALEX_RATE_LIMIT_SLEEP)
    if not oa:
        return record
    if oa.get("year"):
        record.year = int(oa["year"])
    if oa.get("abstract"):
        record.abstract = oa["abstract"][:_MAX_ABSTRACT_CHARS]
    if oa.get("doi"):
        record.doi = oa["doi"]
    if oa.get("cited_by_count") is not None:
        record.cited_by_count = int(oa["cited_by_count"])
    return record


def build_paper_registry(enrich: bool = True) -> list[PaperRecord]:
    """Build registry from id_track.xlsx, optionally enriching via OpenAlex.

    Parameters
    ----------
    enrich : bool
        If True (default), call OpenAlex for each paper with a known openalex_id
        and cache year / abstract / doi / cited_by_count. Set False for a fast
        offline rebuild that doesn't hit the API.
    """
    cfg = get_settings()
    id_track_path = ROOT / cfg["paths"]["id_track_file"]
    df = pd.read_excel(id_track_path, dtype=str).fillna("")

    paper_dir = ROOT / cfg["paths"]["fulltext_paper_dir"]
    records: list[PaperRecord] = []

    for _, row in df.iterrows():
        local_id = str(row.get("local_id", "")).strip()
        openalex_id = row.get("openalex_id", "").strip() or None
        title = row.get("original_title", "").strip()
        filename = row.get("filename", "").strip()

        if not filename.endswith(".pdf"):
            filename = filename + ".pdf"

        pdf_path = paper_dir / filename
        records.append(PaperRecord(
            local_id=local_id,
            openalex_id=openalex_id,
            original_title=title,
            filename=filename,
            pdf_path=str(pdf_path),
        ))

    # ── OpenAlex enrichment (year / abstract / doi / cited_by_count) ─────────
    if enrich:
        hits = 0
        misses = 0
        skipped = 0
        print(f"  Enriching {len(records)} papers via OpenAlex...")
        for i, rec in enumerate(records, 1):
            if not rec.openalex_id:
                skipped += 1
                continue
            before_year = rec.year
            _enrich_from_openalex(rec)
            if rec.year != before_year or rec.abstract:
                hits += 1
            else:
                misses += 1
            if i % 20 == 0:
                print(f"    enriched {i}/{len(records)} (hits={hits}, misses={misses}, skipped={skipped})")
        print(
            f"  Enrichment summary: "
            f"hits={hits}, misses={misses}, skipped (no openalex_id)={skipped}"
        )

    out_path = ROOT / cfg["paths"]["paper_registry_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")

    print(f"Paper registry: {len(records)} records → {out_path}")
    return records


def load_paper_registry() -> list[PaperRecord]:
    cfg = get_settings()
    path = ROOT / cfg["paths"]["paper_registry_path"]
    if not path.exists():
        return build_paper_registry()
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(PaperRecord.model_validate_json(line))
    return records


if __name__ == "__main__":
    # Allow re-building / re-enriching the registry standalone:
    #   python -m app.paper_registry
    build_paper_registry(enrich=True)
