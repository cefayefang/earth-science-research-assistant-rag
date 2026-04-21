import json
from pathlib import Path
import pandas as pd
from .config import get_settings, ROOT
from .schemas import PaperRecord


def build_paper_registry() -> list[PaperRecord]:
    cfg = get_settings()
    id_track_path = ROOT / cfg["paths"]["id_track_file"]
    df = pd.read_excel(id_track_path, dtype=str).fillna("")

    paper_dir = ROOT / cfg["paths"]["fulltext_paper_dir"]
    records = []

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
