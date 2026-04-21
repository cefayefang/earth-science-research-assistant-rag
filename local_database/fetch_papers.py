"""
Paper fetcher: searches OpenAlex for all topics, downloads open-access PDFs,
and produces a manual-download list for paywalled papers.
"""

import json
import os
import time
import requests
import openpyxl
from pathlib import Path
from urllib.parse import quote

# ── Config ────────────────────────────────────────────────────────────────────

PDF_DIR = Path(__file__).parent / "fulltext_paper"
MANIFEST_PATH = Path(__file__).parent / "paper_manifest.json"
MANUAL_LIST_PATH = Path(__file__).parent / "manual_download_list.md"
ID_TRACK_PATH = PDF_DIR / "id_track.xlsx"

PAPERS_PER_TOPIC = 8          # target per topic
MIN_CITATIONS_OLD = 50        # for papers before 2022
MIN_CITATIONS_RECENT = 10     # for papers 2022+
MIN_YEAR = 2015               # hard floor (except classics)
CLASSIC_CITATION_THRESHOLD = 500  # pre-2015 papers allowed if cited this much

OPENALEX_EMAIL = "fayefang@g.harvard.edu"  # polite pool

# ── Topic definitions ─────────────────────────────────────────────────────────

TOPICS = {
    # Hydrology
    "flood_extreme_precipitation": "flood extreme precipitation remote sensing hydrology",
    "soil_moisture": "soil moisture remote sensing satellite retrieval",
    "groundwater": "groundwater remote sensing GRACE satellite",
    "snow_cover_snowmelt": "snow cover snowmelt remote sensing satellite",
    "streamflow_river_discharge": "streamflow river discharge remote sensing estimation",

    # Vegetation
    "ndvi_vegetation_indices": "NDVI vegetation index remote sensing land surface",
    "forest_cover_deforestation": "forest cover change deforestation remote sensing",
    "biomass_estimation": "above ground biomass estimation remote sensing",
    "phenology": "land surface phenology vegetation remote sensing",
    "carbon_cycle_gpp": "gross primary production GPP carbon cycle remote sensing",

    # Climate
    "climate_change_global_warming": "climate change global warming temperature trend remote sensing",
    "heat_wave_temperature_extremes": "heat wave temperature extreme event land surface",
    "precipitation_patterns": "precipitation pattern trend satellite remote sensing",
    "permafrost": "permafrost thaw remote sensing Arctic climate change",

    # Land Surface
    "land_use_land_cover_change": "land use land cover change remote sensing classification",
    "urban_heat_island": "urban heat island remote sensing land surface temperature",
    "desertification_soil_degradation": "desertification soil degradation land degradation remote sensing",

    # Cryosphere
    "glacier_retreat": "glacier retreat mass balance remote sensing satellite",
    "sea_ice": "sea ice extent Arctic Antarctic remote sensing",

    # Ocean
    "sea_level_rise": "sea level rise satellite altimetry coastal",
    "sea_surface_temperature": "sea surface temperature SST remote sensing ocean",
    "ocean_color_chlorophyll": "ocean color chlorophyll phytoplankton remote sensing",

    # Atmosphere
    "aerosols_air_quality": "aerosol optical depth air quality remote sensing satellite",
    "pm25": "PM2.5 particulate matter remote sensing estimation",
    "co2_emissions": "CO2 emissions carbon remote sensing satellite",

    # Natural Hazards
    "wildfire_detection": "wildfire fire detection remote sensing satellite",
    "landslide": "landslide detection mapping remote sensing satellite",
}

# ── OpenAlex helpers ──────────────────────────────────────────────────────────

BASE = "https://api.openalex.org"
HEADERS = {"User-Agent": f"research_assistant/1.0 (mailto:{OPENALEX_EMAIL})"}


def search_openalex(query: str, sort: str, per_page: int = 20) -> list[dict]:
    params = {
        "search": query,
        "sort": sort,
        "per_page": per_page,
        "mailto": OPENALEX_EMAIL,
    }
    try:
        r = requests.get(f"{BASE}/works", params=params, headers=HEADERS, timeout=20)
        r.raise_for_status()
        return r.json().get("results", [])
    except Exception as e:
        print(f"    [warn] OpenAlex error: {e}")
        return []


def passes_criteria(work: dict) -> bool:
    year = work.get("publication_year") or 0
    cites = work.get("cited_by_count") or 0
    if year >= 2022:
        return cites >= MIN_CITATIONS_RECENT
    if year >= MIN_YEAR:
        return cites >= MIN_CITATIONS_OLD
    # pre-2015 classic exception
    return cites >= CLASSIC_CITATION_THRESHOLD


def get_oa_url(work: dict) -> str | None:
    oa = work.get("open_access", {})
    url = oa.get("oa_url")
    if url and url.endswith(".pdf"):
        return url
    # fallback: best_oa_location
    best = work.get("best_oa_location") or {}
    pdf = best.get("pdf_url")
    if pdf:
        return pdf
    return url  # may be HTML; we'll try anyway


def fetch_candidates(topic_key: str, query: str) -> list[dict]:
    """Return up to PAPERS_PER_TOPIC deduplicated candidates."""
    recent = search_openalex(query, sort="publication_date:desc", per_page=25)
    cited = search_openalex(query, sort="cited_by_count:desc", per_page=25)

    seen = {}
    for w in recent + cited:
        wid = w.get("id", "")
        if wid and wid not in seen:
            seen[wid] = w

    filtered = [w for w in seen.values() if passes_criteria(w)]
    # prefer open-access first
    filtered.sort(key=lambda w: (0 if get_oa_url(w) else 1, -(w.get("cited_by_count") or 0)))
    return filtered[:PAPERS_PER_TOPIC]


# ── Download helper ───────────────────────────────────────────────────────────

def download_pdf(url: str, dest: Path) -> bool:
    try:
        r = requests.get(url, timeout=30, headers=HEADERS, allow_redirects=True)
        if r.status_code == 200 and b"%PDF" in r.content[:1024]:
            dest.write_bytes(r.content)
            return True
    except Exception as e:
        print(f"      [warn] download failed: {e}")
    return False


# ── id_track helpers ──────────────────────────────────────────────────────────

def load_id_track() -> tuple[openpyxl.Workbook, int]:
    wb = openpyxl.load_workbook(ID_TRACK_PATH)
    ws = wb.active
    max_local_id = 0
    for row in ws.iter_rows(min_row=2, values_only=True):
        lid = row[1]
        if isinstance(lid, int) and lid > max_local_id:
            max_local_id = lid
    return wb, max_local_id


def append_to_id_track(wb: openpyxl.Workbook, local_id: int, openalex_id: str,
                        title: str, filename: str):
    ws = wb.active
    ws.append([openalex_id, local_id, title, filename])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    wb, next_id = load_id_track()
    next_id += 1

    all_manifest = []
    manual_entries = []

    for topic_key, query in TOPICS.items():
        print(f"\n{'='*60}")
        print(f"Topic: {topic_key}")
        candidates = fetch_candidates(topic_key, query)
        print(f"  Found {len(candidates)} candidates")

        for work in candidates:
            wid = work.get("id", "").replace("https://openalex.org/", "").lower()
            title = (work.get("title") or "untitled")[:120]
            year = work.get("publication_year") or 0
            cites = work.get("cited_by_count") or 0
            doi = (work.get("doi") or "").replace("https://doi.org/", "")
            oa_url = get_oa_url(work)

            authors = work.get("authorships", [])
            first_author = ""
            if authors:
                name = authors[0].get("author", {}).get("display_name", "")
                first_author = name.split()[-1].lower() if name else ""

            slug = f"{topic_key}_{first_author}_{year}"
            filename = slug

            entry = {
                "topic": topic_key,
                "openalex_id": wid,
                "local_id": next_id,
                "title": title,
                "year": year,
                "cited_by_count": cites,
                "doi": doi,
                "oa_url": oa_url,
                "filename": filename,
                "status": None,
            }

            if oa_url:
                dest = PDF_DIR / f"{slug}.pdf"
                print(f"  ↓ Downloading: {title[:60]}...")
                ok = download_pdf(oa_url, dest)
                if ok:
                    print(f"    ✓ Saved as {dest.name}")
                    entry["status"] = "downloaded"
                    append_to_id_track(wb, next_id, wid, title, filename)
                    next_id += 1
                else:
                    print(f"    ✗ Download failed — added to manual list")
                    entry["status"] = "manual_needed"
                    manual_entries.append(entry)
            else:
                print(f"  ✗ No OA URL: {title[:60]}")
                entry["status"] = "manual_needed"
                manual_entries.append(entry)

            all_manifest.append(entry)
            time.sleep(0.3)  # polite rate limiting

        time.sleep(1)

    # Save id_track
    wb.save(ID_TRACK_PATH)
    print(f"\n✓ id_track.xlsx updated")

    # Save full manifest
    with open(MANIFEST_PATH, "w") as f:
        json.dump(all_manifest, f, indent=2, ensure_ascii=False)
    print(f"✓ Full manifest saved to {MANIFEST_PATH}")

    # Save manual download list as markdown
    downloaded = [e for e in all_manifest if e["status"] == "downloaded"]
    manual = [e for e in all_manifest if e["status"] == "manual_needed"]

    with open(MANUAL_LIST_PATH, "w") as f:
        f.write(f"# Manual Download List\n\n")
        f.write(f"**Auto-downloaded:** {len(downloaded)} papers\n")
        f.write(f"**Needs manual download:** {len(manual)} papers\n\n")
        f.write("Use your Harvard VPN + Google Scholar / Sci-Hub to download these.\n\n")
        f.write("Save each PDF to: `local_database/fulltext_paper/`\n\n")
        f.write("Use the filename in the last column.\n\n---\n\n")

        current_topic = None
        for e in sorted(manual, key=lambda x: x["topic"]):
            if e["topic"] != current_topic:
                current_topic = e["topic"]
                f.write(f"\n## {current_topic}\n\n")
                f.write("| Title | Year | Citations | DOI | Filename |\n")
                f.write("|-------|------|-----------|-----|----------|\n")
            doi_link = f"[{e['doi']}](https://doi.org/{e['doi']})" if e["doi"] else "N/A"
            f.write(f"| {e['title'][:80]} | {e['year']} | {e['cited_by_count']} | {doi_link} | `{e['filename']}.pdf` |\n")

    print(f"✓ Manual download list saved to {MANUAL_LIST_PATH}")
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"  Auto-downloaded: {len(downloaded)} PDFs")
    print(f"  Manual needed:   {len(manual)} papers")
    print(f"  Total candidates: {len(all_manifest)}")


if __name__ == "__main__":
    main()
