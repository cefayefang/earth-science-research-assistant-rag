"""
Metadata fetcher: downloads dataset metadata from multiple platforms and saves
them as JSON files in local_database/dataset_metadata/
"""

import json
import time
import requests
from pathlib import Path

METADATA_DIR = Path(__file__).parent / "dataset_metadata"
METADATA_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "research_assistant/1.0 (mailto:fayefang@g.harvard.edu)"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def get(url, params=None):
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  [warn] {url}: {e}")
        return None


def save(data, filename):
    path = METADATA_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved {filename} ({len(data) if isinstance(data, list) else 1} records)")


# ── 1. Copernicus CDS ─────────────────────────────────────────────────────────

def fetch_copernicus_cds():
    print("\n[1/4] Copernicus CDS")
    all_collections = []
    offset = 0
    limit = 50

    while True:
        data = get(
            "https://cds.climate.copernicus.eu/api/catalogue/v1/collections",
            params={"limit": limit, "offset": offset}
        )
        if not data:
            break
        items = data.get("collections", [])
        if not items:
            break
        all_collections.extend(items)
        print(f"  fetched {len(all_collections)} collections...")
        if len(items) < limit:
            break
        offset += limit
        time.sleep(0.5)

    save(all_collections, "copernicus_cds_metadata.json")
    return all_collections


# ── 2. Copernicus Data Space (ESA Sentinel) ───────────────────────────────────

def fetch_copernicus_dataspace():
    print("\n[2/4] Copernicus Data Space (ESA)")
    all_collections = []
    url = "https://catalogue.dataspace.copernicus.eu/stac/collections"

    while url:
        data = get(url)
        if not data:
            break
        items = data.get("collections", [])
        all_collections.extend(items)
        print(f"  fetched {len(all_collections)} collections...")

        # follow next link if present
        next_url = None
        for link in data.get("links", []):
            if link.get("rel") == "next":
                next_url = link.get("href")
                break
        url = next_url
        if url:
            time.sleep(0.5)

    save(all_collections, "copernicus_dataspace_metadata.json")
    return all_collections


# ── 3. NASA CMR (expanded topic queries) ─────────────────────────────────────

def fetch_nasa_cmr_expanded():
    print("\n[4/4] NASA CMR (expanded)")
    all_entries = {}

    topic_queries = [
        "soil moisture SMAP", "flood SAR", "snow cover MODIS", "groundwater GRACE",
        "streamflow river", "NDVI vegetation MODIS", "forest cover Landsat",
        "above ground biomass", "land surface phenology", "gross primary production",
        "land surface temperature", "heat wave urban", "precipitation TRMM GPM",
        "permafrost Arctic", "land cover classification", "glacier mass balance",
        "sea ice Arctic", "sea level altimetry", "sea surface temperature",
        "ocean color chlorophyll", "aerosol optical depth", "PM2.5 particulate",
        "CO2 carbon flux", "wildfire FIRMS MODIS", "landslide InSAR",
    ]

    for query in topic_queries:
        data = get(
            "https://cmr.earthdata.nasa.gov/search/collections.json",
            params={"keyword": query, "page_size": 15, "sort_key": "-usage_score"}
        )
        if not data:
            continue

        entries = data.get("feed", {}).get("entry", [])
        for entry in entries:
            eid = entry.get("id", "")
            if eid not in all_entries:
                all_entries[eid] = entry

        print(f"  [{query[:40]}] +{len(entries)} (total: {len(all_entries)})")
        time.sleep(0.4)

    entries_list = list(all_entries.values())
    save(entries_list, "nasa_cmr_expanded_metadata.json")
    return entries_list


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Dataset Metadata Fetcher")
    print("=" * 60)

    results = {}
    results["copernicus_cds"] = fetch_copernicus_cds()
    results["copernicus_dataspace"] = fetch_copernicus_dataspace()
    results["nasa_cmr_expanded"] = fetch_nasa_cmr_expanded()

    print("\n" + "=" * 60)
    print("SUMMARY")
    for source, records in results.items():
        print(f"  {source}: {len(records)} records")
    total = sum(len(v) for v in results.values())
    print(f"  TOTAL: {total} dataset records")
    print(f"\nAll saved to: {METADATA_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
