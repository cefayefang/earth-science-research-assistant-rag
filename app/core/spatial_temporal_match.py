"""
Structured spatial/temporal matching between a parsed user query and
normalized dataset records.

Mitigates Limitation 1 of Yu et al. 2025 ("text-only retrieval"): instead of
relying solely on the embedding of "Region: -90 -180 90 180" string, we parse
the raw spatial_info / temporal_info fields into geometries/date ranges and
compute explicit overlap scores.

Three dataset-source conventions for spatial_info:
  • NASA CMR:  "-90 -180 90 180"                        (space-separated; lat_min lon_min lat_max lon_max)
  • STAC:      "[-67.9927, 16.8444, -64.1196, 19.9382]"  (JSON-ish list; lon_min lat_min lon_max lat_max)
  • CDSE:      "[-179.95, -81.05, 179.96, 143.99]"       (same as STAC)
  • Copernicus CDS: None  (no spatial info)

temporal_info is uniformly "YYYY-MM-DD to YYYY-MM-DD" with optional empty end
(meaning "to present") or entirely None.

All outputs of this module are in (min_lon, min_lat, max_lon, max_lat) order.
"""
import re
import json
from datetime import date, datetime
from typing import Optional


BBox = tuple[float, float, float, float]   # (min_lon, min_lat, max_lon, max_lat)
DateRange = tuple[date, date]


# ── Parsing dataset side ─────────────────────────────────────────────────────

def parse_dataset_bbox(spatial_info: Optional[str]) -> Optional[BBox]:
    """
    Normalize a spatial_info string to (min_lon, min_lat, max_lon, max_lat).
    Returns None if input is missing / malformed.
    """
    if not spatial_info or not isinstance(spatial_info, str):
        return None
    s = spatial_info.strip()
    if not s:
        return None

    try:
        if s.startswith("["):
            # JSON-ish list: assume [lon_min, lat_min, lon_max, lat_max]
            vals = json.loads(s)
            if isinstance(vals, list) and len(vals) >= 4:
                lon1, lat1, lon2, lat2 = [float(v) for v in vals[:4]]
                return (
                    min(lon1, lon2), min(lat1, lat2),
                    max(lon1, lon2), max(lat1, lat2),
                )
            return None
        else:
            # NASA CMR convention: space-separated "lat_min lon_min lat_max lon_max"
            parts = s.replace(",", " ").split()
            nums = [float(p) for p in parts if _is_number(p)]
            if len(nums) < 4:
                return None
            lat1, lon1, lat2, lon2 = nums[0], nums[1], nums[2], nums[3]
            return (
                min(lon1, lon2), min(lat1, lat2),
                max(lon1, lon2), max(lat1, lat2),
            )
    except (ValueError, json.JSONDecodeError):
        return None


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def parse_dataset_temporal(temporal_info: Optional[str]) -> Optional[DateRange]:
    """
    Parse "YYYY-MM-DD to YYYY-MM-DD" with open end → date.today() as today.
    Returns None if input is missing.
    """
    if not temporal_info or not isinstance(temporal_info, str):
        return None
    if " to " not in temporal_info:
        return None
    try:
        start_s, end_s = temporal_info.split(" to ", 1)
        start_s = start_s.strip()
        end_s = end_s.strip()
        start = _parse_iso_date(start_s)
        end = _parse_iso_date(end_s) if end_s else date.today()
        if start is None or end is None:
            return None
        if start > end:
            start, end = end, start
        return (start, end)
    except Exception:
        return None


def _parse_iso_date(s: str) -> Optional[date]:
    if not s:
        return None
    try:
        # handle YYYY or YYYY-MM too
        parts = s.split("-")
        if len(parts) == 1:
            return date(int(parts[0]), 1, 1)
        if len(parts) == 2:
            return date(int(parts[0]), int(parts[1]), 1)
        return date.fromisoformat(s[:10])
    except (ValueError, TypeError):
        return None


# ── Overlap scoring ──────────────────────────────────────────────────────────

_GLOBAL_BBOX: BBox = (-180.0, -90.0, 180.0, 90.0)


def _is_global(b: BBox, tol: float = 1.0) -> bool:
    """A bbox that covers most of the globe."""
    return (
        b[0] <= -180.0 + tol and b[1] <= -90.0 + tol
        and b[2] >= 180.0 - tol and b[3] >= 90.0 - tol
    )


def _bbox_area(b: BBox) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _bbox_intersect(a: BBox, b: BBox) -> Optional[BBox]:
    lon_min = max(a[0], b[0])
    lat_min = max(a[1], b[1])
    lon_max = min(a[2], b[2])
    lat_max = min(a[3], b[3])
    if lon_max <= lon_min or lat_max <= lat_min:
        return None
    return (lon_min, lat_min, lon_max, lat_max)


def bbox_overlap_score(
    query_bbox: Optional[list[float]],
    dataset_bbox: Optional[BBox],
    default: float = 0.5,
) -> float:
    """
    Returns a 0-1 score. If either side is missing, return `default`.
    If dataset is global, the dataset trivially "covers" any query → 1.0.
    Otherwise use IoU(query, dataset) + containment bonus.
    """
    if not query_bbox or dataset_bbox is None:
        return default
    if len(query_bbox) < 4:
        return default

    q: BBox = (
        min(query_bbox[0], query_bbox[2]),
        min(query_bbox[1], query_bbox[3]),
        max(query_bbox[0], query_bbox[2]),
        max(query_bbox[1], query_bbox[3]),
    )

    if _is_global(dataset_bbox):
        return 1.0

    inter = _bbox_intersect(q, dataset_bbox)
    if inter is None:
        return 0.0

    inter_area = _bbox_area(inter)
    q_area = _bbox_area(q)
    ds_area = _bbox_area(dataset_bbox)
    union = q_area + ds_area - inter_area
    iou = inter_area / union if union > 0 else 0.0

    # containment: fraction of query area covered by dataset
    containment = inter_area / q_area if q_area > 0 else 0.0

    # Blend — containment is often more meaningful (dataset covers query region)
    return min(1.0, 0.5 * iou + 0.5 * containment)


def temporal_overlap_score(
    query_range: Optional[list[str]],
    dataset_range: Optional[DateRange],
    default: float = 0.5,
) -> float:
    """
    Returns 0-1 overlap. If either side missing → default.
    Score = (overlap days) / (query range days).
    """
    if not query_range or dataset_range is None:
        return default
    if len(query_range) < 2:
        return default
    try:
        q_start = _parse_iso_date(query_range[0]) if query_range[0] else None
        q_end = _parse_iso_date(query_range[1]) if query_range[1] else date.today()
    except Exception:
        return default
    if q_start is None or q_end is None:
        return default
    if q_start > q_end:
        q_start, q_end = q_end, q_start

    ds_start, ds_end = dataset_range
    overlap_start = max(q_start, ds_start)
    overlap_end = min(q_end, ds_end)
    if overlap_end < overlap_start:
        return 0.0

    q_span = (q_end - q_start).days + 1
    overlap = (overlap_end - overlap_start).days + 1
    return min(1.0, overlap / q_span) if q_span > 0 else default
