"""Small, stateless helpers.

These are the "dirty work" functions that keep the objects in the pipeline
slim. Nothing here should hold state; everything is a pure function.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


# --------------------------------------------------------------------------- #
# Geography
# --------------------------------------------------------------------------- #
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in kilometers."""
    radius_km = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    return 2 * radius_km * math.asin(math.sqrt(a))


def haversine_km_vectorized(
    lat: pd.Series, lon: pd.Series, lat0: float, lon0: float
) -> pd.Series:
    """Vectorized haversine for a whole column of coordinates."""
    radius_km = 6371.0
    d_lat = (lat - lat0).map(math.radians)
    d_lon = (lon - lon0).map(math.radians)
    lat_rad = lat.map(math.radians)
    lat0_rad = math.radians(lat0)
    a = (
        (d_lat / 2).map(math.sin) ** 2
        + math.cos(lat0_rad)
        * lat_rad.map(math.cos)
        * (d_lon / 2).map(math.sin) ** 2
    )
    return 2 * radius_km * a.map(math.sqrt).map(math.asin)


# --------------------------------------------------------------------------- #
# Filesystem / IO
# --------------------------------------------------------------------------- #
def ensure_dir(path: Path) -> None:
    """Create a directory (and parents) if it does not exist yet."""
    path.mkdir(parents=True, exist_ok=True)


def utc_timestamp() -> str:
    """ISO-8601 UTC timestamp, safe to use inside file names."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def slugify(text: str, max_length: int = 40) -> str:
    """Turn free text into a filesystem-friendly slug."""
    keep = [c.lower() if c.isalnum() else "-" for c in text.strip()]
    slug = "".join(keep)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")[:max_length] or "query"


# --------------------------------------------------------------------------- #
# Formatting candidates for the LLM
# --------------------------------------------------------------------------- #
def candidates_to_payload(candidates: pd.DataFrame) -> list[dict[str, Any]]:
    """Compact, token-cheap representation of the shortlist for the prompt."""
    columns = [
        "restaurant_id",
        "name",
        "style",
        "dietary",
        "avg_score",
        "n_votes",
        "price_range",
        "distance_km",
    ]
    present = [c for c in columns if c in candidates.columns]
    records = candidates[present].to_dict(orient="records")
    for record in records:
        if "avg_score" in record:
            record["avg_score"] = round(float(record["avg_score"]), 2)
        if "distance_km" in record:
            record["distance_km"] = round(float(record["distance_km"]), 2)
    return records
