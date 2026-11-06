"""Air Route Traffic Control Centers: remote radio sites and frequencies from FAA NASR (AFF),
edition effective 2026-10-29 (src/atc/data/centers.json)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .airports import dist_nm

DATA = Path(__file__).parent / "data" / "centers.json"
LOW_USES = ("low", "low/high")


@lru_cache(maxsize=1)
def sites() -> list[dict]:
    try:
        return json.loads(DATA.read_text())
    except (OSError, ValueError):
        return []


def as_airport(site: dict) -> dict:
    """A Center radio site shaped like an airport record, so it can be a Facility's 'airport'."""
    return {"id": site["artcc"], "name": site["name"], "city": site["name"], "lat": site["lat"], "lon": site["lon"],
            "elev": 0, "runways": [], "type": "center", "site": site.get("site", ""),
            "freqs": [{"type": "CENTER", "khz": int(round(f["mhz"] * 1000)), "name": site["name"]}
                      for f in site["freqs"]]}


def nearest_low(lat: float, lon: float, max_nm: float = 250) -> tuple[dict, int, float] | None:
    """(site, kHz, distance nm) of the nearest site with a low-altitude frequency."""
    best = None
    for s in sites():
        lows = [f for f in s["freqs"] if f.get("use") in LOW_USES]
        if not lows:
            continue
        d = dist_nm(lat, lon, s["lat"], s["lon"])
        if d <= max_nm and (best is None or d < best[2]):
            best = (s, int(round(lows[0]["mhz"] * 1000)), d)
    return best


def site_for_freq(khz: int, lat: float, lon: float, max_nm: float = 200) -> tuple[dict, float] | None:
    """The nearest Center site transmitting on `khz` within range."""
    best = None
    for s in sites():
        if any(abs(int(round(f["mhz"] * 1000)) - khz) <= 3 for f in s["freqs"]):
            d = dist_nm(lat, lon, s["lat"], s["lon"])
            if d <= max_nm and (best is None or d < best[1]):
                best = (s, d)
    return best
