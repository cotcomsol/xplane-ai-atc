"""Airport, runway and frequency data from X-Plane's apt.dat.

apt.dat format: https://developer.x-plane.com/article/airport-data-apt-dat-12-00-file-format-specification/
"""

from __future__ import annotations

import gzip
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

APT_DAT_CANDIDATES = [
    "Global Scenery/Global Airports/Earth nav data/apt.dat",
    "Resources/default scenery/default apt dat/Earth nav data/apt.dat",
]

# row code -> facility type. 50-56 are the old 10 kHz rows, 1050-1056 are kHz (8.33 capable).
FREQ_TYPES = {
    50: "ATIS", 51: "CTAF", 52: "DELIVERY", 53: "GROUND", 54: "TOWER", 55: "APPROACH", 56: "DEPARTURE",
}

NM_PER_RAD = 3440.065
CACHE_VERSION = 2


def dist_nm(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * NM_PER_RAD * math.asin(min(1.0, math.sqrt(a)))


def bearing(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


@dataclass
class Facility:
    airport: dict
    kind: str  # TOWER, GROUND, ...
    name: str  # name as listed in apt.dat, e.g. "NORCAL APP"
    khz: int
    distance_nm: float

    @property
    def key(self) -> str:
        """Identifies the controller being talked to (conversation history, voice, handoffs)."""
        if self.kind in ("APPROACH", "DEPARTURE"):
            # One radar controller per frequency: every airport in its area lists that frequency (as
            # approach, departure or both), and which of them is nearest changes as the aircraft flies.
            return f"RADAR:{int(round(self.khz / 25.0)) * 25}"
        return f"{self.airport['id']}:{self.kind}:{self.khz}"


def find_apt_dat(xplane_path: str) -> Path | None:
    root = Path(xplane_path).expanduser()
    for rel in APT_DAT_CANDIDATES:
        p = root / rel
        if p.exists():
            return p
    return None


def _flush(cur: dict | None, out: list) -> None:
    if not cur:
        return
    if cur.get("lat") is None and cur["runways"]:
        r = cur["runways"][0]
        cur["lat"], cur["lon"] = (r["lat1"] + r["lat2"]) / 2, (r["lon1"] + r["lon2"]) / 2
    if cur.get("lat") is not None:
        out.append(cur)


def parse_apt_dat(path: Path) -> list[dict]:
    """Parse airports (header, runways, frequencies). Each airport records the file
    and byte offset of its header so the taxi network can be loaded on demand."""
    airports: list[dict] = []
    cur: dict | None = None
    offset = 0
    with open(path, "rb") as fh:
        for raw in fh:
            line_off = offset
            offset += len(raw)
            parts = raw.decode("latin-1").split()
            if not parts or not parts[0].isdigit():
                continue
            code = int(parts[0])
            if code in (1, 16, 17):
                _flush(cur, airports)
                if len(parts) < 5:
                    cur = None
                    continue
                cur = {
                    "id": parts[4], "name": " ".join(parts[5:]), "elev": int(float(parts[1])),
                    "type": {1: "land", 16: "sea", 17: "heli"}[code],
                    "lat": None, "lon": None, "runways": [], "freqs": [],
                    "file": str(path), "offset": line_off, "taxi": False,
                }
            elif cur is None:
                continue
            elif code == 1302 and len(parts) >= 3:
                k, v = parts[1], " ".join(parts[2:])
                if k == "datum_lat":
                    cur["lat"] = float(v)
                elif k == "datum_lon":
                    cur["lon"] = float(v)
                elif k in ("icao_code", "faa_code", "city", "state"):
                    cur[k] = v
            elif code == 100 and len(parts) >= 20:
                lat1, lon1, lat2, lon2 = map(float, (parts[9], parts[10], parts[18], parts[19]))
                cur["runways"].append({
                    "ids": [parts[8], parts[17]],
                    "lat1": lat1, "lon1": lon1, "lat2": lat2, "lon2": lon2,
                    "length_ft": round(dist_nm(lat1, lon1, lat2, lon2) * 6076.12),
                    "true_hdg": round(bearing(lat1, lon1, lat2, lon2)),
                    "surface": int(parts[2]),
                })
            elif code == 1202:
                cur["taxi"] = True
            elif code in FREQ_TYPES or (code - 1000) in FREQ_TYPES:
                if len(parts) < 2:
                    continue
                khz = int(parts[1]) * (10 if code < 1000 else 1)
                cur["freqs"].append({
                    "type": FREQ_TYPES[code % 1000], "khz": khz, "name": " ".join(parts[2:]),
                })
    _flush(cur, airports)
    return airports


def apt_dat_files(xplane_path: str) -> list[Path]:
    """All apt.dat files X-Plane uses, highest priority first: enabled custom scenery
    packs in scenery_packs.ini order, then the global airports."""
    root = Path(xplane_path).expanduser()
    files: list[Path] = []
    ini = root / "Custom Scenery" / "scenery_packs.ini"
    try:
        for line in ini.read_text(encoding="latin-1").splitlines():
            if not line.startswith("SCENERY_PACK "):
                continue
            pack = Path(line[len("SCENERY_PACK "):].strip())
            if not pack.is_absolute():
                pack = root / pack
            f = pack / "Earth nav data" / "apt.dat"
            if f.exists():
                files.append(f)
    except OSError:
        pass
    glob = find_apt_dat(xplane_path)
    if glob and glob not in files:
        files.append(glob)
    return files


class AirportDB:
    def __init__(self, airports: list[dict]):
        self.airports = [a for a in airports if a["type"] != "heli"]
        self._by_id: dict[str, dict] | None = None

    def by_id(self, ident: str) -> dict | None:
        if self._by_id is None:
            self._by_id = {a["id"]: a for a in self.airports}
        return self._by_id.get(ident.upper())

    @classmethod
    def load(cls, xplane_path: str, cache_dir: Path) -> "AirportDB | None":
        if not xplane_path:
            log.warning("xplane.path not set: no airport/frequency data")
            return None
        files = apt_dat_files(xplane_path)
        if not files:
            log.warning("apt.dat not found under %s", xplane_path)
            return None
        stamp = "|".join(f"{f}:{f.stat().st_size}:{int(f.stat().st_mtime)}" for f in files)
        cache = cache_dir / "airports.json.gz"
        if cache.exists():
            try:
                with gzip.open(cache, "rt") as fh:
                    data = json.load(fh)
                if data.get("stamp") == stamp and data.get("version") == CACHE_VERSION:
                    return cls(data["airports"])
            except (OSError, ValueError):
                pass
        seen: set[str] = set()
        airports: list[dict] = []
        for f in files:
            log.info("Indexing %s ...", f)
            for a in parse_apt_dat(f):
                if a["id"] not in seen:  # custom scenery overrides global airports
                    seen.add(a["id"])
                    airports.append(a)
        cache_dir.mkdir(parents=True, exist_ok=True)
        with gzip.open(cache, "wt") as fh:
            json.dump({"stamp": stamp, "version": CACHE_VERSION, "airports": airports}, fh)
        log.info("Indexed %d airports from %d files", len(airports), len(files))
        return cls(airports)

    def nearby(self, lat: float, lon: float, radius_nm: float = 30, limit: int = 8) -> list[tuple[float, dict]]:
        dlat = radius_nm / 60
        dlon = radius_nm / (60 * max(0.1, math.cos(math.radians(lat))))
        out = []
        for a in self.airports:
            if abs(a["lat"] - lat) > dlat or abs(a["lon"] - lon) > dlon:
                continue
            d = dist_nm(lat, lon, a["lat"], a["lon"])
            if d <= radius_nm:
                out.append((d, a))
        out.sort(key=lambda x: x[0])
        return out[:limit]

    def facility_for(self, khz: int, lat: float, lon: float) -> Facility | None:
        """Find the facility on a tuned frequency, nearest first.

        Tower/ground/delivery/CTAF must be fairly close; approach and departure
        frequencies are listed under airports in their area so allow more range.
        """
        if not khz:
            return None
        best: Facility | None = None
        for d, a in self.nearby(lat, lon, radius_nm=80, limit=400):
            for f in a["freqs"]:
                if abs(f["khz"] - khz) > 3:
                    continue
                max_nm = 80 if f["type"] in ("APPROACH", "DEPARTURE") else 40 if f["type"] == "ATIS" else 30
                if d > max_nm:
                    continue
                # Same frequency listed as tower and CTAF (part-time towers): tower wins
                rank = (d, f["type"] == "CTAF")
                if best is None or rank < (best.distance_nm, best.kind == "CTAF"):
                    best = Facility(a, f["type"], f["name"], f["khz"], d)
        from .centers import as_airport, site_for_freq

        if best is None or best.kind in ("APPROACH", "DEPARTURE"):
            hit = site_for_freq(khz, lat, lon)
            if hit:
                site, d = hit
                # Airports under Center airspace list the Center as their approach ("OAKLAND APCH/DEP")
                center_word = site["name"].split()[0].upper()
                if best is None or center_word in best.name.upper() or "CTR" in best.name.upper() \
                        or "CENTER" in best.name.upper():
                    best = Facility(as_airport(site), "CENTER", site["name"], khz, d)
        return best
