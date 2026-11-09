"""Traffic advisories and traffic facts from X-Plane's TCAS targets."""

from __future__ import annotations

from .airports import bearing, dist_nm
from .phraseology import NUMBERS, type_name
from .phraseology import altitude_words as alt_words

COMPASS8 = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]
ONES = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]


def clock(rel_brg: float) -> str:
    h = int(round((rel_brg % 360) / 30.0)) % 12 or 12
    return f"{ONES[h]} o'clock"


def miles(d: float) -> str:
    n = max(1, int(round(d)))
    return f"{ONES[n] if n <= 12 else ' '.join(NUMBERS[c] for c in str(n))} mile{'s' if n != 1 else ''}"


def direction(hdg_true: float | None) -> str:
    if hdg_true is None:
        return ""
    return COMPASS8[int(((hdg_true % 360) + 22.5) // 45) % 8] + "bound"


def describe_type(t) -> str:
    return type_name(t.icao_type) or ("aircraft" if not t.icao_type else "aircraft type " + t.icao_type)


def advisory(t) -> str:
    parts = ["traffic", clock(t.rel_brg), miles(t.dist_nm)]
    if direction(t.hdg_true):
        parts.append(direction(t.hdg_true))
    parts.append(describe_type(t))
    if t.alt_ft is not None:
        parts.append(f"altitude indicates {alt_words(t.alt_ft)}")
    return ", ".join(parts)


class TrafficWatch:
    """Decides when a radar controller should call traffic (once per target, re-armed after it clears)."""

    def __init__(self):
        self.last_dist: dict[int, float] = {}
        self.called: set[int] = set()

    def due(self, targets, max_nm: float = 6.0, max_ralt: float = 1500.0):
        best = None
        seen = set()
        for t in targets or []:
            seen.add(t.idx)
            prev = self.last_dist.get(t.idx)
            self.last_dist[t.idx] = t.dist_nm
            if t.dist_nm > max_nm + 3:
                self.called.discard(t.idx)  # well clear: may be called again later
                continue
            if t.idx in self.called or t.dist_nm > max_nm or abs(t.rel_alt_ft) > max_ralt:
                continue
            closing = prev is not None and t.dist_nm < prev - 0.02
            if closing or t.dist_nm < 3:
                if best is None or t.dist_nm < best.dist_nm:
                    best = t
        for gone in set(self.last_dist) - seen:
            self.last_dist.pop(gone, None)  # keep `called`: target lists flicker; only moving well clear re-arms
        if best:
            self.called.add(best.idx)
        return best


def facts(targets, airport: dict | None, own_lat: float, own_lon: float, radius_nm: float = 10) -> str:
    """Traffic list for the controller's situation."""
    near = sorted((t for t in targets or [] if t.dist_nm <= radius_nm), key=lambda t: t.dist_nm)[:6]
    if not near:
        return "Other traffic: none nearby. Do not mention traffic or sequence numbers."
    lines = ["TRAFFIC (real aircraft near you; use only these for advisories or sequencing):"]
    for t in near:
        where = ""
        if airport and t.lat is not None and t.lon is not None:
            d = dist_nm(airport["lat"], airport["lon"], t.lat, t.lon)
            b = bearing(airport["lat"], airport["lon"], t.lat, t.lon)
            where = f", {d:.1f} nm {COMPASS8[int(((b % 360) + 22.5) // 45) % 8]} of {airport['id']}"
        alt = f"{t.alt_ft:.0f} ft" if t.alt_ft is not None else "altitude unknown"
        lines.append(f"- {describe_type(t)}: from you {clock(t.rel_brg)}, {t.dist_nm:.1f} nm, {alt} "
                     f"({t.rel_alt_ft:+.0f} ft), {direction(t.hdg_true) or 'heading unknown'}{where}")
    return "\n".join(lines)
