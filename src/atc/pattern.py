"""Traffic pattern geometry: which leg an aircraft is on, and the landing sequence."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .airports import bearing, dist_nm


@dataclass
class PatternPos:
    leg: str  # upwind, crosswind, downwind, base, final, (or "" when not in the pattern)
    side: str  # left / right (pattern side relative to the runway direction)
    along: float  # nm along the runway direction from the landing threshold (negative = before it)
    cross: float  # nm off the extended centerline (positive = right of runway direction)
    path_nm: float  # remaining distance to the threshold flying the pattern
    detail: str  # e.g. "abeam midfield", "1.8 nm final"

    def describe(self, runway: str) -> str:
        if self.leg == "final":
            return f"{self.detail} runway {runway}"
        return f"{self.side} {self.leg} runway {runway}" + (f", {self.detail}" if self.detail else "")


def _runway_frame(airport: dict, runway: str):
    for r in airport.get("runways", []):
        if runway in r["ids"]:
            if r["ids"][0] == runway:
                thr, far = (r["lat1"], r["lon1"]), (r["lat2"], r["lon2"])
            else:
                thr, far = (r["lat2"], r["lon2"]), (r["lat1"], r["lon1"])
            return thr, bearing(thr[0], thr[1], far[0], far[1]), r["length_ft"] / 6076.12
    return None


def _angle(a: float, b: float) -> float:
    return abs((a - b + 540) % 360 - 180)


def position(airport: dict, runway: str, lat: float, lon: float, agl_ft: float, track_true: float | None,
             pattern_side: str = "left") -> PatternPos | None:
    """Classify an airborne aircraft's position in the pattern of `runway`. None if not near the pattern."""
    frame = _runway_frame(airport, runway)
    if not frame or agl_ft > 2500:
        return None
    (tlat, tlon), rhdg, length = frame
    d = dist_nm(tlat, tlon, lat, lon)
    if d > 6:
        return None
    rel = math.radians(bearing(tlat, tlon, lat, lon) - rhdg)
    along, cross = d * math.cos(rel), d * math.sin(rel)
    side = "right" if cross > 0 else "left"
    trk = track_true
    rev = (rhdg + 180) % 360
    if abs(cross) < 0.6 and along < 0 and (trk is None or _angle(trk, rhdg) < 45):
        return PatternPos("final", side, along, cross, -along, f"{-along:.1f} nm final")
    if abs(cross) < 0.6 and along >= 0 and (trk is None or _angle(trk, rhdg) < 45):
        return PatternPos("upwind", side, along, cross, along + 2 * 0.8 + length + 1.0, "")
    if 0.4 <= abs(cross) <= 2.5 and trk is not None and _angle(trk, rev) < 45:
        if along > length:
            detail = "early downwind"
        elif along > length * 0.3:
            detail = "abeam midfield"
        elif along > -0.3:
            detail = "abeam the numbers"
        else:
            detail = f"extended {-along:.1f} nm past the numbers"
        return PatternPos("downwind", side, along, cross, max(0.0, along) + abs(cross) + max(0.0, -along) * 2 + 0.5,
                          detail)
    perp = trk is not None and (_angle(trk, (rhdg + 90) % 360) < 45 or _angle(trk, (rhdg - 90) % 360) < 45)
    if along < -0.2 and abs(cross) >= 0.4 and perp:
        return PatternPos("base", side, along, cross, abs(cross) + abs(along), "")
    if along > length * 0.5 and abs(cross) >= 0.3 and perp:
        return PatternPos("crosswind", side, along, cross, abs(cross) + along + 1.0, "")
    return None


def sequence(airport: dict, runway: str, own: PatternPos | None, others: list[tuple[str, PatternPos]]) -> list[str]:
    """Landing order of aircraft on downwind/base/final, as phrases ('you', 'Piper on left base', ...)."""
    items = []
    if own and own.leg in ("downwind", "base", "final"):
        items.append((own.path_nm, "you"))
    for name, p in others:
        if p.leg in ("downwind", "base", "final"):
            leg = f"{p.side} {p.leg}" if p.leg != "final" else "final"
            items.append((p.path_nm, f"{name} on {leg}"))
    return [label for _, label in sorted(items)]
