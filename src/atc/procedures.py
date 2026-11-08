"""Instrument approach procedures from X-Plane's CIFP files, and vectors to final.

CIFP files (ARINC 424 derived, one per airport) live in
  <X-Plane>/Custom Data/CIFP/<ICAO>.dat   (navdata updates, preferred)
  <X-Plane>/Resources/default data/CIFP/<ICAO>.dat
Fixes are referenced by name only; geometry for vectoring comes from the runway
threshold (RWY records) and the final approach course, which is all we need.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from .airports import bearing, dist_nm
from .phraseology import NUMBERS, spoken_runway

CIFP_DIRS = ["Custom Data/CIFP", "Resources/default data/CIFP"]

KIND_NAMES = {
    "I": "ILS", "L": "LOC", "R": "RNAV (GPS)", "H": "RNAV (RNP)", "P": "GPS", "D": "VOR/DME", "V": "VOR",
    "S": "VOR", "N": "NDB", "Q": "NDB/DME", "B": "LOC back course", "X": "LDA", "U": "SDF", "G": "IGS",
}
SPOKEN_KIND = {"RNAV (GPS)": "RNAV", "RNAV (RNP)": "RNAV RNP", "LOC back course": "localizer back course",
               "LOC": "localizer"}
DESC_FT_PER_NM = 318.0  # 3 degree glidepath


@dataclass
class Transition:
    name: str  # transition id, usually the IAF
    first_fix: str
    altitude: str  # e.g. "at or above 4000"


@dataclass
class Approach:
    ident: str  # CIFP id, e.g. I30L, R30LZ
    kind: str  # ILS, RNAV (GPS), ...
    runway: str  # 30L
    variant: str  # "", "Y", "Z"
    transitions: list[Transition] = field(default_factory=list)
    if_fix: str = ""
    intercept_alt: int | None = None  # glideslope/glidepath intercept or IF altitude
    faf_fix: str = ""
    faf_alt: int | None = None
    course_mag: float | None = None  # final approach course, magnetic
    nav_ident: str = ""  # localizer ident for ILS/LOC

    @property
    def name(self) -> str:
        v = f" {self.variant}" if self.variant else ""
        return f"{self.kind}{v} RWY {self.runway}"

    @property
    def spoken(self) -> str:
        k = SPOKEN_KIND.get(self.kind, self.kind)
        v = {"Y": " Yankee", "Z": " Zulu", "X": " X-ray", "W": " Whiskey"}.get(self.variant, "")
        return f"{k}{v} runway {spoken_runway(self.runway)} approach"


@dataclass
class Threshold:
    runway: str
    lat: float
    lon: float
    elev_ft: int


@dataclass
class Sid:
    ident: str  # ALMDN4
    runways: set[str] = field(default_factory=set)  # CIFP runway transitions: "12B" (both 12L/R), "30L"
    transitions: set[str] = field(default_factory=set)  # enroute transitions: HRNER, MOGEE
    vector: bool = False  # radar vector SID

    def serves(self, runway: str) -> bool:
        """Can a departure from this runway fly the SID? (no runway transitions = any runway)"""
        if not self.runways:
            return True
        num = runway[:2]
        return any(r == runway or r == num or (r[:2] == num and r[2:] == "B") or r == "ALL"
                   for r in self.runways)


@dataclass
class Procedures:
    airport: str
    thresholds: dict[str, Threshold]
    approaches: dict[str, Approach]
    sids: dict[str, Sid] = field(default_factory=dict)

    def for_runway(self, runway: str) -> list[Approach]:
        order = {"ILS": 0, "LOC": 1, "RNAV (GPS)": 2, "RNAV (RNP)": 3, "GPS": 4}
        return sorted((a for a in self.approaches.values() if a.runway == runway),
                      key=lambda a: (order.get(a.kind, 9), a.variant))


def _dms(s: str) -> float:
    """N37212246 -> 37.356239 ; W121552213 -> -121.922814 (deg, min, sec with 2 decimals)."""
    hemi, digits = s[0], s[1:]
    deg_len = 2 if hemi in "NS" else 3
    d = int(digits[:deg_len])
    m = int(digits[deg_len:deg_len + 2])
    sec = int(digits[deg_len + 2:]) / 100.0
    v = d + m / 60 + sec / 3600
    return -v if hemi in "SW" else v


def _alt(desc: str, a1: str) -> tuple[int | None, str]:
    a1 = a1.strip()
    if not a1.isdigit():
        return None, ""
    alt = int(a1)
    words = {"+": "at or above", "-": "at or below", "B": "between", "J": "at or above", "H": "at or above"}
    return alt, f"{words.get(desc.strip(), 'at')} {alt}".strip()


def find_cifp(xplane_path: str, airport_id: str, extra_dirs: list[str] | None = None) -> Path | None:
    roots = [Path(xplane_path).expanduser() / d for d in CIFP_DIRS] if xplane_path else []
    roots += [Path(d) for d in (extra_dirs or [])]
    for r in roots:
        p = r / f"{airport_id}.dat"
        if p.exists():
            return p
    return None


def parse_cifp(path: Path) -> Procedures:
    thresholds: dict[str, Threshold] = {}
    approaches: dict[str, Approach] = {}
    sids: dict[str, Sid] = {}
    for line in path.read_text(encoding="latin-1").splitlines():
        if line.startswith("SID:"):
            f = [x.strip() for x in line[4:].rstrip(";").split(",")]
            if len(f) < 4 or not f[2]:
                continue
            sid = sids.setdefault(f[2], Sid(f[2]))
            rtype, trans = f[1], f[3]
            if rtype in ("T", "V"):
                sid.vector = True
            if trans.startswith("RW"):
                sid.runways.add(trans[2:])
            elif trans and trans != "ALL" and rtype in ("3", "6", "S", "V"):
                sid.transitions.add(trans)
        elif line.startswith("RWY:"):
            head, _, coords = line[4:].partition(";")
            f = [x.strip() for x in head.split(",")]
            c = [x.strip() for x in coords.split(",")]
            if len(c) >= 2 and f[0].startswith("RW"):
                rw = f[0][2:]
                elev = int(f[3]) if len(f) > 3 and f[3].isdigit() else 0
                thresholds[rw] = Threshold(rw, _dms(c[0]), _dms(c[1]), elev)
        elif line.startswith("APPCH:"):
            f = [x for x in line[6:].rstrip(";").split(",")]
            if len(f) < 24:
                continue
            route, ident, trans, fix = f[1].strip(), f[2].strip(), f[3].strip(), f[4].strip()
            desc, path_term = f[8], f[11].strip()
            ap = approaches.get(ident)
            if ap is None:
                kind = KIND_NAMES.get(ident[0], ident[0])
                rest = ident[1:]
                rwy = rest[:2] + (rest[2] if len(rest) > 2 and rest[2] in "LRC" else "")
                variant = rest[len(rwy):] if rwy[:2].isdigit() else ""
                if not rwy[:2].isdigit():  # circling-only approach (e.g. VOR-A); skip
                    continue
                ap = approaches[ident] = Approach(ident, kind, rwy, variant)
            alt, alt_txt = _alt(f[22], f[23])
            if route == "A":  # transition
                t = next((t for t in ap.transitions if t.name == trans), None)
                if t is None and fix:
                    ap.transitions.append(Transition(trans or fix, fix, alt_txt))
                continue
            # final approach segment
            if path_term == "IF" and not ap.if_fix:
                ap.if_fix = fix
                ap.intercept_alt = alt
                if f[13].strip():
                    ap.nav_ident = f[13].strip()
            role = desc[3] if len(desc) > 3 else " "
            if role == "F" and not ap.faf_fix:
                ap.faf_fix = fix
                ap.faf_alt = alt
                crs = f[20].strip()
                if crs.isdigit():
                    ap.course_mag = int(crs) / 10.0
                if f[13].strip():
                    ap.nav_ident = f[13].strip()
    for ap in approaches.values():
        if ap.course_mag is None and ap.runway[:2].isdigit():
            ap.course_mag = float(int(ap.runway[:2]) * 10)  # fallback: runway number
    return Procedures(path.stem, thresholds, approaches, sids)


_cache: dict[str, Procedures | None] = {}


def load(xplane_path: str, airport_id: str, extra_dirs: list[str] | None = None) -> Procedures | None:
    key = f"{xplane_path}|{airport_id}"
    if key not in _cache:
        p = find_cifp(xplane_path, airport_id, extra_dirs)
        _cache[key] = parse_cifp(p) if p else None
    return _cache[key]


def _digits(n: int) -> str:
    return " ".join(NUMBERS[c] for c in str(n))


def _heading_words(h: float) -> str:
    h = int(round(h / 5.0) * 5) % 360 or 360  # controllers assign headings in 5 degree steps
    return _digits(f"{h:03d}")


def _miles_words(n: int) -> str:
    return f"{' '.join(NUMBERS[c] for c in str(n))} mile{'s' if n != 1 else ''}"


def _alt_words(ft: int) -> str:
    th, hu = divmod(int(round(ft / 100.0) * 100), 1000)
    out = []
    if th:
        out.append(f"{_digits(th)} thousand")
    if hu:
        out.append(f"{_digits(hu // 100)} hundred")
    return " ".join(out) or "zero"


def vectors(ap: Approach, thr: Threshold, lat: float, lon: float, magvar: float,
            cur_hdg: float | None = None, cur_alt: float | None = None) -> str:
    return vector_plan(ap, thr, lat, lon, magvar, cur_hdg, cur_alt)[2]


def altitude_change(cur_alt: float | None, target: int) -> str:
    """'descend and maintain X' / 'climb and maintain X' / 'maintain X' relative to the aircraft."""
    words = _alt_words(target)
    if cur_alt is None:
        return f"maintain {words}"
    if cur_alt > target + 200:
        return f"descend and maintain {words}"
    if cur_alt < target - 200:
        return f"climb and maintain {words}"
    return f"maintain {words}"


def turn_words(cur_hdg: float | None, target: float) -> str:
    """'turn left' / 'turn right' (shorter turn), 'fly' when nearly there or heading unknown."""
    if cur_hdg is None:
        return "fly"
    tgt = round(target / 5.0) * 5
    diff = (tgt - cur_hdg + 540) % 360 - 180
    if abs(diff) < 5:
        return "fly"
    return "turn right" if diff > 0 else "turn left"


def vector_plan(ap: Approach, thr: Threshold, lat: float, lon: float, magvar: float,
                cur_hdg: float | None = None, cur_alt: float | None = None) -> tuple[bool, float, str]:
    """(in position to clear, magnetic heading to assign, controller instruction text)."""
    """What a radar controller would say now to get this aircraft onto the approach.
    magvar = true - magnetic (as in AircraftState)."""
    crs_true = (ap.course_mag + magvar) % 360
    out_true = (crs_true + 180) % 360
    d = dist_nm(thr.lat, thr.lon, lat, lon)
    brg = bearing(thr.lat, thr.lon, lat, lon)
    rel = math.radians(brg - out_true)
    along = d * math.cos(rel)  # nm out along the extended centerline
    cross = d * math.sin(rel)  # >0: right of the extended centerline looking outbound = LEFT of course inbound
    faf_nm = ((ap.faf_alt or 2000) - thr.elev_ft) / DESC_FT_PER_NM
    alt = ap.intercept_alt or ap.faf_alt or 3000
    name = ap.spoken
    estab = "the localizer" if ap.kind in ("ILS", "LOC", "LDA", "SDF") else "the final approach course"
    # Join leg: intercept at 30 degrees (20 when close in)
    intercept_angle = 30 if abs(cross) > 1.5 else 20
    hdg_true = crs_true + (intercept_angle if cross > 0 else -intercept_angle)
    intercept_at = along - abs(cross) / math.tan(math.radians(intercept_angle))
    hdg_mag = (hdg_true - magvar) % 360
    turn_ok = 0.3 < abs(cross) < 8 and faf_nm + 2 <= intercept_at <= faf_nm + 9
    # Straight in: already close to the course and heading roughly inbound
    if not turn_ok and abs(cross) <= 1.0 and faf_nm + 2 <= along <= faf_nm + 15 and cur_hdg is not None and abs(
            ((cur_hdg + magvar) - crs_true + 540) % 360 - 180) < 40:
        intercept_angle = 10 if abs(cross) > 0.2 else 0
        hdg_true = crs_true + (intercept_angle if cross > 0 else -intercept_angle)
        hdg_mag = (hdg_true - magvar) % 360
        intercept_at = along
        turn_ok = True
    if turn_ok and cur_alt is not None and cur_alt < alt - 300:
        turn_ok = False  # never clear an approach below the altitude it must be maintained until established
    if turn_ok:
        return True, hdg_mag, (f"Aircraft is {along:.0f} nm out along runway {ap.runway}'s extended centerline and "
                f"{abs(cross):.1f} nm {'left' if cross > 0 else 'right'} of it (looking inbound); a "
                f"{intercept_angle} degree intercept joins {intercept_at:.0f} nm out (FAF {ap.faf_fix} is ~{faf_nm:.0f} nm). "
                f"APPROACH CLEARANCE NOW: \"{_miles_words(max(1, round(along - faf_nm)))} from {ap.faf_fix}, "
                f"{turn_words(cur_hdg, hdg_mag)} heading {_heading_words(hdg_mag)}, maintain {_alt_words(alt)} until "
                f"established on {estab}, cleared {name}\".")
    # Not in position yet: a radar pattern (downwind, base) on the side the aircraft is on
    side = 1 if cross >= 0 else -1  # +1: right of the extended centerline looking outbound
    base_turn = faf_nm + 7  # turn base this far out (leaves a wide intercept window)
    downwind_off = 4.0  # nm abeam the final approach course
    vec_alt = int(math.ceil((alt + 1000) / 500.0) * 500)
    base_h = out_true - side * 90
    on_base_heading = cur_hdg is not None and abs(((cur_hdg + magvar) - base_h + 540) % 360 - 180) < 30
    if on_base_heading and base_turn - 5 <= along <= base_turn + 3 and abs(cross) > 0.3:
        # Already flying base: stay on it (no flip-flopping back to downwind)
        phase, h_true, a_tgt = "base", base_h, alt
    elif along > faf_nm + 14 or abs(cross) > 12:
        # Far away: head for the base-turn area
        plat = thr.lat + (base_turn * math.cos(math.radians(out_true))
                          + side * downwind_off * math.cos(math.radians(out_true + 90))) / 60
        plon = thr.lon + (base_turn * math.sin(math.radians(out_true))
                          + side * downwind_off * math.sin(math.radians(out_true + 90))) / 60 / math.cos(
            math.radians(thr.lat))
        phase, h_true, a_tgt = "inbound to the downwind", bearing(lat, lon, plat, plon), vec_alt
    elif along < base_turn - 1:
        if abs(cross) < downwind_off - 1.5:
            phase, h_true = "building the downwind", out_true + side * 45
        elif abs(cross) > downwind_off + 3:
            phase, h_true = "downwind (closing in)", out_true - side * 20
        else:
            phase, h_true = "downwind", out_true
        a_tgt = vec_alt
    elif along <= base_turn + 3:
        phase, h_true, a_tgt = "base", base_h, alt
    else:
        # Beyond the base-turn area: angle in toward the final approach course for a straight-in
        plat = thr.lat + base_turn * math.cos(math.radians(out_true)) / 60
        plon = thr.lon + base_turn * math.sin(math.radians(out_true)) / 60 / math.cos(math.radians(thr.lat))
        phase, h_true, a_tgt = "vectors to final", bearing(lat, lon, plat, plon), vec_alt
    to_hdg = (h_true - magvar) % 360
    return False, to_hdg, (f"Aircraft is {d:.0f} nm from runway {ap.runway} ({along:.0f} nm out along the extended "
            f"centerline, {abs(cross):.1f} nm to the side), radar pattern phase: {phase}. VECTOR NOW: "
            f"\"{turn_words(cur_hdg, to_hdg)} heading {_heading_words(to_hdg)}, vectors {name.replace(' approach', '')} "
            f"final approach course, {altitude_change(cur_alt, a_tgt)}\". "
            f"When it is in position, the clearance will be given.")


def describe(procs: Procedures, runway: str, lat: float, lon: float, magvar: float,
             cur_hdg: float | None = None, cleared: bool = False, cur_alt: float | None = None) -> str:
    """Facts block for an approach controller about the destination's approaches."""
    aps = procs.for_runway(runway)
    if not aps:
        others = sorted({a.runway for a in procs.approaches.values()})
        return (f"INSTRUMENT APPROACHES at {procs.airport}: none to runway {runway}; approaches exist to "
                f"runway(s) {', '.join(others) or 'none'}")
    lines = [f"INSTRUMENT APPROACHES at {procs.airport} to runway {runway} (use only these):"]
    for i, a in enumerate(aps):
        trans = "; ".join(f"{t.name} ({t.altitude})" if t.altitude else t.name for t in a.transitions[:6])
        default = " DEFAULT: the one to offer/expect unless the pilot asked for another." if i == 0 else ""
        lines.append(f"- {a.name} (spoken \"{a.spoken}\"): final course {a.course_mag:03.0f}, intercept altitude "
                     f"{a.intercept_alt}, FAF {a.faf_fix} at {a.faf_alt}; transitions/IAFs: {trans or 'none'}.{default}")
    thr = procs.thresholds.get(runway)
    if cleared:
        lines.append("The aircraft is already CLEARED for the approach: no more vectors. Do NOT hand it to the tower "
                     "yet; that happens automatically once it is established near the final approach fix.")
    elif thr:
        lines.append("Vectors for the first approach listed (or the one the pilot requested, same geometry): "
                     + vectors(aps[0], thr, lat, lon, magvar, cur_hdg, cur_alt))
    lines.append("Own-navigation alternative (pilot requests the full procedure or a transition): \"cross <IAF> "
                 "<altitude>, cleared <approach>\" using the IAF altitude listed.")
    return "\n".join(lines)


def course_offset(ap: Approach, thr: Threshold, lat: float, lon: float, magvar: float) -> tuple[float, float, float]:
    """(nm out along the extended centerline, nm off the final approach course, FAF distance nm)."""
    out_true = (ap.course_mag + magvar + 180) % 360
    d = dist_nm(thr.lat, thr.lon, lat, lon)
    rel = math.radians(bearing(thr.lat, thr.lon, lat, lon) - out_true)
    faf_nm = ((ap.faf_alt or 2000) - thr.elev_ft) / DESC_FT_PER_NM
    return d * math.cos(rel), abs(d * math.sin(rel)), faf_nm
