"""Filed flight plans from SimBrief.

SimBrief's public fetcher returns a pilot's latest flight plan (OFP) by username or numeric user id,
with no API key:
  https://www.simbrief.com/api/xml.fetcher.php?username=<name>&json=1
We keep only what a clearance needs: origin, destination, route, cruise altitude and the SID/STAR.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

import httpx

from .phraseology import NUMBERS, spell

log = logging.getLogger(__name__)

SIMBRIEF_URL = "https://www.simbrief.com/api/xml.fetcher.php"
_PROC_RE = re.compile(r"^([A-Z]{2,5})(\d)([A-Z]?)$")  # ALMDN4, ULGOT1G, KETO2A
# ICAO speed/level groups: "N0193F140" at the start of atc.route, or "BIL/N0190F150" at a change point
_SPEED_LEVEL_RE = re.compile(r"/?[NKM]\d{3,4}[FASM]\d{3,4}$")
_AIRWAY_RE = re.compile(r"^[A-Z]{1,2}\d{1,4}[A-Z]?$")  # V25, J58, UL9, N746, Q120


@dataclass
class FlightPlan:
    origin: str
    destination: str
    alternate: str = ""
    route: str = ""  # ICAO route string, SID and STAR included when filed
    cruise_ft: int | None = None
    callsign: str = ""
    aircraft: str = ""
    origin_rwy: str = ""
    sid: str = ""  # e.g. ALMDN4
    sid_exit: str = ""  # route fix right after the SID (its enroute transition or end fix)
    star: str = ""
    first_fix: str = ""  # first en-route fix when there's no SID
    generated: int = 0  # unix time SimBrief made the plan
    source: str = "SimBrief"

    @property
    def summary(self) -> str:
        alt = f", cruise {self.cruise_ft} ft" if self.cruise_ft else ""
        return f"{self.origin}-{self.destination}: {self.route or 'DCT'}{alt}"

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _s(v) -> str:
    """SimBrief writes empty fields as {} or None; normalize to a stripped string."""
    return v.strip() if isinstance(v, str) else ""


def _int(v) -> int | None:
    v = _s(v)
    return int(v) if v.isdigit() else None


def parse_simbrief(data: dict) -> FlightPlan:
    status = _s((data.get("fetch") or {}).get("status"))
    if status and status.lower() != "success":
        raise ValueError(f"SimBrief: {status}")
    general, atc = data.get("general") or {}, data.get("atc") or {}
    origin = _s((data.get("origin") or {}).get("icao_code")).upper()
    dest = _s((data.get("destination") or {}).get("icao_code")).upper()
    if not origin or not dest:
        raise ValueError("SimBrief plan has no origin or destination")
    # atc.route is the filed ICAO route with SID/STAR; general.route sometimes leaves them out
    route = _s(atc.get("route")) or _s(general.get("route"))
    route = " ".join(t for t in (_SPEED_LEVEL_RE.sub("", w) for w in route.upper().split()) if t and t != "DCT")
    tokens = route.split()
    fixes = (data.get("navlog") or {}).get("fix") or []
    if isinstance(fixes, dict):  # a single fix comes through as an object
        fixes = [fixes]
    proc_names = {_s(f.get("via_airway")).upper() for f in fixes if _s(f.get("is_sid_star")) == "1"}
    climb_procs = [_s(f.get("via_airway")).upper() for f in fixes
                   if _s(f.get("is_sid_star")) == "1" and _s(f.get("stage")).upper() == "CLB"]

    def is_proc(tok: str) -> bool:
        return bool(_PROC_RE.match(tok)) and (tok in proc_names or not fixes)

    sid = climb_procs[0] if climb_procs else (tokens[0] if tokens and is_proc(tokens[0]) else "")
    rest = tokens[1:] if tokens and sid and tokens[0] == sid else tokens
    star = tokens[-1] if len(tokens) > 1 and is_proc(tokens[-1]) and tokens[-1] != sid else ""
    sid_exit = rest[0] if sid and rest and not _AIRWAY_RE.match(rest[0]) and rest[0] != star else ""
    first_fix = "" if sid else next((t for t in rest if not _AIRWAY_RE.match(t) and t != star), "")
    if not sid and not first_fix:  # no route string: first navlog fix that isn't the airport or a TOC/TOD
        first_fix = next((_s(f.get("ident")).upper() for f in fixes
                          if _s(f.get("type")) not in ("apt", "ltlg") and _s(f.get("ident"))), "")
    cruise = _int(general.get("initial_altitude"))
    if cruise is None:
        ia = _s(atc.get("initial_alt"))  # flight level or hundreds of feet ("360", "080")
        cruise = int(ia) * 100 if ia.isdigit() else None
    return FlightPlan(
        origin=origin, destination=dest,
        alternate=_s((data.get("alternate") or {}).get("icao_code")).upper(),
        route=route,
        cruise_ft=cruise,
        callsign=_s(atc.get("callsign")),
        aircraft=_s((data.get("aircraft") or {}).get("icaocode")),
        origin_rwy=_s((data.get("origin") or {}).get("plan_rwy")),
        sid=sid, sid_exit=sid_exit, star=star, first_fix=first_fix,
        generated=_int((data.get("params") or {}).get("time_generated")) or 0,
    )


async def fetch_simbrief(user: str, timeout: float = 5.0) -> FlightPlan:
    """Latest SimBrief plan for a username (or a numeric SimBrief user id)."""
    key = "userid" if user.isdigit() else "username"
    async with httpx.AsyncClient(timeout=timeout, headers={"User-Agent": "xplane-ai-atc"}) as http:
        r = await http.get(SIMBRIEF_URL, params={key: user, "json": "1"})
    try:
        data = r.json()
    except ValueError:
        raise ValueError(f"SimBrief returned HTTP {r.status_code}") from None
    return parse_simbrief(data)


def spoken_procedure(name: str) -> str:
    """ALMDN4 -> 'ALMDN four', ULGOT1G -> 'ULGOT one Golf' (the name part is read as the fix name)."""
    m = _PROC_RE.match(name.upper())
    if not m:
        return name
    out = f"{m.group(1)} {NUMBERS[m.group(2)]}"
    return f"{out} {spell(m.group(3))}" if m.group(3) else out


def plan_age_text(plan: FlightPlan, now: float | None = None) -> str:
    if not plan.generated:
        return ""
    mins = int(((now or time.time()) - plan.generated) / 60)
    return f"{mins} min old" if mins < 120 else f"{mins // 60} h old"
