"""The ATC "brain": works out the facts in code (runway in use, facility name,
squawk codes) and hands them to the language model, which only has to
understand the pilot and phrase the reply."""

from __future__ import annotations

import math
import random
import time as _time
import re
from collections import deque

from .airports import AirportDB, Facility, bearing, dist_nm
from .airspace import CLASS_B, airspace_class
from .flightplan import FlightPlan, spoken_procedure
from .phraseology import NUMBERS, altimeter_phrase, callsign_forms, region_for, spoken_runway, spoken_wind
from .phraseology import altitude_words as _alt_words
from .state import AircraftState, fmt_freq
from .procedures import describe as describe_procedures
from .procedures import load as load_procedures
from .procedures import course_offset as approach_course_offset
from .procedures import vector_plan as approach_vector_plan
from .pattern import position as pattern_position
from .pattern import sequence as pattern_sequence
from .traffic import TrafficWatch, advisory as traffic_advisory
from .traffic import describe_type as traffic_type
from .traffic import facts as traffic_facts
from .taxi import on_runway, route_to_parking, route_to_runway, runway_exit, spoken_taxiway

SILENCE = "<silence>"
RADAR_KINDS = ("APPROACH", "DEPARTURE", "CENTER")
# A pilot transmission with any of these wants to go somewhere, not to park
DEPARTURE_WORDS = (r"\b(depart|departure|take ?off|for runway|to runway|vfr|ifr|[a-z]+bound|pattern|closed traffic|"
                   r"touch and go|flight following|run ?up|to the (north|south|east|west))")
ACK_TIMEOUT_S = 20  # chase an unacknowledged instruction after this long (real time)
ACK_CHASE = "The pilot has not acknowledged your last transmission"  # start of that event's text
ALT_CONFORMANCE_S = 60  # an aircraft level this long away from its assigned altitude gets queried
RECONTACT_S = 180  # a call this long after a facility's handoff is a new contact, not the goodbye
FULL_STOP_KT = 25  # slower than this on the ground after a landing: the flight is over (not a touch and go)
GROUND_TRAFFIC_AGL_FT = 100  # a target lower than this over a nearby airport is on the ground
INBOUND_WORDS = r"\b(inbound|to land|for landing|full stop|touch and go|for the option)\b"
EVENT_LABEL = "EVENT (no pilot transmission - you initiate this call)"

SYSTEM_PROMPT = f"""You are an air traffic controller working one radio frequency in a flight simulator.
You get the current SITUATION (facts computed from the simulator and airport database) and the pilot's
latest transmission, transcribed by speech recognition (expect transcription errors in names, callsigns
and numbers; interpret them sensibly, e.g. "Salisbury suck" said to Salzburg Tower is "Salzburg Tower").

Rules:
- Speak only as the facility in the situation, using the spoken facility name given there.
- Use the PHRASEOLOGY named in the situation:
  FAA = FAA JO 7110.65 (US): "altimeter two niner niner two", "wind two seven zero at one zero",
    "runway three one, line up and wait", "taxi to runway three one", "Cessna three alpha bravo".
  ICAO = ICAO Doc 4444 / EASA (Europe and most of the world): "QNH one zero one three",
    "wind two seven zero degrees, one zero knots", "taxi to holding point runway three three",
    "line up runway three three", "climb to altitude five thousand feet", "squawk four six two one".
  Be brief and professional, like a real controller. One transmission only.
- Callsigns: use the spoken callsign forms from the situation, or how the pilot said it if it is the same
  aircraft (FAA allows make + last three, e.g. "Cirrus zero Charlie Romeo"). Never say a bare letter like
  "N"; registrations are always spoken with the phonetic alphabet ("November five five zero Charlie Romeo").
  Use the full callsign on first contact; abbreviate only after you have used it yourself. Keep using the
  same form the pilot uses (if they say "Cirrus ...", don't switch to "November ...").
- The SITUATION is the truth. Use only the runway in use, wind, altimeter/QNH, frequencies and airports it
  lists. Never invent frequencies, taxiways, hold-short points, traffic or sequence numbers.
- Taxi: if TAXI ROUTE is given, use it exactly. Otherwise you have no taxiway data: say e.g. "runway three
  one, taxi via the parallel taxiway" (FAA) or "taxi to holding point runway three three" (ICAO).
- Tower: give pattern entry or straight-in instructions; clear to land or take off only on the runway in use.
  Don't combine a pattern entry and a landing clearance in the same call unless the aircraft is on final.
  Order: "wind three two zero at one two, runway three zero left, cleared for takeoff" (same for
  "cleared to land"). If a VFR departure's direction of flight is unknown, ask "say direction of flight"
  before clearing it for takeoff, unless DEPARTURE FLOW says it was already handled. When it is known,
  approve it before the clearance using the shortest turn from the runway heading, e.g. runway heading
  three zero zero, southbound: "left turn southbound approved, wind ..., runway ..., cleared for takeoff";
  straight out: "straight out departure approved".
- ATIS: only when the situation says the pilot has NOT reported the current ATIS at this airport, include
  "information <current letter> current" (ground/tower) or the altimeter. Otherwise don't mention ATIS.
- VFR aircraft are never "cleared to" a destination; clearance delivery gives "<callsign>, maintain VFR at or
  below ..., departure frequency ..., squawk ...".
- Readbacks: pilots must read back runway assignments with takeoff, landing and crossing clearances and all
  hold-short instructions, and should read back taxi routes, headings, altitudes, squawk codes and
  frequencies. If the pilot answers one of these with only "roger"/"wilco"/a callsign, or leaves out the
  runway or a hold-short, ask for it with a standard phrase: "<callsign>, read back taxi instructions",
  "<callsign>, read back hold short instructions", "<callsign>, verify runway three zero left".
  Don't accept "roger" for a clearance.
- Runway crossings: when TAXI ROUTE lists runways to cross, the taxi instruction must include
  "cross runway <x>" (or "hold short of runway <x>") for each one, in route order.
- Weather requests: give wind and altimeter/QNH from the situation (and ATIS information if listed).
- VFR aircraft choose their own altitude: never say "climb", "descend" or "maintain <altitude>" to a VFR
  aircraft unless it is a real restriction (e.g. "maintain VFR at or below two thousand five hundred" on a
  Class B/C departure). Acknowledge their altitude instead, and use "maintain VFR". When a VFR pilot asks to
  climb or descend: "VFR altitude your discretion" (add "remain clear of Class Bravo airspace" if the
  situation lists Class B nearby). Never say "climb approved" or "proceed direct" to a VFR aircraft;
  after radar contact say "resume own navigation" if they were given a heading or restriction.
- Instrument approaches (approach/departure controllers): offer or issue only approaches listed under
  INSTRUMENT APPROACHES. When a pilot requests an approach, say "expect <approach>" and give the VECTOR NOW
  instruction from the situation. Clear the approach only with the APPROACH CLEARANCE NOW wording from the
  situation (heading, "maintain <altitude> until established on the localizer/final approach course",
  "cleared <approach>"), or for own navigation "cross <IAF> at or above <altitude>, cleared <approach>".
  VFR practice approaches: use the wording given with "Flight rules" in the situation.
- IFR: follow "IFR CLEARANCE" / "IFR DEPARTURE" in the situation exactly. A correct IFR clearance readback gets
  "<callsign>, readback correct".
- DEPARTURE FLOW in the situation (Class B/C airports) says what a VFR departure still needs; follow it.
- Radar services (flight following, IFR, flight information, Class B/C departures): if the situation says
  no code has been assigned, assign the NEXT SQUAWK CODE given. Say "radar contact" (or "identified" for ICAO) with position
  only when the situation says the transponder now shows the assigned code, and give the altimeter with it:
  "radar contact, three miles northwest of San Jose, altimeter three zero zero one".
- An EVENT instead of a pilot transmission means you are making an unprompted call (e.g. a frequency change
  or radar contact): address the aircraft by callsign and say only what the event asks for.
- If the transmission is unintelligible or incomplete, ask the pilot to say again.
- Every transmission starts with the aircraft's callsign.
- Pattern: use "Aircraft position", "Pattern traffic" and LANDING SEQUENCE from the situation for pattern
  calls ("number two, follow the Piper on left base", "extend downwind, I'll call your base", "make left closed
  traffic", "report midfield downwind"). Pattern work (closed traffic / touch and go / option) never gets a
  frequency change after takeoff.
- Traffic: use only aircraft listed under TRAFFIC. When a pilot reports "traffic in sight" or "looking" after an
  advisory, no reply is needed ({SILENCE}) unless you want to add an instruction (e.g. "follow that traffic").
  Tower may sequence with it: "number two, follow the Cessna on left downwind".
- If the pilot reads back an instruction incorrectly or leaves items out, correct just those items, all of them:
  "<callsign>, negative, fly heading one three five, climb and maintain six thousand one hundred"; don't repeat
  items they got right. If the readback is correct and nothing
  new is needed, output {SILENCE}. If the transmission is not for you or needs no reply, output {SILENCE}.
- A readback is correct when it has the right numbers, even in short form ("right one zero zero", "seven
  thousand", "two five zero, six thousand until established"). Aircraft take time to turn, climb and descend:
  never "verify" a heading or altitude the aircraft is still turning/climbing toward, and never re-issue an
  instruction the pilot just read back correctly; output {SILENCE}.
- After you hand an aircraft to another frequency ("contact ..."), its readback or goodbye gets {SILENCE}.
  Never tell it to remain with you.
- "Negative" is only for a pilot error. When you add or change something you didn't say before (a route, a
  crossing), the pilot's readback of your earlier words was correct: just give the addition or say "amend".
- "Loud and clear" (answering your radio check) gets {SILENCE}, or the instruction it never acknowledged, once.
- Write the reply exactly as it should be spoken: every number as words digit by digit in aviation style
  ("one one eight point six", "squawk four six two one"), except altitudes ("five thousand five hundred").
- No quotes, labels or explanations. Output only the spoken words or {SILENCE}.
"""

ALPHABET_WORDS = ["alpha", "alfa", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india",
                  "juliet", "juliett", "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo",
                  "sierra", "tango", "uniform", "victor", "whiskey", "x-ray", "xray", "yankee", "zulu"]

COMPASS = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]

KIND_SUFFIX = {
    "TOWER": "Tower", "GROUND": "Ground", "DELIVERY": "Clearance Delivery",
    "APPROACH": "Approach", "DEPARTURE": "Departure",
}
ABBREV = {"TWR": "", "GND": "", "APP": "", "DEP": "", "CLNC": "", "DEL": "", "CD": "", "TRACON": "",
          "APCH": "", "CLR": "", "CTL": "", "ARR": "", "TOWER": "", "GROUND": "", "APPROACH": "",
          "DEPARTURE": "", "CLEARANCE": "", "DELIVERY": "", "CTAF": "", "UNICOM": "", "UNIC": "", "ATCT": "",
          "MULTICOM": ""}
DIGITS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
          "seven": "7", "eight": "8", "niner": "9", "nine": "9"}


def _compass(deg: float) -> str:
    return COMPASS[int(((deg % 360) + 22.5) // 45) % 8]


def spoken_facility(fac: Facility) -> str:
    """'NORCAL APP' -> 'NorCal Approach', 'PALO ALTO TWR' -> 'Palo Alto Tower'."""
    if fac.kind == "CENTER":
        return fac.name  # "Oakland Center"
    words = [w for w in re.split(r"[\s/]+", fac.name) if w and w.upper() not in ABBREV]
    base = " ".join(w.capitalize() for w in words).replace("Norcal", "NorCal").replace("Socal", "SoCal")
    if not base:
        a = fac.airport
        base = a.get("city") or re.sub(
            r"\(.*?\)|\b(Rgnl|Muni|Intl|Arpt|Airport|Regional|Municipal|International)\b", "",
            a["name"], flags=re.I).strip()
    return f"{base} {KIND_SUFFIX.get(fac.kind, fac.kind.title())}".strip()


_NAME_NOISE = re.compile(r"\b(intl|international|muni|municipal|rgnl|regional|airport|arpt|field|county|"
                         r"metro|metropolitan|executive|exec|airpark|apt)\b", re.I)


def short_name(a: dict) -> str:
    """How controllers name an airport: 'Monterey', 'San Jose' (city, or the name without Regl/Intl/...)."""
    if a.get("city"):
        return a["city"].title() if a["city"].isupper() else a["city"]
    clean = re.sub(r"\s+", " ", _NAME_NOISE.sub(" ", a["name"])).strip(" -")
    return clean or a["name"]


# Airport ids that are also everyday words ("we'd like to fly the ILS" is not a flight to KFLY)
_ID_WORDS = frozenset(
    "act add age ago aid aim air all and any are arm art ask bad bag bar bay bed big bit bow box boy bus but buy "
    "can car cut day did dry due eat end far few fit fix fly for fun gas get got had has her him his hot how its "
    "jet job key law lay let lot low man map may men new nor not now off oil old one our out own pay pie put ran "
    "red run sad saw say see set she sit six sky sun ten the top try two use via war was way who why win yes yet "
    "you able also area away back base best both call came come copy does done down each east even ever fast "
    "feet find five four from full gave gear give gone good half hand have head hear here high hold home just "
    "keep kind know land last left less like line long look made make many mile more most much must near need "
    "next nine none once only onto open over past plan read rest road roll same show side some soon stay stop "
    "sure take taxi tell than that them then they this time told took turn unit upon very wait want well went "
    "were west what when wind wing with word work your zero".split())


def _airport_names(a: dict) -> list[str]:
    names = {a["id"].lower()}
    if len(a["id"]) == 4 and a["id"][0] in "KP":
        names.add(a["id"][1:].lower())
    names -= _ID_WORDS
    for n in (a.get("city"), a["name"]):
        if n:
            clean = re.sub(r"\s+", " ", _NAME_NOISE.sub(" ", re.sub(r"[^A-Za-z ]", " ", n))).strip().lower()
            if len(clean) >= 4:
                names.add(clean)
    return [n for n in names if len(n) >= 3]


def find_destination(db, transcript: str, fac) -> dict | None:
    """The airport a pilot says they're going to ("VFR to Monterey", "inbound for landing")."""
    low = transcript.lower()
    a = fac.airport
    # Calling an airport's tower/ground to land there
    if fac.kind == "TOWER" and re.search(r"\b(inbound|landing|full stop|touch and go|for the option|"
                                         r"for landing|to land)\b", low):
        return a
    tails = [m.group(1) for m in re.finditer(
        r"\b(?:to|for|destination|inbound(?: to| for)?|landing(?: at)?|into|"
        r"(?:approach|ils|rnav|localizer|gps|runway [a-z ]+?) at)\s+(.+)", low)
        if not re.match(r"(the )?(runway|parking|ramp|transient|hold|taxi|tie|fbo|gate|hangar|depart|land\b|"
                        r"climb|descend|go|get|request|the option)", m.group(1))]
    if not tails:
        return None
    best = None
    for d, cand in db.nearby(a["lat"], a["lon"], radius_nm=250, limit=4000):
        for name in _airport_names(cand):
            for ti, tail in enumerate(tails):
                if name not in tail:  # cheap test first: thousands of airports are checked per transmission
                    continue
                hit = re.search(r"\b" + re.escape(name) + r"\b", tail)
                if hit:
                    key = (ti, hit.start(), -len(name), d)  # earliest, longest, nearest
                    if best is None or key < best[0]:
                        best = (key, cand)
    return best[1] if best else None


_ALT_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
              "niner": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
              "sixteen": 16, "seventeen": 17}


def parse_altitude(text: str) -> int | None:
    """'seven thousand five hundred', 'one zero thousand', '7,500', '7500 feet' -> feet."""
    t = text.lower().replace(",", "")
    m = re.search(r"\b(?:flight level|fl)\s*(\d{3})\b", t)
    if m:
        return int(m.group(1)) * 100
    m = re.search(r"\bflight level\s+(\w+)[\s-]+(\w+)[\s-]+(\w+)", t)
    if m and all(w in DIGITS for w in m.groups()):
        return int("".join(DIGITS[w] for w in m.groups())) * 100
    # a transponder code is not an altitude ("squawking 1200")
    m = re.search(r"(?<!squawk )(?<!squawking )\b(\d{1,2})(\d)00\b(?:\s*(?:feet|ft))?", t)
    if m and int(m.group(1) + m.group(2) + "00") >= 1000:
        return int(m.group(1) + m.group(2) + "00")
    m = re.search(r"\b((?:one zero|one one|one two|one three|one four|one five|one six|one seven)|"
                  r"(?:one|two|three|four|five|six|seven|eight|nine|niner|ten|eleven|twelve|thirteen|fourteen|"
                  r"fifteen|sixteen|seventeen))\s+thousand(?:\s+(one|two|three|four|five|six|seven|eight|nine|niner)"
                  r"\s+hundred)?", t)
    if not m:
        return None
    th = m.group(1)
    n = 10 + _ALT_WORDS[th.split()[1]] if th.startswith("one ") else _ALT_WORDS[th]
    return n * 1000 + (_ALT_WORDS[m.group(2)] * 100 if m.group(2) else 0)


def ifr_cruise(field_elev: float, course_mag: float) -> int:
    """Lowest sensible IFR cruise: odd thousands eastbound (000-179), even westbound."""
    alt = max(4000, int(math.ceil((field_elev + 3000) / 1000.0) * 1000))
    east = course_mag < 180
    while (alt // 1000) % 2 != (1 if east else 0):
        alt += 1000
    return alt


def altitude_words(ft: int) -> str:
    """'four thousand five hundred'; 18,000 ft and up as a flight level ('flight level three six zero')."""
    if ft >= 18000:
        return "flight level " + " ".join(NUMBERS[c] for c in f"{int(round(ft / 100.0)):03d}")
    return _alt_words(ft)


DIRECTIONS = {"north": 360, "northeast": 45, "east": 90, "southeast": 135, "south": 180,
              "southwest": 225, "west": 270, "northwest": 315}


def departure_turn(transcript: str, runway_hdg: float) -> str | None:
    """'...departing southbound' + runway heading 300 -> 'left turn southbound approved'."""
    m = re.search(r"\b(north|south)?(east|west)?\s*-?\s*bound\b|\bto the (north|south|east|west)\w*", transcript,
                  re.I)
    if not m:
        return None
    word = (m.group(3) or ((m.group(1) or "") + (m.group(2) or ""))).lower()
    if word not in DIRECTIONS:
        return None
    diff = (DIRECTIONS[word] - runway_hdg + 540) % 360 - 180  # -180..180, negative = left
    if abs(diff) <= 30:
        return "straight out departure approved"
    return f"{'right' if diff > 0 else 'left'} turn {word}bound approved"


def _listed_freqs(a: dict) -> list[dict]:
    """Frequencies to show the controller; CTAF is hidden at towered airports (towers are
    modeled as always open)."""
    towered = any(f["type"] == "TOWER" for f in a["freqs"])
    return [f for f in a["freqs"] if not (towered and f["type"] == "CTAF")]


def departure_freq(a: dict) -> tuple[str, int] | None:
    """(spoken facility name, kHz) VFR departures are handed to: departure, else approach."""
    for kind in ("DEPARTURE", "APPROACH"):
        f = next((f for f in a["freqs"] if f["type"] == kind), None)
        if f:
            return spoken_facility(Facility(a, kind, f["name"], f["khz"], 0)), f["khz"]
    return None


def _freq_label(airport: dict, f: dict) -> str:
    """'approach "NorCal Approach" 120.100'; tower/ground etc. just by type."""
    if f["type"] in ("APPROACH", "DEPARTURE"):
        name = spoken_facility(Facility(airport, f["type"], f["name"], f["khz"], 0))
        return f"{f['type'].lower()} \"{name}\" {fmt_freq(f['khz'])}"
    return f"{f['type'].lower()} {fmt_freq(f['khz'])}"


def runway_in_use(airport: dict, wind_dir_true: float | None, wind_kt: float | None) -> tuple[str, float] | None:
    """Pick the runway end with the most headwind; longest runway when calm.
    Returns (runway id, true heading)."""
    ends = []
    for r in airport.get("runways", []):
        ends.append((r["ids"][0], r["true_hdg"], r["length_ft"]))
        ends.append((r["ids"][1], (r["true_hdg"] + 180) % 360, r["length_ft"]))
    if not ends:
        return None
    if wind_dir_true is None or not wind_kt or wind_kt < 4:
        # Calm: longest runway, prefer the end facing the prevailing west wind
        rid, hdg, _ = max(ends, key=lambda e: (e[2], math.cos(math.radians(e[1] - 270))))
        return rid, hdg
    rid, hdg, _ = max(ends, key=lambda e: (round(wind_kt * math.cos(math.radians(wind_dir_true - e[1]))), e[2]))
    return rid, hdg


def runway_true_hdg(airport: dict, rid: str) -> float | None:
    for r in airport.get("runways", []):
        if rid == r["ids"][0]:
            return r["true_hdg"]
        if len(r["ids"]) > 1 and rid == r["ids"][1]:
            return (r["true_hdg"] + 180) % 360
    return None


def tailwind_kt(airport: dict, rid: str, wind_dir_true: float | None, wind_kt: float | None) -> float:
    hdg = runway_true_hdg(airport, rid)
    if hdg is None or wind_dir_true is None or not wind_kt:
        return 0.0
    return -wind_kt * math.cos(math.radians(wind_dir_true - hdg))


MAX_TAILWIND_KT = 5  # an approach to the other end of the runway in use is fine up to this much tailwind
_APPROACH_KINDS = {"ils": "ILS", "localizer": "LOC", "loc": "LOC", "rnav": "RNAV", "gps": "RNAV", "vor": "VOR"}


def parse_approach_request(text: str) -> tuple[str, str] | None:
    """'request the practice ILS runway three zero approach' -> ('ILS', '30'); 'RNAV3-0' -> ('RNAV', '30')."""
    t = re.sub(r"(?<=[a-z])(?=\d)", " ", text.lower())
    t = re.sub(r"(?<=\d)-(?=\d)", "", t)
    m = re.search(r"\b(ils|localizer|loc|rnav|gps|vor)\b", t)
    if not m:
        return None
    digits, side = "", ""
    for w in re.findall(r"[a-z]+|\d+", t[m.end():])[:10]:
        if w.isdigit() and not digits and len(w) <= 2:
            digits = w
        elif w in DIGITS and len(digits) < 2:
            digits += DIGITS[w]
        elif digits and w in ("left", "right", "center", "l", "r", "c"):
            side = w[0].upper()
            break
        elif digits:
            break
    if not digits or not 1 <= int(digits) <= 36:
        return None
    return _APPROACH_KINDS[m.group(1)], f"{int(digits)}{side}"


def approach_matches(kind: str, ap) -> bool:
    """Does a published approach satisfy a requested kind (ILS, LOC, RNAV, VOR)?"""
    k = ap.kind
    return {"ILS": k == "ILS", "LOC": k in ("LOC", "ILS"), "RNAV": k.startswith("RNAV") or k == "GPS",
            "VOR": k.startswith("VOR")}.get(kind, False)


def runway_matches(requested: str, rwy: str) -> bool:
    """'30' matches 30, 30L, 30R; '30L' only 30L; leading zeros ignored ('3' = '03')."""
    num = lambda r: re.match(r"0*(\d+)", r).group(1) if re.match(r"\d", r) else r  # noqa: E731
    side = lambda r: re.sub(r"^\d+", "", r)  # noqa: E731
    return num(rwy) == num(requested) and (not side(requested) or side(rwy) == side(requested))


def runway_end_position(airport: dict, lat: float, lon: float) -> str | None:
    """'on runway 28R' / 'at the approach end of runway 28R (holding short)' for an aircraft on the ground."""
    best = None
    for r in airport.get("runways", []):
        for i, rid in enumerate(r["ids"][:2]):
            tlat, tlon = (r["lat1"], r["lon1"]) if i == 0 else (r["lat2"], r["lon2"])
            d = dist_nm(lat, lon, tlat, tlon)
            if d < 0.15 and (best is None or d < best[0]):
                best = (d, rid)
    if not best:
        return None
    if on_runway(airport, lat, lon):
        return f"on runway {best[1]} (lined up or entering)"
    return f"at the approach end of runway {best[1]} (holding short of it)"


def clean_reply(text: str) -> str:
    """Keep plain ASCII speech: models occasionally emit stray foreign characters,
    quotes or markdown, which the voice would try to read."""
    text = text.replace("\u2019", "'").replace("\u2014", ", ").replace("\u2013", "-")
    # the no-reply token, however the model mangles it ("<s ilence>", "[silence]", "(silence)")
    text = re.sub(r"[<\[(]\s*s\s*i\s*l\s*e\s*n\s*c\s*e\s*[>\])]", SILENCE, text, flags=re.I)
    text = re.sub(r"[^A-Za-z0-9 ,.'?!<>/-]+", " ", text)
    text = re.sub(r"\s+([,.?!])", r"\1", re.sub(r"\s+", " ", text))
    return text.strip(" ,")


def parse_heading(text: str) -> int | None:
    """'turn right heading one zero zero' -> 100 (None for 'runway heading' or no heading)."""
    m = re.search(r"\bheading\s+(\w+)[\s,]+(\w+)[\s,]+(\w+)", text.lower())
    if not m or not all(w in DIGITS for w in m.groups()):
        return None
    h = int("".join(DIGITS[w] for w in m.groups()))
    return h if 1 <= h <= 360 else None


def instruction_altitude(text: str) -> int | None:
    """Altitude from 'climb/descend and maintain ...' or 'maintain ...' (not 'until established')."""
    low = text.lower()
    m = re.search(r"\b(?:climb and maintain|descend and maintain|maintain)\s+"
                  r"(flight level \w+[\s-]+\w+[\s-]+\w+|.+?thousand(?: \w+ hundred)?)", low)
    if m and "until established" not in low[m.start():m.end() + 25] and "vfr" not in m.group(0):
        return parse_altitude(m.group(1))
    return None


def words_to_code(text: str) -> str | None:
    """Find 'squawk four five one two' (or 4512) and return '4512'."""
    m = re.search(r"squawk\s+((?:(?:zero|one|two|three|four|five|six|seven|\d)[\s,-]*){4})", text, re.I)
    if not m:
        return None
    toks = re.findall(r"zero|one|two|three|four|five|six|seven|\d", m.group(1), re.I)
    return "".join(DIGITS.get(t.lower(), t) for t in toks)[:4]


class Controller:
    def __init__(self, db: AirportDB | None, history_turns: int = 12, xplane_path: str = "",
                 cifp_dirs: list[str] | None = None):
        self.db = db
        self.xplane_path = xplane_path
        self.cifp_dirs = cifp_dirs or []
        self.flight_rules = "VFR"  # becomes IFR when the pilot says so
        self.traffic_watch = TrafficWatch()
        self.pattern_work = False  # pilot asked for closed traffic / touch and goes
        self.pattern_kind: str | None = None  # touch and go / stop and go / option / low approach / None (full stop)
        self.landing_cleared = False  # landing / touch-and-go / option clearance given this lap
        self._lap: dict = {}  # per-lap flags (extend downwind, go-around, chased clearance)
        self.requested_alt: int | None = None
        self.ifr_plan: dict | None = None  # clearance we're about to issue
        self.ifr_clearance: dict | None = None  # clearance issued: {dest, initial, cruise, route, sid, fix}
        self.flight_plan: FlightPlan | None = None  # filed plan (SimBrief)
        self.requested_approach: tuple[str, str] | None = None  # (ILS/LOC/RNAV/VOR, runway) the pilot asked for
        self.arrival_rwy: tuple[str, str] | None = None  # (airport id, runway) when it isn't the wind runway
        self._last_pilot_fac: str | None = None  # facility key of the pilot's previous transmission
        self.intent_said = False  # the pilot said what they'll do at the end of the approach / this lap
        self.missed_reported = False  # the pilot reported a missed approach (practice), not yet answered for
        self._expect_prompted = False
        self.ifr_climb_given = False
        self.assigned_alt: int | None = None
        self.assigned_hdg: int | None = None  # last vector heading (magnetic)
        self.handoff_said: dict[str, str] = {}  # facility key -> the frequency change it issued
        self._alt_off_since: float | None = None
        self._alt_chased = False
        self.atis_told: dict[str, str] = {}  # airport id -> letter ATC told the pilot
        self.ifr_approach = False  # vectors for / cleared for an instrument approach
        self.awaiting_ack: tuple[float, str, str] | None = None  # (time, facility key, instruction)
        self.ack_chased = False
        self.approach_cleared = False
        self.history: dict[str, deque] = {}
        self.history_turns = history_turns
        self.assigned_code: str | None = None
        self.identified = False  # radar contact given for the assigned code
        self.takeoff_cleared: set[str] = set()  # facility keys that cleared us for takeoff
        self.handed_off: set[str] = set()  # facility keys that already issued a frequency change
        self.atis_reported: dict[str, str] = {}  # airport id -> ATIS letter the pilot reported
        self.destination: dict | None = None  # airport the pilot said they're going to
        self.landed_at: str | None = None  # airport id of the last landing (cleared when airborne)
        self._was_airborne = False
        self._rollout = False  # landed, still rolling out: a full stop ends the flight, a touch and go doesn't
        self._clock: float | None = None  # last value of _now(): sim time that never wraps or runs backwards
        self._zulu_prev: float | None = None
        self._sim_t = 0.0
        self._pending_code: str | None = None  # squawk offered to the model, not yet assigned
        self._clearance_prompted = False  # the "in position, clear the approach" call was made
        self._last_vector: tuple[float | None, int | None] = (None, None)  # (clock, heading) of the last vector
        self.handoff_at: dict[str, float] = {}  # facility key -> when its frequency change was said (real time)
        self.exit_issued = False
        self._rng = random.Random()

    def _next_code(self) -> str:
        while True:
            code = "".join(str(self._rng.randint(0, 7)) for _ in range(4))
            if code[0] in "0123456" and code not in ("1200", "7500", "7600", "7700", "7777", "0000"):
                return code

    def observe(self, transcript: str, fac: Facility | None) -> None:
        """Pick up flight rules, destination and requested altitude from a pilot transmission."""
        if not transcript:
            return
        if re.search(r"\b(ifr|i f r|instrument flight|on an instrument)\b", transcript, re.I):
            self.flight_rules = "IFR"
        elif re.search(r"\bvfr\b", transcript, re.I):
            self.flight_rules = "VFR"
        elif fac and self._filed_for(fac.airport) and re.search(
                r"\b(clearance|ready to copy|as filed)\b", transcript, re.I):
            self.flight_rules = "IFR"  # asking for the clearance of the plan filed from this airport
        if fac and self.db:
            dest = find_destination(self.db, transcript, fac)
            if dest:
                self.destination = dest
        if re.search(r"closed traffic|pattern work|touch and go|stop and go|the option|(remain|stay) in the pattern|"
                     r"pattern for landing practice", transcript, re.I):
            self.pattern_work = True
        elif re.search(r"\b(depart(ing)?|frequency change|leaving the pattern|out of the pattern|full stop)\b",
                       transcript, re.I):
            self.pattern_work = False
        req = parse_approach_request(transcript)
        if req:  # a new approach: its ending (full stop, low approach, option) isn't known yet
            self.requested_approach = req
            self.intent_said = self.missed_reported = False
            self.pattern_kind = None
        m_kind = re.search(r"touch and go|stop and go|the option|low approach|full stop", transcript, re.I)
        if m_kind:
            k = m_kind.group(0).lower().replace("the option", "option")
            self.pattern_kind = None if k == "full stop" else k
            self.intent_said = True
        if re.search(r"\bgo(ing)? around\b", transcript, re.I):
            self._lap["go_around_reported"] = True
        if re.search(r"\b(missed approach|going missed|on the missed|executing the missed)\b", transcript, re.I):
            self.missed_reported = self.intent_said = True
            self.pattern_kind = "low approach"
            self._lap["go_around_reported"] = True  # not a surprise go-around
            self.ifr_approach = self.approach_cleared = False  # that approach is over
            self.requested_approach = None
            # ... and the next one gets its own vectors, approach clearance and landing clearance
            self._clearance_prompted = self.landing_cleared = False
            self._last_vector = (None, None)
            self._lap = {"go_around_reported": True}
        alt = parse_altitude(transcript)
        if alt and re.search(r"\b(request|filed|cruise|cruising|at|altitude|to)\b", transcript, re.I):
            self.requested_alt = alt

    def set_flight_plan(self, plan: FlightPlan | None) -> None:
        self.flight_plan = plan

    def _filed_for(self, a: dict | None) -> FlightPlan | None:
        """The filed plan if it departs from this airport."""
        p = self.flight_plan
        return p if p and a and p.origin == a["id"] else None

    def _recontact(self, fac: Facility | None, transcript: str) -> None:
        """The pilot is back on a frequency that handed it off earlier (e.g. the tower after departure, now
        inbound): that facility starts over instead of treating the call as a goodbye."""
        if not transcript or not fac:
            return
        if fac.key in self.handed_off:
            said = self.handoff_at.get(fac.key)
            elsewhere = bool(self._last_pilot_fac) and self._last_pilot_fac != fac.key
            later = said is not None and _time.monotonic() - said > RECONTACT_S
            inbound = fac.kind == "TOWER" and bool(re.search(INBOUND_WORDS, transcript, re.I))
            if elsewhere or later or inbound:
                self._forget_handoff(fac.key)
                self.takeoff_cleared.discard(fac.key)
                self.history.pop(fac.key, None)
        self._last_pilot_fac = fac.key

    def _forget_handoff(self, key: str) -> None:
        self.handed_off.discard(key)
        self.handoff_said.pop(key, None)
        self.handoff_at.pop(key, None)

    def _practice_intent_unknown(self) -> bool:
        return (self.flight_rules == "VFR" and not self.intent_said and not self.landing_cleared
                and bool(self.ifr_approach or self.approach_cleared or self.requested_approach))

    def _runway(self, a: dict, s: AircraftState) -> tuple[str, float] | None:
        """Runway in use for this airport: the wind's, or the arrival runway an approach was approved to."""
        ar = self.arrival_rwy
        if ar and ar[0] == a["id"] and not s.on_ground:
            hdg = runway_true_hdg(a, ar[1])
            if hdg is not None:
                return ar[1], hdg
        return runway_in_use(a, s.wind_dir, s.wind_kt)

    def _request_note(self, procs, rwy: str, apt: dict, s: AircraftState) -> str | None:
        """What to tell a pilot whose requested approach isn't the one being set up."""
        req = self.requested_approach
        if not req or self.approach_cleared:
            return None
        kind, rq = req
        wind_rwy = (runway_in_use(apt, s.wind_dir, s.wind_kt) or ("",))[0]
        here = [ap for ap in procs.for_runway_all() if approach_matches(kind, ap) and runway_matches(rq, ap.runway)]
        if here and here[0].runway == rwy:
            if rwy != wind_rwy:
                return (f"The pilot's requested {here[0].spoken} is approved (the wind allows runway {rwy}); "
                        f"the tower will land it on runway {rwy}.")
            return None
        if here:
            return (f"The pilot asked for the {here[0].spoken}, but the wind is too strong for runway {rq}: say "
                    f"\"unable {here[0].spoken.replace(' approach', '')}, wind favors runway "
                    f"{spoken_runway(wind_rwy)}\" and offer the default approach listed above.")
        alts = sorted((ap for ap in procs.for_runway_all() if approach_matches(kind, ap)
                       and tailwind_kt(apt, ap.runway, s.wind_dir, s.wind_kt) <= MAX_TAILWIND_KT),
                      key=lambda ap: ap.runway[:2] != wind_rwy[:2])  # the runway in use first
        txt = (f"PILOT ASKED FOR the {kind} runway {rq}: there is NO {kind} approach to runway {rq} at {apt['id']}. "
               f"Don't switch approaches without saying so: tell the pilot (e.g. \"no {kind} runway "
               f"{spoken_runway(rq)}\")")
        if alts:
            return txt + (f", and offer the {alts[0].spoken} (the wind allows it) or the default approach listed "
                          f"above: \"the {alts[0].spoken.replace(' approach', '')} is available, say "
                          "intentions\". Until the pilot picks, don't clear any approach.")
        return txt + " and say \"expect <the default approach listed above>\"."

    def situation(self, s: AircraftState, fac: Facility | None, callsign: str, transcript: str = "") -> str:
        self._recontact(fac, transcript)
        self.observe(transcript, fac)
        plan = self._filed_for(fac.airport) if fac and fac.kind != "CENTER" and s.on_ground else None
        if plan and not self.ifr_clearance and (self.destination is None or self.flight_rules == "IFR"):
            dest = self.db.by_id(plan.destination) if self.db else None
            if dest:
                self.destination = dest  # the filed destination, before the pilot says it
        magvar = s.magvar
        region = "FAA" if not fac or fac.kind == "CENTER" else region_for(fac.airport["id"])
        full, abbrev = callsign_forms(callsign, region, s.icao_type)
        lines = [
            f"PHRASEOLOGY: {'ICAO' if region == 'ICAO' else 'FAA'}",
            f"Aircraft registration: {callsign or 'unknown'}  type: {s.icao_type or 'unknown'}",
            f"Spoken callsign: full \"{full}\", abbreviated \"{abbrev}\"" if full else "Callsign: unknown",
            "FIRST CONTACT on this frequency: use the FULL callsign (make + full registration is fine, e.g. "
            "\"Cirrus five five zero Charlie Romeo\")" if not self._hist(fac) else
            "Already in contact: abbreviated callsign is fine",
            f"{'On the ground' if s.on_ground else 'Airborne'}; altitude {s.alt_ft:.0f} ft MSL"
            f" ({s.agl_ft:.0f} AGL); heading {s.hdg_mag:.0f}; groundspeed {s.gs_kt:.0f} kt;"
            f" vertical speed {s.vs_fpm:+.0f} fpm",
        ]
        if s.wind_dir is None or s.wind_kt is None:
            lines.append("Wind: not available (leave wind out of the reply entirely)")
        else:
            if s.wind_kt < 3:
                lines.append("Wind: calm")
            else:
                wmag = round(((s.wind_dir - magvar) % 360) / 10) * 10 or 360
                lines.append(f"Wind: {wmag:03d} at {s.wind_kt} kt")
        alt = altimeter_phrase(s.altimeter_inhg, region)
        if alt:
            lines.append(alt)

        if fac:
            a = fac.airport
            brg = bearing(a["lat"], a["lon"], s.lat, s.lon)
            if fac.kind == "CENTER":
                near = self.db.nearby(s.lat, s.lon, radius_nm=60, limit=1) if self.db else []
                ref = (f"{near[0][0]:.0f} nm {_compass(bearing(near[0][1]['lat'], near[0][1]['lon'], s.lat, s.lon))} "
                       f"of {short_name(near[0][1])}") if near else "en route"
                lines += ["", f"YOU ARE: \"{fac.name}\" (Air Route Traffic Control Center, radio site "
                              f"{a.get('site', '')}) on {fmt_freq(fac.khz)}",
                          f"Aircraft is {ref} (use this for position reports)"]
            else:
                lines += [
                    "",
                    f"YOU ARE: \"{spoken_facility(fac)}\" ({fac.kind.lower()} for {a['name']}, {a['id']}) on "
                    f"{fmt_freq(fac.khz)}",
                    f"Aircraft is {fac.distance_nm:.1f} nm {_compass(brg)} of {a['id']} "
                    f"(field elevation {a['elev']} ft)",
                ]
            aclass = airspace_class(a) if region == "FAA" else None
            if aclass and fac.kind in ("TOWER", "GROUND", "DELIVERY"):
                lines.append(f"AIRSPACE: {a['id']} is Class {aclass}")
            if plan and fac.kind in ("GROUND", "DELIVERY", "TOWER"):
                from .flightplan import plan_age_text
                age = plan_age_text(plan)
                lines.append(f"FILED FLIGHT PLAN ({plan.source}{', ' + age if age else ''}): {plan.origin} to "
                             f"{plan.destination}, route {plan.route or 'direct'}"
                             + (f", cruise {plan.cruise_ft} ft" if plan.cruise_ft else "")
                             + (f", aircraft {plan.aircraft}" if plan.aircraft else "")
                             + ". Treat it as on file (IFR); don't ask the pilot for the route or destination.")
            if self.flight_rules == "IFR" and s.on_ground and fac.kind in ("GROUND", "DELIVERY", "TOWER"):
                lines.append(self._ifr_departure(a, fac, s))
            elif aclass in ("B", "C") and s.on_ground and fac.kind in ("GROUND", "DELIVERY", "TOWER"):
                lines.append(self._departure_flow(a, fac, aclass))
            if fac.kind in ("TOWER", "GROUND", "DELIVERY"):
                rw = self._runway(a, s)
                if rw:
                    rhdg = round((rw[1] - magvar) % 360) or 360
                    lines.append(f"RUNWAY IN USE: {rw[0]} (heading {rhdg:03d})")
                    if s.on_ground and fac.kind in ("TOWER", "GROUND"):
                        at = runway_end_position(a, s.lat, s.lon)
                        if at:
                            lines.append(f"Aircraft position: {at}")
                    if fac.kind == "TOWER" and region == "FAA":
                        w = spoken_wind(s.wind_dir, s.wind_kt, magvar)
                        r = spoken_runway(rw[0])
                        wc = (w + ", ") if w else ""
                        if s.on_ground:
                            closed = ", make left closed traffic" if self.pattern_work else ""
                            lines.append(f"Takeoff clearance wording: \"{wc}runway {r}, cleared for takeoff{closed}\"")
                        else:
                            lines.append(f"Landing clearance wording: \"{wc}runway {r}, {self._landing_phrase()}\" "
                                         "(when the aircraft is on downwind/base/final and no clearance was given yet)")
                            if fac.kind == "TOWER" and self.missed_reported:
                                dep = departure_freq(a)
                                dep_txt = f"contact {dep[0]} {fmt_freq(dep[1])}" if dep else "frequency change approved"
                                lines.append(
                                    "MISSED APPROACH reported (practice approach): don't clear it to land. Reply "
                                    "\"<callsign>, roger\" plus what comes next: another approach -> "
                                    f"\"{dep_txt}\"; the pattern -> \"make left traffic runway {r}\"; leaving "
                                    "-> \"frequency change approved\"; if they haven't said, \"say intentions\".")
                            elif fac.kind == "TOWER" and self._practice_intent_unknown():
                                lines.append(
                                    "PRACTICE APPROACH inbound, the pilot hasn't said how it ends: on the check-in say "
                                    "\"<callsign>, <tower name>, continue, say intentions after the approach\" (full "
                                    "stop, low approach or the option). Clear it once known: full stop -> cleared to "
                                    "land; low approach -> \"cleared low approach\"; option -> \"cleared for the "
                                    "option\".")
                    turn = departure_turn(transcript, rhdg)
                    if fac.kind == "TOWER" and s.on_ground and turn:
                        lines.append(f"Requested departure direction: right after the callsign, approve it with "
                                     f"\"{turn}\", then wind, runway, cleared for takeoff")
                    has_ground = any(f["type"] == "GROUND" for f in a["freqs"])
                    if s.on_ground and (fac.kind == "GROUND" or (fac.kind == "TOWER" and not has_ground)):
                        arriving = self.landed_at == a["id"] or (re.search(
                            r"\bpark|tie ?down|clear of (the )?runway|to the (ramp|fbo|hangars?)\b", transcript, re.I)
                            and not re.search(DEPARTURE_WORDS, transcript, re.I))
                        if arriving:
                            pk = route_to_parking(a, s.lat, s.lon, transcript)
                            if pk:
                                lines.append(pk[0].describe() + f" (parking: {pk[1].name or pk[1].kind})"
                                             " - the aircraft has landed / wants to park; say \"taxi to parking via ...\"")
                        else:
                            route = route_to_runway(a, s.lat, s.lon, rw[0])
                            if route:
                                lines.append(route.describe() + " (use this when the pilot wants to taxi for departure)")
            lines.append(self._airport_block(a))
            if (any(f["type"] == "ATIS" for f in a["freqs"]) and fac.kind in ("TOWER", "GROUND", "DELIVERY")
                    and self.landed_at != a["id"]):
                from .atis import info_letter

                rw = runway_in_use(a, s.wind_dir, s.wind_kt)
                cur = info_letter(a['id'], s.zulu_sec, rw[0] if rw else None)
                rep = self.atis_reported.get(a["id"])
                told = self.atis_told.get(a["id"])
                status = ("pilot has reported it" if rep == cur else
                          f"pilot reported an OLD letter ({rep})" if rep and told != cur else
                          "pilot has not reported it, but you already told them it is current - don't repeat it"
                          if told == cur else "pilot has NOT reported it")
                lines.append(f"Current ATIS information: {cur} ({status})")
            if fac.kind == "TOWER" and not s.on_ground:
                lines += self._pattern_facts(a, s)
            if s.traffic is None:
                lines.append("Other traffic: none known. Do not mention traffic or sequence numbers.")
            else:
                lines.append(traffic_facts(s.traffic, a if fac.kind in ("TOWER", "GROUND") else None, s.lat, s.lon))
            if fac.kind in RADAR_KINDS:
                dst_apt = self.destination or fac.airport
                sep = ("" if airspace_class(dst_apt) in ("B", "C")
                       else ", no separation services provided")
                lines.append(f"Flight rules: {self.flight_rules}" + (
                    " (an instrument approach request from a VFR pilot is a PRACTICE approach: add "
                    f"\"maintain VFR, practice approach approved{sep}\" only to the transmission that clears the "
                    "approach, never to vectors)"
                    if self.flight_rules == "VFR" else ""))
            if fac.kind in RADAR_KINDS and not s.on_ground:
                check = self._readback_check(transcript)
                if check:
                    lines.append(check)
                if self.assigned_alt:
                    diff = s.alt_ft - self.assigned_alt
                    converging = (diff > 0 and s.vs_fpm < -200) or (diff < 0 and s.vs_fpm > 200)
                    lines.append(f"Assigned altitude: {self.assigned_alt}; aircraft at {s.alt_ft:.0f}"
                                 + ((f", {'climbing' if diff < 0 else 'descending'} toward it (fine)" if converging
                                     else f" - MISMATCH ({diff:+.0f} ft): if the pilot reported a different "
                                     f"altitude, say \"verify assigned altitude {altitude_words(self.assigned_alt)}\"")
                                    if abs(diff) > 300 else ""))
                if self.assigned_hdg is not None:
                    lines.append(f"Assigned heading: {self.assigned_hdg:03d}; aircraft heading {s.hdg_mag:03.0f}"
                                 + (" (still turning toward it - fine)"
                                    if abs((self.assigned_hdg - s.hdg_mag + 540) % 360 - 180) > 10 else ""))
                if not self._hist(fac) and self.identified:
                    lines.append("CHECK-IN after a handoff: radar contact was already given by the previous controller. "
                                 "Reply with callsign, your facility name, the altimeter and any instruction still due; "
                                 "never say \"radar contact\" again.")
                if self.destination:
                    dst = self.destination
                    if any(f["type"] == "ATIS" for f in dst["freqs"]) and dist_nm(
                            s.lat, s.lon, dst["lat"], dst["lon"]) < 40:
                        from .atis import info_letter
                        drw = runway_in_use(dst, s.wind_dir, s.wind_kt)
                        lines.append(f"Destination {dst['id']} ATIS: information "
                                     f"{info_letter(dst['id'], s.zulu_sec, drw[0] if drw else None)} (when giving the "
                                     f"altimeter say \"{short_name(dst)} altimeter ...\")")
            if fac.kind in RADAR_KINDS and self.ifr_clearance and not self.ifr_climb_given \
                    and not s.on_ground:
                c = self.ifr_clearance
                lines.append(f"IFR DEPARTURE checking in: after the radar contact call, say \"{self._climb_phrase(c)}\" "
                             "(no other frequencies).")
            if fac.kind in RADAR_KINDS and self.destination:
                dst = self.destination
                dd = dist_nm(s.lat, s.lon, dst["lat"], dst["lon"])
                dtwr = next((f for f in dst["freqs"] if f["type"] == "TOWER"), None)
                lines.append(f"Pilot's destination: {dst['id']} {dst['name']}, {dd:.0f} nm "
                             f"{_compass(bearing(s.lat, s.lon, dst['lat'], dst['lon']))}; "
                             + (f"tower {fmt_freq(dtwr['khz'])}" if dtwr else "non-towered"))
            if fac.kind in RADAR_KINDS and self._handles_arrival(s, fac, transcript):
                ctx = self._approach_ctx(s, fac)
                if ctx:
                    procs, rwy, _ = ctx
                    block = describe_procedures(procs, rwy, s.lat, s.lon, s.magvar, s.hdg_mag,
                                                cleared=self.approach_cleared, cur_alt=s.alt_ft)
                    block = self._vector_already_issued(block)
                    if fac.key in self.handoff_said:
                        block = re.sub(r" Do NOT hand it to the tower yet;[^.]*\.", "", block)
                    if not self._radar_identified(s):
                        block = block.split("\nVectors for")[0] + (
                            "\nNOT RADAR IDENTIFIED YET: no vectors or approach clearance until radar contact. "
                            "For now assign the squawk (if none) and say \"expect <approach>\".")
                    lines.append(block)
                    note = self._request_note(procs, rwy, ctx[2], s)
                    if note:
                        lines.append(note)
                    others = sorted({a.name for a in procs.approaches.values() if a.runway != rwy})
                    if others:
                        lines.append(f"Other approaches at {procs.airport} (other runways): {', '.join(others)}")
            if fac.kind in RADAR_KINDS and self.db:
                for d, b in self.db.nearby(s.lat, s.lon, radius_nm=30, limit=2000):
                    if b["id"] in CLASS_B:
                        lines.append(f"Class B airspace nearby: {b['id']} Class Bravo, {d:.0f} nm "
                                     f"{_compass(bearing(s.lat, s.lon, b['lat'], b['lon']))}")
                        break
            if fac.kind in RADAR_KINDS:
                if self.assigned_code is None:
                    lines.append(f"Squawk: no code assigned yet. NEXT SQUAWK CODE: {self._next_code_cached()}")
                elif self.identified:
                    lines.append(f"Squawk: {self.assigned_code} assigned, radar contact already given")
                elif f"{s.xpdr_code:04d}" == self.assigned_code and "alt" in s.xpdr_mode:
                    lines.append(f"Squawk: you assigned {self.assigned_code}; transponder shows it with altitude"
                                 " -> radar identified")
                else:
                    lines.append(f"Squawk: you assigned {self.assigned_code}; transponder shows "
                                 f"{s.xpdr_code:04d} ({s.xpdr_mode}) -> NOT radar identified yet")
            if fac.key in self.handed_off and fac.key in self.handoff_said:
                lines.append(f"ALREADY HANDED OFF: you told this aircraft \"{self.handoff_said[fac.key]}\". Its readback "
                             f"or goodbye gets {SILENCE}; if it asks again, repeat the frequency. Never tell it to "
                             "remain with you.")
        if self.db:
            near = self.db.nearby(s.lat, s.lon, radius_nm=40, limit=6)
            others = [(d, a) for d, a in near if not fac or a["id"] != fac.airport["id"]]
            if others:
                lines += ["", "Other nearby airports:"]
                for d, a in others:
                    fr = "; ".join(_freq_label(a, f) for f in _listed_freqs(a) if f["type"] != "ATIS")
                    lines.append(f"- {a['id']} {a['name']}: {d:.0f} nm "
                                 f"{_compass(bearing(s.lat, s.lon, a['lat'], a['lon']))}; {fr or 'no frequencies'}")
        return "\n".join(lines)

    def _readback_check(self, transcript: str) -> str | None:
        """Compare the numbers the pilot said with the assigned heading/altitude."""
        if not transcript or (self.assigned_hdg is None and not self.assigned_alt):
            return None
        low = transcript.lower()
        said = []
        if self.assigned_hdg is not None:
            nums = {int(n) for n in re.findall(r"\b\d{3}\b", low)}
            words = re.findall(r"\b(" + "|".join(DIGITS) + r")[\s,-]+(" + "|".join(DIGITS) + r")[\s,-]+("
                               + "|".join(DIGITS) + r")\b", low)
            nums |= {int("".join(DIGITS[w] for w in g)) for g in words}
            if self.assigned_hdg in nums:
                said.append(f"heading {self.assigned_hdg:03d}")
        if self.assigned_alt:
            alt = parse_altitude(low)
            if alt == self.assigned_alt or str(self.assigned_alt) in low.replace(",", ""):
                said.append(f"altitude {self.assigned_alt}")
        if not said:
            return None
        return (f"READBACK CHECK: the pilot said the assigned {' and '.join(said)} - correct. Unless something else "
                f"in the transmission needs an answer, reply {SILENCE}.")

    def _vector_already_issued(self, block: str) -> str:
        """Replace VECTOR NOW with a note when the same heading/altitude was already assigned."""
        m = re.search(r'VECTOR NOW: "([^"]*)"', block)
        if not m or self.assigned_hdg is None:
            return block
        hdg = parse_heading(m.group(1))
        alt = instruction_altitude(m.group(1))
        if hdg is None or abs((hdg - self.assigned_hdg + 540) % 360 - 180) > 10 or (
                alt and self.assigned_alt and alt != self.assigned_alt):
            return block
        return block.replace(m.group(0), f"VECTOR ALREADY ISSUED (heading {self.assigned_hdg:03d}"
                             + (f", {self.assigned_alt}" if self.assigned_alt else "")
                             + f"): nothing new to say; a correct readback gets {SILENCE}.")

    def _ifr_departure(self, a: dict, fac: Facility, s: AircraftState) -> str:
        dep = departure_freq(a)
        dep_txt = f"{dep[0]} {fmt_freq(dep[1])}" if dep else "departure (no frequency listed)"
        delivery = next((f for f in a["freqs"] if f["type"] == "DELIVERY"), None)
        ground = next((f for f in a["freqs"] if f["type"] == "GROUND"), None)
        clr = self.ifr_clearance
        if clr:
            if fac.kind == "TOWER":
                if clr.get("sid"):
                    return (f"IFR DEPARTURE (clearance issued, flying the {spoken_procedure(clr['sid'])} departure): "
                            "a normal takeoff clearance (callsign, wind, runway, \"cleared for takeoff\"), no heading "
                            "- the departure procedure gives the route. Don't add a frequency; the handoff to "
                            "departure comes after takeoff.")
                return ("IFR DEPARTURE (clearance issued): the takeoff clearance must start with \"fly runway "
                        "heading\" (callsign, \"fly runway heading\", wind, runway, \"cleared for takeoff\"). Don't add "
                        "a frequency; the handoff to departure comes after takeoff.")
            return (f"IFR clearance already issued: cleared to {short_name(clr['dest'])} via {clr['route']}, "
                    f"maintain {clr['initial']}, expect {clr['cruise']} ten minutes after departure, departure "
                    f"{dep_txt}, squawk {self.assigned_code}. If the pilot just read it back correctly, say only "
                    "\"<callsign>, readback correct\"" + (f" (and \"contact ground {fmt_freq(ground['khz'])} when "
                                                           "ready to taxi\")" if fac.kind == "DELIVERY" and ground else "")
                    + ". Otherwise correct the wrong item.")
        if fac.kind == "GROUND" and delivery:
            return (f"IFR: the pilot needs an IFR clearance from clearance delivery: \"contact clearance delivery "
                    f"{fmt_freq(delivery['khz'])} for your IFR clearance\" (unless they already have it).")
        plan = self._filed_for(a)
        dest = (self.db.by_id(plan.destination) if plan and self.db else None) or self.destination
        if not dest:
            if plan:
                return (f"IFR: the filed destination {plan.destination} isn't in the airport data; ask \"say "
                        "destination\".")
            return "IFR: the pilot wants an IFR clearance but hasn't said the destination: ask \"say destination\"."
        crs = (bearing(a["lat"], a["lon"], dest["lat"], dest["lon"]) - s.magvar) % 360
        cruise = (plan.cruise_ft if plan else None) or self.requested_alt or ifr_cruise(a["elev"], crs)
        initial = min(cruise, int(math.ceil((a["elev"] + 2500) / 1000.0) * 1000))
        code = self._next_code_cached()
        route, sid, fix, note = self._clearance_route(a, s, plan)
        self.ifr_plan = {"dest": dest, "initial": initial, "cruise": cruise, "route": route, "sid": sid, "fix": fix}
        return (f"IFR CLEARANCE to issue now (CRAFT), exactly: \"<callsign>, cleared to {short_name(dest)} airport via "
                f"{route}. Maintain {altitude_words(initial)}, expect {altitude_words(cruise)} "
                f"one zero minutes after departure. Departure frequency {dep_txt}, squawk {code}.\" After a correct "
                "readback say \"readback correct\"." + (f" {note}" if note else ""))

    def _clearance_route(self, a: dict, s: AircraftState, plan: FlightPlan | None
                         ) -> tuple[str, str, str, str]:
        """(route words for the clearance, SID flown, first fix to go direct to, note for the controller)."""
        if not plan:
            return "radar vectors, then as filed", "", "", ""
        fix = plan.sid_exit or plan.first_fix
        vectors = f"radar vectors {fix}, then as filed" if fix else "radar vectors, then as filed"
        if not plan.sid:
            return vectors, "", fix, ""
        procs = load_procedures(self.xplane_path, a["id"], self.cifp_dirs)
        sid = procs.sids.get(plan.sid) if procs else None
        rw = runway_in_use(a, s.wind_dir, s.wind_kt)
        if procs and procs.sids and not sid:
            return vectors, "", fix, (f"(The filed {plan.sid} departure isn't in the nav data, so the clearance "
                                      "uses radar vectors instead; don't mention it unless the pilot asks.)")
        if sid and rw and not sid.serves(rw[0]):
            return vectors, "", fix, (f"(The filed {plan.sid} departure doesn't serve runway {rw[0]} in use, so the "
                                      "route is amended to radar vectors; if the pilot asks, say it's for the "
                                      "runway in use.)")
        words = f"the {spoken_procedure(plan.sid)} departure"
        if sid and plan.sid_exit in sid.transitions:
            words += f", {plan.sid_exit} transition"
        return words + ", then as filed", plan.sid, "", ""

    def _climb_phrase(self, c: dict) -> str:
        """What departure says to an IFR departure once identified: the climb, and the route if needed."""
        climb = f"climb and maintain {altitude_words(c['cruise'])}"
        if c.get("sid"):
            return climb  # already on the departure procedure
        return f"{climb}, cleared direct {c.get('fix') or short_name(c['dest'])}"

    def _departure_flow(self, a: dict, fac: Facility, aclass: str) -> str:
        dep = departure_freq(a)
        dep_txt = f"{dep[0]} {fmt_freq(dep[1])}" if dep else "departure (no frequency listed)"
        delivery = next((f for f in a["freqs"] if f["type"] == "DELIVERY"), None)
        if self.assigned_code:
            return (f"DEPARTURE FLOW (Class {aclass}): the VFR departure already has squawk {self.assigned_code} and "
                    f"departure frequency {dep_txt}; nothing more needed before taxi/takeoff.")
        top = int(round((a["elev"] + 2000) / 500.0) * 500)
        issue = (f"if the pilot hasn't said a destination or direction of flight, ask for it; otherwise issue now "
                 f"(no need to ask for an altitude): \"maintain VFR at or below {top}, departure frequency {dep_txt}, "
                 f"squawk {self._next_code_cached()}\"")
        if fac.kind == "GROUND" and delivery:
            return (f"DEPARTURE FLOW (Class {aclass}): VFR departures get a squawk from clearance delivery "
                    f"({fmt_freq(delivery['khz'])}) before taxi. If this pilot wants to depart and has no squawk, "
                    f"tell them to contact clearance delivery on that frequency.")
        if fac.kind == "TOWER":
            return (f"DEPARTURE FLOW (Class {aclass}): this departure never got a squawk on the ground; before "
                    f"takeoff clearance, {issue}.")
        return f"DEPARTURE FLOW (Class {aclass}): for a VFR departure, {issue}."

    def _next_code_cached(self) -> str:
        if not self._pending_code:
            self._pending_code = self._next_code()
        return self._pending_code

    @staticmethod
    def _airport_block(a: dict) -> str:
        rw = ", ".join(f"{r['ids'][0]}/{r['ids'][1]} ({r['length_ft']} ft)" for r in a["runways"])
        fr = "; ".join(_freq_label(a, f) for f in _listed_freqs(a))
        return f"Runways: {rw or 'none listed'}\nFrequencies at {a['id']}: {fr or 'none listed'}"

    def _hist(self, fac: Facility | None) -> deque:
        key = fac.key if fac else "none"
        return self.history.setdefault(key, deque(maxlen=self.history_turns * 2))

    def messages(self, fac: Facility | None, situation: str, transcript: str, event: bool = False) -> list[dict]:
        label = EVENT_LABEL if event else "PILOT SAID"
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self._hist(fac),
            {"role": "user", "content": f"SITUATION:\n{situation}\n\n{label}: {transcript}"},
        ]

    def remember(self, fac: Facility | None, transcript: str, reply: str, event: bool = False) -> None:
        hist = self._hist(fac)
        hist.append({"role": "user", "content": f"{EVENT_LABEL if event else 'PILOT SAID'}: {transcript}"})
        hist.append({"role": "assistant", "content": reply})
        if not event:
            self.awaiting_ack = None  # the pilot spoke: whatever we were waiting for got an answer
            self.observe(transcript, fac)
        if (self.missed_reported and fac and fac.kind == "TOWER" and reply and SILENCE not in reply
                and "intentions" not in reply.lower()):
            self.missed_reported = False  # answered: what's next is pattern / another approach / leaving
            self.pattern_kind = None
            self.intent_said = False
        low = reply.lower()
        said = re.sub(r"\bradar contact\b", " ", low)  # "radar contact" is neither an instruction nor a handoff
        # the chase for an unanswered instruction is made once: repeating the instruction doesn't restart it
        chase = event and transcript.startswith(ACK_CHASE)
        if fac and not chase and SILENCE not in reply and re.search(
                r"\b(heading|maintain|descend|climb|cleared|contact|squawk|hold short|cross runway|line up|taxi)\b",
                said):
            self.awaiting_ack = (_time.monotonic(), fac.key, reply)
            self.ack_chased = False
        if fac and not event:
            m = re.search(r"\b(?:information|with|have)\s+(" + "|".join(ALPHABET_WORDS) + r")\b", transcript, re.I)
            if m:
                self.atis_reported[fac.airport["id"]] = m.group(1).capitalize().replace("Xray", "X-ray")
        code = words_to_code(reply)
        if code and code != self.assigned_code:
            self.assigned_code = code
            self.identified = False
            self._pending_code = None
        if "radar contact" in low or re.search(r"\bidentified\b", low):
            self.identified = True
        m_atis = re.search(r"information (" + "|".join(ALPHABET_WORDS) + r") (is )?current", low)
        if fac and m_atis:
            self.atis_told[fac.airport["id"]] = m_atis.group(1).capitalize().replace("Xray", "X-ray")
        alt = instruction_altitude(reply)
        if alt:
            self.assigned_alt = alt
            self._alt_off_since = None
            self._alt_chased = False
        hdg = parse_heading(reply)
        if hdg is not None:
            self.assigned_hdg = hdg
            # an event won't repeat it right away
            self._last_vector = (self._clock if self._clock is not None else _time.time(), hdg)
        if re.search(r"resume own navigation|proceed direct|cleared direct|own navigation", low):
            self.assigned_hdg = None
        if "climb and maintain" in low:
            self.ifr_climb_given = True
        if self.ifr_plan and "cleared to" in low and "squawk" in low:
            self.ifr_clearance = self.ifr_plan
        if re.search(r"\bvectors\b.*\b(approach|final approach course)\b", low) or re.search(
                r"\bexpect (the )?(ils|rnav|localizer|gps|vor)\b", low):
            self.ifr_approach = True
        if re.search(r"\bcleared\b.*\bapproach\b", low) and "cleared to land" not in low:
            self.ifr_approach = True
            self.approach_cleared = True
        if re.search(r"cleared (to land|touch and go|for the option|stop and go|low approach)", low):
            self.landing_cleared = True
        if fac and "cleared for takeoff" in low:
            self.takeoff_cleared.add(fac.key)
            self._forget_handoff(fac.key)  # a new departure: an earlier flight's frequency change is history
        if fac and (re.search(r"\bcontact\b.*\b(departure|approach|center|radar|tower|ground)\b", said)
                    or "frequency change approved" in low) and SILENCE not in reply:
            self.handed_off.add(fac.key)
            self.handoff_said[fac.key] = re.sub(r"^[^,]*,\s*", "", reply.strip())
            self.handoff_at[fac.key] = _time.monotonic()

    def _now(self, s: AircraftState) -> float:
        """Seconds on a clock that follows sim time (so it stops with the sim) but, unlike zulu time,
        never wraps at 0000Z or runs backwards. Wall clock when the sim gives no time."""
        z = s.zulu_sec
        if z is None:
            self._clock = _time.time()
            return self._clock
        if self._zulu_prev is None:
            self._sim_t = z
        else:
            d = z - self._zulu_prev
            if d < -43200:  # passed 0000Z
                d += 86400
            self._sim_t += max(0.0, d)
        self._zulu_prev = z
        self._clock = self._sim_t
        return self._clock

    def shift_timers(self, dt: float) -> None:
        """The sim was paused for dt seconds: real-time timers don't count them."""
        if self.awaiting_ack:
            t, key, text = self.awaiting_ack
            self.awaiting_ack = (t + dt, key, text)
        if self._alt_off_since is not None:
            self._alt_off_since += dt
        self.handoff_at = {k: t + dt for k, t in self.handoff_at.items()}

    def _end_flight(self) -> None:
        """Full stop after a landing: the next departure starts clean (rules, clearance, squawk, destination)."""
        self.flight_rules = "VFR"
        self.ifr_plan = self.ifr_clearance = None
        self.ifr_climb_given = self._expect_prompted = False
        self.assigned_alt = self.requested_alt = None
        self._alt_off_since = None
        self._alt_chased = False
        self.assigned_code = self._pending_code = None
        self.identified = False
        self.destination = None
        self.takeoff_cleared.clear()

    def _track_phase(self, s: AircraftState) -> None:
        """Notice takeoffs and landings."""
        self._now(s)
        if s.on_ground and self._rollout and s.gs_kt < FULL_STOP_KT:
            self._rollout = False
            self._end_flight()
        if not s.on_ground:
            self._rollout = False
            if not self._was_airborne:
                self.landed_at = None
                self.exit_issued = False
                self.landing_cleared = False
                self._lap = {}
            self._was_airborne = True
        elif self._was_airborne:
            self._was_airborne = False
            if self.db:
                near = self.db.nearby(s.lat, s.lon, radius_nm=3, limit=1)
                if near and on_runway(near[0][1], s.lat, s.lon):
                    self.landed_at = near[0][1]["id"]
                    self.ifr_approach = self.approach_cleared = False
                    self.requested_approach = self.arrival_rwy = None
                    self.intent_said = self.missed_reported = False
                    self.assigned_hdg = None
                    self._clearance_prompted = False
                    self._last_vector = (None, None)
                    self._rollout = True

    def _handles_arrival(self, s: AircraftState, fac: Facility, transcript: str = "") -> bool:
        """Is this radar controller the one to set up the destination approach?"""
        dst = self.destination
        if not dst:
            return bool(re.search(r"\b(approach|ils|rnav|localizer|gps)\b", transcript, re.I)) or self.ifr_approach
        app = next((f for f in dst["freqs"] if f["type"] == "APPROACH"), None)
        if app and abs(app["khz"] - fac.khz) > 3:
            return False  # another approach control owns the destination; hand off first
        return self.ifr_approach or dist_nm(s.lat, s.lon, dst["lat"], dst["lon"]) < 35 or bool(
            re.search(r"\b(approach|ils|rnav|localizer|gps)\b", transcript, re.I))

    def _radar_identified(self, s: AircraftState) -> bool:
        return bool(self.assigned_code and (self.identified or (
            f"{s.xpdr_code:04d}" == self.assigned_code and "alt" in s.xpdr_mode)))

    def _approach_ctx(self, s: AircraftState, fac: Facility):
        """(procedures, runway in use, airport) for the destination (or the facility's own airport)."""
        apt = self.destination or fac.airport
        procs = load_procedures(self.xplane_path, apt["id"], self.cifp_dirs)
        if not procs or not procs.approaches:
            return None
        rw = self._runway(apt, s)
        if not rw:
            return None
        rwy = rw[0]
        req = self.requested_approach
        if req and not self.approach_cleared:  # the pilot's requested approach, if it exists and the wind allows
            ok = [ap for ap in procs.for_runway_all() if approach_matches(req[0], ap) and runway_matches(req[1], ap.runway)
                  and tailwind_kt(apt, ap.runway, s.wind_dir, s.wind_kt) <= MAX_TAILWIND_KT
                  and runway_true_hdg(apt, ap.runway) is not None]
            if ok:
                rwy = ok[0].runway
                wind = runway_in_use(apt, s.wind_dir, s.wind_kt)
                self.arrival_rwy = (apt["id"], rwy) if wind and wind[0] != rwy else None
        return procs, rwy, apt

    def _approach_event(self, s: AircraftState, fac: Facility) -> str | None:
        if fac.kind not in RADAR_KINDS or not self.ifr_approach or s.on_ground:
            return None
        if not self._handles_arrival(s, fac):
            return None
        ctx = self._approach_ctx(s, fac)
        if not ctx:
            return None
        procs, rwy, apt = ctx
        aps = procs.for_runway(rwy)
        thr = procs.thresholds.get(rwy)
        if not aps or not thr:
            return None
        if not self.approach_cleared and self._radar_identified(s):
            ready, hdg, v = approach_vector_plan(aps[0], thr, s.lat, s.lon, s.magvar, s.hdg_mag, s.alt_ft)
            if ready and not self._clearance_prompted:
                self._clearance_prompted = True
                return ("The aircraft is now in position to join final. Start with the callsign and issue the "
                        "approach clearance (the approach you told the pilot to expect) exactly as worded: " + v)
            off = abs((hdg - s.hdg_mag + 540) % 360 - 180)
            now = self._now(s)
            last_t, last_h = self._last_vector
            since = float("inf") if last_t is None or last_t > now else now - last_t
            # Re-vector when well off the needed heading; give the pilot time to comply with the last one
            if not ready and off > 25 and since > 45 and (
                    last_h is None or abs((hdg - last_h + 540) % 360 - 180) > 15 or since > 120):
                self._last_vector = (now, hdg)
                return ("Vector the aircraft toward the final approach course (it is flying heading "
                        f"{s.hdg_mag:03.0f}): " + v.split("VECTOR NOW: ")[-1] + " Start with the callsign, then "
                        "say the quoted instruction exactly.")
            return None
        if fac.key in self.handed_off or (self.awaiting_ack and self.awaiting_ack[1] == fac.key):
            return None
        ap = aps[0]
        along, off, faf_nm = approach_course_offset(ap, thr, s.lat, s.lon, s.magvar)
        d = dist_nm(s.lat, s.lon, thr.lat, thr.lon)
        if along < faf_nm + 2 and off < 1.0:
            self.handed_off.add(fac.key)
            twr = next((f for f in apt["freqs"] if f["type"] == "TOWER"), None)
            if twr:
                name = spoken_facility(Facility(apt, "TOWER", twr["name"], twr["khz"], d))
                return (f"The aircraft is cleared for the approach and near the final approach fix ({d:.0f} nm from "
                        f"the runway). Hand it to the tower: \"contact {name} {fmt_freq(twr['khz'])}\".")
            return ("The aircraft is cleared for the approach and near the final approach fix at a non-towered "
                    "airport: \"change to advisory frequency approved\" (and for IFR: \"report cancellation of IFR "
                    "on this frequency or with flight service\").")
        return None

    def _center_event(self, s: AircraftState, fac: Facility, dst: dict, d_dst: float, app: dict | None) -> str | None:
        """Handoffs into, between and out of Center airspace."""
        from .centers import nearest_low

        if fac.kind in ("APPROACH", "DEPARTURE"):
            # Leaving the approach control's area with a long way still to go: hand to Center
            d_home = dist_nm(s.lat, s.lon, fac.airport["lat"], fac.airport["lon"])
            if d_home > 35 and d_dst > 35 and not (app and abs(app["khz"] - fac.khz) <= 3):
                ctr = nearest_low(s.lat, s.lon)
                if ctr:
                    self.handed_off.add(fac.key)
                    return (f"The aircraft is leaving your airspace ({d_home:.0f} nm out). Hand it to Center: "
                            f"\"contact {ctr[0]['name']} {fmt_freq(ctr[1])}\".")
            return None
        if fac.kind != "CENTER":
            return None
        if app and abs(app["khz"] - fac.khz) <= 3:
            app = None  # the destination's "approach" is this Center frequency: keep the aircraft
        if d_dst < 35:
            self.handed_off.add(fac.key)
            if app:
                nf = self.db.facility_for(app["khz"], dst["lat"], dst["lon"]) if self.db else None
                name = spoken_facility(nf) if nf else spoken_facility(
                    Facility(dst, "APPROACH", app["name"], app["khz"], d_dst))
                return (f"The aircraft is {d_dst:.0f} nm from its destination. Hand it to approach: "
                        f"\"contact {name} {fmt_freq(app['khz'])}\".")
            if d_dst < 15:
                self.assigned_code = None
                self.identified = False
                if self.flight_rules == "IFR":
                    return ("The IFR aircraft is near its destination, which has no approach control. Say: "
                            "\"<callsign>, change to advisory frequency approved, report cancellation of IFR on this "
                            "frequency or with flight service\" (after issuing the approach clearance if they asked).")
                return ("The aircraft is near its destination. Say: \"<callsign>, radar service terminated, squawk "
                        "VFR, frequency change approved\".")
            self.handed_off.discard(fac.key)
            return None
        # Center to Center / sector to sector
        ctr = nearest_low(s.lat, s.lon)
        if ctr and abs(ctr[1] - fac.khz) > 3:
            here = fac.airport
            if ctr[0]["artcc"] != here["id"] or dist_nm(s.lat, s.lon, here["lat"], here["lon"]) > ctr[2] + 40:
                self.handed_off.add(fac.key)
                return (f"Hand the aircraft to the next sector: \"contact {ctr[0]['name']} {fmt_freq(ctr[1])}\".")
        return None

    def _enroute_event(self, s: AircraftState, fac: Facility) -> str | None:
        dst = self.destination
        if fac.kind not in RADAR_KINDS or not dst or s.on_ground or fac.key in self.handed_off:
            return None
        d_dst = dist_nm(s.lat, s.lon, dst["lat"], dst["lon"])
        app = next((f for f in dst["freqs"] if f["type"] == "APPROACH"), None)
        ev = self._center_event(s, fac, dst, d_dst, app)
        if ev:
            return ev
        # Sector handoff to the approach control serving the destination
        if fac.kind != "CENTER" and app and abs(app["khz"] - fac.khz) > 3 and d_dst > 12 and \
                d_dst < dist_nm(s.lat, s.lon, fac.airport["lat"], fac.airport["lon"]) - 5:
            self.handed_off.add(fac.key)
            nf = self.db.facility_for(app["khz"], dst["lat"], dst["lon"]) if self.db else None
            name = spoken_facility(nf) if nf else spoken_facility(
                Facility(dst, "APPROACH", app["name"], app["khz"], d_dst))
            return (f"The aircraft is now closer to its destination's approach control. Hand it off: "
                    f"\"contact {name} {fmt_freq(app['khz'])}\".")
        # IFR arrival: tell the pilot what approach to expect, with a descent
        if self.flight_rules == "IFR" and not self.ifr_approach and not self._expect_prompted and d_dst < 30 \
                and self._radar_identified(s) and self._handles_arrival(s, fac):
            ctx = self._approach_ctx(s, fac)
            if ctx:
                procs, rwy, apt = ctx
                aps = procs.for_runway(rwy)
                if aps:
                    self._expect_prompted = True
                    alt = max(3000, int(math.ceil(((aps[0].intercept_alt or 3000) + 1000) / 1000.0) * 1000))
                    alt_txt = (f", \"descend and maintain {altitude_words(alt)}\"" if s.alt_ft > alt + 300 else "")
                    return (f"The IFR arrival is {d_dst:.0f} nm from {apt['name']}. Say: callsign, \"expect "
                            f"{aps[0].spoken}\", the destination altimeter{alt_txt}.")
        return None

    def _landing_phrase(self) -> str:
        return {"touch and go": "cleared touch and go", "stop and go": "cleared stop and go",
                "option": "cleared for the option", "low approach": "cleared low approach"}.get(
            self.pattern_kind or "", "cleared to land")

    def _pattern_state(self, a: dict, s: AircraftState):
        rw = self._runway(a, s)
        if not rw:
            return None
        rwy = rw[0]
        own = pattern_position(a, rwy, s.lat, s.lon, s.agl_ft, (s.hdg_mag + s.magvar) % 360)
        others = []
        for t in s.traffic or []:
            if t.lat is None or t.alt_ft is None or t.alt_ft - a["elev"] < GROUND_TRAFFIC_AGL_FT:
                continue  # unknown position, or taxiing / parked
            p = pattern_position(a, rwy, t.lat, t.lon, t.alt_ft - a["elev"], t.hdg_true)
            if p:
                others.append((traffic_type(t), p))
        return rwy, own, others

    def _airborne_traffic(self, s: AircraftState, fac: Facility) -> list:
        """Targets worth a traffic call: not the ones taxiing or parked at an airport."""
        if fac.kind == "TOWER":
            fields = [fac.airport]
        else:
            fields = [a for _, a in self.db.nearby(s.lat, s.lon, radius_nm=12, limit=30)] if self.db else []
        out = []
        for t in s.traffic or []:
            if t.lat is not None and t.lon is not None and t.alt_ft is not None and any(
                    t.alt_ft - a["elev"] < GROUND_TRAFFIC_AGL_FT and dist_nm(t.lat, t.lon, a["lat"], a["lon"]) < 2.5
                    for a in fields):
                continue
            out.append(t)
        return out

    def _pattern_facts(self, a: dict, s: AircraftState) -> list[str]:
        st = self._pattern_state(a, s)
        if not st:
            return []
        rwy, own, others = st
        out = [f"Pattern altitude {int(round((a['elev'] + 1000) / 100.0) * 100)} MSL; standard left traffic"]
        if own:
            out.append(f"Aircraft position: {own.describe(rwy)}, {s.agl_ft:.0f} ft AGL")
        for name, p in others:
            out.append(f"Pattern traffic: {name} on {p.describe(rwy)}")
        seq = pattern_sequence(a, rwy, own, others)
        if own and len(seq) > 1 and "you" in seq:
            n = seq.index("you") + 1
            ahead = seq[n - 2] if n > 1 else None
            out.append(f"LANDING SEQUENCE: {', '.join(seq)} -> this aircraft is number {n}"
                       + (f", following the {ahead}" if ahead else "")
                       + (" (say \"number <n>, follow the <traffic>\")" if n > 1 else ""))
        return out

    def _tower_event(self, s: AircraftState, fac: Facility) -> str | None:
        if fac.kind != "TOWER" or s.on_ground:
            return None
        a = fac.airport
        st = self._pattern_state(a, s)
        if not st:
            return None
        rwy, own, others = st
        lap = self._lap
        w = spoken_wind(s.wind_dir, s.wind_kt, s.magvar)
        r = spoken_runway(rwy)
        # Go-around: was on short final, now climbing away without landing
        if own and own.leg == "final" and own.path_nm < 1.0 and s.agl_ft < 400:
            lap["short_final"] = True
        if lap.get("short_final") and s.agl_ft > 500 and s.vs_fpm > 300:
            expected = lap.get("go_around_reported") or self.pattern_kind in ("low approach", "option")
            # climbing away without landing: a new lap, with its own sequencing and landing clearance
            self._lap = {}
            self.landing_cleared = self._clearance_prompted = False
            if expected:
                return None  # reported, or a low approach / option it was cleared for
            return (f"The aircraft went around without saying so. Acknowledge and give pattern instructions: "
                    f"\"<callsign>, roger, make left traffic runway {r}, report downwind\".")
        if not own:
            return None
        seq = pattern_sequence(a, rwy, own, others)
        n = seq.index("you") + 1 if "you" in seq else 1
        ahead = seq[n - 2] if n > 1 else None
        # Extend downwind when the traffic ahead is still well out
        if own.leg == "downwind" and own.along < 0.3 and ahead and not lap.get("extended"):
            ahead_pos = next((p for name, p in others if ahead.startswith(name) and p.leg in ("base", "final")), None)
            if ahead_pos and ahead_pos.path_nm > 1.2:
                lap["extended"] = True
                return (f"Traffic ahead ({ahead}) is still {ahead_pos.path_nm:.1f} nm from the runway. Say: "
                        f"\"<callsign>, extend downwind, number {n}, I'll call your base\".")
        if lap.get("extended") and not lap.get("base_called") and own.leg == "downwind":
            ahead_pos = next((p for name, p in others if ahead and ahead.startswith(name)), None)
            if not ahead_pos or ahead_pos.path_nm < 0.8:
                lap["base_called"] = True
                follow = f", follow the {ahead}" if ahead else ""
                return f"Call the aircraft's base turn: \"<callsign>, turn base now{follow}\"."
        # Landing clearance when on base/final without one
        if own.leg in ("base", "final") and own.path_nm < 3 and not self.landing_cleared and not lap.get("clr_chased"):
            lap["clr_chased"] = True
            # a practice approach whose ending the pilot never said: the option covers all of them
            land = "cleared for the option" if self._practice_intent_unknown() else self._landing_phrase()
            num = f"number {n}, follow the {ahead}, " if ahead else ""
            return (f"The aircraft is on {own.describe(rwy)} without a landing clearance. Say: \"<callsign>, {num}"
                    f"{w + ', ' if w else ''}runway {r}, {land}\".")
        return None

    def _arrival_event(self, s: AircraftState, fac: Facility) -> str | None:
        ev = self._tower_event(s, fac)
        if ev:
            return ev
        ev = self._enroute_event(s, fac)
        if ev:
            return ev
        ev = self._approach_event(s, fac)
        if ev or self.ifr_approach and fac.kind in RADAR_KINDS:
            return ev  # instrument approaches use the FAF handoff, not the 12 nm VFR one
        # Approach/departure: hand off to the destination tower, or terminate radar service
        dst = self.destination
        if fac.kind in RADAR_KINDS and dst and fac.key not in self.handed_off and not s.on_ground:
            d = dist_nm(s.lat, s.lon, dst["lat"], dst["lon"])
            twr = next((f for f in dst["freqs"] if f["type"] == "TOWER"), None)
            if twr and d < 12:
                self.handed_off.add(fac.key)
                name = spoken_facility(Facility(dst, "TOWER", twr["name"], twr["khz"], d))
                return (f"The aircraft is {d:.0f} nm from its destination {dst['name']}. Hand it to the tower: "
                        f"\"contact {name} {fmt_freq(twr['khz'])}\".")
            if not twr and d < 10:
                self.handed_off.add(fac.key)
                self.assigned_code = None
                self.identified = False
                return (f"The aircraft is {d:.0f} nm from its non-towered destination {dst['name']}. Terminate "
                        "radar service: \"radar service terminated, squawk VFR, frequency change approved\".")
        # Tower after landing: exit and ground frequency
        a = fac.airport
        if (fac.kind == "TOWER" and s.on_ground and self.landed_at == a["id"] and not self.exit_issued
                and 3 < s.gs_kt < 45 and on_runway(a, s.lat, s.lon)):
            self.exit_issued = True
            ex = runway_exit(a, s.lat, s.lon, (s.hdg_mag + s.magvar) % 360)
            gnd = next((f for f in a["freqs"] if f["type"] == "GROUND"), None)
            turn = (f"turn {ex[1]} at {spoken_taxiway(ex[0])}" if ex else "turn off when able")
            after = (f"contact ground {fmt_freq(gnd['khz'])}" if gnd else "taxi to parking")
            return (f"The aircraft just landed and is slowing on the runway. Tell it: \"{turn}, {after}\" "
                    "(you may add \"remain this frequency\" instead if there is no ground controller).")
        return None

    def pending_event(self, s: AircraftState, fac: Facility | None) -> str | None:
        """A call the controller should make without being called first, or None.
        Each event fires once."""
        self._track_phase(s)
        if not fac or not self._hist(fac):
            return None  # only facilities we're already talking to
        if self.awaiting_ack and self.awaiting_ack[1] == fac.key and not self.ack_chased \
                and _time.monotonic() - self.awaiting_ack[0] > ACK_TIMEOUT_S:
            self.ack_chased = True
            return (f"{ACK_CHASE} (\"{self.awaiting_ack[2]}\"). Call them: "
                    "the callsign and \"how do you hear?\", or repeat the instruction once.")
        if fac.kind in RADAR_KINDS and self.assigned_alt and not s.on_ground and not self._alt_chased:
            diff = s.alt_ft - self.assigned_alt
            converging = (diff > 0 and s.vs_fpm < -300) or (diff < 0 and s.vs_fpm > 300)
            if abs(diff) > 300 and not converging:
                if self._alt_off_since is None:
                    self._alt_off_since = _time.monotonic()
                elif _time.monotonic() - self._alt_off_since > ALT_CONFORMANCE_S:
                    self._alt_chased = True
                    return (f"The aircraft is level at {s.alt_ft:.0f} ft but assigned {self.assigned_alt}. Say: "
                            f"callsign, \"verify {'climbing' if diff < 0 else 'descending'} to "
                            f"{altitude_words(self.assigned_alt)}\".")
            else:
                self._alt_off_since = None
        if (fac.kind in RADAR_KINDS and self.assigned_code and not self.identified
                and f"{s.xpdr_code:04d}" == self.assigned_code and "alt" in s.xpdr_mode):
            self.identified = True
            if self.ifr_clearance and not s.on_ground and not self.ifr_climb_given:
                c = self.ifr_clearance
                return ("The IFR departure's transponder now shows its code. Say: callsign, \"radar contact\" with "
                        f"position, then \"{self._climb_phrase(c)}\".")
            return ("The aircraft's transponder now shows the code you assigned. Call radar contact with the "
                    "aircraft's position relative to the nearest airport, and give the altimeter.")
        ev = self._arrival_event(s, fac)
        if ev:
            return ev
        # IFR departure that was identified but never got its climb / direct (e.g. the check-in went sideways)
        if fac.kind in RADAR_KINDS and self.ifr_clearance and not self.ifr_climb_given and self.identified \
                and not s.on_ground and not self._lap.get("climb_prompted") and not self.awaiting_ack:
            self._lap["climb_prompted"] = True
            c = self.ifr_clearance
            return f"Give the IFR departure its climb and route: \"<callsign>, {self._climb_phrase(c)}\"."
        # Traffic advisories: radar controllers for identified aircraft, tower near its airport
        if not s.on_ground and s.traffic and (
                (fac.kind in RADAR_KINDS and self._radar_identified(s)) or
                (fac.kind == "TOWER" and dist_nm(s.lat, s.lon, fac.airport["lat"], fac.airport["lon"]) < 6)):
            t = self.traffic_watch.due(self._airborne_traffic(s, fac))
            if t:
                return f"Issue a traffic advisory, exactly: \"<callsign>, {traffic_advisory(t)}\"."
        if fac.kind == "TOWER" and fac.key in self.takeoff_cleared and fac.key not in self.handed_off \
                and not s.on_ground and not self.pattern_work:
            a = fac.airport
            d = dist_nm(a["lat"], a["lon"], s.lat, s.lon)
            bc = airspace_class(a) in ("B", "C") or self.flight_rules == "IFR"
            if (d > 1.5 or s.alt_ft - a["elev"] > 800) if bc else (d > 3 or s.alt_ft - a["elev"] > 1500):
                self.handed_off.add(fac.key)
                dep = departure_freq(a)
                if (airspace_class(a) in ("B", "C") or self.flight_rules == "IFR") and dep:
                    return (f"The aircraft you cleared for takeoff is now {d:.1f} nm from the airport at "
                            f"{s.alt_ft:.0f} ft. Hand it to departure: \"contact {dep[0]} "
                            f"{fmt_freq(dep[1])}\".")
                return (f"The aircraft you cleared for takeoff is now {d:.1f} nm from the airport at "
                        f"{s.alt_ft:.0f} ft. Issue the frequency change: if the pilot asked for flight following "
                        "or is IFR, 'contact <departure or approach facility> <frequency>' using a departure or "
                        "approach frequency from the situation; otherwise 'frequency change approved'.")
        return None
