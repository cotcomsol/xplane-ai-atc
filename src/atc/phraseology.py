"""Region rules and spoken forms: FAA vs ICAO phraseology, callsigns, altimeter."""

from __future__ import annotations

import math
import re

ALPHABET = {
    "A": "Alpha", "B": "Bravo", "C": "Charlie", "D": "Delta", "E": "Echo", "F": "Foxtrot", "G": "Golf",
    "H": "Hotel", "I": "India", "J": "Juliett", "K": "Kilo", "L": "Lima", "M": "Mike", "N": "November",
    "O": "Oscar", "P": "Papa", "Q": "Quebec", "R": "Romeo", "S": "Sierra", "T": "Tango", "U": "Uniform",
    "V": "Victor", "W": "Whiskey", "X": "X-ray", "Y": "Yankee", "Z": "Zulu",
}
NUMBERS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six",
           "7": "seven", "8": "eight", "9": "niner"}

# ICAO aircraft type designator prefix -> name used on the radio (FAA 7110.65 2-4-20)
TYPE_NAMES = [
    ("SR2", "Cirrus"), ("S22T", "Cirrus"), ("SF50", "Cirrus"), ("C1", "Cessna"), ("C2", "Cessna"),
    ("C3", "Cessna"), ("C4", "Cessna"), ("C5", "Citation"), ("C6", "Citation"), ("C7", "Citation"),
    ("P28", "Piper"), ("PA", "Piper"), ("P32", "Piper"), ("P46", "Piper"), ("M20", "Mooney"),
    ("BE35", "Bonanza"), ("BE33", "Bonanza"), ("BE36", "Bonanza"), ("BE5", "Baron"), ("BE9", "King Air"),
    ("B350", "King Air"), ("DA4", "Diamond"), ("DA6", "Diamond"), ("DA2", "Diamond"), ("TBM", "TBM"),
    ("PC12", "Pilatus"), ("PC24", "Pilatus"), ("RV", "Experimental"), ("A5", "Icon"), ("GLAS", "Experimental"),
    ("EC", "Helicopter"), ("R22", "Robinson"), ("R44", "Robinson"), ("R66", "Robinson"),
]


def type_name(icao_type: str) -> str | None:
    t = (icao_type or "").upper()
    for prefix, name in TYPE_NAMES:
        if t.startswith(prefix):
            return name
    return None


# ICAO prefixes of airports that use FAA phraseology (US states and territories).
FAA_PREFIXES = ("K", "PA", "PH", "PG", "PF", "PO", "PP", "PJ", "PM", "PW", "TJ", "TI")


def region_for(airport_id: str) -> str:
    """'FAA', 'CANADA' or 'ICAO'. US airports without an ICAO code (e.g. '0Q5') count as FAA."""
    a = airport_id.upper()
    if a.startswith(FAA_PREFIXES) or (len(a) <= 4 and any(c.isdigit() for c in a)):
        return "FAA"
    if a.startswith("C") and len(a) == 4:
        return "CANADA"
    return "ICAO"


def spell(text: str) -> str:
    """'N550CR' -> 'November five five zero Charlie Romeo'."""
    out = []
    for ch in re.sub(r"[^A-Za-z0-9]", "", text).upper():
        out.append(ALPHABET.get(ch) or NUMBERS.get(ch, ch))
    return " ".join(out)


def callsign_forms(registration: str, region: str, icao_type: str = "") -> tuple[str, str]:
    """Return (full spoken callsign, abbreviated spoken callsign) for a registration."""
    reg = re.sub(r"[^A-Za-z0-9]", "", registration).upper()
    if not reg:
        return "", ""
    full = spell(reg)
    if region == "ICAO" or not reg.startswith("N"):
        # ICAO Annex 10: first character plus at least the last two
        abbrev = spell(reg[0] + reg[-2:]) if len(reg) > 3 else full
        return full, abbrev
    # FAA: make + number without the N ("Cirrus five five zero Charlie Romeo"), abbreviated to
    # make + last three ("Cirrus zero Charlie Romeo")
    make = type_name(icao_type)
    if make:
        return f"{make} {spell(reg[1:])}", f"{make} {spell(reg[-3:])}"
    return full, "<type> " + spell(reg[-3:])


def altimeter_phrase(inhg: float | None, region: str) -> str | None:
    if not inhg:
        return None
    if region == "ICAO":
        return f"QNH {math.floor(inhg * 33.8639 + 0.01)} hPa"  # QNH is rounded down
    return f"Altimeter {inhg:.2f} inHg"


def digits(n) -> str:
    return " ".join(NUMBERS.get(c, c) for c in str(n))


def altitude_words(ft: float) -> str:
    """4500 -> 'four thousand five hundred', 12000 -> 'one two thousand' (nearest hundred feet)."""
    th, hu = divmod(int(round(ft / 100.0) * 100), 1000)
    parts = []
    if th:
        parts.append(f"{digits(th)} thousand")
    if hu:
        parts.append(f"{digits(hu // 100)} hundred")
    return " ".join(parts) or "zero"


def spoken_runway(rwy: str) -> str:
    """'30L' -> 'three zero left'."""
    return " ".join(NUMBERS.get(c, {"L": "left", "R": "right", "C": "center"}.get(c, c)) for c in rwy)


def spoken_wind(wind_dir_true, wind_kt, magvar: float = 0.0) -> str:
    """'wind three one zero at niner' (magnetic, nearest ten degrees), 'wind calm', or ''."""
    if wind_dir_true is None or wind_kt is None:
        return ""
    if wind_kt < 3:
        return "wind calm"
    wmag = round(((wind_dir_true - magvar) % 360) / 10) * 10 or 360
    return f"wind {digits(f'{wmag:03d}')} at {digits(int(wind_kt))}"
