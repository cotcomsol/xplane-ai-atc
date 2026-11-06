"""ATIS broadcasts built in code from sim weather (no language model needed)."""

from __future__ import annotations

import time
import zlib

from .airports import Facility
from .controller import runway_in_use
from .phraseology import ALPHABET, NUMBERS, region_for

LETTERS = list(ALPHABET.values())


def digits(n: int | str) -> str:
    """1013 -> 'one zero one three' (aviation style)."""
    return " ".join(NUMBERS.get(c, c) for c in str(n))


def altitude_words(ft: int) -> str:
    """4500 -> 'four thousand five hundred', 12000 -> 'one two thousand'."""
    ft = int(round(ft / 100.0) * 100)
    th, hu = divmod(ft, 1000)
    parts = []
    if th:
        parts.append(f"{digits(th)} thousand")
    if hu:
        parts.append(f"{digits(hu // 100)} hundred")
    return " ".join(parts) or "zero"


def zulu_hour_min(zulu_sec: float | None) -> tuple[int, int]:
    if zulu_sec is None:
        t = time.gmtime()
        return t.tm_hour, t.tm_min
    zs = int(zulu_sec) % 86400
    return zs // 3600, (zs % 3600) // 60


def info_letter(airport_id: str, zulu_sec: float | None, runway: str | None) -> str:
    """Changes every hour, and when the runway in use changes."""
    hour, _ = zulu_hour_min(zulu_sec)
    return LETTERS[(hour + zlib.crc32(f"{airport_id}{runway}".encode())) % 26]


def _sky(s, field_elev: int) -> str:
    if s.clouds is None:
        return ""
    layers = []
    for base_msl, cover in s.clouds:
        agl = base_msl - field_elev
        if agl < 0 or agl > 25000:
            continue
        kind = "few clouds" if cover < 0.25 else "scattered" if cover < 0.5 else "broken" if cover < 0.875 else "overcast"
        if kind in ("broken", "overcast") and not any("ceiling" in x for x in layers):
            kind = f"ceiling {altitude_words(agl)} {kind}"
            layers.append(kind)
        else:
            layers.append(f"{kind} at {altitude_words(agl)}")
    return ", ".join(layers) if layers else "sky clear"


def field_temp_dew(s, elev_ft: float) -> tuple[float | None, float | None]:
    """Temperature and dew point at field elevation (standard lapse rates from sea level).
    Falls back to the temperature at the aircraft when it is on the ground."""
    temp = dew = None
    if s.temp_sl_c is not None:
        temp = s.temp_sl_c - 1.98 * elev_ft / 1000
    elif s.temp_c is not None and s.on_ground:
        temp = s.temp_c
    if s.dew_sl_c is not None:
        dew = s.dew_sl_c - 0.55 * elev_ft / 1000
    return temp, dew


def _temp(c: float) -> str:
    n = int(round(c))
    return ("minus " if n < 0 else "") + digits(abs(n))


def build(fac: Facility, s) -> tuple[str, str]:
    """Return (ATIS text to speak, information letter)."""
    a = fac.airport
    rw = runway_in_use(a, s.wind_dir, s.wind_kt)
    rwy = rw[0] if rw else None
    letter = info_letter(a["id"], s.zulu_sec, rwy)
    hh, mm = zulu_hour_min(s.zulu_sec)
    if mm < 53:  # ATIS observations are issued at 53 past the hour
        hh = (hh - 1) % 24
    icao = region_for(a["id"]) == "ICAO"
    name = a.get("city") or a["name"]

    out = [f"{name} information {letter}", f"{digits(f'{hh:02d}53')} zulu"]
    if s.wind_dir is not None and s.wind_kt is not None:
        if s.wind_kt < 3:
            out.append("wind calm")
        else:
            wmag = round(((s.wind_dir - s.magvar) % 360) / 10) * 10 or 360
            out.append(f"wind {digits(f'{wmag:03d}')} at {digits(s.wind_kt)}")
    if s.vis_sm is not None:
        if icao:
            out.append("visibility one zero kilometers or more" if s.vis_sm >= 6.2
                       else f"visibility {digits(int(s.vis_sm * 1609 // 100 * 100))} meters")
        else:
            v = min(10, int(s.vis_sm)) if s.vis_sm >= 1 else None
            out.append(f"visibility {digits(v)}" if v else "visibility less than one")
    sky = _sky(s, a["elev"])
    if sky:
        out.append(sky)
    temp, dew = field_temp_dew(s, a["elev"])
    if temp is not None:
        t = f"temperature {_temp(temp)}"
        if dew is not None:
            t += f", dew point {_temp(min(dew, temp))}"
        out.append(t)
    if s.altimeter_inhg:
        if icao:
            out.append(f"QNH {digits(int(s.altimeter_inhg * 33.8639 + 0.01))}")
        else:
            out.append(f"altimeter {digits(f'{s.altimeter_inhg:.2f}'.replace('.', ''))}")
    if rwy:
        rspoken = " ".join(NUMBERS.get(c, {"L": "left", "R": "right", "C": "center"}.get(c, c)) for c in rwy)
        out.append(f"{'visual approaches in use, ' if not icao else ''}landing and departing runway {rspoken}")
    out.append(f"advise on initial contact you have information {letter}")
    return ". ".join(x[0].upper() + x[1:] for x in out) + ".", letter
