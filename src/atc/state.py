"""Aircraft state derived from raw X-Plane dataref values."""

from __future__ import annotations

import math
from dataclasses import dataclass

M_TO_FT = 3.28084
MS_TO_KT = 1.943844


def _f(v, default=0.0) -> float:
    if isinstance(v, list):
        v = v[0] if v else default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# sim/cockpit2/radios/actuators/transponder_mode; 6 and 7 are TCAS-equipped aircraft, which also report altitude
XPDR_MODES = {0: "off", 1: "standby", 2: "on (mode A)", 3: "alt (mode C)", 4: "test", 5: "ground",
              6: "TA only (mode S, alt)", 7: "TA/RA (mode S, alt)"}


@dataclass
class AircraftState:
    lat: float
    lon: float
    alt_ft: float  # indicated (baro) altitude if available, else MSL
    agl_ft: float
    gs_kt: float
    ias_kt: float
    hdg_mag: float
    vs_fpm: float
    on_ground: bool
    com1_khz: int
    com2_khz: int
    tx_com: int  # 1 or 2
    xpdr_code: int
    xpdr_mode: str
    tailnum: str
    icao_type: str
    altimeter_inhg: float | None
    wind_dir: float | None
    wind_kt: float | None
    magvar: float = 0.0  # true - magnetic, degrees (east variation is positive)
    # Weather (None when X-Plane doesn't provide it)
    vis_sm: float | None = None
    temp_c: float | None = None  # at the aircraft
    temp_sl_c: float | None = None  # sea level (ATIS adjusts it to field elevation)
    dew_sl_c: float | None = None
    clouds: list[tuple[float, float]] | None = None  # (base ft MSL, coverage 0-1) per layer
    zulu_sec: float | None = None
    traffic: list = None  # list[Target]
    rx_com1: bool = True  # audio panel receive switches
    rx_com2: bool = True
    paused: bool = False

    @property
    def tx_khz(self) -> int:
        return self.com2_khz if self.tx_com == 2 else self.com1_khz

    @classmethod
    def from_values(cls, v: dict) -> "AircraftState":
        msl_ft = _f(v.get("elev_m")) * M_TO_FT
        baro = v.get("baro_alt_ft")
        # audio_com_selection(_man): 6 = COM1, 7 = COM2
        sel = int(_f(v.get("mic_sel", v.get("mic_sel_auto", 6)), 6))
        sl = v.get("sl_press_pa")
        wind_dir = v.get("wind_dir")
        wind = v.get("wind_kt")
        on_ground = bool(_f(v.get("on_ground")))
        if not on_ground and v.get("wind_sfc_dir") is not None and v.get("wind_sfc_ms") is not None:
            # Airborne: report the surface layer, not the wind at our altitude
            wind_dir, wind = v["wind_sfc_dir"], v["wind_sfc_ms"]
        return cls(
            lat=_f(v.get("lat")),
            lon=_f(v.get("lon")),
            alt_ft=_f(baro) if baro is not None else msl_ft,
            agl_ft=_f(v.get("agl_m")) * M_TO_FT,
            gs_kt=_f(v.get("gs_ms")) * MS_TO_KT,
            ias_kt=_f(v.get("ias_kt")),
            hdg_mag=_f(v.get("hdg_mag")) % 360,
            vs_fpm=_f(v.get("vs_fpm")),
            on_ground=on_ground,
            com1_khz=int(_f(v.get("com1_khz"))),
            com2_khz=int(_f(v.get("com2_khz"))),
            tx_com=2 if sel == 7 else 1,
            xpdr_code=int(_f(v.get("xpdr_code"))),
            xpdr_mode=XPDR_MODES.get(int(_f(v.get("xpdr_mode"))), "unknown"),
            tailnum=str(v.get("tailnum") or ""),
            icao_type=str(v.get("icao_type") or ""),
            altimeter_inhg=round(_f(sl) / 3386.389, 2) if sl else None,
            wind_dir=round(_f(wind_dir)) % 360 if wind_dir is not None else None,
            wind_kt=round(_f(wind) * MS_TO_KT) if wind is not None else None,
            magvar=((_f(v.get("hdg_true")) - _f(v.get("hdg_mag")) + 180) % 360 - 180)
            if v.get("hdg_true") is not None else 0.0,
            vis_sm=_vis(v),
            temp_c=_f(v["temp_c"]) if v.get("temp_c") is not None else None,
            temp_sl_c=_f(v["temp_sl_c"]) if v.get("temp_sl_c") is not None else None,
            dew_sl_c=_dew_sl(v),
            clouds=_clouds(v),
            zulu_sec=_f(v["zulu_sec"]) if v.get("zulu_sec") is not None else None,
            traffic=_traffic(v),
            rx_com1=bool(_f(v.get("rx_com1", 1), 1)),
            rx_com2=bool(_f(v.get("rx_com2", 1), 1)),
            paused=bool(_f(v.get("paused"))),
        )


def _dew_sl(v: dict) -> float | None:
    """Sea-level dew point from humidity + temperature (Magnus formula), else X-Plane's own value."""
    rh, t = v.get("rh_sl_pct"), v.get("temp_sl_c")
    if rh is not None and t is not None and _f(rh) > 0:
        a, b = 17.62, 243.12
        rhf = min(100.0, _f(rh))
        rhf = rhf * 100 if rhf <= 1.0 else rhf  # ratio or percent
        g = math.log(rhf / 100) + a * _f(t) / (b + _f(t))
        return b * g / (a - g)
    return _f(v["dew_sl_c"]) if v.get("dew_sl_c") is not None else None


@dataclass
class Target:
    idx: int
    rel_brg: float  # degrees relative to our heading, clockwise
    dist_nm: float
    rel_alt_ft: float  # + above us
    alt_ft: float | None  # MSL
    hdg_true: float | None
    icao_type: str
    lat: float | None = None
    lon: float | None = None


def _arr(v: dict, key: str) -> list:
    x = v.get(key)
    return x if isinstance(x, list) else []


def _traffic(v: dict) -> list[Target]:
    brg, dist, ralt = _arr(v, "tcas_brg"), _arr(v, "tcas_dist_m"), _arr(v, "tcas_ralt_m")
    if not dist:
        return []
    n = int(_f(v.get("tcas_n"), len(dist)) or len(dist))
    ele, psi, lat, lon = _arr(v, "tcas_ele_m"), _arr(v, "tcas_psi"), _arr(v, "tcas_lat"), _arr(v, "tcas_lon")
    raw = v.get("tcas_type") if isinstance(v.get("tcas_type"), (bytes, bytearray)) else b""
    out = []
    for i in range(1, min(n, len(dist))):
        d = _f(dist[i])
        if d <= 0:
            continue
        t = raw[i * 8:(i + 1) * 8].split(b"\x00", 1)[0].decode("latin-1").strip() if raw else ""
        out.append(Target(
            i, _f(brg[i]) if i < len(brg) else 0.0, d / 1852.0, _f(ralt[i]) * M_TO_FT if i < len(ralt) else 0.0,
            _f(ele[i]) * M_TO_FT if i < len(ele) else None, _f(psi[i]) if i < len(psi) else None, t,
            _f(lat[i]) if i < len(lat) else None, _f(lon[i]) if i < len(lon) else None))
    return out


def _vis(v: dict) -> float | None:
    if v.get("vis_sm") is not None:
        return _f(v["vis_sm"])
    if v.get("vis_m") is not None:
        return _f(v["vis_m"]) / 1609.34
    return None


def _clouds(v: dict) -> list[tuple[float, float]] | None:
    bases, covers = v.get("cloud_base_m"), v.get("cloud_cover")
    if not isinstance(bases, list) or not isinstance(covers, list):
        return None
    out = []
    for b, c in zip(bases, covers):
        c = _f(c)
        if c > 1.0:  # percent (XP12) or 0-6 enum (XP11) -> fraction
            c = c / 100 if c > 6 else c / 6
        if c > 0.05:
            out.append((_f(b) * M_TO_FT, c))
    return sorted(out)


def fmt_freq(khz: int) -> str:
    """As said on the radio: 118000 -> '118.0', 121300 -> '121.3', 132805 -> '132.805'."""
    if not khz:
        return "---.-"
    txt = f"{khz / 1000:.3f}".rstrip("0")
    return txt + "0" if txt.endswith(".") else txt
