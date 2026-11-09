"""X-Plane 12 web API client (REST for lookups, websocket for live data).

Docs: https://developer.x-plane.com/article/x-plane-web-api/
The web server only accepts connections from the local machine.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import Callable

import httpx
import websockets

log = logging.getLogger(__name__)

# key -> X-Plane dataref name, or a list of candidate names (first one that exists
# wins). Missing datarefs are skipped, so guesses for version-specific names are fine.
DATAREFS: dict[str, str | list[str]] = {
    "lat": "sim/flightmodel/position/latitude",
    "lon": "sim/flightmodel/position/longitude",
    "elev_m": "sim/flightmodel/position/elevation",
    "agl_m": "sim/flightmodel/position/y_agl",
    "gs_ms": "sim/flightmodel/position/groundspeed",
    "ias_kt": "sim/flightmodel/position/indicated_airspeed",
    "hdg_mag": "sim/flightmodel/position/mag_psi",
    "hdg_true": "sim/flightmodel/position/psi",
    "vs_fpm": "sim/flightmodel/position/vh_ind_fpm",
    "on_ground": "sim/flightmodel/failures/onground_any",
    "baro_alt_ft": "sim/cockpit2/gauges/indicators/altitude_ft_pilot",
    "com1_khz": "sim/cockpit2/radios/actuators/com1_frequency_hz_833",
    "com2_khz": "sim/cockpit2/radios/actuators/com2_frequency_hz_833",
    "mic_sel": "sim/cockpit2/radios/actuators/audio_com_selection_man",
    "mic_sel_auto": "sim/cockpit2/radios/actuators/audio_com_selection",
    "rx_com1": "sim/cockpit2/radios/actuators/audio_selection_com1",
    "rx_com2": "sim/cockpit2/radios/actuators/audio_selection_com2",
    "xpdr_code": "sim/cockpit/radios/transponder_code",
    "xpdr_mode": "sim/cockpit2/radios/actuators/transponder_mode",
    "tailnum": "sim/aircraft/view/acf_tailnum",
    "icao_type": "sim/aircraft/view/acf_ICAO",
    # QNH as reported by stations (XP12), falling back to sea-level pressure
    "sl_press_pa": ["sim/weather/aircraft/qnh_pas", "sim/weather/region/qnh_pas",
                    "sim/weather/region/sealevel_pressure_pas"],
    # Wind at the aircraft (used on the ground) and the surface layer (used when airborne)
    "wind_dir": "sim/weather/aircraft/wind_now_direction_degt",
    "wind_kt": "sim/weather/aircraft/wind_now_speed_msc",
    "wind_sfc_dir": "sim/weather/region/wind_direction_degt",
    "wind_sfc_ms": "sim/weather/region/wind_speed_msc",
    "paused": "sim/time/paused",
    "zulu_sec": "sim/time/zulu_time_sec",
    # Weather for ATIS (XP12 names first, older names as fallbacks)
    "vis_sm": ["sim/weather/aircraft/visibility_reported_sm"],
    "vis_m": ["sim/weather/visibility_reported_m"],
    "temp_c": ["sim/weather/aircraft/temperature_ambient_deg_c", "sim/weather/temperature_ambient_c"],
    "temp_sl_c": ["sim/weather/region/sealevel_temperature_c", "sim/weather/temperature_sealevel_c"],
    "rh_sl_pct": ["sim/weather/aircraft/relative_humidity_sealevel_percent",
                  "sim/weather/relative_humidity_sealevel_percent"],
    "dew_sl_c": "sim/weather/dewpoi_sealevel_c",
    # Traffic (TCAS targets: AI and multiplayer aircraft; index 0 is our own aircraft)
    "tcas_n": "sim/cockpit2/tcas/indicators/tcas_num_acf",
    "tcas_brg": "sim/cockpit2/tcas/indicators/relative_bearing_degs",
    "tcas_dist_m": "sim/cockpit2/tcas/indicators/relative_distance_mtrs",
    "tcas_ralt_m": "sim/cockpit2/tcas/indicators/relative_altitude_mtrs",
    "tcas_ele_m": "sim/cockpit2/tcas/targets/position/ele",
    "tcas_psi": "sim/cockpit2/tcas/targets/position/psi",
    "tcas_lat": "sim/cockpit2/tcas/targets/position/lat",
    "tcas_lon": "sim/cockpit2/tcas/targets/position/lon",
    "tcas_vs": "sim/cockpit2/tcas/targets/position/vertical_speed",
    "tcas_type": "sim/cockpit2/tcas/targets/icao_type",
    "cloud_base_m": ["sim/weather/aircraft/cloud_base_msl_m", "sim/weather/region/cloud_base_msl_m",
                     "sim/weather/cloud_base_msl_m"],
    "cloud_cover": ["sim/weather/aircraft/cloud_coverage_percent", "sim/weather/region/cloud_coverage_percent",
                    "sim/weather/cloud_coverage"],
}


RAW_DATA_KEYS = {"tcas_type"}


def _vnum(version: str) -> int:
    """'v2' -> 2 (so that v10 sorts after v9)."""
    digits = "".join(c for c in str(version) if c.isdigit())
    return int(digits) if digits else 0


class XPlane:
    def __init__(self, host: str = "localhost", port: int = 8086):
        self.base = f"http://{host}:{port}/api"
        self.ws_base = f"ws://{host}:{port}/api"
        self.version = "v2"
        self.values: dict[str, object] = {}
        self.connected = False
        self._ids: dict[int, tuple[str, str]] = {}  # id -> (name, value_type)
        self._cmd_ids: dict[int, str] = {}
        self._cmd_cbs: dict[str, list[Callable[[bool], None]]] = {}
        self._req = 0
        self._active: set[str] = set()  # commands currently held down
        self.missing: list[str] = []

    def on_command(self, command: str, cb: Callable[[bool], None]) -> None:
        """Register cb(is_active) for when an X-Plane command starts/stops."""
        self._cmd_cbs.setdefault(command, []).append(cb)

    async def _pick_version(self, http: httpx.AsyncClient) -> None:
        try:
            r = await http.get(f"{self.base}/capabilities")
            versions = r.json()["api"]["versions"]
            self.version = max(versions, key=_vnum)
            xp = r.json().get("x-plane", {}).get("version")
            log.info("X-Plane %s, web API %s", xp, self.version)
        except Exception:
            self.version = "v1"
            log.warning("No /api/capabilities; assuming web API v1 (no command support)")

    async def _lookup(self, http: httpx.AsyncClient, kind: str, name: str) -> dict | None:
        r = await http.get(f"{self.base}/{self.version}/{kind}", params={"filter[name]": name})
        if r.status_code != 200:
            return None
        data = r.json().get("data") or []
        return data[0] if data else None

    async def _resolve(self) -> None:
        self._ids.clear()
        self._cmd_ids.clear()
        self.missing = []
        async with httpx.AsyncClient(timeout=5) as http:
            await self._pick_version(http)
            for key, names in DATAREFS.items():
                for name in [names] if isinstance(names, str) else names:
                    d = await self._lookup(http, "datarefs", name)
                    if d:
                        self._ids[d["id"]] = (key, d.get("value_type", ""))
                        break
                else:
                    self.missing.append(key)
            if _vnum(self.version) >= 2:
                for cmd in self._cmd_cbs:
                    c = await self._lookup(http, "commands", cmd)
                    if c:
                        self._cmd_ids[c["id"]] = cmd
                    else:
                        log.error("X-Plane command not found: %s", cmd)
        if self.missing:
            log.info("Datarefs not available (ignored): %s", ", ".join(self.missing))
        if not self._ids:
            raise RuntimeError("no datarefs resolved (is a flight loaded?)")

    async def search_datarefs(self, text: str) -> list[dict]:
        """All dataref names containing `text` (for finding names in a given X-Plane version)."""
        async with httpx.AsyncClient(timeout=30) as http:
            await self._pick_version(http)
            r = await http.get(f"{self.base}/{self.version}/datarefs", params={"fields": "name,value_type"})
            r.raise_for_status()
            return [d for d in r.json()["data"] if text.lower() in d["name"].lower()]

    def _decode(self, key: str, vtype: str, v):
        if key in RAW_DATA_KEYS and vtype == "data" and isinstance(v, str):
            return base64.b64decode(v)  # several fixed-width strings packed together
        if vtype == "data" and isinstance(v, str):
            raw = base64.b64decode(v)
            return raw.split(b"\x00", 1)[0].decode("latin-1").strip()
        return v

    async def _send(self, ws, type_: str, params: dict) -> None:
        self._req += 1
        await ws.send(json.dumps({"req_id": self._req, "type": type_, "params": params}))

    async def run(self) -> None:
        """Connect and keep self.values updated forever, reconnecting as needed."""
        delay = 2
        while True:
            try:
                await self._resolve()
                async with websockets.connect(f"{self.ws_base}/{self.version}", max_size=None) as ws:
                    await self._send(ws, "dataref_subscribe_values",
                                     {"datarefs": [{"id": i} for i in self._ids]})
                    if self._cmd_ids:
                        await self._send(ws, "command_subscribe_is_active",
                                         {"commands": [{"id": i} for i in self._cmd_ids]})
                    self.connected = True
                    delay = 2
                    log.info("Connected to X-Plane (%d datarefs, %d commands)",
                             len(self._ids), len(self._cmd_ids))
                    async for msg in ws:
                        self._handle(json.loads(msg))
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - keep retrying whatever happened
                if self.connected:
                    log.warning("X-Plane connection lost: %s", e)
                else:
                    log.info("Waiting for X-Plane at %s (%s)", self.base, e)
            self.connected = False
            for cmd in list(self._active):  # e.g. PTT held when the connection dropped: it won't be released
                self._fire(cmd, False)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 15)

    def _fire(self, cmd: str, active: bool) -> None:
        (self._active.add if active else self._active.discard)(cmd)
        for cb in self._cmd_cbs.get(cmd, []):
            try:
                cb(active)
            except Exception:  # noqa: BLE001 - a failing callback must not drop the connection
                log.exception("Handler for X-Plane command %s failed", cmd)

    def _handle(self, m: dict) -> None:
        t = m.get("type")
        if t == "dataref_update_values":
            for sid, v in m["data"].items():
                key, vtype = self._ids.get(int(sid.strip()), (None, None))
                if key:
                    self.values[key] = self._decode(key, vtype, v)
        elif t == "command_update_is_active":
            for sid, active in m["data"].items():
                cmd = self._cmd_ids.get(int(sid.strip()))
                if cmd:
                    self._fire(cmd, bool(active))
        elif t == "result" and not m.get("success", True):
            log.error("X-Plane request %s failed: %s %s", m.get("req_id"),
                      m.get("error_code"), m.get("error_message"))
