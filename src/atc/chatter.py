"""Background radio chatter: other aircraft working the same controller.

Uses X-Plane's AI/multiplayer traffic when there is any near the facility, so what you hear
matches what you can see; otherwise (mode "always") fictional aircraft make plausible calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
import zlib

from .airports import bearing, dist_nm
from .controller import runway_in_use, short_name, spoken_facility
from .pattern import position as pattern_position
from .phraseology import spell, spoken_wind, type_name
from .state import AircraftState, fmt_freq

log = logging.getLogger(__name__)

COMPASS8 = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]
GA_TYPES = ["Skyhawk", "Cessna", "Piper", "Cherokee", "Bonanza", "Mooney", "Cirrus", "Diamond", "Skylane", "Archer"]
# Routine things another aircraft does on each kind of frequency
PHASES = {
    "GROUND": ["requesting taxi for departure from the ramp", "just landed, clear of the runway, requesting taxi to parking",
               "requesting taxi to the fuel pumps / FBO"],
    "DELIVERY": ["requesting a VFR departure (squawk and departure frequency)", "requesting its IFR clearance"],
    "TOWER": ["holding short, ready for departure", "inbound from 10 miles out for landing",
              "in the pattern doing touch and goes", "departing the area, requesting a frequency change"],
    "APPROACH": ["inbound VFR requesting landing at the airport", "requesting flight following en route",
                 "checking in level at its altitude", "requesting a practice approach"],
    "DEPARTURE": ["checking in after departure, climbing", "requesting flight following en route",
                  "leaving the area, requesting a frequency change"],
    "CENTER": ["checking in level at its cruising altitude", "requesting a different altitude for ride or winds",
               "requesting a frequency change to its destination", "IFR, checking in with a new altimeter request"],
}
PILOT_VOICES_KOKORO = ["am_fenrir", "am_puck", "am_echo", "bm_lewis", "bm_george", "af_bella", "af_sky", "af_kore"]

CHATTER_PROMPT = """You write one short, realistic radio exchange heard on a US ATC frequency between the
controller and ANOTHER aircraft (not the user's). FAA phraseology (AIM 4-2, JO 7110.65). Keep it routine and brief:
2 or 3 transmissions (pilot call, controller reply, and optionally the pilot's short readback ending with the
abbreviated callsign). Use only the facts given (runway, wind, altimeter, frequencies, the aircraft's position and
altitude). Never mention the user's aircraft. Write every number as spoken words in aviation style ("runway three
one", "one two zero point one", "four thousand five hundred", "altimeter three zero zero one").
Output only JSON: {"lines": [{"from": "pilot", "text": "..."}, {"from": "atc", "text": "..."}]}"""


def _n_number(seed: int) -> str:
    rng = random.Random(seed)
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    return "N" + str(rng.randint(100, 999)) + rng.choice(letters) + rng.choice(letters)


class Chatter:
    def __init__(self, sess, mode: str = "always", min_gap: float = 45, max_gap: float = 120):
        self.sess = sess
        self.mode = mode
        self.min_gap, self.max_gap = min_gap, max_gap
        self.next_t = time.monotonic() + random.uniform(15, 40)
        self.task: asyncio.Task | None = None
        self.fictional: dict[str, dict] = {}  # facility key -> a made-up aircraft working it

    def _voices(self) -> list[str]:
        cfg = self.sess.cfg
        if self.sess.local or "kokoro" in cfg.tts_model:
            return PILOT_VOICES_KOKORO
        return list(reversed(cfg.tts_voices)) or cfg.tts_voices

    def due(self, fac, busy: bool) -> bool:
        if self.mode == "off" or fac is None or fac.kind in ("ATIS", "CTAF") or busy:
            return False
        if self.task and not self.task.done():
            return False
        return time.monotonic() >= self.next_t

    def start(self, s: AircraftState, fac, is_idle) -> None:
        self.next_t = time.monotonic() + random.uniform(self.min_gap, self.max_gap)
        self.task = asyncio.get_running_loop().create_task(self._run(s, fac, is_idle))

    def _subject(self, s: AircraftState, fac) -> dict | None:
        """The other aircraft to voice: a real AI target near the facility, else a fictional one."""
        a = fac.airport
        cands = []
        for t in s.traffic or []:
            if t.lat is None or t.alt_ft is None:
                continue
            d = dist_nm(a["lat"], a["lon"], t.lat, t.lon)
            agl = t.alt_ft - a["elev"]
            if fac.kind == "TOWER" and d < 6 and agl < 3000 or \
                    fac.kind == "GROUND" and d < 1.5 and agl < 100 or \
                    fac.kind in ("APPROACH", "DEPARTURE") and d < 40 and agl > 300:
                cands.append(t)
        if cands:
            t = random.choice(cands)
            return {"callsign": _n_number(t.idx * 7919 + zlib.crc32(a["id"].encode())),
                    "type": type_name(t.icao_type) or random.Random(t.idx).choice(GA_TYPES),
                    "lat": t.lat, "lon": t.lon, "alt": t.alt_ft, "hdg": t.hdg_true, "real": True}
        if self.mode != "always":
            return None
        f = self.fictional.get(fac.key)
        # The same aircraft for a couple of exchanges (its next steps), then someone new
        if not f or f["n"] >= 2 or random.random() < 0.3:
            seed = random.randint(0, 10 ** 6)
            f = {"callsign": _n_number(seed), "type": random.Random(seed).choice(GA_TYPES), "real": False,
                 "phase": random.choice(PHASES.get(fac.kind, PHASES["APPROACH"])), "n": 0, "history": []}
            self.fictional[fac.key] = f
        f["n"] += 1
        return f

    def _facts(self, s: AircraftState, fac, subj: dict) -> str:
        a = fac.airport
        rw = runway_in_use(a, s.wind_dir, s.wind_kt)
        lines = [f"Facility: \"{spoken_facility(fac)}\" on {fmt_freq(fac.khz)} ({fac.kind.lower()} for "
                 f"{short_name(a)}, {a['id']}, field elevation {a['elev']} ft)",
                 f"Other aircraft: {subj['type']} {spell(subj['callsign'][1:])} (registration {subj['callsign']})"]
        if rw:
            lines.append(f"Runway in use: {rw[0]}")
        w = spoken_wind(s.wind_dir, s.wind_kt, s.magvar)
        if w:
            lines.append(f"Wind: {w}")
        if s.altimeter_inhg:
            lines.append(f"Altimeter: {s.altimeter_inhg:.2f}")
        if any(f["type"] == "ATIS" for f in a["freqs"]):
            from .atis import info_letter

            lines.append(f"Current ATIS: information {info_letter(a['id'], s.zulu_sec, rw[0] if rw else None)}")
        if fac.kind in ("TOWER", "GROUND") and dist_nm(a["lat"], a["lon"], s.lat, s.lon) < 4 and (
                s.on_ground or s.agl_ft < 1500):
            lines.append("The listener's own aircraft is about to use the runway (taxiing, holding short, departing "
                         "or landing): this other aircraft must NOT be cleared for takeoff, to land or onto the runway "
                         "right now. Give it something else (a taxi instruction, \"hold short\", \"continue\", a "
                         "pattern sequence or extend-downwind instruction).")
        for f in a["freqs"]:
            if f["type"] in ("TOWER", "GROUND", "APPROACH", "DEPARTURE"):
                lines.append(f"{f['type'].lower()} frequency {fmt_freq(f['khz'])}")
        if subj["real"]:
            d = dist_nm(a["lat"], a["lon"], subj["lat"], subj["lon"])
            b = bearing(a["lat"], a["lon"], subj["lat"], subj["lon"])
            where = f"{d:.1f} nm {COMPASS8[int(((b % 360) + 22.5) // 45) % 8]} of the airport at {subj['alt']:.0f} ft MSL"
            if fac.kind == "TOWER" and rw:
                p = pattern_position(a, rw[0], subj["lat"], subj["lon"], subj["alt"] - a["elev"], subj["hdg"])
                if p:
                    where = f"{p.describe(rw[0])} ({subj['alt']:.0f} ft MSL)"
            lines.append(f"Its position: {where}")
            if subj.get("hdg") is not None:
                lines.append(f"Its heading: {subj['hdg']:.0f} true")
        else:
            lines.append(f"It is a fictional aircraft, {subj['phase']}. Invent a plausible routine call for that "
                         f"which fits this {fac.kind.lower()} frequency (and nothing a {fac.kind.lower()} controller "
                         "wouldn't handle).")
            if subj.get("history"):
                lines.append("Its previous exchange on this frequency (this call must be its NEXT step, not a "
                             "repeat):\n  " + "\n  ".join(subj["history"][-3:]))
        return "\n".join(lines)

    async def _run(self, s: AircraftState, fac, is_idle) -> None:
        sess = self.sess
        try:
            subj = self._subject(s, fac)
            if not subj:
                return
            raw = await sess.api.chat(sess.cfg.chat_model, [
                {"role": "system", "content": CHATTER_PROMPT},
                {"role": "user", "content": self._facts(s, fac, subj)},
            ], max_tokens=600, reasoning_effort="none")
            m = re.search(r"\{.*\}", raw, re.S)
            lines = json.loads(m.group(0))["lines"] if m else []
            lines = [ln for ln in lines if ln.get("text")][:3]
            if not lines or not is_idle():
                return
            if not subj["real"]:
                subj.setdefault("history", []).extend(f"{ln['from']}: {ln['text']}" for ln in lines)
            pilot_voice = self._voices()[zlib.crc32(subj["callsign"].encode()) % len(self._voices())]
            async with sess._lock:  # the frequency is busy while it plays
                if sess.log:
                    sess.log.write("chatter", facility=f"{fac.airport['id']} {fac.kind}",
                                   lines=[f"{ln['from']}: {ln['text']}" for ln in lines])
                for ln in lines:
                    if not is_idle(ignore_lock=True):
                        return
                    who = "OTHER PILOT" if ln["from"] == "pilot" else f"ATC ({fac.name})"
                    log.info("  %s: %s", who, ln["text"])
                    voice = pilot_voice if ln["from"] == "pilot" else None
                    await sess.speak(ln["text"], fac, voice=voice)
                    await asyncio.sleep(random.uniform(0.4, 1.0))
        except Exception as e:  # noqa: BLE001 - chatter must never break the session
            log.debug("chatter failed: %s", e)
