"""Generate synthetic pilot transmissions for speech-to-text benchmarks (tools/stt_bench.py).

Speaks a set of typical pilot calls with the local Kokoro voices, at 16 kHz like the microphone
recording, clean and with cockpit noise, and writes a manifest.jsonl next to the WAVs.

    uv run python tools/stt_clips.py tmp/stt_bench/clips

Synthetic speech is clearer than a real pilot on a headset mic, so treat the results as a first
filter; real recordings ([audio] save_recordings) are the real test.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt

from atc.audio import MIC_RATE, to_wav
from atc.local_tts import RATE, LocalVoice

# (text as spoken, callsign as spoken, speech-to-text hint like Session.stt_hint)
CALLS = [
    ("Palo Alto Ground, Skyhawk one two three alpha bravo, at the transient ramp with information Charlie, "
     "request taxi VFR to Monterey", "Skyhawk one two three alpha bravo", "KPAO", "N123AB", "C172"),
    ("Palo Alto Tower, Skyhawk one two three alpha bravo, holding short runway three one, ready for departure, "
     "left downwind departure", "Skyhawk one two three alpha bravo", "KPAO", "N123AB", "C172"),
    ("Runway three one, cleared for takeoff, left downwind departure, Skyhawk three alpha bravo",
     "Skyhawk three alpha bravo", "KPAO", "N123AB", "C172"),
    ("NorCal Approach, Cirrus eight two eight zulu zulu, ten miles south of Billings, level eight thousand "
     "five hundred, request flight following to Spokane", "Cirrus eight two eight zulu zulu", "KBIL", "N828ZZ", "S22T"),
    ("Billings Clearance, Cirrus eight two eight zulu zulu, IFR to Spokane, ready to copy",
     "Cirrus eight two eight zulu zulu", "KBIL", "N828ZZ", "S22T"),
    ("Cleared to Spokane airport via radar vectors BIL, then as filed, maintain seven thousand, expect one four "
     "thousand one zero minutes after departure, departure frequency one one niner point five, squawk four two "
     "one four, Cirrus eight two eight zulu zulu", "Cirrus eight two eight zulu zulu", "KBIL", "N828ZZ", "S22T"),
    ("Climb and maintain one four thousand, cleared direct ARMTY, Cirrus eight zulu zulu",
     "Cirrus eight zulu zulu", "KBIL", "N828ZZ", "S22T"),
    ("Salt Lake Center, Cirrus eight two eight zulu zulu, level one four thousand",
     "Cirrus eight two eight zulu zulu", "KBIL", "N828ZZ", "S22T"),
    ("Turn right heading two five zero, maintain six thousand until established, cleared ILS runway two eight "
     "right approach, Bonanza four seven niner kilo papa", "Bonanza four seven niner kilo papa", "KSJC", "N479KP", "BE36"),
    ("San Jose Tower, Bonanza four seven niner kilo papa, ten mile final runway three zero left",
     "Bonanza four seven niner kilo papa", "KSJC", "N479KP", "BE36"),
    ("Cleared to land runway three zero left, Bonanza niner kilo papa", "Bonanza niner kilo papa", "KSJC",
     "N479KP", "BE36"),
    ("San Jose Ground, Bonanza four seven niner kilo papa, clear of three zero left at Zulu, taxi to "
     "Signature", "Bonanza four seven niner kilo papa", "KSJC", "N479KP", "BE36"),
    ("Squawk seven seven zero zero, Skyhawk three alpha bravo", "Skyhawk three alpha bravo", "KPAO", "N123AB", "C172"),
    ("Say again for Skyhawk one two three alpha bravo", "Skyhawk one two three alpha bravo", "KPAO", "N123AB", "C172"),
    ("Altimeter two niner niner two, information Delta, Skyhawk three alpha bravo", "Skyhawk three alpha bravo",
     "KPAO", "N123AB", "C172"),
    ("Monterey Tower, Skyhawk one two three alpha bravo, request the RNAV runway one zero right approach, full stop",
     "Skyhawk one two three alpha bravo", "KMRY", "N123AB", "C172"),
    ("Contact departure one two zero point one, Skyhawk three alpha bravo", "Skyhawk three alpha bravo", "KPAO",
     "N123AB", "C172"),
    ("Salzburg Tower, Delta Echo Kilo Lima Mike, on the ground, request taxi to the holding point runway one five",
     "Delta Echo Kilo Lima Mike", "LOWS", "DEKLM", "C172"),
    ("Traffic in sight, Skyhawk three alpha bravo, request left closed traffic, touch and go",
     "Skyhawk three alpha bravo", "KPAO", "N123AB", "C172"),
    ("Going around, Bonanza niner kilo papa", "Bonanza niner kilo papa", "KSJC", "N479KP", "BE36"),
]
VOICES = [("am_michael", 1.15), ("af_sarah", 1.0), ("am_eric", 1.25)]


def cockpit_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Engine drone (prop harmonics) plus broadband wind/cabin noise."""
    t = np.arange(n) / MIC_RATE
    drone = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) / k for k, f in enumerate((82, 164, 246, 328), 1))
    wind = sosfilt(butter(2, 1200, "low", fs=MIC_RATE, output="sos"), rng.standard_normal(n))
    x = drone / np.max(np.abs(drone)) + 2.0 * wind / np.std(wind)
    return x / np.std(x)


def headset_mic(x: np.ndarray) -> np.ndarray:
    return sosfilt(butter(2, [250, 3800], "bandpass", fs=MIC_RATE, output="sos"), x)


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    voice = LocalVoice(Path("cache"), threads=8)
    rng = np.random.default_rng(7)
    rows = []
    for i, (text, callsign, airport, reg, typ) in enumerate(CALLS):
        name, speed = VOICES[i % len(VOICES)]
        speech = resample_poly(voice.synth(text, name, speed), MIC_RATE, RATE).astype(np.float32)
        speech = 0.5 * speech / np.max(np.abs(speech))
        pad = np.zeros(int(0.3 * MIC_RATE), dtype=np.float32)
        clean = np.concatenate([pad, speech, pad])
        hint = (f"Pilot radio call to air traffic control. Callsign {reg}. Aircraft type {typ}. Airport {airport}. "
                "Words: runway, taxi, holding point, hold short, QNH, altimeter, squawk, niner, information, "
                "ready for departure, request, VFR, flight following.")
        for cond in ("clean", "noisy"):
            x = clean
            if cond == "noisy":  # about 12 dB speech-to-noise, through a headset mic band
                noise = cockpit_noise(x.size, rng)
                x = headset_mic(x + noise * np.sqrt(np.mean(speech ** 2)) / 4.0).astype(np.float32)
                x = 0.7 * x / np.max(np.abs(x))
            fn = f"{i:02d}-{cond}.wav"
            (out / fn).write_bytes(to_wav(x))
            rows.append({"audio": fn, "ref": text, "callsign": callsign, "hint": hint, "cond": cond,
                         "voice": name, "secs": round(x.size / MIC_RATE, 2)})
        print(f"{i:02d} {name:10s} {clean.size / MIC_RATE:4.1f}s  {text[:70]}")
    with open(out / "manifest.jsonl", "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    print(f"{len(rows)} clips in {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "tmp/stt_bench/clips"))
