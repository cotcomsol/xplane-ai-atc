"""Session log: one JSON object per line, written as the flight happens."""

from __future__ import annotations

import json
import time
from pathlib import Path


class FlightLog:
    def __init__(self, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / time.strftime("flight-%Y%m%d-%H%M%S.jsonl")
        self._fh = open(self.path, "a", encoding="utf-8")

    def write(self, kind: str, **data) -> None:
        rec = {"t": time.time(), "kind": kind, **data}
        self._fh.write(json.dumps(rec, ensure_ascii=True) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def snapshot(s) -> dict:
    """The aircraft state fields worth keeping per transmission."""
    return {
        "lat": round(s.lat, 5), "lon": round(s.lon, 5), "alt_ft": round(s.alt_ft), "agl_ft": round(s.agl_ft),
        "gs_kt": round(s.gs_kt), "hdg": round(s.hdg_mag), "vs_fpm": round(s.vs_fpm), "on_ground": s.on_ground,
        "com1": s.com1_khz, "com2": s.com2_khz, "tx_com": s.tx_com,
        "xpdr": f"{s.xpdr_code:04d}", "xpdr_mode": s.xpdr_mode, "zulu_sec": s.zulu_sec,
    }


def latest(log_dir: Path) -> Path | None:
    logs = sorted(log_dir.glob("flight-*.jsonl"))
    return logs[-1] if logs else None
