"""Configuration loading from atc.toml, environment and .env files."""

from __future__ import annotations

import logging
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    xplane_host: str = "localhost"
    xplane_port: int = 8086
    xplane_path: str = ""
    ptt_command: str = "sim/operation/contact_atc"

    callsign: str = ""
    simbrief_user: str = ""  # SimBrief username or numeric user id: the filed flight plan

    api_key: str = ""
    stt_model: str = "openai/gpt-4o-mini-transcribe"
    chat_model: str = "openai/gpt-5.6-luna"
    reasoning_effort: str = "none"
    tts_model: str = "hexgrad/kokoro-82m"
    tts_voices: list[str] = field(default_factory=lambda: [
        "am_michael", "am_adam", "am_eric", "am_liam", "am_onyx", "af_heart", "af_sarah", "af_nicole",
    ])
    tts_speed: float = 1.3
    tts_instructions: str = (
        "You are a busy US air traffic controller on the radio. "
        "Speak fast, flat and clipped, no emotion."
    )

    input_device: str | int | None = None
    output_device: str | int | None = None
    radio_effect: bool = True
    stream_voice: bool = True
    voice_engine: str = "local"  # local (Kokoro on this computer) / openrouter (tts_model)
    voice_threads: int = 4  # CPU threads for the local voice
    stt_engine: str = "openrouter"  # openrouter (stt_model) / local (Parakeet on this computer)
    stt_threads: int = 4  # CPU threads for local speech-to-text
    save_recordings: bool = False  # keep each PTT recording as a WAV next to the flight log

    chatter_mode: str = "always"  # always / traffic (only real AI traffic) / off
    chatter_min_gap: float = 45
    chatter_max_gap: float = 120

    cache_dir: Path = Path("cache")
    log_dir: Path = Path("logs")
    cifp_dirs: list[str] = field(default_factory=list)  # extra CIFP folders (besides X-Plane's)
    debrief_model: str = "anthropic/claude-opus-5.5"


def _read_dotenv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip().removeprefix("export ").strip()] = v.strip().strip("'\"")
    except OSError:
        pass
    return out


log = logging.getLogger(__name__)


def _bool(v, default: bool) -> bool:
    """true/false as a TOML boolean, or the same written as a string ("false", "off", "0")."""
    if isinstance(v, str):
        low = v.strip().lower()
        if low in ("true", "yes", "on", "1"):
            return True
        if low in ("false", "no", "off", "0", ""):
            return False
        log.warning("Config: %r is not true or false; using %s", v, str(default).lower())
        return default
    return default if v is None else bool(v)


def _choice(name: str, v, allowed: tuple[str, ...], default: str) -> str:
    low = str(v).strip().lower()
    if low not in allowed:
        log.warning("Config: %s = %r is not one of %s; using %r", name, v, ", ".join(allowed), default)
        return default
    return low


def _device(v):
    if v in ("", None):
        return None
    if isinstance(v, int):
        return v
    return int(v) if str(v).isdigit() else v


def load(path: str | Path = "atc.toml") -> Config:
    cfg = Config()
    p = Path(path)
    data = tomllib.loads(p.read_text()) if p.exists() else {}

    xp = data.get("xplane", {})
    cfg.xplane_host = xp.get("host", cfg.xplane_host)
    cfg.xplane_port = int(xp.get("port", cfg.xplane_port))
    cfg.xplane_path = xp.get("path", cfg.xplane_path)
    cfg.ptt_command = xp.get("ptt_command", cfg.ptt_command)
    cfg.cifp_dirs = xp.get("cifp_dirs", cfg.cifp_dirs)

    cfg.callsign = data.get("pilot", {}).get("callsign", cfg.callsign)
    cfg.simbrief_user = str(data.get("simbrief", {}).get("username", cfg.simbrief_user)).strip()

    orr = data.get("openrouter", {})
    cfg.api_key = orr.get("api_key", "")
    cfg.stt_model = orr.get("stt_model", cfg.stt_model)
    cfg.chat_model = orr.get("chat_model", cfg.chat_model)
    cfg.reasoning_effort = orr.get("reasoning_effort", cfg.reasoning_effort)
    cfg.tts_model = orr.get("tts_model", cfg.tts_model)
    cfg.tts_speed = float(orr.get("tts_speed", cfg.tts_speed))
    cfg.tts_voices = orr.get("tts_voices", cfg.tts_voices)
    cfg.tts_instructions = orr.get("tts_instructions", cfg.tts_instructions)
    cfg.debrief_model = orr.get("debrief_model", cfg.debrief_model)

    au = data.get("audio", {})
    cfg.input_device = _device(au.get("input_device"))
    cfg.output_device = _device(au.get("output_device"))
    cfg.radio_effect = _bool(au.get("radio_effect"), True)
    cfg.stream_voice = _bool(au.get("stream_voice"), True)
    cfg.voice_engine = _choice("[audio] voice_engine", au.get("voice_engine", cfg.voice_engine),
                               ("local", "openrouter"), cfg.voice_engine)
    cfg.voice_threads = int(au.get("voice_threads", cfg.voice_threads))
    cfg.stt_engine = _choice("[audio] stt_engine", au.get("stt_engine", cfg.stt_engine),
                             ("openrouter", "local"), cfg.stt_engine)
    cfg.stt_threads = int(au.get("stt_threads", cfg.stt_threads))
    cfg.save_recordings = _bool(au.get("save_recordings"), cfg.save_recordings)

    ch = data.get("chatter", {})
    cfg.chatter_mode = _choice("[chatter] mode", ch.get("mode", cfg.chatter_mode),
                               ("always", "traffic", "off"), cfg.chatter_mode)
    cfg.chatter_min_gap = float(ch.get("min_gap", cfg.chatter_min_gap))
    cfg.chatter_max_gap = float(ch.get("max_gap", cfg.chatter_max_gap))

    if not cfg.api_key:
        cfg.api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not cfg.api_key:
        for env in (Path(".env"), Path.home() / ".env"):
            cfg.api_key = _read_dotenv(env).get("OPENROUTER_API_KEY", "")
            if cfg.api_key:
                break
    return cfg
