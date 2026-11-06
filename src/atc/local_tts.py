"""Local text-to-speech: Kokoro-82M run on the CPU with onnxruntime (kokoro-onnx).

Same voices as OpenRouter's hexgrad/kokoro-82m, but no network queue: the remote service
answered in 0.5 - 20 s for the same sentence; locally a clause takes ~0.2-0.4 s. Text is
voiced clause by clause so playback starts after the first clause while the rest is generated.

The model files (~120 MB) are downloaded once into the cache directory.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
MODEL_FILE = "kokoro-v1.0.int8.onnx"
VOICES_FILE = "voices-v1.0.bin"
RATE = 24000
DEFAULT_VOICE = "am_michael"

# Acronyms espeak reads as words ("ills", "iffer"); spoken letter by letter instead
_SPELL = re.compile(r"\b(IFR|ILS|GPS|VOR|DME|NDB|LOC|ATC|FBO|IAF|FAF|MVA)\b")


def prepare_text(text: str) -> str:
    """Pronunciation fixes for espeak-based phonemization."""
    text = _SPELL.sub(lambda m: " ".join(m.group(1)), text)
    return text.replace("/", " ")


def clauses(text: str, min_chars: int = 18) -> list[str]:
    """Split at commas / sentence ends into speakable pieces (very short pieces are merged)."""
    parts = [p.strip() for p in re.split(r"(?<=[,.;:?!])\s+", text) if p.strip()]
    out: list[str] = []
    for p in parts:
        if out and (len(out[-1]) < min_chars or len(p) < min_chars):
            out[-1] = f"{out[-1]} {p}"
        else:
            out.append(p)
    return out


def _trim(x: np.ndarray, keep_tail: float, thr: float = 0.01) -> np.ndarray:
    """Cut leading silence and leave `keep_tail` seconds after the last sound."""
    if x.size == 0:
        return x
    nz = np.nonzero(np.abs(x) > thr * (np.max(np.abs(x)) or 1.0))[0]
    if nz.size == 0:
        return x[:0]
    return x[nz[0]:min(x.size, nz[-1] + 1 + int(keep_tail * RATE))]


class LocalVoice:
    def __init__(self, cache_dir: Path, threads: int = 4):
        self.dir = Path(cache_dir) / "kokoro"
        self.threads = threads
        self._kokoro = None
        self._voices: set[str] = set()
        self._load_lock = threading.Lock()
        self._run_lock = threading.Lock()  # one generation at a time (onnxruntime already uses several threads)

    @staticmethod
    def available() -> bool:
        try:
            import kokoro_onnx  # noqa: F401
            import onnxruntime  # noqa: F401
        except ImportError:
            return False
        return True

    def _download(self, name: str) -> Path:
        import httpx

        dest = self.dir / name
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        self.dir.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(dest.suffix + ".part")
        log.info("Downloading the local voice model %s (one time) ...", name)
        with httpx.stream("GET", RELEASE + name, follow_redirects=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length") or 0)
            done = next_report = 0
            with open(part, "wb") as fh:
                for chunk in r.iter_bytes(1 << 20):
                    fh.write(chunk)
                    done += len(chunk)
                    if total and done >= next_report:
                        log.info("  %s: %d%%", name, 100 * done // total)
                        next_report += total // 5
        part.replace(dest)
        return dest

    def load(self) -> None:
        """Download (first time) and load the model. Blocking; safe to call repeatedly."""
        with self._load_lock:
            if self._kokoro is not None:
                return
            import onnxruntime as ort
            from kokoro_onnx import Kokoro

            for name in ("phonemizer", "kokoro_onnx"):  # chatty INFO/WARNING lines
                logging.getLogger(name).setLevel(logging.ERROR)

            model, voices = self._download(MODEL_FILE), self._download(VOICES_FILE)
            so = ort.SessionOptions()
            so.intra_op_num_threads = self.threads
            so.inter_op_num_threads = 1
            sess = ort.InferenceSession(str(model), so, providers=["CPUExecutionProvider"])
            k = Kokoro.from_session(sess, str(voices))
            self._voices = set(k.get_voices())
            k.create("Ready.", voice=DEFAULT_VOICE, speed=1.0, lang="en-us")  # warm up
            self._kokoro = k
            log.info("Local voice ready (Kokoro, %d CPU threads)", self.threads)

    def _voice(self, voice: str | None) -> str:
        return voice if voice in self._voices else DEFAULT_VOICE

    def synth(self, text: str, voice: str | None, speed: float = 1.0, tail: float = 0.12) -> np.ndarray:
        """Blocking: text -> float32 samples at RATE."""
        self.load()
        with self._run_lock:
            samples, _ = self._kokoro.create(prepare_text(text), voice=self._voice(voice), speed=speed, lang="en-us")
        return _trim(np.asarray(samples, dtype=np.float32), tail)

    def render(self, text: str, voice: str | None, speed: float = 1.0) -> np.ndarray:
        """Blocking: a whole (possibly long) text, sentence by sentence."""
        gap = np.zeros(int(0.25 * RATE), dtype=np.float32)
        parts = []
        for s in [p for p in re.split(r"(?<=\.)\s+", text) if p.strip()]:
            parts += [self.synth(s, voice, speed, tail=0.05), gap]
        return np.concatenate(parts[:-1]) if parts else np.zeros(0, dtype=np.float32)

    async def stream(self, text: str, voice: str | None, speed: float = 1.0):
        """Yield (RATE, float32 samples) clause by clause; the next clause is generated while the
        previous one plays."""
        pieces = clauses(text)
        if not pieces:
            return
        nxt = asyncio.ensure_future(asyncio.to_thread(self.synth, pieces[0], voice, speed))
        try:
            for i in range(len(pieces)):
                cur = await nxt
                if i + 1 < len(pieces):
                    nxt = asyncio.ensure_future(asyncio.to_thread(self.synth, pieces[i + 1], voice, speed))
                yield RATE, cur
        finally:
            if not nxt.done():
                nxt.cancel()
