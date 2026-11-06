"""Microphone capture, playback and the VHF radio sound effect."""

from __future__ import annotations

import io
import threading
import wave

import numpy as np
from scipy import signal

MIC_RATE = 16000


class Recorder:
    """Records from the microphone between start() and stop()."""

    def __init__(self, device=None):
        self.device = device
        self._chunks: list[np.ndarray] = []
        self._stream = None
        self._lock = threading.Lock()

    def _cb(self, indata, frames, time_info, status):
        with self._lock:
            self._chunks.append(indata[:, 0].copy())

    def start(self) -> None:
        if self._stream:
            return
        import sounddevice as sd

        self._chunks = []
        self._stream = sd.InputStream(samplerate=MIC_RATE, channels=1, dtype="float32",
                                      device=self.device, callback=self._cb)
        self._stream.start()

    @property
    def recording(self) -> bool:
        return self._stream is not None

    def stop(self) -> np.ndarray:
        if not self._stream:
            return np.zeros(0, dtype=np.float32)
        self._stream.stop()
        self._stream.close()
        self._stream = None
        with self._lock:
            return np.concatenate(self._chunks) if self._chunks else np.zeros(0, dtype=np.float32)


def to_wav(samples: np.ndarray, rate: int = MIC_RATE) -> bytes:
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def pcm16_to_float(data: bytes) -> np.ndarray:
    return np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2").astype(np.float32) / 32768


def radio_effect(x: np.ndarray, rate: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Make clean speech sound like it came over a VHF AM radio."""
    rng = rng or np.random.default_rng()
    if x.size == 0:
        return x
    x = x / (np.max(np.abs(x)) + 1e-9)
    # Narrow voice band
    sos = signal.butter(4, [350, 2800], btype="bandpass", fs=rate, output="sos")
    y = signal.sosfilt(sos, x)
    # Compression / mild clipping like an AM transmitter
    y = np.tanh(3.0 * y) / np.tanh(3.0)
    # Hiss under the voice, plus a squelch-tail burst at the end
    hiss = signal.sosfilt(signal.butter(2, [400, 4000], btype="bandpass", fs=rate, output="sos"),
                          rng.normal(0, 1, y.size))
    y = 0.85 * y + 0.04 * hiss
    lead = np.zeros(int(0.08 * rate))
    tail_n = int(0.18 * rate)
    tail = rng.normal(0, 1, tail_n) * np.linspace(0.35, 0, tail_n)
    tail = signal.sosfilt(signal.butter(2, [300, 5000], btype="bandpass", fs=rate, output="sos"), tail)
    out = np.concatenate([lead, y, tail])
    return (0.8 * out / (np.max(np.abs(out)) + 1e-9)).astype(np.float32)


class RadioStream:
    """The radio effect applied chunk by chunk (filter state carried across chunks, fixed gain
    instead of whole-clip normalization), for audio that is played while it downloads."""

    def __init__(self, rate: int, gain: float = 1.6, rng: np.random.Generator | None = None):
        self.rate = rate
        self.gain = gain
        self.rng = rng or np.random.default_rng()
        self.sos = signal.butter(4, [350, 2800], btype="bandpass", fs=rate, output="sos")
        self.zi = np.zeros((self.sos.shape[0], 2))
        self.hsos = signal.butter(2, [400, 4000], btype="bandpass", fs=rate, output="sos")
        self.hzi = np.zeros((self.hsos.shape[0], 2))
        self.started = False

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x.astype(np.float32)
        y, self.zi = signal.sosfilt(self.sos, x * self.gain, zi=self.zi)
        y = np.tanh(3.0 * y) / np.tanh(3.0)
        h, self.hzi = signal.sosfilt(self.hsos, self.rng.normal(0, 1, y.size), zi=self.hzi)
        out = 0.8 * (0.85 * y + 0.04 * h)
        if not self.started:
            self.started = True
            out = np.concatenate([np.zeros(int(0.08 * self.rate)), out])
        return out.astype(np.float32)

    def tail(self) -> np.ndarray:
        n = int(0.18 * self.rate)
        t = self.rng.normal(0, 1, n) * np.linspace(0.35, 0, n)
        t = signal.sosfilt(signal.butter(2, [300, 5000], btype="bandpass", fs=self.rate, output="sos"), t)
        return (0.8 * 0.5 * t / (np.max(np.abs(t)) + 1e-9)).astype(np.float32)


class AudioStream:
    """Plays chunks as they are fed in. feed() from any thread; wait() blocks until done."""

    def __init__(self, rate: int, device=None):
        import collections

        import sounddevice as sd

        self._q: collections.deque[np.ndarray] = collections.deque()
        self._cur = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()
        self._finished = False
        self._done = threading.Event()
        self._stream = sd.OutputStream(samplerate=rate, channels=1, dtype="float32", device=device,
                                       callback=self._cb)
        self._stream.start()

    def _cb(self, outdata, frames, time_info, status):
        out = np.zeros(frames, dtype=np.float32)
        filled = 0
        with self._lock:
            while filled < frames:
                if self._cur.size == 0:
                    if not self._q:
                        break
                    self._cur = self._q.popleft()
                    continue
                n = min(frames - filled, self._cur.size)
                out[filled:filled + n] = self._cur[:n]
                self._cur = self._cur[n:]
                filled += n
            drained = self._finished and not self._q and self._cur.size == 0
        outdata[:, 0] = out
        if drained and filled < frames:
            self._done.set()

    def feed(self, x: np.ndarray) -> None:
        with self._lock:
            self._q.append(x.astype(np.float32))

    def finish(self) -> None:
        with self._lock:
            self._finished = True

    def wait(self) -> None:
        self._done.wait()
        self.close()

    def abort(self) -> None:
        self._done.set()
        self.close()

    def close(self) -> None:
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:  # noqa: BLE001 - already closed
            pass

    @property
    def active(self) -> bool:
        return not self._done.is_set()


class Player:
    def __init__(self, device=None):
        self.device = device
        self._stream: AudioStream | None = None

    def stream(self, rate: int) -> AudioStream:
        """Start a streaming playback (stops anything already playing)."""
        self.stop()
        self._stream = AudioStream(rate, self.device)
        return self._stream

    def play(self, samples: np.ndarray, rate: int) -> None:
        """Play and block until done (call from a worker thread)."""
        import sounddevice as sd

        sd.play(samples, rate, device=self.device)
        sd.wait()

    def start(self, samples: np.ndarray, rate: int) -> None:
        """Start playing without waiting."""
        import sounddevice as sd

        sd.play(samples, rate, device=self.device)

    def busy(self) -> bool:
        import sounddevice as sd

        if self._stream and self._stream.active:
            return True
        try:
            return bool(sd.get_stream().active)
        except RuntimeError:  # nothing played yet
            return False

    def stop(self) -> None:
        import sounddevice as sd

        if self._stream:
            self._stream.abort()
            self._stream = None
        sd.stop()
