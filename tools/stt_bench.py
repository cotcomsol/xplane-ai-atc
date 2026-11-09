# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "numpy", "onnx-asr[cpu,hub]>=0.12", "faster-whisper>=1.2"]
# ///
"""Compare speech-to-text engines on pilot transmissions: accuracy and decode time.

Runs in its own environment (the local models aren't app dependencies):

    uv run tools/stt_bench.py tmp/stt_bench/clips                    # synthetic clips (tools/stt_clips.py)
    uv run tools/stt_bench.py from-log logs/flight-....jsonl out_dir # real recordings -> manifest to check/edit
    uv run tools/stt_bench.py out_dir --engines parakeet,whisper-small.en --threads 4

A clip set is a folder with WAVs (16 kHz mono) and manifest.jsonl rows {audio, ref, callsign, hint, cond}.
For recordings from a flight log, ref starts as what the OpenRouter model heard: correct the wrong ones by
hand, or the score only measures agreement with that model.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

PHONETIC = {"alpha": "a", "alfa": "a", "bravo": "b", "charlie": "c", "delta": "d", "echo": "e", "foxtrot": "f",
            "golf": "g", "hotel": "h", "india": "i", "juliet": "j", "juliett": "j", "kilo": "k", "lima": "l",
            "mike": "m", "november": "n", "oscar": "o", "papa": "p", "quebec": "q", "romeo": "r", "sierra": "s",
            "tango": "t", "uniform": "u", "victor": "v", "whiskey": "w", "xray": "x", "yankee": "y",
            "zulu": "z", "left": "l", "right": "r", "center": "c"}
NUM_WORDS = {w: str(i) for i, w in enumerate("zero one two three four five six seven eight nine".split())}
NUM_WORDS.update({"niner": "9", "ten": "10", "eleven": "11", "twelve": "12"})
ENGINES = ["openrouter", "parakeet", "parakeet-int8", "whisper-base.en", "whisper-small.en", "whisper-distil-large-v3"]


def normalize(text: str) -> list[str]:
    """Comparable tokens: numbers as single digits ('one four thousand' = '14,000' = 1 4 0 0 0), phonetic
    letters and runway sides as letters ('alpha bravo' = 'AB', 'three zero left' = '30L'), no punctuation.
    Formatting differences don't count as errors; the app reads either form."""
    t = re.sub(r"[Xx]-ray", "xray", text)
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)  # 14,000
    t = re.sub(r"(?<=\d)-(?=\d)", "", t)  # 3-0
    t = re.sub(r"(?<=\d)er\b|(?<=\d)er(?=\d)", "", t)  # 479er, 29er9er2 (niner)
    t = re.sub(r"(\d)\.(\d)", r"\1 point \2", t)
    raw = []
    for w in re.split(r"[\s,;:!?\"/]+|\.(?!\d)|-", t):
        w = w.strip(".'()")
        if not w:
            continue
        if re.fullmatch(r"[A-Za-z]*\d[A-Za-z0-9]*", w) or (w.isupper() and 1 < len(w) <= 5):
            raw += list(w.lower())  # N123AB, 30L, BIL: one token per character
        else:
            low = w.lower()
            raw += list(NUM_WORDS[low]) if low in NUM_WORDS else [PHONETIC.get(low, low)]
    out: list[str] = []
    for tok in raw:  # "1 4 thousand" -> 1 4 0 0 0 ; "7 thousand 5 hundred" -> 7 5 0 0
        if tok in ("thousand", "hundred") and out and out[-1].isdigit():
            zeros = 3 if tok == "thousand" else 2
            if tok == "hundred" and len(out) >= 4 and out[-4:-1] == ["0", "0", "0"]:
                out[-4:] = [out[-1], "0", "0"]  # 7000 + 5 hundred -> 7500
                continue
            out += ["0"] * zeros
        else:
            out.append(tok)
    return out


def wer(ref: list[str], hyp: list[str]) -> tuple[int, int]:
    """(word edits, reference words)."""
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[-1], len(ref)


def load_wav(path: Path):
    """16 kHz mono float32 samples (the app records at 16 kHz)."""
    import wave

    import numpy as np

    with wave.open(str(path)) as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1:
            raise ValueError(f"{path}: need 16 kHz mono")
        return np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768


def contains(hyp: list[str], sub: list[str]) -> bool:
    return any(hyp[i:i + len(sub)] == sub for i in range(len(hyp) - len(sub) + 1))


class OpenRouterEngine:
    """The app's current engine (network round trip included)."""

    def __init__(self, threads: int, model: str = "openai/gpt-4o-mini-transcribe"):
        import base64

        import httpx

        self.b64, self.model = base64, model
        key = os.environ.get("OPENROUTER_API_KEY", "")
        for env in (Path(".env"), Path.home() / ".env"):
            if not key and env.exists():
                m = re.search(r"^(?:export\s+)?OPENROUTER_API_KEY\s*=\s*['\"]?([^'\"\n]+)", env.read_text(), re.M)
                key = m.group(1).strip() if m else ""
        if not key:
            raise RuntimeError("no OPENROUTER_API_KEY")
        self.http = httpx.Client(base_url="https://openrouter.ai/api/v1", timeout=30,
                                 headers={"Authorization": f"Bearer {key}"})

    def __call__(self, path: Path, hint: str) -> str:
        body = {"model": self.model, "language": "en", "prompt": hint,
                "input_audio": {"data": self.b64.b64encode(path.read_bytes()).decode(), "format": "wav"}}
        r = self.http.post("/audio/transcriptions", json=body)
        r.raise_for_status()
        return (r.json().get("text") or "").strip()


class ParakeetEngine:
    """NVIDIA Parakeet TDT 0.6B v2 (English) on onnxruntime. No vocabulary hint."""

    def __init__(self, threads: int, quantization: str | None = None):
        import onnx_asr
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
        # a plain folder, not the Hugging Face cache: onnxruntime refuses its symlinked weight files
        path = Path.home() / ".cache" / "atc-stt-bench" / f"parakeet-tdt-0.6b-v2{'-' + quantization if quantization else ''}"
        self.m = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v2", path, quantization=quantization, sess_options=so,
                                     providers=["CPUExecutionProvider"])

    def __call__(self, path: Path, hint: str) -> str:
        return self.m.recognize(load_wav(path), sample_rate=16000)


class WhisperEngine:
    """faster-whisper (CTranslate2, int8 on CPU), greedy decoding, the hint as initial_prompt."""

    def __init__(self, threads: int, size: str):
        from faster_whisper import WhisperModel

        self.m = WhisperModel(size, device="cpu", compute_type="int8", cpu_threads=threads)

    def __call__(self, path: Path, hint: str) -> str:
        segs, _ = self.m.transcribe(load_wav(path), language="en", beam_size=1, initial_prompt=hint or None,
                                    vad_filter=False, condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segs)


def make_engine(name: str, threads: int):
    if name == "openrouter":
        return OpenRouterEngine(threads)
    if name == "parakeet":
        return ParakeetEngine(threads)
    if name == "parakeet-int8":
        return ParakeetEngine(threads, "int8")
    if name.startswith("whisper-"):
        return WhisperEngine(threads, name.removeprefix("whisper-"))
    raise SystemExit(f"unknown engine {name}; known: {', '.join(ENGINES)}")


def from_log(log: Path, out: Path) -> None:
    """Flight log (with [audio] save_recordings) -> manifest.jsonl in out/, WAVs referenced in place."""
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for line in log.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("kind") == "recording":
            wav = (log.parent / r["audio"]).resolve()
            rows.append({"audio": str(wav), "ref": r.get("text", ""), "callsign": "", "hint": r.get("hint", ""),
                         "cond": "real", "secs": r.get("secs"), "app_stt_s": r.get("stt_s")})
    (out / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} recordings -> {out / 'manifest.jsonl'} (check/correct the 'ref' texts)")


def bench(clips: Path, engines: list[str], threads: int, show: bool) -> None:
    rows = [json.loads(x) for x in (clips / "manifest.jsonl").read_text().splitlines() if x.strip()]
    results = {}
    for name in engines:
        print(f"\n== {name} ({threads} threads)", flush=True)
        t0 = time.monotonic()
        try:
            eng = make_engine(name, threads)
        except Exception as e:  # noqa: BLE001
            print(f"   skipped: {e}")
            continue
        print(f"   loaded in {time.monotonic() - t0:.1f}s", flush=True)
        eng(clips / rows[0]["audio"], rows[0].get("hint", ""))  # warm-up, not timed
        res = []
        for r in rows:
            t0 = time.monotonic()
            hyp = eng(clips / r["audio"], r.get("hint", ""))
            dt = time.monotonic() - t0
            ref_t, hyp_t = normalize(r["ref"]), normalize(hyp)
            e, n = wer(ref_t, hyp_t)
            cs = normalize(r["callsign"]) if r.get("callsign") else None
            res.append({"cond": r.get("cond", ""), "edits": e, "words": n, "dt": dt, "secs": r.get("secs") or 0,
                        "callsign_ok": contains(hyp_t, cs) if cs else None})
            if show and e:
                print(f"   {r['audio']} ({e} err)\n     ref: {r['ref']}\n     hyp: {hyp}")
        results[name] = res
    print(f"\n{'engine':26s} {'cond':6s} {'WER':>6s} {'callsign':>9s} {'mean s':>7s} {'p90 s':>6s} {'max s':>6s}")
    for name, res in results.items():
        for cond in sorted({x["cond"] for x in res}) + ["all"]:
            sub = [x for x in res if cond == "all" or x["cond"] == cond]
            e, n = sum(x["edits"] for x in sub), sum(x["words"] for x in sub)
            cs = [x["callsign_ok"] for x in sub if x["callsign_ok"] is not None]
            dts = sorted(x["dt"] for x in sub)
            print(f"{name:26s} {cond:6s} {100 * e / max(n, 1):5.1f}% "
                  f"{(f'{sum(cs)}/{len(cs)}' if cs else '-'):>9s} {statistics.mean(dts):7.2f} "
                  f"{dts[int(0.9 * (len(dts) - 1))]:6.2f} {dts[-1]:6.2f}")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "from-log":
        p = argparse.ArgumentParser(prog="stt_bench.py from-log")
        p.add_argument("cmd")
        p.add_argument("log", type=Path)
        p.add_argument("out", type=Path)
        a = p.parse_args()
        return from_log(a.log, a.out)
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("clips", type=Path, help="folder with manifest.jsonl")
    p.add_argument("--engines", default=",".join(ENGINES), help=f"comma list from: {', '.join(ENGINES)}")
    p.add_argument("--threads", type=int, default=4, help="CPU threads per local model (the sim needs the rest)")
    p.add_argument("--show", action="store_true", help="print every clip with errors")
    a = p.parse_args()
    bench(a.clips, [e.strip() for e in a.engines.split(",") if e.strip()], a.threads, a.show)


if __name__ == "__main__":
    main()
