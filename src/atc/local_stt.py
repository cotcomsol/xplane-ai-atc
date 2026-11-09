"""Speech-to-text on this computer: NVIDIA Parakeet TDT 0.6B v2 (English), int8, on onnxruntime.

About 0.2 s per transmission on 4 CPU threads, no network wait. The model (~620 MB) is downloaded from
Hugging Face into cache/ the first time. It takes no vocabulary hint (OpenRouter's model gets the
callsign and facility), so it can miss "niner" now and then; see tools/stt_bench.py.
"""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

MODEL = "nemo-parakeet-tdt-0.6b-v2"
REPO = "istupakov/parakeet-tdt-0.6b-v2-onnx"  # Hugging Face
QUANTIZATION = "int8"
NAME = f"parakeet-tdt-0.6b-v2-{QUANTIZATION}"
MODEL_FILES = ("config.json", "vocab.txt", f"encoder-model.{QUANTIZATION}.onnx",
               f"decoder_joint-model.{QUANTIZATION}.onnx")


_PHONETIC = ("alpha|bravo|charlie|delta|echo|foxtrot|golf|hotel|india|juliet|juliett|kilo|lima|mike|november|"
             "oscar|papa|quebec|romeo|sierra|tango|uniform|victor|whiskey|x-ray|xray|yankee|zulu")


def clean(text: str) -> str:
    """Undo Parakeet's ways of writing 'niner', only where the result can't be ordinary English:
    '479er' -> '479', '29er9er2' -> '2992', '119 or 0.5' (one one niner point five) -> '119.5',
    '479 or Kilo Papa' / '9 are kilo' -> '479 Kilo Papa' / '9 kilo'. 'one or two' is left alone."""
    t = re.sub(r"(?<=\d)er(?=\d|\b)", "", text)
    t = re.sub(r"(\d) (?:or|are) 0\.(\d)", r"\1.\2", t)
    t = re.sub(rf"(\d),? (?:or|are),? (?=(?:{_PHONETIC})\b)", r"\1 ", t, flags=re.I)
    return t.strip()


class LocalSTT:
    def __init__(self, cache_dir: Path, threads: int = 4):
        # a plain folder, not the Hugging Face cache: onnxruntime refuses that cache's symlinked weight files
        self.dir = Path(cache_dir) / NAME
        self.threads = threads
        self._model = None
        self._load_lock = threading.Lock()
        self._run_lock = threading.Lock()

    @property
    def name(self) -> str:
        return NAME

    @staticmethod
    def available() -> bool:
        try:
            import onnx_asr  # noqa: F401
            import onnxruntime  # noqa: F401
        except ImportError:
            return False
        return True

    def load(self) -> None:
        """Download (first time) and load the model. Blocking; safe to call repeatedly."""
        with self._load_lock:
            if self._model is not None:
                return
            import onnx_asr
            import onnxruntime as ort

            logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
            if not all((self.dir / f).exists() for f in MODEL_FILES):  # first time, or an interrupted download
                from huggingface_hub import snapshot_download

                log.info("Downloading the local speech-to-text model (~620 MB, one time) ...")
                snapshot_download(REPO, local_dir=self.dir, allow_patterns=list(MODEL_FILES))
            so = ort.SessionOptions()
            so.intra_op_num_threads = self.threads
            so.inter_op_num_threads = 1
            m = onnx_asr.load_model(MODEL, self.dir, quantization=QUANTIZATION, sess_options=so,
                                    providers=["CPUExecutionProvider"])
            m.recognize(np.zeros(16000, dtype=np.float32), sample_rate=16000)  # warm up
            self._model = m
            log.info("Local speech-to-text ready (Parakeet, %d CPU threads)", self.threads)

    def transcribe(self, samples: np.ndarray, rate: int = 16000) -> str:
        """Blocking: float32 mono samples -> text."""
        self.load()
        with self._run_lock:
            text = self._model.recognize(np.ascontiguousarray(samples, dtype=np.float32), sample_rate=rate)
        return clean(text if isinstance(text, str) else str(text))
