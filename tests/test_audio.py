import unittest

import numpy as np

from atc.audio import RadioStream, radio_effect


class RadioStreamTest(unittest.TestCase):
    def test_chunked_matches_whole_clip_level(self):
        rate = 24000
        t = np.arange(rate * 2) / rate
        speech = (0.5 * np.sin(2 * np.pi * 700 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)
        fx = RadioStream(rate, rng=np.random.default_rng(1))
        chunks = [fx.process(c) for c in np.array_split(speech, 37)] + [fx.tail()]
        streamed = np.concatenate(chunks)
        whole = radio_effect(speech, rate, np.random.default_rng(1))
        self.assertEqual(streamed.size, whole.size)  # same lead-in and squelch tail lengths
        self.assertLess(np.max(np.abs(streamed)), 1.0)
        rms = lambda a: float(np.sqrt(np.mean(a ** 2)))  # noqa: E731
        self.assertAlmostEqual(rms(streamed), rms(whole), delta=0.25 * rms(whole))

    def test_no_clicks_at_chunk_boundaries(self):
        rate = 24000
        x = np.sin(2 * np.pi * 800 * np.arange(rate) / rate).astype(np.float32) * 0.5
        fx = RadioStream(rate, rng=np.random.default_rng(2))
        a = np.concatenate([fx.process(c) for c in np.array_split(x, 50)])
        fx2 = RadioStream(rate, rng=np.random.default_rng(2))
        b = fx2.process(x)
        self.assertLess(np.max(np.abs(a[int(0.3 * rate):] - b[int(0.3 * rate):])), 0.2)  # same filter output


if __name__ == "__main__":
    unittest.main()
