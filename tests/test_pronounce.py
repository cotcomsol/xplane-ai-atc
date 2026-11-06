import unittest

from atc.pronounce import for_tts


class PronounceTest(unittest.TestCase):
    def test_kokoro_wind_forced_to_noun(self):
        out = for_tts("Wind three zero zero at eight, winds calm, rewind", "hexgrad/kokoro-82m")
        self.assertEqual(out, "[Wind](/w\u02c8\u026and/) three zero zero at eight, "
                              "[winds](/w\u02c8\u026andz/) calm, rewind")

    def test_other_models_untouched(self):
        self.assertEqual(for_tts("wind three zero zero", "x-ai/grok-voice-tts-1.0"), "wind three zero zero")


if __name__ == "__main__":
    unittest.main()
