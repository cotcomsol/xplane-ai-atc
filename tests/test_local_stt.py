import unittest

from atc.local_stt import clean


class CleanTest(unittest.TestCase):
    def test_niner_suffix(self):
        self.assertEqual(clean("Bonanza 479er Kilo Papa"), "Bonanza 479 Kilo Papa")
        self.assertEqual(clean("Altimeter 29er9er2"), "Altimeter 2992")

    def test_niner_heard_as_or(self):
        self.assertEqual(clean("Departure frequency 119 or 0.5."), "Departure frequency 119.5.")
        self.assertEqual(clean("Bonanza 479 or Kilo Pop."), "Bonanza 479 Kilo Pop.")
        self.assertEqual(clean("Going around, Bonanza 9 are kilo papa."), "Going around, Bonanza 9 kilo papa.")

    def test_ordinary_english_untouched(self):
        for s in ("request one or two touch and goes", "runway 31 or 13", "information Delta, other traffic"):
            self.assertEqual(clean(s), s)


if __name__ == "__main__":
    unittest.main()
