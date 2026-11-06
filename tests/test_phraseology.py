import unittest

from atc.phraseology import altimeter_phrase, callsign_forms, region_for, spell


class PhraseologyTest(unittest.TestCase):
    def test_region(self):
        self.assertEqual(region_for("KPAO"), "FAA")
        self.assertEqual(region_for("PHNL"), "FAA")
        self.assertEqual(region_for("0Q5"), "FAA")
        self.assertEqual(region_for("LOWS"), "ICAO")
        self.assertEqual(region_for("EGLL"), "ICAO")
        self.assertEqual(region_for("CYVR"), "CANADA")

    def test_spell(self):
        self.assertEqual(spell("N550CR"), "November five five zero Charlie Romeo")
        self.assertEqual(spell("D-EABC"), "Delta Echo Alpha Bravo Charlie")

    def test_callsign_forms(self):
        self.assertEqual(callsign_forms("N123AB", "FAA"),
                         ("November one two three Alpha Bravo", "<type> three Alpha Bravo"))
        self.assertEqual(callsign_forms("N550CR", "ICAO")[1], "November Charlie Romeo")
        self.assertEqual(callsign_forms("N550CR", "FAA", "SR22"),
                         ("Cirrus five five zero Charlie Romeo", "Cirrus zero Charlie Romeo"))
        self.assertEqual(callsign_forms("N123AB", "FAA", "C172")[1], "Cessna three Alpha Bravo")
        self.assertEqual(callsign_forms("D-EABC", "ICAO")[1], "Delta Bravo Charlie")

    def test_spoken_wind_and_runway(self):
        from atc.phraseology import spoken_runway, spoken_wind
        self.assertEqual(spoken_wind(310, 9), "wind three one zero at niner")
        self.assertEqual(spoken_wind(5, 12, magvar=14), "wind three five zero at one two")
        self.assertEqual(spoken_wind(200, 2), "wind calm")
        self.assertEqual(spoken_runway("30L"), "three zero left")

    def test_altimeter(self):
        self.assertEqual(altimeter_phrase(29.92, "ICAO"), "QNH 1013 hPa")
        self.assertEqual(altimeter_phrase(29.93, "ICAO"), "QNH 1013 hPa")
        self.assertEqual(altimeter_phrase(29.92, "FAA"), "Altimeter 29.92 inHg")


if __name__ == "__main__":
    unittest.main()
