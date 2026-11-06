import unittest
from pathlib import Path

from atc.airports import apt_dat_files, parse_apt_dat
from atc.taxi import route_to_runway, spoken_taxiway

FIX = Path(__file__).parent / "fixtures" / "xp"


class TaxiTest(unittest.TestCase):
    def setUp(self):
        self.apt = next(a for a in parse_apt_dat(apt_dat_files(str(FIX))[0]) if a["id"] == "XTXI")

    def test_flagged_as_having_network(self):
        self.assertTrue(self.apt["taxi"])

    def test_route_from_ramp_to_north_end(self):
        r = route_to_runway(self.apt, 40.0005, -99.990, "18")
        self.assertEqual(r.via, ["B", "A"])
        self.assertEqual(r.crossings, ["17/35"])
        self.assertFalse(r.backtaxi)

    def test_route_from_ramp_to_south_end(self):
        r = route_to_runway(self.apt, 40.0005, -99.990, "36")
        self.assertEqual(r.via, ["B", "A"])

    def test_no_network(self):
        pao = next(a for a in parse_apt_dat(apt_dat_files(str(FIX))[0]) if a["id"] == "KPAO")
        self.assertIsNone(route_to_runway(pao, 37.46, -122.11, "31"))

    def test_spoken(self):
        self.assertEqual(spoken_taxiway("A1"), "Alpha one")
        self.assertEqual(spoken_taxiway("TWY B"), "Bravo")
        self.assertIn("Bravo, Alpha", route_to_runway(self.apt, 40.0005, -99.990, "18").describe())


if __name__ == "__main__":
    unittest.main()
