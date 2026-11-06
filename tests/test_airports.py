import unittest
from pathlib import Path

from atc.airports import AirportDB, find_apt_dat, parse_apt_dat

FIX = Path(__file__).parent / "fixtures" / "xp"


class AirportsTest(unittest.TestCase):
    def setUp(self):
        self.airports = parse_apt_dat(find_apt_dat(str(FIX)))
        self.db = AirportDB(self.airports)

    def test_parse(self):
        pao = next(a for a in self.airports if a["id"] == "KPAO")
        self.assertEqual(pao["runways"][0]["ids"], ["13", "31"])
        self.assertTrue(2000 < pao["runways"][0]["length_ft"] < 3000)
        self.assertIn({"type": "TOWER", "khz": 118600, "name": "PALO ALTO TWR"}, pao["freqs"])

    def test_facility_on_ground_freq(self):
        f = self.db.facility_for(125000, 37.4611, -122.1150)
        self.assertEqual((f.airport["id"], f.kind), ("KPAO", "GROUND"))

    def test_nearest_approach_wins(self):
        f = self.db.facility_for(120100, 37.45, -122.10)
        self.assertEqual((f.airport["id"], f.kind), ("KPAO", "APPROACH"))

    def test_tower_out_of_range(self):
        self.assertIsNone(self.db.facility_for(118600, 38.5, -122.1))

    def test_tower_beats_ctaf_on_same_freq(self):
        db = AirportDB([{"id": "KSQL", "name": "San Carlos", "type": "land", "lat": 37.51, "lon": -122.25,
                         "runways": [], "freqs": [{"type": "CTAF", "khz": 119000, "name": "CTAF"},
                                                  {"type": "TOWER", "khz": 119000, "name": "TWR"}]}])
        self.assertEqual(db.facility_for(119000, 37.51, -122.25).kind, "TOWER")

    def test_unknown_freq(self):
        self.assertIsNone(self.db.facility_for(123450, 37.4611, -122.1150))


if __name__ == "__main__":
    unittest.main()
