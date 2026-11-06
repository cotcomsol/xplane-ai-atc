import math
import unittest
from pathlib import Path

from atc.airports import AirportDB, bearing, find_apt_dat, parse_apt_dat
from atc.pattern import position, sequence

FIX = Path(__file__).parent / "fixtures" / "xp"


class PatternTest(unittest.TestCase):
    def setUp(self):
        db = AirportDB(parse_apt_dat(find_apt_dat(str(FIX))))
        self.a = next(x for x in db.airports if x["id"] == "KPAO")
        r = self.a["runways"][0]
        self.thr = (r["lat2"], r["lon2"])  # runway 31 threshold
        self.rhdg = bearing(r["lat2"], r["lon2"], r["lat1"], r["lon1"])

    def at(self, along, cross):
        """Point `along` nm down the runway direction from the 31 threshold, `cross` nm to its right."""
        h, c = math.radians(self.rhdg), math.radians(self.rhdg + 90)
        lat = self.thr[0] + (along * math.cos(h) + cross * math.cos(c)) / 60
        lon = self.thr[1] + (along * math.sin(h) + cross * math.sin(c)) / 60 / math.cos(math.radians(self.thr[0]))
        return lat, lon

    def pos(self, along, cross, trk_off, agl=1000):
        return position(self.a, "31", *self.at(along, cross), agl, (self.rhdg + trk_off) % 360)

    def test_legs(self):
        self.assertEqual(self.pos(-2.0, 0.0, 0).leg, "final")
        self.assertEqual(self.pos(-2.0, 0.0, 0).detail, "2.0 nm final")
        self.assertEqual(self.pos(0.5, 0.0, 0, agl=400).leg, "upwind")
        dw = self.pos(0.2, -1.0, 180)
        self.assertEqual((dw.leg, dw.side), ("downwind", "left"))
        self.assertEqual(self.pos(-0.8, -0.8, 90).leg, "base")  # turning toward the centerline from the left
        self.assertEqual(self.pos(0.2, 1.0, 180).side, "right")
        self.assertIsNone(self.pos(0.2, -1.0, 180, agl=3500))
        self.assertIsNone(self.pos(10, 0, 0))

    def test_sequence(self):
        own = self.pos(0.2, -1.0, 180)  # left downwind abeam the numbers
        piper = self.pos(-1.5, 0.0, 0)  # 1.5 nm final
        cessna = self.pos(0.6, -1.0, 180)  # behind us on downwind
        self.assertEqual(sequence(self.a, "31", own, [("Piper", piper), ("Cessna", cessna)]),
                         ["Piper on final", "you", "Cessna on left downwind"])


if __name__ == "__main__":
    unittest.main()
