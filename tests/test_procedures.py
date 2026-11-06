import unittest
from pathlib import Path

from atc.procedures import find_cifp, parse_cifp, vectors

FIX = Path(__file__).parent / "fixtures" / "xp"


class ProceduresTest(unittest.TestCase):
    def setUp(self):
        self.p = parse_cifp(find_cifp(str(FIX), "KSJC"))

    def test_parse(self):
        ils = self.p.approaches["I30L"]
        self.assertEqual((ils.kind, ils.runway, ils.variant), ("ILS", "30L", ""))
        self.assertEqual((ils.if_fix, ils.intercept_alt, ils.faf_fix, ils.faf_alt), ("JENES", 3200, "HIVAK", 2700))
        self.assertEqual(ils.course_mag, 306.0)
        self.assertIn(("KLIDE", "at or above 4000"), [(t.name, t.altitude) for t in ils.transitions])
        self.assertEqual(self.p.approaches["R30LZ"].spoken, "RNAV Zulu runway three zero left approach")
        self.assertAlmostEqual(self.p.thresholds["30L"].lat, 37.3562, places=3)
        self.assertEqual([a.ident for a in self.p.for_runway("30L")], ["I30L", "R30LY", "R30LZ"])

    def test_intercept_when_in_position(self):
        # 18 nm out on the extended centerline (outbound true course 139), offset ~3 nm to the side
        thr = self.p.thresholds["30L"]
        import math
        out = math.radians(139)
        lat = thr.lat + 18 / 60 * math.cos(out) + 3 / 60 * math.cos(out + math.pi / 2)
        lon = thr.lon + (18 / 60 * math.sin(out) + 3 / 60 * math.sin(out + math.pi / 2)) / math.cos(math.radians(thr.lat))
        v = vectors(self.p.approaches["I30L"], thr, lat, lon, 13.0)
        self.assertIn("APPROACH CLEARANCE NOW", v)
        self.assertIn("maintain three thousand two hundred until established on the localizer, cleared ILS runway "
                      "three zero left approach", v)
        self.assertIn("heading three three five", v)
        v2 = vectors(self.p.approaches["I30L"], thr, lat, lon, 13.0, cur_hdg=170)
        self.assertIn("turn right heading three three five", v2)  # left of course inbound: 306 + 30, 5-degree steps

    def test_vector_when_too_close(self):
        v = vectors(self.p.approaches["I30L"], self.p.thresholds["30L"], 37.30, -121.78, 13.0)
        self.assertIn("VECTOR NOW", v)


if __name__ == "__main__":
    unittest.main()


class AltitudeChangeTest(unittest.TestCase):
    def test_words(self):
        from atc.procedures import altitude_change
        self.assertEqual(altitude_change(5000, 6100), "climb and maintain six thousand one hundred")
        self.assertEqual(altitude_change(7000, 3200), "descend and maintain three thousand two hundred")
        self.assertEqual(altitude_change(3150, 3200), "maintain three thousand two hundred")


class VectoringSimulationTest(unittest.TestCase):
    def test_every_start_ends_cleared(self):
        """Fly an aircraft that follows the vectors from 16 directions x 2 ranges: all must get cleared."""
        import math

        from atc.procedures import vector_plan
        p = parse_cifp(find_cifp(str(FIX), "KSJC"))
        ap, thr, mv = p.approaches["I30L"], p.thresholds["30L"], 13.0
        for k in range(16):
            for rng in (10, 22):
                b = k * 22.5
                lat = thr.lat + rng / 60 * math.cos(math.radians(b))
                lon = thr.lon + rng / 60 * math.sin(math.radians(b)) / math.cos(math.radians(thr.lat))
                hdg, alt, ready = (b + 180) % 360, 6000, False
                for _ in range(1500):
                    ready, h, _txt = vector_plan(ap, thr, lat, lon, mv, hdg, alt)
                    if ready:
                        break
                    if abs((h - hdg + 540) % 360 - 180) > 25:
                        hdg = h
                    t = math.radians(hdg + mv)
                    lat += 0.04 * math.cos(t) / 60
                    lon += 0.04 * math.sin(t) / 60 / math.cos(math.radians(lat))
                    alt = max(alt - 60, ap.intercept_alt)
                self.assertTrue(ready, f"not cleared starting {b} deg, {rng} nm")
