import base64
import unittest

from atc.state import AircraftState, Target
from atc.traffic import TrafficWatch, advisory, clock, facts


def tgt(idx=1, brg=60, d=3.2, ralt=200, alt=3500, hdg=95, typ="C172", lat=None, lon=None):
    return Target(idx, brg, d, ralt, alt, hdg, typ, lat, lon)


class TrafficTest(unittest.TestCase):
    def test_clock(self):
        self.assertEqual(clock(0), "twelve o'clock")
        self.assertEqual(clock(62), "two o'clock")
        self.assertEqual(clock(-90), "nine o'clock")
        self.assertEqual(clock(350), "twelve o'clock")

    def test_advisory_wording(self):
        self.assertEqual(advisory(tgt()), "traffic, two o'clock, three miles, eastbound, Cessna, "
                                          "altitude indicates three thousand five hundred")

    def test_watch_calls_closing_target_once(self):
        w = TrafficWatch()
        self.assertIsNone(w.due([tgt(d=5.5)]))  # first sample: can't tell if closing
        self.assertEqual(w.due([tgt(d=5.0)]).idx, 1)  # closing
        self.assertIsNone(w.due([tgt(d=4.5)]))  # already called
        self.assertIsNone(w.due([tgt(idx=2, d=4.0, ralt=3000)]))  # too far above
        self.assertIsNone(w.due([tgt(d=4.2)]))  # flickered out for a sample: still not called again
        w.due([tgt(d=10)])  # moved well clear: re-armed
        self.assertEqual(w.due([tgt(d=5.8)]).idx, 1)

    def test_from_xplane_arrays(self):
        types = b"\x00" * 8 + b"C172\x00\x00\x00\x00" + b"\x00" * 8
        v = {"on_ground": 0, "tcas_n": 2, "tcas_brg": [0, 90.0, 0], "tcas_dist_m": [0, 3704.0, 0],
             "tcas_ralt_m": [0, 152.4, 0], "tcas_ele_m": [0, 1066.8, 0], "tcas_psi": [0, 180.0, 0],
             "tcas_type": types}
        s = AircraftState.from_values(v)
        self.assertEqual(len(s.traffic), 1)
        t = s.traffic[0]
        self.assertAlmostEqual(t.dist_nm, 2.0, places=2)
        self.assertEqual((t.icao_type, round(t.rel_alt_ft), round(t.alt_ft)), ("C172", 500, 3500))

    def test_facts(self):
        apt = {"id": "KPAO", "lat": 37.46, "lon": -122.11}
        f = facts([tgt(lat=37.43, lon=-122.11)], apt, 37.4, -122.0)
        self.assertIn("Cessna: from you two o'clock, 3.2 nm, 3500 ft", f)
        self.assertIn("nm south of KPAO", f)
        self.assertIn("none nearby", facts([], apt, 37.4, -122.0))
