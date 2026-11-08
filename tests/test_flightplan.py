import unittest
from pathlib import Path

from atc.airports import AirportDB, find_apt_dat, parse_apt_dat
from atc.controller import Controller, altitude_words
from atc.flightplan import FlightPlan, parse_simbrief, spoken_procedure
from atc.procedures import find_cifp, parse_cifp
from atc.state import AircraftState

FIX = Path(__file__).parent / "fixtures" / "xp"
SJC = dict(lat=37.3626, lon=-121.9291)


def ofp(route: str, navlog: list[tuple[str, str, str, str]], cruise="9000", status="Success", origin="KSJC",
        dest="KSQL", atc_route: str = "") -> dict:
    """A trimmed SimBrief OFP (json=1) with the fields we read. navlog: (ident, type, via_airway, stage)."""
    procs = {t for t in (atc_route or route).split() if t[-1].isdigit() or (len(t) > 2 and t[-2].isdigit())}
    return {
        "fetch": {"status": status},
        "params": {"time_generated": "1792045800"},
        "general": {"route": route, "initial_altitude": cruise},
        "origin": {"icao_code": origin, "plan_rwy": "30L"},
        "destination": {"icao_code": dest},
        "alternate": {"icao_code": "KOAK"},
        "atc": {"callsign": "N123AB", "route": atc_route or {}, "initial_alt": {}},
        "aircraft": {"icaocode": "C172"},
        "navlog": {"fix": [{"ident": i, "type": ty, "via_airway": via, "stage": st,
                            "is_sid_star": "1" if via in procs else "0"} for i, ty, via, st in navlog]},
    }


def ground_state(**kw) -> AircraftState:
    base = dict(lat=SJC["lat"], lon=SJC["lon"], alt_ft=62, agl_ft=0, gs_kt=0, ias_kt=0, hdg_mag=300, vs_fpm=0,
                on_ground=True, com1_khz=118000, com2_khz=0, tx_com=1, xpdr_code=1200,
                xpdr_mode="alt (mode C)", tailnum="N123AB", icao_type="C172", altimeter_inhg=29.92,
                wind_dir=300, wind_kt=8)
    base.update(kw)
    return AircraftState(**base)


class ParseTest(unittest.TestCase):
    def test_us_rnav_sid_with_transition(self):
        p = parse_simbrief(ofp("LOUPE1 HRNER V25 SNS", [
            ("LOUPE", "wpt", "LOUPE1", "CLB"), ("HRNER", "wpt", "LOUPE1", "CLB"),
            ("SNS", "vor", "V25", "CRZ")]))
        self.assertEqual((p.origin, p.destination, p.alternate), ("KSJC", "KSQL", "KOAK"))
        self.assertEqual((p.sid, p.sid_exit, p.star, p.first_fix), ("LOUPE1", "HRNER", "", ""))
        self.assertEqual(p.cruise_ft, 9000)
        self.assertEqual(p.route, "LOUPE1 HRNER V25 SNS")

    def test_icao_route_from_atc_block_with_star(self):
        d = ofp("DCT ULGOT N746 ABETI DCT", [
            ("TLL", "vor", "ULGOT1G", "CLB"), ("ULGOT", "wpt", "ULGOT1G", "CLB"),
            ("ABETI", "wpt", "N746", "DSC"), ("RIX", "vor", "ABETI2A", "DSC")], cruise="36000",
            origin="EETN", dest="EVRA", atc_route="ULGOT1G ULGOT N746 ABETI ABETI2A")  # general.route lacks SID/STAR
        p = parse_simbrief(d)
        self.assertEqual((p.sid, p.sid_exit, p.star), ("ULGOT1G", "ULGOT", "ABETI2A"))
        self.assertEqual(p.cruise_ft, 36000)

    def test_speed_level_groups_are_not_fixes(self):
        # a real OFP (KBIL-KGEG): atc.route starts with the speed/level group
        p = parse_simbrief(ofp("DCT BIL DCT ARMTY DCT HLN DCT MLP HILIE3", [
            ("BIL", "vor", "DCT", "CLB"), ("ARMTY", "wpt", "DCT", "CRZ"), ("HILIE", "wpt", "HILIE3", "DSC")],
            cruise="14000", origin="KBIL", dest="KGEG",
            atc_route="N0193F140 DCT BIL DCT ARMTY DCT HLN/N0190F160 DCT MLP HILIE3"))
        self.assertEqual(p.route, "BIL ARMTY HLN MLP HILIE3")
        self.assertEqual((p.sid, p.first_fix, p.star, p.cruise_ft), ("", "BIL", "HILIE3", 14000))

    def test_no_sid_first_fix(self):
        p = parse_simbrief(ofp("DCT SUNOL V334 SNS", [("SUNOL", "wpt", "DCT", "CLB"), ("SNS", "vor", "V334", "CRZ")]))
        self.assertEqual((p.sid, p.first_fix), ("", "SUNOL"))

    def test_error_status(self):
        with self.assertRaises(ValueError):
            parse_simbrief({"fetch": {"status": "Error: Unknown UserID"}})

    def test_spoken_procedure(self):
        self.assertEqual(spoken_procedure("ALMDN4"), "ALMDN four")
        self.assertEqual(spoken_procedure("ULGOT1G"), "ULGOT one Golf")

    def test_flight_level_words(self):
        self.assertEqual(altitude_words(36000), "flight level three six zero")
        self.assertEqual(altitude_words(9000), "niner thousand")


class SidTest(unittest.TestCase):
    def test_sids_and_runways(self):
        p = parse_cifp(find_cifp(str(FIX), "KSJC"))
        self.assertIn("HRNER", p.sids["LOUPE1"].transitions)
        self.assertTrue(p.sids["LOUPE1"].serves("30L"))
        self.assertFalse(p.sids["ALMDN4"].serves("30L"))
        self.assertTrue(p.sids["ALMDN4"].serves("12R"))  # RW12B = both parallels


class ClearanceTest(unittest.TestCase):
    def setUp(self):
        self.db = AirportDB(parse_apt_dat(find_apt_dat(str(FIX))))
        self.ctl = Controller(self.db, xplane_path=str(FIX))
        self.s = ground_state()
        self.dlv = self.db.facility_for(118000, self.s.lat, self.s.lon)
        self.assertEqual((self.dlv.airport["id"], self.dlv.kind), ("KSJC", "DELIVERY"))

    def clearance(self, route, navlog, text="San Jose clearance, Skyhawk one two three alpha bravo, ready to copy"):
        self.ctl.set_flight_plan(parse_simbrief(ofp(route, navlog)))
        return self.ctl.situation(self.s, self.dlv, "N123AB", text)

    def test_filed_sid_and_transition(self):
        sit = self.clearance("LOUPE1 HRNER V25 SNS", [("LOUPE", "wpt", "LOUPE1", "CLB"),
                                                     ("HRNER", "wpt", "LOUPE1", "CLB")])
        self.assertEqual(self.ctl.flight_rules, "IFR")  # "ready to copy" with a plan on file
        self.assertIn("FILED FLIGHT PLAN (SimBrief", sit)
        self.assertIn("cleared to San Carlos airport via the LOUPE one departure, HRNER transition, then as filed", sit)
        self.assertIn("expect niner thousand", sit)

    def test_sid_for_other_runway_is_amended(self):
        sit = self.clearance("ALMDN4 HRNER V25 SNS", [("ALMDN", "wpt", "ALMDN4", "CLB"),
                                                     ("HRNER", "wpt", "ALMDN4", "CLB")])
        self.assertIn("via radar vectors HRNER, then as filed", sit)
        self.assertIn("doesn't serve runway 30", sit)

    def test_no_sid_vectors_to_first_fix_then_direct(self):
        sit = self.clearance("SUNOL V334 SNS", [("SUNOL", "wpt", "DCT", "CLB")])
        self.assertIn("via radar vectors SUNOL, then as filed", sit)
        self.ctl.remember(self.dlv, "x", "Skyhawk one two three alpha bravo, cleared to San Carlos airport via radar "
                          "vectors SUNOL, then as filed, maintain three thousand, expect niner thousand one zero "
                          "minutes after departure, departure frequency one two one point three, squawk four six "
                          "two one")
        self.assertEqual(self.ctl._climb_phrase(self.ctl.ifr_clearance),
                         "climb and maintain niner thousand, cleared direct SUNOL")

    def test_plan_from_another_airport_is_ignored(self):
        self.ctl.set_flight_plan(parse_simbrief(ofp("LOUPE1 HRNER", [("LOUPE", "wpt", "LOUPE1", "CLB")],
                                                    origin="KOAK")))
        sit = self.ctl.situation(self.s, self.dlv, "N123AB", "San Jose clearance, Skyhawk one two three alpha "
                                                             "bravo, IFR to San Carlos, ready to copy")
        self.assertNotIn("FILED FLIGHT PLAN", sit)
        self.assertIn("via radar vectors, then as filed", sit)


if __name__ == "__main__":
    unittest.main()
