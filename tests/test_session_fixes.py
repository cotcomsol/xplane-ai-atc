"""Fixes from the KBZN flight test (2026-11-09)."""
import unittest
from pathlib import Path

from atc.airports import AirportDB, Facility, find_apt_dat, parse_apt_dat
from atc.controller import Controller, clean_reply, parse_approach_request, spoken_facility
from atc.local_stt import fix_make
from atc.main import readback_window
from atc.procedures import find_cifp, parse_cifp
from atc.state import AircraftState

FIX = Path(__file__).parent / "fixtures" / "xp"


def state(**kw):
    base = dict(lat=37.20, lon=-121.75, alt_ft=4000, agl_ft=3900, gs_kt=120, ias_kt=120, hdg_mag=300, vs_fpm=0,
                on_ground=False, com1_khz=121300, com2_khz=0, tx_com=1, xpdr_code=1200,
                xpdr_mode="alt (mode C)", tailnum="N828ZZ", icao_type="SR22", altimeter_inhg=29.92,
                wind_dir=None, wind_kt=0)
    base.update(kw)
    return AircraftState(**base)


class FacilityNameTest(unittest.TestCase):
    def test_ctaf_label_is_not_a_name(self):
        a = {"id": "KBZN", "name": "Bozeman Yellowstone Intl", "city": "Bozeman"}
        self.assertEqual(spoken_facility(Facility(a, "TOWER", "CTAF", 118200, 0)), "Bozeman Tower")
        self.assertEqual(spoken_facility(Facility(a, "TOWER", "BOZEMAN TWR/CTAF", 118200, 0)), "Bozeman Tower")


class RecontactTest(unittest.TestCase):
    def test_tower_starts_over_when_the_pilot_comes_back(self):
        db = AirportDB(parse_apt_dat(find_apt_dat(str(FIX))))
        ctl = Controller(db)
        twr = db.facility_for(118600, 37.46, -122.11)  # Palo Alto Tower
        app = db.facility_for(120100, 37.40, -122.05)  # NorCal
        s = state(lat=37.47, lon=-122.12, com1_khz=118600)
        ctl.situation(s, twr, "N828ZZ", "Palo Alto Tower, Cirrus 828ZZ, ready for departure")
        ctl.remember(twr, "ready", "Cirrus eight two eight Zulu Zulu, runway three one, cleared for takeoff.")
        ctl.remember(twr, "", "Cirrus eight Zulu Zulu, frequency change approved.", event=True)
        self.assertIn(twr.key, ctl.handed_off)
        ctl.situation(s, twr, "N828ZZ", "frequency change approved, Cirrus 8ZZ")  # goodbye: still handed off
        self.assertIn(twr.key, ctl.handed_off)
        ctl.situation(s, app, "N828ZZ", "NorCal Approach, Cirrus 828ZZ, request the ILS")
        sit = ctl.situation(s, twr, "N828ZZ", "Palo Alto Tower, Cirrus 828ZZ, inbound on the ILS")
        self.assertNotIn(twr.key, ctl.handed_off)
        self.assertNotIn("ALREADY HANDED OFF", sit)


class ApproachRequestTest(unittest.TestCase):
    def setUp(self):
        self.db = AirportDB(parse_apt_dat(find_apt_dat(str(FIX))))
        self.ctl = Controller(self.db, xplane_path=str(FIX))
        self.sjc = self.db.by_id("KSJC")
        self.procs = parse_cifp(find_cifp(str(FIX), "KSJC"))
        self.dep = self.db.facility_for(121300, 37.30, -121.85)

    def ctx(self, text, **kw):
        self.ctl.requested_approach = parse_approach_request(text)
        self.ctl.destination = self.sjc
        s = state(**kw)
        procs, rwy, apt = self.ctl._approach_ctx(s, self.dep)
        return s, rwy, self.ctl._request_note(procs, rwy, apt, s)

    def test_parse(self):
        self.assertEqual(parse_approach_request("request vectors for I practice ILS runway 3-0 approach"), ("ILS", "30"))
        self.assertEqual(parse_approach_request("we'll expect the RNAV3-0"), ("RNAV", "30"))
        self.assertEqual(parse_approach_request("request the ILS runway one two right"), ("ILS", "12R"))
        self.assertIsNone(parse_approach_request("request the ILS"))

    def test_requested_approach_to_the_other_end_when_calm(self):
        s, rwy, note = self.ctx("request the ILS runway one two right")
        self.assertEqual(rwy, "12R")
        self.assertIn("approved", note)
        self.assertEqual(self.ctl._runway(self.sjc, s)[0], "12R")  # the tower lands it there too

    def test_requested_approach_refused_with_tailwind(self):
        _, rwy, note = self.ctx("request the ILS runway one two right", wind_dir=300, wind_kt=12)
        self.assertTrue(rwy.startswith("30"))
        self.assertIn("unable", note)

    def test_nonexistent_approach_is_said_and_alternative_offered(self):
        _, rwy, note = self.ctx("request the practice ILS runway three zero right approach")  # ILS is 30L only
        self.assertIn("NO ILS approach to runway 30R", note)
        self.assertIn("ILS runway three zero left", note)  # the runway in use's ILS first
        _, _, note = self.ctx("request the VOR runway three zero")
        self.assertIn("NO VOR approach", note)


class PracticeApproachTest(unittest.TestCase):
    def setUp(self):
        self.db = AirportDB(parse_apt_dat(find_apt_dat(str(FIX))))
        self.ctl = Controller(self.db)
        self.twr = self.db.facility_for(118600, 37.46, -122.11)  # Palo Alto Tower
        self.s = state(lat=37.40, lon=-122.05, com1_khz=118600, alt_ft=2000, agl_ft=1990)

    def test_tower_asks_how_a_practice_approach_ends(self):
        self.ctl.ifr_approach = True  # approach control set it up
        sit = self.ctl.situation(self.s, self.twr, "N828ZZ", "Palo Alto Tower, Cirrus 828ZZ, on the RNAV 31")
        self.assertIn("say intentions after the approach", sit)
        sit = self.ctl.situation(self.s, self.twr, "N828ZZ", "low approach please")
        self.assertNotIn("say intentions after the approach", sit)
        self.assertIn("cleared low approach", sit)

    def test_missed_approach_report(self):
        self.ctl.ifr_approach = True
        self.ctl.situation(self.s, self.twr, "N828ZZ", "Palo Alto Tower, Cirrus 828ZZ, on the RNAV 31")
        sit = self.ctl.situation(self.s, self.twr, "N828ZZ", "Cirrus 828ZZ, missed approach")
        self.assertIn("MISSED APPROACH reported", sit)
        self.assertFalse(self.ctl.ifr_approach)
        self.ctl.remember(self.twr, "Cirrus 828ZZ, missed approach",
                          "Cirrus eight Zulu Zulu, roger, make left traffic runway three one.")
        self.assertFalse(self.ctl.missed_reported)  # answered: the next lap is ordinary pattern work
        sit = self.ctl.situation(self.s, self.twr, "N828ZZ", "left downwind three one")
        self.assertNotIn("MISSED APPROACH", sit)
        self.assertIn("cleared to land", sit)


class ReplyAndTimingTest(unittest.TestCase):
    def test_mangled_silence(self):
        self.assertEqual(clean_reply("<s ilence>"), "<silence>")

    def test_readback_window_grows_with_the_instruction(self):
        short = readback_window("Cirrus eight Zulu Zulu, roger.")
        clearance = readback_window("Cirrus eight Zulu Zulu, eight miles from SITRE, turn left heading three three "
                                    "zero, maintain five thousand seven hundred until established on the final "
                                    "approach course, cleared RNAV runway three zero approach.")
        self.assertLess(short, 5)
        self.assertGreater(clearance, 10)
        self.assertLessEqual(clearance, 14)

    def test_make_mishearings(self):
        self.assertEqual(fix_make("Bozman Ground Series 828 Zulu Zulu", "Cirrus"), "Bozman Ground Cirrus 828 Zulu Zulu")
        self.assertEqual(fix_make("Serious 800", "Cirrus"), "Cirrus 800")
        self.assertEqual(fix_make("a series of calls", "Bonanza"), "a series of calls")  # other makes untouched


if __name__ == "__main__":
    unittest.main()
