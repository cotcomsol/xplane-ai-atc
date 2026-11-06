"""Fixes from the KBIL flight test (2026-11-06)."""

import unittest
from pathlib import Path

from atc.airports import AirportDB, find_apt_dat, parse_apt_dat
from atc.controller import Controller, instruction_altitude, parse_heading, runway_end_position
from atc.state import AircraftState

FIX = Path(__file__).parent / "fixtures" / "xp"


def state(**kw):
    base = dict(lat=37.35, lon=-122.0, alt_ft=3500, agl_ft=3400, gs_kt=100, ias_kt=100, hdg_mag=310, vs_fpm=0,
                on_ground=False, com1_khz=120100, com2_khz=0, tx_com=1, xpdr_code=1200,
                xpdr_mode="alt (mode C)", tailnum="N123AB", icao_type="C172", altimeter_inhg=29.92,
                wind_dir=300, wind_kt=8)
    base.update(kw)
    return AircraftState(**base)


class HelpersTest(unittest.TestCase):
    def test_parse_heading(self):
        self.assertEqual(parse_heading("Cirrus zero Charlie Romeo, turn right heading one zero zero, vectors"), 100)
        self.assertEqual(parse_heading("fly heading zero five five"), 55)
        self.assertIsNone(parse_heading("fly runway heading, runway two eight right, cleared for takeoff"))
        self.assertIsNone(parse_heading("maintain seven thousand"))

    def test_instruction_altitude(self):
        self.assertEqual(instruction_altitude("turn right heading zero five five, climb and maintain seven thousand"),
                         7000)
        self.assertIsNone(instruction_altitude("maintain six thousand until established on the localizer"))
        self.assertIsNone(instruction_altitude("maintain VFR at or below five thousand five hundred"))


class FlightFixesTest(unittest.TestCase):
    def setUp(self):
        self.db = AirportDB(parse_apt_dat(find_apt_dat(str(FIX))))
        self.ctl = Controller(self.db)
        self.gnd = self.db.facility_for(121700, 40.0, -99.996)  # XTXI ground
        self.twr = self.db.facility_for(118300, 40.0, -99.996)

    def test_hangars_and_departure_gets_runway_route(self):
        s = state(lat=40.0005, lon=-99.9895, on_ground=True, gs_kt=0, com1_khz=121700, wind_dir=180, wind_kt=10)
        sit = self.ctl.situation(s, self.gnd, "N123AB",
                                 "east end GA hangars, ready to taxi for VFR departure westbound")
        self.assertIn("TAXI ROUTE to runway", sit)
        self.assertNotIn("TAXI ROUTE to parking", sit)
        sit = self.ctl.situation(s, self.gnd, "N123AB", "request taxi to parking")
        self.assertIn("TAXI ROUTE to parking", sit)

    def test_holding_short_position(self):
        a = self.gnd.airport
        self.assertIn("approach end of runway 18", runway_end_position(a, 40.0102, -99.9985))
        self.assertIn("on runway 18", runway_end_position(a, 40.0098, -100.0))
        self.assertIsNone(runway_end_position(a, 40.0005, -99.9895))
        sit = self.ctl.situation(state(lat=40.0102, lon=-99.9985, on_ground=True, gs_kt=0, com1_khz=118300,
                                       wind_dir=180, wind_kt=10), self.twr, "N123AB", "holding short runway one eight")
        self.assertIn("Aircraft position: at the approach end of runway 18", sit)

    def test_handed_off_readback_is_silent(self):
        self.ctl.remember(self.twr, "ready for takeoff", "Cessna three alpha bravo, cleared for takeoff")
        self.ctl.remember(self.twr, "<EVENT>", "Cessna three alpha bravo, contact Taxitown Departure one two zero "
                          "point five.", event=True)
        sit = self.ctl.situation(state(lat=40.02, lon=-100.0, com1_khz=118300), self.twr, "N123AB",
                                 "departure now, see ya")
        self.assertIn("ALREADY HANDED OFF", sit)
        self.assertIn("contact Taxitown Departure one two zero point five", sit)

    def test_vector_heading_remembered(self):
        app = self.db.facility_for(120100, 37.40, -122.05)
        self.ctl.remember(app, "request vectors ILS", "Cirrus zero Charlie Romeo, turn right heading zero five five, "
                          "vectors ILS runway three one final approach course, climb and maintain seven thousand.")
        self.assertEqual(self.ctl.assigned_hdg, 55)
        self.assertEqual(self.ctl.assigned_alt, 7000)
        sit = self.ctl.situation(state(alt_ft=5900, vs_fpm=900, hdg_mag=340), app, "N123AB", "right zero five five")
        self.assertIn("climbing toward it (fine)", sit)
        self.assertNotIn("MISMATCH", sit)
        self.assertIn("still turning toward it", sit)
        self.ctl.remember(app, "x", "Cirrus zero Charlie Romeo, resume own navigation.")
        self.assertIsNone(self.ctl.assigned_hdg)

    def test_level_off_mismatch_still_flagged(self):
        app = self.db.facility_for(120100, 37.40, -122.05)
        self.ctl.remember(app, "x", "Cessna three alpha bravo, climb and maintain seven thousand.")
        sit = self.ctl.situation(state(alt_ft=5900, vs_fpm=0), app, "N123AB", "level five thousand nine hundred")
        self.assertIn("MISMATCH", sit)

    def test_readback_check(self):
        self.ctl.assigned_hdg, self.ctl.assigned_alt = 100, 7000
        self.assertIn("heading 100", self.ctl._readback_check("Right turn one zero zero zero zero Charlie Romeo."))
        self.assertIn("altitude 7000", self.ctl._readback_check("Seven Thousand Sierra Zero Charlie Romeo"))
        self.assertIn("heading 100 and altitude 7000",
                      self.ctl._readback_check("right 100, maintain 7,000, zero Charlie Romeo"))
        self.assertIsNone(self.ctl._readback_check("right turn one one zero"))

    def test_vector_already_issued(self):
        self.ctl.assigned_hdg, self.ctl.assigned_alt = 100, 7000
        block = ('radar pattern phase: downwind. VECTOR NOW: "turn right heading one zero zero, vectors ILS runway '
                 'two eight right final approach course, maintain seven thousand". When it is in position')
        out = self.ctl._vector_already_issued(block)
        self.assertIn("VECTOR ALREADY ISSUED (heading 100, 7000)", out)
        self.ctl.assigned_hdg = 55
        self.assertIn("VECTOR NOW", self.ctl._vector_already_issued(block))


class LocalVoiceTextTest(unittest.TestCase):
    def test_prepare_and_clauses(self):
        from atc.local_tts import clauses, prepare_text
        self.assertEqual(prepare_text("cleared ILS runway two eight, IFR, VOR/DME, RNAV"),
                         "cleared I L S runway two eight, I F R, V O R D M E, RNAV")
        self.assertEqual(clauses("Cirrus zero Charlie Romeo, roger."), ["Cirrus zero Charlie Romeo, roger."])
        self.assertEqual(clauses("Cirrus zero Charlie Romeo, turn right heading one niner zero, descend and maintain "
                                 "six thousand."),
                         ["Cirrus zero Charlie Romeo,", "turn right heading one niner zero,",
                          "descend and maintain six thousand."])


if __name__ == "__main__":
    unittest.main()
