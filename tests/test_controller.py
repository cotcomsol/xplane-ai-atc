import unittest
from pathlib import Path

from atc.airports import AirportDB, find_apt_dat, parse_apt_dat
from atc.controller import Controller, clean_reply, runway_in_use, spoken_facility, words_to_code
from atc.state import AircraftState

FIX = Path(__file__).parent / "fixtures" / "xp"


def state(**kw):
    base = dict(lat=37.35, lon=-122.0, alt_ft=3500, agl_ft=3400, gs_kt=100, ias_kt=100, hdg_mag=310, vs_fpm=0,
                on_ground=False, com1_khz=120100, com2_khz=0, tx_com=1, xpdr_code=1200,
                xpdr_mode="alt (mode C)", tailnum="N123AB", icao_type="C172", altimeter_inhg=29.92,
                wind_dir=300, wind_kt=8)
    base.update(kw)
    return AircraftState(**base)


class ControllerTest(unittest.TestCase):
    def setUp(self):
        self.db = AirportDB(parse_apt_dat(find_apt_dat(str(FIX))))
        self.pao = next(a for a in self.db.airports if a["id"] == "KPAO")

    def test_runway_into_wind(self):
        self.assertEqual(runway_in_use(self.pao, 300, 8)[0], "31")
        self.assertEqual(runway_in_use(self.pao, 130, 8)[0], "13")

    def test_runway_calm_prefers_west(self):
        self.assertEqual(runway_in_use(self.pao, None, None)[0], "31")

    def test_spoken_facility(self):
        f = self.db.facility_for(120100, 37.4, -122.1)
        self.assertEqual(spoken_facility(f), "NorCal Approach")
        f = self.db.facility_for(118600, 37.4, -122.1)
        self.assertEqual(spoken_facility(f), "Palo Alto Tower")

    def test_spoken_facility_uses_city(self):
        f = self.db.facility_for(118100, 47.79, 13.0)
        self.assertEqual(spoken_facility(f), "Salzburg Tower")

    def test_clean_reply(self):
        self.assertEqual(clean_reply("Cirrus five, taxi via Bravo. \u0437\u0445\u0443 "), "Cirrus five, taxi via Bravo.")
        self.assertEqual(clean_reply("**Roger**, <silence>"), "Roger, <silence>")

    def test_words_to_code(self):
        self.assertEqual(words_to_code("Cessna three alpha bravo, squawk four six two one."), "4621")
        self.assertEqual(words_to_code("squawk 0412"), "0412")
        self.assertIsNone(words_to_code("squawk VFR"))

    def test_squawk_flow(self):
        ctl = Controller(self.db)
        s = state()
        fac = self.db.facility_for(s.tx_khz, s.lat, s.lon)
        self.assertIn("NEXT SQUAWK CODE", ctl.situation(s, fac, "N123AB"))
        ctl.remember(fac, "request flight following", "Cessna one two three alpha bravo, squawk four six two one")
        self.assertIn("NOT radar identified", ctl.situation(s, fac, "N123AB"))
        self.assertIn("radar identified", ctl.situation(state(xpdr_code=4621), fac, "N123AB"))
        self.assertNotIn("NOT", ctl.situation(state(xpdr_code=4621), fac, "N123AB").split("Squawk:")[1])


if __name__ == "__main__":
    unittest.main()


class EventTest(unittest.TestCase):
    def setUp(self):
        self.db = AirportDB(parse_apt_dat(find_apt_dat(str(FIX))))
        self.ctl = Controller(self.db)

    def test_handoff_after_takeoff_fires_once(self):
        s = state(com1_khz=118600, lat=37.4611, lon=-122.115, alt_ft=500, on_ground=False)
        fac = self.db.facility_for(118600, s.lat, s.lon)
        self.ctl.remember(fac, "ready for departure", "Cessna three alpha bravo, runway three one, cleared for takeoff")
        self.assertIsNone(self.ctl.pending_event(s, fac))  # still close and low
        far = state(com1_khz=118600, lat=37.50, lon=-122.16, alt_ft=1500, on_ground=False)
        self.assertIn("frequency change", self.ctl.pending_event(far, fac))
        self.assertIsNone(self.ctl.pending_event(far, fac))

    def test_no_handoff_if_tower_already_did_it(self):
        fac = self.db.facility_for(118600, 37.46, -122.11)
        self.ctl.remember(fac, "x", "Cessna three alpha bravo, cleared for takeoff")
        self.ctl.remember(fac, "x", "Cessna three alpha bravo, frequency change approved")
        self.assertIsNone(self.ctl.pending_event(state(com1_khz=118600, lat=37.6, lon=-122.3), fac))

    def test_radar_contact_when_code_set(self):
        s = state()
        fac = self.db.facility_for(s.tx_khz, s.lat, s.lon)
        self.ctl.remember(fac, "request flight following", "Cessna one two three alpha bravo, squawk four six two one")
        self.assertIsNone(self.ctl.pending_event(s, fac))
        self.assertIn("radar contact", self.ctl.pending_event(state(xpdr_code=4621), fac))
        self.assertIsNone(self.ctl.pending_event(state(xpdr_code=4621), fac))

    def test_no_events_without_contact(self):
        s = state(xpdr_code=4621)
        self.ctl.assigned_code = "4621"
        self.assertIsNone(self.ctl.pending_event(s, self.db.facility_for(s.tx_khz, s.lat, s.lon)))


class StateTest(unittest.TestCase):
    def test_airborne_uses_surface_wind_and_computes_dew(self):
        v = {"lat": 37.0, "lon": -122.0, "on_ground": 0, "wind_dir": 250.0, "wind_kt": 20.0,
             "wind_sfc_dir": [300.0, 280.0], "wind_sfc_ms": [5.0, 15.0], "temp_sl_c": 20.0, "rh_sl_pct": 50.0}
        s = AircraftState.from_values(v)
        self.assertEqual((s.wind_dir, s.wind_kt), (300, 10))
        self.assertAlmostEqual(s.dew_sl_c, 9.3, delta=0.2)

    def test_on_ground_uses_local_wind(self):
        s = AircraftState.from_values({"on_ground": 1, "wind_dir": 250.0, "wind_kt": 5.0,
                                       "wind_sfc_dir": [300.0], "wind_sfc_ms": [9.0]})
        self.assertEqual((s.wind_dir, s.wind_kt), (250, 10))


class AirspaceTest(unittest.TestCase):
    def apt(self, ident, freqs, elev=60):
        return {"id": ident, "name": ident, "elev": elev, "type": "land", "lat": 37.36, "lon": -121.93,
                "runways": [], "freqs": freqs}

    def test_classes(self):
        from atc.airspace import airspace_class
        twr = [{"type": "TOWER", "khz": 124000, "name": "TWR"}]
        self.assertEqual(airspace_class(self.apt("KSFO", twr)), "B")
        self.assertEqual(airspace_class(self.apt("KLGA", twr)), "B")  # hand-added multi-airport Class B
        self.assertEqual(airspace_class(self.apt("KSJC", twr)), "C")
        self.assertEqual(airspace_class(self.apt("KPAO", twr)), "D")
        self.assertEqual(airspace_class(self.apt("XFIX", [])), "G/E")

    def test_departure_flow(self):
        from atc.airports import Facility
        freqs = [{"type": "DELIVERY", "khz": 118000, "name": "CLNC DEL"},
                 {"type": "GROUND", "khz": 121700, "name": "GND"},
                 {"type": "TOWER", "khz": 124000, "name": "TWR"},
                 {"type": "CTAF", "khz": 124000, "name": "CTAF"},
                 {"type": "DEPARTURE", "khz": 121300, "name": "NORCAL DEP"}]
        a = self.apt("KSJC", freqs)
        ctl = Controller(None)
        gnd = ctl._departure_flow(a, Facility(a, "GROUND", "GND", 121700, 0), "C")
        self.assertIn("contact clearance delivery", gnd)
        dlv = ctl._departure_flow(a, Facility(a, "DELIVERY", "CLNC DEL", 118000, 0), "C")
        self.assertIn("maintain VFR at or below 2000, departure frequency NorCal Departure 121.3, squawk", dlv)
        ctl.assigned_code = "4601"
        self.assertIn("already has squawk 4601", ctl._departure_flow(a, Facility(a, "GROUND", "GND", 121700, 0), "C"))
        self.assertNotIn("ctaf", Controller._airport_block(a))


class AtisTrackingTest(unittest.TestCase):
    def test_reported_letter_is_tracked_per_airport(self):
        db = AirportDB(parse_apt_dat(find_apt_dat(str(FIX))))
        ctl = Controller(db)
        s = state(com1_khz=125000, lat=37.4611, lon=-122.115, on_ground=True, alt_ft=7)
        gnd = db.facility_for(125000, s.lat, s.lon)
        self.assertIn("pilot has NOT reported it", ctl.situation(s, gnd, "N123AB"))
        ctl.remember(gnd, "Palo Alto ground, Cessna one two three alpha bravo, ramp, with information Zulu", "x")
        sit = ctl.situation(s, gnd, "N123AB")
        self.assertTrue("pilot has reported it" in sit or "OLD letter (Zulu)" in sit)


class DepartureTurnTest(unittest.TestCase):
    def test_turns(self):
        from atc.controller import departure_turn
        self.assertEqual(departure_turn("ready for departure, southbound", 300), "left turn southbound approved")
        self.assertEqual(departure_turn("departing to the east", 300), "right turn eastbound approved")
        self.assertEqual(departure_turn("northwest bound", 300), "straight out departure approved")
        self.assertIsNone(departure_turn("ready for departure", 300))


class ArrivalTest(unittest.TestCase):
    def setUp(self):
        self.db = AirportDB(parse_apt_dat(find_apt_dat(str(FIX))))
        self.ctl = Controller(self.db)

    def test_destination_parsing(self):
        from atc.controller import find_destination
        app = self.db.facility_for(135650, 37.51, -122.25)  # NorCal via San Carlos
        self.assertEqual(find_destination(self.db, "request flight following VFR to Palo Alto", app)["id"], "KPAO")
        self.assertIsNone(find_destination(self.db, "taxi to runway three one", app))
        twr = self.db.facility_for(118600, 37.46, -122.11)
        self.assertEqual(find_destination(self.db, "ten miles south inbound full stop", twr)["id"], "KPAO")

    def test_handoff_to_destination_tower(self):
        app = self.db.facility_for(120100, 37.40, -122.05)
        self.ctl.remember(app, "flight following VFR to Palo Alto", "squawk four six two one")
        far = state(lat=37.25, lon=-121.95)  # ~14 nm
        self.assertIsNone(self.ctl.pending_event(far, app))
        near = state(lat=37.35, lon=-122.02)  # ~8 nm
        ev = self.ctl.pending_event(near, app)
        self.assertIn("contact Palo Alto Tower 118.6", ev)
        self.assertIsNone(self.ctl.pending_event(near, app))

    def test_landing_exit_and_parking(self):
        twr = self.db.facility_for(118300, 40.0, -99.996)  # XTXI tower
        self.ctl.remember(twr, "inbound full stop", "cleared to land runway one eight")
        self.ctl.pending_event(state(lat=40.02, lon=-100.0, on_ground=False, com1_khz=118300), twr)
        rolling = state(lat=40.004, lon=-100.0, on_ground=True, gs_kt=30, hdg_mag=180, com1_khz=118300)
        ev = self.ctl.pending_event(rolling, twr)
        self.assertIn("turn left at Alpha one", ev)  # next exit ahead (south); A1 heads east = left
        self.assertIn("contact ground 121.7", ev)
        self.assertIsNone(self.ctl.pending_event(rolling, twr))
        gnd = self.db.facility_for(121700, 40.0, -99.996)
        sit = self.ctl.situation(state(lat=40.01, lon=-99.998, on_ground=True, com1_khz=121700), gnd, "N123AB",
                                 "clear of the runway, taxi to parking")
        self.assertIn("TAXI ROUTE to parking", sit)


class AckTest(unittest.TestCase):
    def test_chases_unacknowledged_instruction_once(self):
        import atc.controller as c
        db = AirportDB(parse_apt_dat(find_apt_dat(str(FIX))))
        ctl = Controller(db)
        s = state()
        fac = db.facility_for(s.tx_khz, s.lat, s.lon)
        ctl.remember(fac, "request flight following", "Cessna one two three alpha bravo, squawk four six two one")
        self.assertIsNone(ctl.pending_event(s, fac))
        old = c.ACK_TIMEOUT_S
        c.ACK_TIMEOUT_S = -1
        try:
            self.assertIn("how do you hear", ctl.pending_event(s, fac))
            self.assertNotIn("how do you hear", ctl.pending_event(s, fac) or "")
            ctl.remember(fac, "squawk four six two one", "<silence>")
            self.assertIsNone(ctl.awaiting_ack)
        finally:
            c.ACK_TIMEOUT_S = old


class IfrHelpersTest(unittest.TestCase):
    def test_parse_altitude(self):
        from atc.controller import parse_altitude
        self.assertEqual(parse_altitude("IFR to Monterey at seven thousand"), 7000)
        self.assertEqual(parse_altitude("request one zero thousand"), 10000)
        self.assertEqual(parse_altitude("filed for 8,500 feet"), 8500)
        self.assertEqual(parse_altitude("four thousand five hundred"), 4500)
        self.assertIsNone(parse_altitude("runway three zero left"))

    def test_ifr_cruise_hemispheric(self):
        from atc.controller import ifr_cruise
        self.assertEqual(ifr_cruise(60, 140), 5000)  # eastbound: odd
        self.assertEqual(ifr_cruise(60, 300), 4000)  # westbound: even
        self.assertEqual(ifr_cruise(5000, 90), 9000)

    def test_ifr_clearance_facts(self):
        db = AirportDB(parse_apt_dat(find_apt_dat(str(FIX))))
        ctl = Controller(db)
        s = state(com1_khz=125000, lat=37.4611, lon=-122.115, on_ground=True, alt_ft=7)
        gnd = db.facility_for(125000, s.lat, s.lon)
        sit = ctl.situation(s, gnd, "N123AB", "Palo Alto ground, Cessna one two three alpha bravo, IFR to San Carlos "
                                              "at four thousand, ready to copy")
        self.assertIn("cleared to San Carlos airport via radar vectors", sit)
        self.assertIn("expect four thousand", sit)
        ctl.remember(gnd, "x", "Cessna one two three alpha bravo, cleared to San Carlos airport via radar vectors, "
                                "then as filed, maintain three thousand, expect four thousand one zero minutes after "
                                "departure, departure frequency one two zero point one, squawk four six two one")
        self.assertIsNotNone(ctl.ifr_clearance)
        self.assertIn("readback correct", ctl.situation(s, gnd, "N123AB", "cleared to San Carlos ..."))


class ConformanceTest(unittest.TestCase):
    def test_verify_altitude_when_level_off_assignment(self):
        import atc.controller as c
        db = AirportDB(parse_apt_dat(find_apt_dat(str(FIX))))
        ctl = Controller(db)
        s = state(alt_ft=5000, vs_fpm=0)
        fac = db.facility_for(s.tx_khz, s.lat, s.lon)
        ctl.remember(fac, "x", "Cessna three alpha bravo, climb and maintain six thousand one hundred")
        ctl.remember(fac, "climb and maintain six thousand one hundred", "<silence>")
        self.assertEqual(ctl.assigned_alt, 6100)
        old = c.ALT_CONFORMANCE_S
        c.ALT_CONFORMANCE_S = -1
        try:
            self.assertIsNone(ctl.pending_event(state(alt_ft=5500, vs_fpm=500), fac))  # climbing: fine
            ctl.pending_event(s, fac)  # starts the timer
            self.assertIn("verify climbing to six thousand one hundred", ctl.pending_event(s, fac))
        finally:
            c.ALT_CONFORMANCE_S = old
