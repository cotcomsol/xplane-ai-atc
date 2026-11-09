"""Regression tests for the problems found in the code scan."""

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from atc import config as config_mod
from atc import debrief
from atc.airports import AirportDB, Facility, find_apt_dat, parse_apt_dat
from atc.atis import info_letter
from atc.controller import ACK_CHASE, Controller, _airport_names, instruction_altitude, parse_altitude
from atc.openrouter import OpenRouter
from atc.state import AircraftState, Target
from atc.xplane import XPlane, _vnum

FIX = Path(__file__).parent / "fixtures" / "xp"


def state(**kw):
    base = dict(lat=37.35, lon=-122.0, alt_ft=3500, agl_ft=3400, gs_kt=100, ias_kt=100, hdg_mag=310, vs_fpm=0,
                on_ground=False, com1_khz=120100, com2_khz=0, tx_com=1, xpdr_code=1200,
                xpdr_mode="alt (mode C)", tailnum="N123AB", icao_type="C172", altimeter_inhg=29.92,
                wind_dir=300, wind_kt=8)
    base.update(kw)
    return AircraftState(**base)


class ControllerFixes(unittest.TestCase):
    def setUp(self):
        self.db = AirportDB(parse_apt_dat(find_apt_dat(str(FIX))))
        self.ctl = Controller(self.db)
        self.sjc, self.pao = self.db.by_id("KSJC"), self.db.by_id("KPAO")
        self.app = self.db.facility_for(120100, 37.40, -122.05)
        self.twr = self.db.facility_for(118600, 37.46, -122.11)  # Palo Alto Tower

    def test_one_radar_controller_per_frequency(self):
        # 121.3 listed as San Jose's departure frequency and as Palo Alto's approach: the same controller
        near_sjc = Facility(self.sjc, "DEPARTURE", "NORCAL DEP", 121300, 2.0)
        near_pao = Facility(self.pao, "APPROACH", "NORCAL APP", 121300, 1.0)
        self.assertEqual(near_sjc.key, near_pao.key)
        self.ctl.remember(near_sjc, "with you", "Cessna three alpha bravo, NorCal Departure, roger")
        self.assertEqual(len(self.ctl._hist(near_pao)), 2)
        self.assertNotEqual(Facility(self.sjc, "TOWER", "TWR", 124000, 1).key,
                            Facility(self.pao, "TOWER", "TWR", 124000, 1).key)

    def test_radar_contact_is_not_a_handoff(self):
        self.ctl.remember(self.app, "request the ILS", "Cessna three alpha bravo, radar contact, five miles south "
                          "of San Jose, expect ILS runway three zero left approach")
        self.assertNotIn(self.app.key, self.ctl.handed_off)
        self.ctl.remember(self.app, "x", "Cessna three alpha bravo, radar contact")
        self.assertIsNone(self.ctl.awaiting_ack)
        self.ctl.remember(self.app, "x", "Cessna three alpha bravo, contact San Jose Tower one two four point zero")
        self.assertIn(self.app.key, self.ctl.handed_off)

    def test_ack_chase_is_made_once(self):
        said = "Cessna three alpha bravo, descend and maintain three thousand"
        self.ctl.remember(self.app, "hello", said)
        chases = 0
        for _ in range(4):
            self.ctl.awaiting_ack = (time.monotonic() - 30, self.app.key, said)
            ev = self.ctl.pending_event(state(), self.app)
            if ev and ev.startswith(ACK_CHASE):
                chases += 1
                self.ctl.remember(self.app, ev, said, event=True)  # the model repeats the instruction
        self.assertEqual(chases, 1)

    def test_flight_levels(self):
        self.assertEqual(instruction_altitude("climb and maintain flight level three six zero"), 36000)
        self.assertEqual(instruction_altitude("maintain five thousand, expect flight level three six zero one zero "
                                              "minutes after departure"), 5000)
        self.assertEqual(parse_altitude("climbing to FL230"), 23000)
        self.ctl.remember(self.app, "x", "Citation one two three, maintain five thousand")
        self.ctl.remember(self.app, "x", "Citation one two three, climb and maintain flight level two three zero")
        self.assertEqual(self.ctl.assigned_alt, 23000)

    def test_squawk_is_not_an_altitude(self):
        self.assertEqual(parse_altitude("squawking 1200 at 4500"), 4500)
        self.assertIsNone(parse_altitude("squawking 1200"))
        self.assertEqual(parse_altitude("with you at 4,500"), 4500)

    def test_tcas_transponder_modes_report_altitude(self):
        for mode in (3, 6, 7):
            self.assertIn("alt", AircraftState.from_values({"xpdr_mode": mode}).xpdr_mode)
        self.assertNotIn("alt", AircraftState.from_values({"xpdr_mode": 2}).xpdr_mode)

    def test_missed_approach_starts_a_new_approach(self):
        c = self.ctl
        c._clearance_prompted = c.ifr_approach = c.approach_cleared = c.landing_cleared = True
        c._lap = {"clr_chased": True, "short_final": True}
        c.observe("going missed", self.app)
        self.assertFalse(c._clearance_prompted)
        self.assertFalse(c.landing_cleared)
        self.assertEqual(c._lap, {"go_around_reported": True})

    def test_full_stop_ends_the_flight_but_a_touch_and_go_does_not(self):
        c = self.ctl
        c.flight_rules, c.ifr_clearance, c.assigned_code, c.destination = "IFR", {"dest": self.sjc}, "4512", self.sjc
        c._rollout = True  # just touched down
        c._track_phase(state(on_ground=True, gs_kt=60))
        self.assertEqual(c.flight_rules, "IFR")
        c._track_phase(state(on_ground=False, gs_kt=70))  # touch and go: airborne again
        c._track_phase(state(on_ground=True, gs_kt=10))
        self.assertIsNotNone(c.ifr_clearance)
        c._rollout = True
        c._track_phase(state(on_ground=True, gs_kt=10))  # full stop
        self.assertEqual((c.flight_rules, c.ifr_clearance, c.assigned_code, c.destination), ("VFR", None, None, None))

    def test_sim_clock_survives_midnight_zulu(self):
        c = self.ctl
        t0 = c._now(state(zulu_sec=86390.0))
        self.assertEqual(c._now(state(zulu_sec=20.0)) - t0, 30.0)
        self.assertEqual(c._now(state(zulu_sec=10.0)) - t0, 30.0)  # time set back: the clock just holds

    def test_tower_starts_over_for_an_inbound_call(self):
        c = self.ctl
        s = state(lat=37.47, lon=-122.12, com1_khz=118600)
        c.situation(s, self.twr, "N828ZZ", "Palo Alto Tower, Cirrus 828ZZ, ready for departure")
        c.remember(self.twr, "ready", "Cirrus eight two eight Zulu Zulu, runway three one, cleared for takeoff.")
        c.remember(self.twr, "", "Cirrus eight Zulu Zulu, frequency change approved.", event=True)
        c.situation(s, self.twr, "N828ZZ", "frequency change approved, Cirrus 8ZZ")
        self.assertIn(self.twr.key, c.handed_off)
        sit = c.situation(s, self.twr, "N828ZZ", "Palo Alto Tower, Cirrus 828ZZ, eight miles south, inbound full stop")
        self.assertNotIn(self.twr.key, c.handed_off)
        self.assertNotIn("ALREADY HANDED OFF", sit)

    def test_a_new_takeoff_clearance_clears_the_old_handoff(self):
        c = self.ctl
        c.remember(self.twr, "x", "Cirrus eight Zulu Zulu, frequency change approved.")
        c.remember(self.twr, "ready", "Cirrus eight Zulu Zulu, runway three one, cleared for takeoff.")
        self.assertNotIn(self.twr.key, c.handed_off)

    def test_traffic_on_the_ground_is_not_called(self):
        a = self.pao
        parked = Target(1, 0.0, 1.0, -900.0, a["elev"] + 5, 90.0, "C172", a["lat"], a["lon"])
        flying = Target(2, 0.0, 1.0, 0.0, a["elev"] + 1000, 90.0, "C172", a["lat"] + 0.01, a["lon"])
        s = state(lat=a["lat"], lon=a["lon"], traffic=[parked, flying])
        self.assertEqual([t.idx for t in self.ctl._airborne_traffic(s, self.twr)], [2])
        self.assertEqual([t.idx for t in self.ctl._airborne_traffic(s, self.app)], [2])

    def test_airport_ids_that_are_words(self):
        self.assertNotIn("fly", _airport_names({"id": "KFLY", "name": "Meadow Lake"}))
        self.assertIn("sjc", _airport_names({"id": "KSJC", "name": "San Jose Intl"}))


class OtherFixes(unittest.TestCase):
    def test_atis_letter_changes_with_the_observation(self):
        before, after = info_letter("KSJC", 17 * 3600 + 52 * 60, "30L"), info_letter("KSJC", 17 * 3600 + 53 * 60, "30L")
        self.assertNotEqual(before, after)
        self.assertEqual(after, info_letter("KSJC", 18 * 3600 + 30 * 60, "30L"))

    def test_config_values(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "atc.toml"
            p.write_text('[audio]\nradio_effect = "false"\nvoice_engine = "Local"\n[chatter]\nmode = "sometimes"\n')
            with self.assertLogs("atc.config", level="WARNING"):
                cfg = config_mod.load(p)
        self.assertFalse(cfg.radio_effect)
        self.assertEqual(cfg.voice_engine, "local")
        self.assertEqual(cfg.chatter_mode, "always")

    def test_debrief_skips_a_cut_off_line(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "flight.jsonl"
            p.write_text(json.dumps({"t": 1, "kind": "pilot", "text": "hello"}) + '\n{"t": 2, "kind": "at')
            self.assertEqual([r["kind"] for r in debrief.load(p)], ["pilot"])

    def test_openrouter_error_in_a_200_response(self):
        r = SimpleNamespace(json=lambda: {"error": {"message": "Provider returned error", "code": 502}}, text="")
        with self.assertRaisesRegex(RuntimeError, "Provider returned error"):
            OpenRouter._json(r, "/chat/completions")
        self.assertEqual(OpenRouter._json(SimpleNamespace(json=lambda: {"choices": []}, text=""), "/x"), {"choices": []})

    def test_xplane_commands(self):
        self.assertEqual(max(["v1", "v2", "v10", "v9"], key=_vnum), "v10")
        xp = XPlane()
        seen = []

        def bad(active):
            raise RuntimeError("boom")

        xp.on_command("ptt", bad)
        xp.on_command("ptt", seen.append)
        with self.assertLogs("atc.xplane", level="ERROR"):
            xp._fire("ptt", True)  # a failing handler doesn't stop the others
        self.assertEqual((seen, xp._active), ([True], {"ptt"}))


if __name__ == "__main__":
    unittest.main()
