import unittest
from pathlib import Path

from atc.airports import AirportDB, apt_dat_files, parse_apt_dat
from atc.atis import altitude_words, build, digits
from atc.state import AircraftState

FIX = Path(__file__).parent / "fixtures" / "xp"


def st(**kw):
    b = dict(lat=37.4611, lon=-122.115, alt_ft=7, agl_ft=0, gs_kt=0, ias_kt=0, hdg_mag=310, vs_fpm=0,
             on_ground=True, com1_khz=135275, com2_khz=0, tx_com=1, xpdr_code=1200, xpdr_mode="alt",
             tailnum="N123AB", icao_type="C172", altimeter_inhg=29.92, wind_dir=310, wind_kt=9,
             vis_sm=10, temp_c=5.0, temp_sl_c=18.4, dew_sl_c=9.6, clouds=[(4507, 0.2), (8000, 0.7)], zulu_sec=17 * 3600 + 58 * 60)
    b.update(kw)
    return AircraftState(**b)


class AtisTest(unittest.TestCase):
    def setUp(self):
        self.db = AirportDB(parse_apt_dat(apt_dat_files(str(FIX))[0]))

    def test_words(self):
        self.assertEqual(digits("2992"), "two niner niner two")
        self.assertEqual(altitude_words(4500), "four thousand five hundred")
        self.assertEqual(altitude_words(12000), "one two thousand")

    def test_faa_atis(self):
        fac = self.db.facility_for(135275, 37.4611, -122.115)
        text, letter = build(fac, st())
        self.assertIn(f"information {letter}", text)
        self.assertIn("One seven five three zulu", text)
        self.assertIn("Wind three one zero at niner", text)
        self.assertIn("Visibility one zero", text)
        self.assertIn("Few clouds at four thousand five hundred, ceiling eight thousand broken", text)
        self.assertIn("Temperature one eight, dew point one zero", text)
        self.assertIn("Altimeter two niner niner two", text)
        self.assertIn("runway three one", text)
        self.assertTrue(text.isascii())

    def test_missing_weather_is_left_out(self):
        fac = self.db.facility_for(135275, 37.4611, -122.115)
        text, _ = build(fac, st(vis_sm=None, temp_c=None, temp_sl_c=None, dew_sl_c=None, clouds=None,
                                 wind_dir=None, wind_kt=None))
        self.assertNotIn("Visibility", text)
        self.assertNotIn("Temperature", text)
        self.assertNotIn("Wind", text)

    def test_field_temperature_from_sea_level(self):
        # Salzburg is 1411 ft: 18.4 - 2.8 = 15.6 -> "one six"; dew 9.6 - 0.8 = 8.8 -> "niner"
        fac = self.db.facility_for(125730, 47.79, 13.0)
        text, _ = build(fac, st(lat=47.79, lon=13.0))
        self.assertIn("Temperature one six, dew point niner", text)

    def test_icao_atis(self):
        fac = self.db.facility_for(125730, 47.79, 13.0)
        text, _ = build(fac, st(lat=47.79, lon=13.0, altimeter_inhg=29.92))
        self.assertIn("QNH one zero one three", text)
        self.assertIn("Salzburg information", text)


if __name__ == "__main__":
    unittest.main()
