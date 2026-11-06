import unittest

from atc.airports import AirportDB
from atc.centers import nearest_low, site_for_freq, sites


class CentersTest(unittest.TestCase):
    def test_data_loaded(self):
        self.assertGreater(len(sites()), 500)

    def test_nearest_low(self):
        site, khz, d = nearest_low(37.0, -121.5)
        self.assertEqual(site["artcc"], "ZOA")
        self.assertTrue(118000 <= khz <= 136975)
        self.assertEqual(nearest_low(39.8, -104.7)[0]["artcc"], "ZDV")

    def test_center_frequency_is_a_facility(self):
        site, khz, _ = nearest_low(37.0, -121.5)
        self.assertEqual(site_for_freq(khz, 37.0, -121.5)[0]["artcc"], "ZOA")
        fac = AirportDB([]).facility_for(khz, 37.0, -121.5)
        self.assertEqual((fac.kind, fac.name), ("CENTER", "Oakland Center"))


if __name__ == "__main__":
    unittest.main()
