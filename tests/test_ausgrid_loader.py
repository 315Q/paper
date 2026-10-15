"""
Unit tests for AusgridDataLoader and real task extraction.
"""

import os
import unittest
import numpy as np
from config import default_cfg
from data.ausgrid_loader import AusgridDataLoader
from data.task_distribution import TaskDistribution


class TestAusgridDataLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.csv_path = "Ausgrid_solar_home_data/Solar home 2010-2011.csv"
        if not os.path.exists(cls.csv_path):
            raise unittest.SkipTest("Ausgrid dataset not present in local folder.")
        cls.loader = AusgridDataLoader(csv_path=cls.csv_path, cfg=default_cfg)
        cls.loader.load_data()

    def test_tasks_count(self):
        # 4 customers x 4 seasons x 3 variations = 48 tasks
        self.assertEqual(len(self.loader.tasks_data), 48)

    def test_archetype_peaks(self):
        # Cust 1 (Afternoon) should peak in afternoon/evening (>= 12:00)
        p1 = self.loader.get_task_profiles(1, "summer", 0)["load_total"]
        peak1 = np.argmax(p1)
        self.assertGreaterEqual(peak1, 24)  # after 12:00

        # Cust 2 (Morning) has a prominent morning peak (slots 10 to 20, 05:00 - 10:00)
        p2 = self.loader.get_task_profiles(2, "summer", 2)["load_total"]
        morning_max = np.max(p2[10:20])
        self.assertGreater(morning_max, 1.5)

        # Cust 3 (Midday) should peak around noon (slots 18 to 26, 09:00 - 13:00)
        p3 = self.loader.get_task_profiles(3, "summer", 0)["load_total"]
        midday_max = np.max(p3[18:26])
        self.assertGreater(midday_max, 1.5)

    def test_solar_seasonality(self):
        # Summer solar should be significantly greater than winter solar
        summer_pv = self.loader.get_task_profiles(1, "summer", 0)["pv_profile"].sum()
        winter_pv = self.loader.get_task_profiles(1, "winter", 0)["pv_profile"].sum()
        self.assertGreater(summer_pv, winter_pv * 1.5)

    def test_task_distribution_integration(self):
        dist = TaskDistribution(cfg=default_cfg, ausgrid_csv=self.csv_path)
        train_tasks, test_tasks = dist.get_tasks_for_customer_transfer()
        self.assertEqual(len(train_tasks), 36)
        self.assertEqual(len(test_tasks), 12)
        # Verify Customer 4 is in test_tasks
        for t in test_tasks:
            self.assertEqual(t.customer_id, 4)


if __name__ == "__main__":
    unittest.main()
