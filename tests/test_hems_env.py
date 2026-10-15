"""
Unit tests for HEMSEnv and Physical Constraints.
"""

import unittest
import numpy as np
from config import default_cfg
from data.task_distribution import TaskDistribution
from envs.hems_env import HEMSEnv


class TestHEMSEnv(unittest.TestCase):
    def setUp(self):
        self.tasks = TaskDistribution(cfg=default_cfg, seed=123)
        self.sample_task = self.tasks.all_tasks[0]
        self.env = HEMSEnv(cfg=default_cfg, task=self.sample_task)

    def test_dimensions(self):
        state, lstm_input = self.env.reset()
        self.assertEqual(state.shape, (5,))
        self.assertEqual(lstm_input.shape, (9,))
        self.assertEqual(self.env.action_space_dim, 25)

    def test_battery_soc_bounds(self):
        self.env.reset()
        for action_idx in range(25):
            _, _, _, done, info = self.env.step(action_idx)
            soc = info["e_soc"]
            self.assertGreaterEqual(soc, default_cfg.e_cap * default_cfg.soc_min - 1e-4)
            self.assertLessEqual(soc, default_cfg.e_cap * default_cfg.soc_max + 1e-4)
            if done:
                break

    def test_power_balance_and_costs(self):
        self.env.reset()
        _, _, reward, done, info = self.env.step(0)  # maximum discharge + load reduction
        self.assertGreaterEqual(info["p_buy"], 0.0)
        self.assertGreaterEqual(info["electricity_cost"], 0.0)
        self.assertGreaterEqual(info["battery_wear_cost"], 0.0)
        self.assertLessEqual(reward, 0.0)  # Costs are negative rewards

    def test_full_episode_termination(self):
        self.env.reset()
        done = False
        steps = 0
        while not done:
            _, _, _, done, _ = self.env.step(12)  # idle action
            steps += 1
        self.assertEqual(steps, default_cfg.time_steps)
        self.assertTrue(done)


if __name__ == "__main__":
    unittest.main()
