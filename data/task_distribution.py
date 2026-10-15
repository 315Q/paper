"""
Task Distribution and Sampler for Meta-RL HEMS.
Manages the 48 scheduling tasks across 4 customers and 4 seasons with realistic variations.
Automatically prefers the authentic Ausgrid smart meter dataset when available.
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from config import HEMSConfig, default_cfg
from data.profile_generator import ProfileGenerator
from data.ausgrid_loader import AusgridDataLoader


@dataclass
class HEMSTask:
    task_id: int
    customer_id: int
    season: str
    variation_id: int
    tariff: np.ndarray       # shape (48,)
    pv_profile: np.ndarray   # shape (48,)
    load_total: np.ndarray   # shape (48,)
    load_ul: np.ndarray      # shape (48,)
    load_ps: np.ndarray      # shape (48,)
    load_ts: np.ndarray      # shape (48,)
    alpha_td_max: float      # customer comfort threshold


class TaskDistribution:
    def __init__(
        self,
        cfg: Optional[HEMSConfig] = None,
        seed: int = 42,
        ausgrid_csv: str = "Ausgrid_solar_home_data/Solar home 2010-2011.csv",
    ):
        self.cfg = cfg or default_cfg
        self.rng = np.random.default_rng(seed)
        self.all_tasks: List[HEMSTask] = []
        self.ausgrid_csv = ausgrid_csv

        if os.path.exists(self.ausgrid_csv):
            print(f"[TaskDistribution] Found real Ausgrid dataset at: {self.ausgrid_csv}. Loading 48 real tasks...")
            self.loader = AusgridDataLoader(csv_path=self.ausgrid_csv, cfg=self.cfg)
            self.loader.load_data()
            self._build_real_tasks()
        else:
            print("[TaskDistribution] Ausgrid CSV not found; using calibrated ProfileGenerator fallback.")
            self.generator = ProfileGenerator(time_steps=self.cfg.time_steps, seed=seed)
            self._build_synthetic_tasks()

    def _build_real_tasks(self):
        """Builds 48 tasks using authentic Ausgrid smart meter trajectories."""
        seasons = ["spring", "summer", "autumn", "winter"]
        customers = [1, 2, 3, 4]
        variations = [0, 1, 2]

        task_id = 0
        for cust in customers:
            for season in seasons:
                for var_idx in variations:
                    p = self.loader.get_task_profiles(cust, season, var_idx)
                    task = HEMSTask(
                        task_id=task_id,
                        customer_id=cust,
                        season=season,
                        variation_id=var_idx,
                        tariff=p["tariff"],
                        pv_profile=p["pv_profile"],
                        load_total=p["load_total"],
                        load_ul=p["load_ul"],
                        load_ps=p["load_ps"],
                        load_ts=p["load_ts"],
                        alpha_td_max=self.cfg.alpha_td_max,
                    )
                    self.all_tasks.append(task)
                    task_id += 1

    def _build_synthetic_tasks(self):
        """Synthetic fallback matching Fig 4, 5, 6 curves."""
        seasons = ["spring", "summer", "autumn", "winter"]
        customers = [1, 2, 3, 4]
        variations = [0, 1, 2]

        task_id = 0
        for cust in customers:
            for season in seasons:
                for var_idx in variations:
                    noise = 0.02 + 0.02 * var_idx
                    tariff = self.cfg.get_tou_tariff(season)
                    pv = self.generator.generate_pv_profile(season, noise_level=noise)
                    load_dict = self.generator.generate_customer_load(cust, noise_level=noise)

                    task = HEMSTask(
                        task_id=task_id,
                        customer_id=cust,
                        season=season,
                        variation_id=var_idx,
                        tariff=tariff,
                        pv_profile=pv,
                        load_total=load_dict["total"],
                        load_ul=load_dict["ul"],
                        load_ps=load_dict["ps"],
                        load_ts=load_dict["ts"],
                        alpha_td_max=self.cfg.alpha_td_max,
                    )
                    self.all_tasks.append(task)
                    task_id += 1

    def get_tasks_for_customer_transfer(self) -> Tuple[List[HEMSTask], List[HEMSTask]]:
        """
        Customer Transfer Setup:
        - Meta-Training: Customer 1, 2, 3 (36 tasks)
        - Meta-Testing: Customer 4 (12 tasks)
        """
        train_tasks = [t for t in self.all_tasks if t.customer_id in [1, 2, 3]]
        test_tasks = [t for t in self.all_tasks if t.customer_id == 4]
        return train_tasks, test_tasks

    def get_tasks_for_season_transfer(self, test_season: str) -> Tuple[List[HEMSTask], List[HEMSTask]]:
        """
        Season Transfer Setup:
        - test_season = 'winter' (Train: spring, summer, autumn; Test: winter)
        - test_season = 'summer' (Train: spring, autumn, winter; Test: summer)
        """
        test_season_lower = test_season.lower()
        train_tasks = [t for t in self.all_tasks if t.season.lower() != test_season_lower]
        test_tasks = [t for t in self.all_tasks if t.season.lower() == test_season_lower]
        return train_tasks, test_tasks

    def sample_batch(self, task_pool: List[HEMSTask], batch_size: int) -> List[HEMSTask]:
        """Samples a batch of tasks from the designated task pool."""
        indices = self.rng.choice(len(task_pool), size=batch_size, replace=False if len(task_pool) >= batch_size else True)
        return [task_pool[i] for i in indices]
