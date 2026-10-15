"""
Global Configuration and Hyperparameters for Meta-RL HEMS Reproduction.
Paper: Meta-Reinforcement Learning-Based Transferable Scheduling Strategy for Energy Management
       (IEEE TCSI 2023, Luolin Xiong et al., DOI: 10.1109/TCSI.2023.3240702)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class HEMSConfig:
    # --- Time Horizon ---
    time_steps: int = 48             # 48 intervals per day (24 hours)
    delta_t: float = 0.5             # Half-hour interval (0.5 hour)

    # --- Energy Storage Equipment (ESE / Battery) ---
    e_cap: float = 10.0              # Battery capacity: 10.0 kWh
    soc_min: float = 0.1             # Minimum SoC (10%)
    soc_max: float = 0.9             # Maximum SoC (90%)
    p_ese_max: float = 2.5           # Max charge/discharge power (kW)
    eta_charge: float = 0.95         # Charging efficiency
    eta_discharge: float = 0.95      # Discharging efficiency
    c_ese: float = 3500.0            # Battery replacement cost (A$)
    k_ese: float = 0.05              # Slope of linear life approximation
    # α_ESE = c_ESE * |k_ESE / 100| * (1 / E_cap)
    alpha_ese: float = 3500.0 * (0.05 / 100.0) / 10.0  # ~0.0175 A$/kWh degradation

    # --- Customer Comfort Parameters ---
    c_ts_cus: float = 0.05           # Discomfort coefficient for time-shiftable load
    c_ps_cus: float = 0.08           # Discomfort coefficient for power-shiftable load
    alpha_td_max: float = 1.0        # Max allowable comfort parameter (default: 1.0)
    penalty_weight: float = 5.0      # Penalty weight for constraint violations in reward

    # --- TOU Electricity Tariffs (A$/kWh) as per Paper Fig. 4 ---
    # Prices: Off-peak = 0.1892, Shoulder = 0.3102, On-peak = 0.5929
    price_off_peak: float = 0.1892
    price_shoulder: float = 0.3102
    price_on_peak: float = 0.5929

    # --- Meta-RL Algorithm Hyperparameters (Section IV-A) ---
    gamma: float = 0.9               # Discount factor
    beta_v: float = 0.05             # Weight for value loss L_v
    beta_e: float = 0.05             # Weight for entropy regularization L_env
    alpha_prime: float = 0.001       # Inner level adaptation learning rate
    beta: float = 0.001              # Outer level meta learning rate
    lstm_hidden_dim: int = 48        # Core actor and critic LSTM units
    meta_batch_size: int = 5         # Synchronous task threads / batch size
    meta_train_episodes: int = 1000  # Number of meta-training episodes
    meta_test_episodes: int = 100    # Number of meta-testing episodes

    # --- Random Seed ---
    seed: int = 42

    def get_tou_tariff(self, season: str) -> np.ndarray:
        """
        Returns the 48-slot price vector (A$/kWh) for a given season based on Fig. 4.
        Summer: Nov. 1 - Mar. 31
          - Off-peak: 22:00 - 07:00 (slots 0-13, 44-47)
          - Shoulder: 07:00 - 14:00 (slots 14-27), 20:00 - 22:00 (slots 40-43)
          - On-peak: 14:00 - 20:00 (slots 28-39)
        Winter: Jun. 1 - Aug. 31
          - Off-peak: 22:00 - 07:00 (slots 0-13, 44-47)
          - Shoulder: 07:00 - 17:00 (slots 14-33), 21:00 - 22:00 (slots 42-43)
          - On-peak: 17:00 - 21:00 (slots 34-41)
        """
        tariff = np.zeros(self.time_steps, dtype=np.float32)
        season_lower = season.lower()

        if season_lower == "summer":
            # Off-peak: 22:00 - 07:00
            tariff[0:14] = self.price_off_peak
            tariff[44:48] = self.price_off_peak
            # Shoulder: 07:00 - 14:00 and 20:00 - 22:00
            tariff[14:28] = self.price_shoulder
            tariff[40:44] = self.price_shoulder
            # On-peak: 14:00 - 20:00
            tariff[28:40] = self.price_on_peak
        elif season_lower == "winter":
            # Off-peak: 22:00 - 07:00
            tariff[0:14] = self.price_off_peak
            tariff[44:48] = self.price_off_peak
            # Shoulder: 07:00 - 17:00 and 21:00 - 22:00
            tariff[14:34] = self.price_shoulder
            tariff[42:44] = self.price_shoulder
            # On-peak: 17:00 - 21:00
            tariff[34:42] = self.price_on_peak
        elif season_lower in ["spring", "autumn"]:
            # Transition seasons: blend / shoulder dominant
            tariff[0:14] = self.price_off_peak
            tariff[44:48] = self.price_off_peak
            tariff[14:30] = self.price_shoulder
            tariff[38:44] = self.price_shoulder
            tariff[30:38] = self.price_on_peak
        else:
            raise ValueError(f"Unknown season: {season}")

        return tariff


# Default global instance
default_cfg = HEMSConfig()
