"""
High-Fidelity Profile Generator for Ausgrid Customers, PV Generation, and Appliances.
Mathematically calibrated to match Fig. 4, Fig. 5, and Fig. 6 from the paper:
- 4 representative customers with distinct electricity preferences (Fig. 5)
- 4 seasonal photovoltaic generation profiles (Fig. 6)
- Load breakdown: Uncontrollable (UL), Power-Shiftable (PS), Time-Shiftable (TS)
"""

import numpy as np
from typing import Dict, Tuple


class ProfileGenerator:
    def __init__(self, time_steps: int = 48, seed: int = 42):
        self.time_steps = time_steps
        self.dt = 24.0 / time_steps  # 0.5 hour
        self.hours = np.linspace(0, 24, time_steps, endpoint=False)
        self.rng = np.random.default_rng(seed)

    def generate_pv_profile(self, season: str, noise_level: float = 0.03) -> np.ndarray:
        """
        Generates 48-slot PV generation profile (kW) matching Fig. 6:
        - Summer: Sunrise ~05:30, Sunset ~19:30, Peak ~1.5 kW around 12:30
        - Winter: Sunrise ~07:00, Sunset ~17:00, Peak ~0.5 kW around 12:00
        - Spring: Sunrise ~06:00, Sunset ~18:30, Peak ~1.05 kW around 12:00
        - Autumn: Sunrise ~06:30, Sunset ~18:00, Peak ~0.95 kW around 12:00
        """
        season_lower = season.lower()
        if season_lower == "summer":
            t_rise, t_set, p_peak = 5.5, 19.5, 1.48
        elif season_lower == "winter":
            t_rise, t_set, p_peak = 7.0, 17.0, 0.52
        elif season_lower == "spring":
            t_rise, t_set, p_peak = 6.0, 18.5, 1.05
        elif season_lower == "autumn":
            t_rise, t_set, p_peak = 6.5, 18.0, 0.95
        else:
            raise ValueError(f"Unknown season: {season}")

        pv = np.zeros(self.time_steps, dtype=np.float32)
        daylight_mask = (self.hours >= t_rise) & (self.hours <= t_set)
        rel_time = (self.hours[daylight_mask] - t_rise) / (t_set - t_rise)
        pv[daylight_mask] = p_peak * np.sin(np.pi * rel_time) ** 1.3

        if noise_level > 0:
            noise = self.rng.normal(0, noise_level, size=self.time_steps)
            pv = np.clip(pv + noise * (pv > 0.05), 0.0, p_peak * 1.1)

        return pv.astype(np.float32)

    def generate_customer_load(self, customer_id: int, noise_level: float = 0.04) -> Dict[str, np.ndarray]:
        """
        Generates customer demand profile (kW) matching Fig. 5:
        - Customer 1 (Afternoon preference): peaks between 13:00 - 18:00 (peak ~2.0 kW)
        - Customer 2 (Morning preference): peaks between 06:00 - 10:00 (peak ~2.4 kW)
        - Customer 3 (Midday preference): peaks between 10:00 - 12:00 (peak ~1.8 kW)
        - Customer 4 (Night preference): peaks between 18:00 - 23:00 (peak ~1.8 kW)

        Returns dictionary with:
        - total: Total baseline load
        - ul: Uncontrollable load (~35%)
        - ps: Power-shiftable load (~40%)
        - ts: Time-shiftable load (~25%)
        """
        h = self.hours
        base_standby = 0.25

        if customer_id == 1:
            # Afternoon consumer
            peak_afternoon = 1.7 * np.exp(-((h - 15.5) ** 2) / (2 * 2.2 ** 2))
            secondary_noon = 0.8 * np.exp(-((h - 12.0) ** 2) / (2 * 1.5 ** 2))
            morning_bump = 0.5 * np.exp(-((h - 7.5) ** 2) / (2 * 1.2 ** 2))
            total = base_standby + peak_afternoon + secondary_noon + morning_bump

        elif customer_id == 2:
            # Morning consumer
            peak_morning = 2.1 * np.exp(-((h - 7.5) ** 2) / (2 * 1.6 ** 2))
            secondary_noon = 0.7 * np.exp(-((h - 13.0) ** 2) / (2 * 1.8 ** 2))
            evening_bump = 0.6 * np.exp(-((h - 19.5) ** 2) / (2 * 1.5 ** 2))
            total = base_standby + peak_morning + secondary_noon + evening_bump

        elif customer_id == 3:
            # Midday consumer (10:00 - 12:00)
            peak_midday = 1.6 * np.exp(-((h - 11.0) ** 2) / (2 * 1.4 ** 2))
            secondary_afternoon = 0.9 * np.exp(-((h - 16.0) ** 2) / (2 * 2.0 ** 2))
            morning_bump = 0.4 * np.exp(-((h - 8.0) ** 2) / (2 * 1.5 ** 2))
            total = base_standby + peak_midday + secondary_afternoon + morning_bump

        elif customer_id == 4:
            # Night consumer (18:00 - 23:00)
            peak_night = 1.6 * np.exp(-((h - 20.5) ** 2) / (2 * 2.0 ** 2))
            evening_tail = 1.0 * np.exp(-((h - 23.0) ** 2) / (2 * 1.8 ** 2))
            morning_small = 0.35 * np.exp(-((h - 8.0) ** 2) / (2 * 1.2 ** 2))
            total = base_standby + peak_night + evening_tail + morning_small
        else:
            raise ValueError(f"Customer ID must be 1, 2, 3, or 4. Got {customer_id}")

        if noise_level > 0:
            noise = self.rng.normal(0, noise_level, size=self.time_steps)
            total = np.clip(total + noise, 0.1, 3.5)

        # Decompose total load into UL, PS, TS components
        ul = 0.35 * total + 0.05
        ps = 0.40 * total
        ts = total - ul - ps
        ts = np.maximum(0.0, ts)

        return {
            "total": total.astype(np.float32),
            "ul": ul.astype(np.float32),
            "ps": ps.astype(np.float32),
            "ts": ts.astype(np.float32),
        }
