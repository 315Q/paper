"""
Gymnasium-compatible Home Energy Management System (HEMS) Environment.
Formulated strictly based on Section II of IEEE TCSI 2023 paper:
'Meta-Reinforcement Learning-Based Transferable Scheduling Strategy for Energy Management'
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
from config import HEMSConfig, default_cfg
from data.task_distribution import HEMSTask


class HEMSEnv:
    """
    HEMS Markov Decision Process (MDP) Environment:
    - State: s_t = [P_PV, P_UL, P_PS, P_TS, E_SoC]  (5-dimensional vector)
    - Action: a_t = [Delta P_ESE, Delta P_Sh] (25 discrete action pairs via Softmax)
    - Reward: r_t = - (P_buy * lambda_t + alpha_ESE * |Delta P_ESE| * Delta t) - penalty
    """

    def __init__(self, cfg: Optional[HEMSConfig] = None, task: Optional[HEMSTask] = None):
        self.cfg = cfg or default_cfg
        self.task = task
        self.time_steps = self.cfg.time_steps
        self.dt = self.cfg.delta_t

        # Action space discretization (25 actions = 5 battery levels x 5 load levels)
        self.ese_levels = np.array([-2.5, -1.25, 0.0, 1.25, 2.5], dtype=np.float32)
        self.sh_levels = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
        self.action_space_dim = len(self.ese_levels) * len(self.sh_levels)  # 25

        # State dimensions
        self.state_dim = 5  # [P_PV, P_UL, P_PS, P_TS, E_SoC]
        self.history_dim = self.state_dim + 2 + 1 + 1  # [state, prev_a_ese, prev_a_sh, prev_r, t/T] = 9

        # Internal state tracking
        self.current_step = 0
        self.e_soc = self.cfg.e_cap * 0.5  # Initial battery energy: 5.0 kWh
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.prev_reward = 0.0
        self.ts_cumulative_delta = 0.0
        self.episode_cost = 0.0
        self.episode_comfort_violation = 0.0

    def set_task(self, task: HEMSTask):
        """Switches the current environment to a new MDP task."""
        self.task = task

    def reset(self, task: Optional[HEMSTask] = None, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resets environment at the start of a 24-hour episode (t = 0).
        Returns:
            state: s_0 (shape: 5)
            lstm_input: x_0 (shape: 9)
        """
        if task is not None:
            self.task = task
        if self.task is None:
            raise ValueError("Task must be set before resetting HEMSEnv.")

        self.current_step = 0
        # Random initial battery energy near 50% capacity
        self.e_soc = float(np.clip(self.cfg.e_cap * 0.5 + np.random.uniform(-0.5, 0.5),
                                   self.cfg.e_cap * self.cfg.soc_min,
                                   self.cfg.e_cap * self.cfg.soc_max))
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.prev_reward = 0.0
        self.ts_cumulative_delta = 0.0
        self.episode_cost = 0.0
        self.episode_comfort_violation = 0.0

        state = self._get_state()
        lstm_input = self._get_lstm_input(state)
        return state, lstm_input

    def _get_state(self) -> np.ndarray:
        t = self.current_step
        p_pv = float(self.task.pv_profile[t])
        p_ul = float(self.task.load_ul[t])
        p_ps = float(self.task.load_ps[t])
        p_ts = float(self.task.load_ts[t])
        e_soc = float(self.e_soc)
        return np.array([p_pv, p_ul, p_ps, p_ts, e_soc], dtype=np.float32)

    def _get_lstm_input(self, state: np.ndarray) -> np.ndarray:
        # [s_t (5), a_{t-1} (2), r_{t-1} (1), normalized_t (1)]
        norm_t = float(self.current_step) / float(self.time_steps)
        return np.concatenate([
            state,
            self.prev_action,
            np.array([self.prev_reward, norm_t], dtype=np.float32)
        ])

    def decode_action(self, action_idx: int) -> Tuple[float, float]:
        """Decodes discrete action index (0..24) into (Delta P_ESE, Delta P_Sh)."""
        ese_idx = action_idx // len(self.sh_levels)
        sh_idx = action_idx % len(self.sh_levels)
        return float(self.ese_levels[ese_idx]), float(self.sh_levels[sh_idx])

    def step(self, action: int) -> Tuple[np.ndarray, np.ndarray, float, bool, Dict[str, Any]]:
        """
        Executes one 30-min scheduling step.
        Args:
            action: discrete action index (0..24)
        Returns:
            next_state: s_{t+1} (shape: 5)
            next_lstm_input: x_{t+1} (shape: 9)
            reward: r_t (float)
            done: bool (True when t == 48)
            info: dict containing cost breakdown and metrics
        """
        t = self.current_step
        p_pv = float(self.task.pv_profile[t])
        p_ul = float(self.task.load_ul[t])
        p_ps = float(self.task.load_ps[t])
        p_ts = float(self.task.load_ts[t])
        price_t = float(self.task.tariff[t])

        delta_p_ese_req, delta_p_sh_req = self.decode_action(action)

        # 1. Physical battery constraints: limit charge/discharge by SoC
        max_charge = max(0.0, (self.cfg.e_cap * self.cfg.soc_max - self.e_soc) / (self.dt * self.cfg.eta_charge))
        max_discharge = max(0.0, (self.e_soc - self.cfg.e_cap * self.cfg.soc_min) * self.cfg.eta_discharge / self.dt)
        actual_delta_p_ese = float(np.clip(delta_p_ese_req, -max_discharge, max_charge))

        # Battery state transition: E_{t+1} = E_t + Delta P_ESE * Delta t
        if actual_delta_p_ese >= 0:
            self.e_soc += actual_delta_p_ese * self.dt * self.cfg.eta_charge
        else:
            self.e_soc += actual_delta_p_ese * self.dt / self.cfg.eta_discharge
        self.e_soc = float(np.clip(self.e_soc, self.cfg.e_cap * self.cfg.soc_min, self.cfg.e_cap * self.cfg.soc_max))

        # 2. Shiftable load allocation: schedule TS first (Eq. 6b, 6d)
        # Delta P_TS in {0, P_TS, -P_TS}
        if delta_p_sh_req < -0.3 and p_ts > 0.05:
            # Postpone/defer TS load during peak or high-demand periods
            delta_p_ts = -min(p_ts, abs(delta_p_sh_req))
        elif delta_p_sh_req > 0.3 and self.ts_cumulative_delta < -0.01:
            # Fulfill/repay previously deferred TS load during off-peak or solar hours
            delta_p_ts = min(abs(self.ts_cumulative_delta), delta_p_sh_req)
        elif delta_p_sh_req > 0.3 and p_ts > 0.05:
            delta_p_ts = min(p_ts * 0.5, delta_p_sh_req)
        else:
            delta_p_ts = 0.0
        delta_p_ts = max(-p_ts, delta_p_ts)
        self.ts_cumulative_delta += delta_p_ts

        # Delta P_PS adjusts flexibly
        delta_p_ps = delta_p_sh_req - delta_p_ts
        delta_p_ps = max(-p_ps, delta_p_ps)
        actual_delta_p_sh = delta_p_ts + delta_p_ps

        # 3. Power balance equation (Eq. 6a):
        # P_buy = P_UL + P_PS + P_TS - P_PV + Delta P_PS + Delta P_TS + Delta P_ESE
        net_demand = p_ul + p_ps + p_ts - p_pv + actual_delta_p_sh + actual_delta_p_ese
        p_buy = max(0.0, net_demand)  # Grid purchases cannot be negative (surplus PV curtailed or free)

        # 4. Economic costs:
        # Supply charge (Reference [50]): 1.0604 A$/day divided across 48 intervals
        step_supply_charge = 1.0604 / float(self.time_steps)
        electricity_cost = p_buy * price_t * self.dt
        battery_wear_cost = self.cfg.alpha_ese * abs(actual_delta_p_ese) * self.dt
        step_cost = electricity_cost + battery_wear_cost + step_supply_charge

        # 5. Customer comfort penalty (Eq. 6f-6j):
        # Quadratic discomfort
        alpha_ts_d = self.cfg.c_ts_cus * (delta_p_ts ** 2)
        alpha_ps_d = self.cfg.c_ps_cus * (delta_p_ps ** 2)
        alpha_td = alpha_ts_d + alpha_ps_d
        alpha_tol = getattr(self.task, "alpha_td_max", self.cfg.alpha_td_max)
        comfort_violation = max(0.0, alpha_td - alpha_tol)
        self.episode_comfort_violation += comfort_violation

        penalty = self.cfg.penalty_weight * comfort_violation

        # 6. Reward (Eq. 4 & Eq. 6)
        reward = -(step_cost + penalty)

        # Update historical actions & rewards for next LSTM input
        self.prev_action = np.array([actual_delta_p_ese, actual_delta_p_sh], dtype=np.float32)
        self.prev_reward = reward

        self.current_step += 1
        done = (self.current_step >= self.time_steps)

        # If day completed, check TS 24-hour demand fulfillment constraint (Eq. 6f)
        if done and self.ts_cumulative_delta < -0.1:
            ts_penalty = self.cfg.penalty_weight * abs(self.ts_cumulative_delta)
            reward -= ts_penalty

        # Total objective cost C as strictly defined in Eq. (6): C = - sum(r_t)
        self.episode_cost += (-reward)

        next_state = self._get_state() if not done else np.zeros(self.state_dim, dtype=np.float32)
        next_lstm_input = self._get_lstm_input(next_state) if not done else np.zeros(self.history_dim, dtype=np.float32)

        info = {
            "step_cost": step_cost,
            "electricity_cost": electricity_cost,
            "battery_wear_cost": battery_wear_cost,
            "episode_cost": self.episode_cost,
            "p_buy": p_buy,
            "e_soc": self.e_soc,
            "comfort_violation": comfort_violation,
            "delta_p_ese": actual_delta_p_ese,
            "delta_p_sh": actual_delta_p_sh,
        }

        return next_state, next_lstm_input, float(reward), done, info
