"""
Standard Baseline RL Algorithms for HEMS.
Implements non-meta RL baselines (Standard A2C, PPO, and heuristic benchmark)
to compare cross-task transferability as presented in Tables I, II, III of the paper.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import List, Dict, Tuple, Optional

from config import HEMSConfig, default_cfg
from data.task_distribution import HEMSTask
from envs.hems_env import HEMSEnv
from models.meta_lstm_ac import MetaLSTMActorCritic


class StandardA2CBaseline:
    """
    Standard Advantage Actor-Critic (AC) trained on the pooled tasks without MAML meta-gradients.
    When tested on unseen tasks, it uses fixed weights (or standard slow SGD) which exhibits
    poor transferability compared to Meta-RL.
    """

    def __init__(self, cfg: Optional[HEMSConfig] = None, lr: float = 0.001, device: Optional[torch.device] = None):
        self.cfg = cfg or default_cfg
        self.device = device or torch.device("cpu")
        self.model = MetaLSTMActorCritic(
            input_dim=9,
            hidden_dim=self.cfg.lstm_hidden_dim,
            action_dim=25
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.env = HEMSEnv(cfg=self.cfg)

    def train_episode(self, task: HEMSTask) -> float:
        _, lstm_input = self.env.reset(task=task)
        hidden = self.model.init_hidden(batch_size=1, device=self.device)

        rewards, values, log_probs, entropies = [], [], [], []
        done = False
        step = 0

        while not done and step < self.cfg.time_steps:
            x_t = torch.tensor(lstm_input, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits, value, hidden = self.model(x_t, hidden)
            dist = Categorical(logits=logits)
            action = dist.sample()

            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())
            values.append(value)

            _, next_lstm_input, reward, done, _ = self.env.step(int(action.item()))
            rewards.append(reward)
            lstm_input = next_lstm_input
            step += 1

        returns = []
        R = 0.0
        for r in reversed(rewards):
            R = r + self.cfg.gamma * R
            returns.insert(0, R)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device).unsqueeze(1)
        values_t = torch.cat(values, dim=0)
        log_probs_t = torch.cat(log_probs, dim=0)
        entropies_t = torch.cat(entropies, dim=0)

        advantages = (returns_t - values_t).detach()
        loss_pi = -(log_probs_t.unsqueeze(1) * advantages).mean()
        loss_v = 0.5 * ((returns_t - values_t) ** 2).mean()
        loss_env = -entropies_t.mean()

        total_loss = loss_pi + self.cfg.beta_v * loss_v + self.cfg.beta_e * loss_env

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return self.env.episode_cost

    def train(self, tasks: List[HEMSTask], num_episodes: int = 1000) -> List[float]:
        rng = np.random.default_rng(self.cfg.seed + 100)
        history = []
        for ep in range(1, num_episodes + 1):
            task = tasks[rng.choice(len(tasks))]
            cost = self.train_episode(task)
            history.append(float(cost))
        return history

    def evaluate(self, test_tasks: List[HEMSTask], num_episodes: int = 100) -> List[float]:
        rng = np.random.default_rng(self.cfg.seed + 200)
        costs = []

        for _ in range(num_episodes):
            task = test_tasks[rng.choice(len(test_tasks))]
            _, lstm_input = self.env.reset(task=task)
            hidden = self.model.init_hidden(batch_size=1, device=self.device)
            done = False
            step = 0
            while not done and step < self.cfg.time_steps:
                x_t = torch.tensor(lstm_input, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    logits, _, hidden = self.model(x_t, hidden)
                action = torch.argmax(logits, dim=-1)
                _, lstm_input, _, done, _ = self.env.step(int(action.item()))
                step += 1
            costs.append(float(self.env.episode_cost))
        return costs
