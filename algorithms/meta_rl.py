"""
Meta-Reinforcement Learning Trainer for HEMS.
Implements Algorithm 1 from IEEE TCSI 2023 paper:
'Meta-Reinforcement Learning-Based Transferable Scheduling Strategy for Energy Management'
Integrates:
- Advantage Actor-Critic (A2C)
- LSTM recurrent memory
- MAML inner-outer optimization for transferable initial parameter learning
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import List, Dict, Tuple, Optional, Any

from config import HEMSConfig, default_cfg
from data.task_distribution import HEMSTask, TaskDistribution
from envs.hems_env import HEMSEnv
from models.meta_lstm_ac import MetaLSTMActorCritic


class MetaRLTrainer:
    def __init__(
        self,
        cfg: Optional[HEMSConfig] = None,
        model: Optional[MetaLSTMActorCritic] = None,
        device: Optional[torch.device] = None,
    ):
        self.cfg = cfg or default_cfg
        self.device = device or torch.device("cpu")
        self.model = model or MetaLSTMActorCritic(
            input_dim=9,
            hidden_dim=self.cfg.lstm_hidden_dim,
            action_dim=25
        ).to(self.device)

        # Meta outer-loop optimizer
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.beta)

        # Environment
        self.env = HEMSEnv(cfg=self.cfg)

    def rollout_trajectory(
        self,
        task: HEMSTask,
        params: Optional[Dict[str, torch.Tensor]] = None,
        explore: bool = True,
    ) -> Dict[str, Any]:
        """
        Collects a full 48-step trajectory for a given task.
        Returns:
            trajectory dictionary with inputs, actions, rewards, values, log_probs, entropies, episode_cost
        """
        _, lstm_input = self.env.reset(task=task)
        hidden = self.model.init_hidden(batch_size=1, device=self.device)

        inputs_list = []
        actions_list = []
        rewards_list = []
        values_list = []
        log_probs_list = []
        entropies_list = []

        done = False
        step = 0
        while not done and step < self.cfg.time_steps:
            x_t = torch.tensor(lstm_input, dtype=torch.float32, device=self.device).unsqueeze(0)
            inputs_list.append(x_t)

            logits, value, hidden = self.model(x_t, hidden, params=params)
            dist = Categorical(logits=logits)

            if explore:
                action = dist.sample()
            else:
                action = torch.argmax(logits, dim=-1)

            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            action_idx = int(action.item())
            _, next_lstm_input, reward, done, info = self.env.step(action_idx)

            actions_list.append(action)
            rewards_list.append(reward)
            values_list.append(value)
            log_probs_list.append(log_prob)
            entropies_list.append(entropy)

            lstm_input = next_lstm_input
            step += 1

        # Compute discounted returns R_t (Eq. 20)
        returns = []
        R = 0.0
        for r in reversed(rewards_list):
            R = r + self.cfg.gamma * R
            returns.insert(0, R)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device).unsqueeze(1)

        values_t = torch.cat(values_list, dim=0)
        log_probs_t = torch.cat(log_probs_list, dim=0)
        entropies_t = torch.cat(entropies_list, dim=0)

        # Advantage TD error: R_t - V(s_t)
        advantages = returns_t - values_t

        return {
            "inputs": inputs_list,
            "actions": actions_list,
            "rewards": rewards_list,
            "returns": returns_t,
            "values": values_t,
            "log_probs": log_probs_t,
            "entropies": entropies_t,
            "advantages": advantages,
            "episode_cost": self.env.episode_cost,
        }

    def compute_loss(self, traj: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes A2C objective loss as defined in Eq. (16)-(19):
        L = L_pi + beta_v * L_v + beta_e * L_env
        """
        advantages = traj["advantages"].detach()
        log_probs = traj["log_probs"]
        values = traj["values"]
        returns = traj["returns"]
        entropies = traj["entropies"]

        # Policy loss: - sum log_pi * A
        loss_pi = -(log_probs.unsqueeze(1) * advantages).mean()

        # Value loss: 0.5 * (R - V)^2
        loss_v = 0.5 * ((returns - values) ** 2).mean()

        # Entropy loss: - H(pi)
        loss_env = -entropies.mean()

        total_loss = loss_pi + self.cfg.beta_v * loss_v + self.cfg.beta_e * loss_env
        return total_loss, loss_pi, loss_v, loss_env

    def adapt_inner(self, task: HEMSTask) -> Dict[str, torch.Tensor]:
        """
        Inner loop adaptation (Eq. 7 & 12):
        theta_k <- theta - alpha_prime * grad(L_{tau_k})
        """
        traj = self.rollout_trajectory(task, params=None, explore=True)
        loss, _, _, _ = self.compute_loss(traj)

        # Compute gradients with respect to model parameters
        grads = torch.autograd.grad(loss, self.model.parameters(), retain_graph=False, create_graph=False)

        # Apply 1 gradient step to obtain adapted weights theta_k
        adapted_params = {}
        for (name, param), grad in zip(self.model.named_parameters(), grads):
            adapted_params[name] = param - self.cfg.alpha_prime * grad

        return adapted_params

    def train_meta_step(self, task_batch: List[HEMSTask]) -> Dict[str, float]:
        """
        Executes one meta-training iteration across a batch of tasks (Algorithm 1).
        """
        self.meta_optimizer.zero_grad()
        meta_loss_val = 0.0
        avg_cost_val = 0.0

        batch_losses = []
        for task in task_batch:
            # 1. Inner adaptation: get adapted theta_k
            adapted_params = self.adapt_inner(task)

            # 2. Evaluate on adapted policy (query rollout)
            query_traj = self.rollout_trajectory(task, params=adapted_params, explore=True)
            loss_k, _, _, _ = self.compute_loss(query_traj)

            batch_losses.append(loss_k)
            avg_cost_val += query_traj["episode_cost"]

        # Outer meta-update: theta <- theta - beta * grad(sum L_k)
        meta_loss = torch.stack(batch_losses).mean()
        meta_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.meta_optimizer.step()

        return {
            "meta_loss": float(meta_loss.item()),
            "avg_cost": avg_cost_val / len(task_batch),
        }

    def train(
        self,
        train_tasks: List[HEMSTask],
        num_episodes: int = 1000,
        batch_size: int = 5,
        log_interval: int = 50,
    ) -> List[float]:
        """
        Full Meta-RL training loop (1000 episodes).
        Returns list of training episode costs.
        """
        cost_history = []
        rng = np.random.default_rng(self.cfg.seed)

        for ep in range(1, num_episodes + 1):
            batch_indices = rng.choice(len(train_tasks), size=batch_size, replace=True)
            task_batch = [train_tasks[i] for i in batch_indices]

            metrics = self.train_meta_step(task_batch)
            cost_history.append(metrics["avg_cost"])

            if ep % log_interval == 0 or ep == 1:
                print(f"[Meta-RL Train] Episode {ep:4d}/{num_episodes} | Avg Cost: {metrics['avg_cost']:6.2f} A$ | Meta Loss: {metrics['meta_loss']:.4f}")

        return cost_history

    def evaluate(
        self,
        test_tasks: List[HEMSTask],
        num_episodes: int = 100,
        fine_tune_steps: int = 1,
    ) -> List[float]:
        """
        Meta-Testing evaluation on unseen tasks (100 episodes).
        Evaluates rapid adaptation performance.
        """
        cost_history = []
        rng = np.random.default_rng(self.cfg.seed + 999)

        for ep in range(1, num_episodes + 1):
            task = test_tasks[rng.choice(len(test_tasks))]

            if fine_tune_steps > 0:
                adapted_params = self.adapt_inner(task)
            else:
                adapted_params = None

            traj = self.rollout_trajectory(task, params=adapted_params, explore=False)
            cost_history.append(traj["episode_cost"])

        mean_cost = float(np.mean(cost_history))
        print(f"[Meta-RL Test] Over {num_episodes} episodes | Mean Energy Cost: {mean_cost:6.2f} A$")
        return cost_history
