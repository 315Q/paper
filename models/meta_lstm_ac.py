"""
Recurrent Actor-Critic Network with LSTM Memory for Meta-RL HEMS.
As specified in Section III & IV-A of the paper:
- Shared LSTM feature extractor with 48 hidden units
- Linear Actor head with Softmax output over discrete actions
- Linear Critic head with scalar state-value estimation
- Supports functional execution with adapted weights (theta_k) for MAML inner loops
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List


class MetaLSTMActorCritic(nn.Module):
    def __init__(self, input_dim: int = 9, hidden_dim: int = 48, action_dim: int = 25):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # Shared LSTM cell (48 units)
        self.lstm = nn.LSTMCell(input_dim, hidden_dim)

        # Actor head: 48 -> 25 (Softmax via categorical distribution)
        self.actor_head = nn.Linear(hidden_dim, action_dim)

        # Critic head: 48 -> 1 (Scalar Value)
        self.critic_head = nn.Linear(hidden_dim, 1)

        # Orthogonal initialization for stability
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.orthogonal_(param, gain=0.01 if "actor" in name else 1.0)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

    def init_hidden(self, batch_size: int = 1, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initializes (h_0, c_0) to zeros."""
        dev = device or next(self.parameters()).device
        h_0 = torch.zeros(batch_size, self.hidden_dim, device=dev)
        c_0 = torch.zeros(batch_size, self.hidden_dim, device=dev)
        return h_0, c_0

    def forward(
        self,
        x: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        Args:
            x: Input tensor [batch_size, input_dim] (s_t, a_{t-1}, r_{t-1}, t/T)
            hidden: Tuple (h_t, c_t)
            params: Optional dict of adapted parameters (for MAML inner loop theta_k)
        Returns:
            action_logits: [batch_size, action_dim]
            value: [batch_size, 1]
            next_hidden: Tuple (h_{t+1}, c_{t+1})
        """
        h_prev, c_prev = hidden

        if params is None:
            # Standard PyTorch forward pass using module parameters
            h_next, c_next = self.lstm(x, (h_prev, c_prev))
            action_logits = self.actor_head(h_next)
            value = self.critic_head(h_next)
        else:
            # Functional forward pass using adapted parameter dict
            w_ih = params["lstm.weight_ih"]
            w_hh = params["lstm.weight_hh"]
            b_ih = params["lstm.bias_ih"]
            b_hh = params["lstm.bias_hh"]

            # LSTM cell computation: gates = x @ w_ih.T + b_ih + h_prev @ w_hh.T + b_hh
            gates = F.linear(x, w_ih, b_ih) + F.linear(h_prev, w_hh, b_hh)
            i_gate, f_gate, c_cand, o_gate = gates.chunk(4, dim=-1)

            i = torch.sigmoid(i_gate)
            f = torch.sigmoid(f_gate)
            c = torch.tanh(c_cand)
            o = torch.sigmoid(o_gate)

            c_next = f * c_prev + i * c
            h_next = o * torch.tanh(c_next)

            # Actor & Critic heads
            w_act = params["actor_head.weight"]
            b_act = params["actor_head.bias"]
            w_crit = params["critic_head.weight"]
            b_crit = params["critic_head.bias"]

            action_logits = F.linear(h_next, w_act, b_act)
            value = F.linear(h_next, w_crit, b_crit)

        return action_logits, value, (h_next, c_next)

    def get_parameter_dict(self) -> Dict[str, torch.Tensor]:
        """Returns a cloned dictionary of current parameters for inner loop copying."""
        return {name: param.clone() for name, param in self.named_parameters()}
