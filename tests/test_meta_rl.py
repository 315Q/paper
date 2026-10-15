"""
Unit tests for Meta-RL Model and Inner-Outer Loop Optimization.
"""

import unittest
import numpy as np
import math
import torch
from config import default_cfg
from data.task_distribution import TaskDistribution
from models.meta_lstm_ac import MetaLSTMActorCritic
from algorithms.meta_rl import MetaRLTrainer


class TestMetaRL(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.model = MetaLSTMActorCritic(input_dim=9, hidden_dim=48, action_dim=25).to(self.device)
        self.tasks = TaskDistribution(cfg=default_cfg, seed=42)
        self.trainer = MetaRLTrainer(cfg=default_cfg, model=self.model, device=self.device)

    def test_model_forward_shape(self):
        x = torch.randn(1, 9)
        hidden = self.model.init_hidden(batch_size=1)
        logits, value, next_hidden = self.model(x, hidden)
        self.assertEqual(logits.shape, (1, 25))
        self.assertEqual(value.shape, (1, 1))
        self.assertEqual(next_hidden[0].shape, (1, 48))
        self.assertEqual(next_hidden[1].shape, (1, 48))

    def test_functional_forward(self):
        x = torch.randn(1, 9)
        hidden = self.model.init_hidden(batch_size=1)
        params = self.model.get_parameter_dict()

        # Modify a weight slightly in params dict
        params["actor_head.bias"] = params["actor_head.bias"] + 0.1

        logits_orig, _, _ = self.model(x, hidden, params=None)
        logits_func, _, _ = self.model(x, hidden, params=params)

        self.assertFalse(torch.allclose(logits_orig, logits_func))

    def test_meta_training_step(self):
        train_tasks, _ = self.tasks.get_tasks_for_customer_transfer()
        batch = train_tasks[:2]
        metrics = self.trainer.train_meta_step(batch)
        self.assertIn("meta_loss", metrics)
        self.assertIn("avg_cost", metrics)
        self.assertFalse(np.isnan(metrics["meta_loss"]))


if __name__ == "__main__":
    import numpy as np
    unittest.main()
