"""
Experiment 1: Generalization & Transferability Between Different Customers.
Reproduces Table I and Fig. 7 from Section IV-B of IEEE TCSI 2023 paper:
- Meta-Training: Customer 1, 2, 3 (1000 episodes)
- Meta-Testing: Customer 4 (100 episodes)
"""

import os
import argparse
import numpy as np
from typing import Optional
from config import HEMSConfig, default_cfg
from data.task_distribution import TaskDistribution
from algorithms.meta_rl import MetaRLTrainer
from algorithms.baselines import StandardA2CBaseline
from visualization.plotter import HEMSVisualizer


def run_customer_transfer_experiment(
    train_episodes: int = 1000,
    test_episodes: int = 100,
    cfg: Optional[HEMSConfig] = None,
    save_fig: bool = True,
):
    print("=" * 70)
    print(">>> RUNNING EXPERIMENT 1: CUSTOMER TRANSFER (TABLE I & FIG. 7) <<<")
    print(f"    Train: Customers 1, 2, 3 ({train_episodes} episodes)")
    print(f"    Test:  Customer 4 ({test_episodes} episodes)")
    print("=" * 70)

    cfg = cfg or default_cfg
    tasks = TaskDistribution(cfg=cfg, seed=cfg.seed)
    train_tasks, test_tasks = tasks.get_tasks_for_customer_transfer()

    # 1. Train Meta-RL
    print("\n--- [1/2] Training Meta-RL Agent (Algorithm 1) ---")
    meta_trainer = MetaRLTrainer(cfg=cfg)
    meta_train_costs = meta_trainer.train(
        train_tasks=train_tasks,
        num_episodes=train_episodes,
        batch_size=cfg.meta_batch_size,
        log_interval=max(1, train_episodes // 10),
    )
    print("\n--- Evaluating Meta-RL on Unseen Customer 4 ---")
    meta_test_costs = meta_trainer.evaluate(test_tasks=test_tasks, num_episodes=test_episodes)

    # 2. Train Standard A2C Baseline
    print("\n--- [2/2] Training Standard Non-Meta A2C Baseline ---")
    a2c_baseline = StandardA2CBaseline(cfg=cfg)
    a2c_train_costs = a2c_baseline.train(tasks=train_tasks, num_episodes=train_episodes)
    print("\n--- Evaluating Standard A2C on Unseen Customer 4 ---")
    a2c_test_costs = a2c_baseline.evaluate(test_tasks=test_tasks, num_episodes=test_episodes)

    meta_train_mean = float(np.mean(meta_train_costs[-100:])) if len(meta_train_costs) >= 100 else float(np.mean(meta_train_costs))
    meta_test_mean = float(np.mean(meta_test_costs))
    a2c_train_mean = float(np.mean(a2c_train_costs[-100:])) if len(a2c_train_costs) >= 100 else float(np.mean(a2c_train_costs))
    a2c_test_mean = float(np.mean(a2c_test_costs))

    # Format Benchmark Table I
    columns = ["Algorithm", "Cost During Training (A$) [Ours / Paper]", "Cost During Testing (A$) [Ours / Paper]"]
    rows = [
        ["Meta-RL (Proposed)", f"**{meta_train_mean:.2f}** / 11.11", f"**{meta_test_mean:.2f}** / 9.42"],
        ["Standard AC [35]", f"{a2c_train_mean:.2f} / 24.41", f"{a2c_test_mean:.2f} / 25.31"],
        ["TRPO [29] (Paper Ref)", "- / 73.18", "- / 77.80"],
        ["PPO [41] (Paper Ref)", "- / 33.14", "- / 40.80"],
        ["DDPG [42] (Paper Ref)", "- / 18.61", "- / 28.46"],
    ]

    visualizer = HEMSVisualizer()
    visualizer.print_benchmark_table("TABLE I: HOME ENERGY COST OF THE CUSTOMERS WITHIN 24 HOURS", columns, rows)

    if save_fig:
        train_dict = {
            "Meta-RL": meta_train_costs,
            "AC": a2c_train_costs,
        }
        test_dict = {
            "Meta-RL": meta_test_costs,
            "AC": a2c_test_costs,
        }
        max_tr = max(meta_train_costs + a2c_train_costs) if (meta_train_costs and a2c_train_costs) else 35.0
        max_te = max(meta_test_costs + a2c_test_costs) if (meta_test_costs and a2c_test_costs) else 35.0
        visualizer.plot_transfer_comparison(
            train_curves=train_dict,
            test_curves=test_dict,
            title="The home energy cost of Customer 1, 2, and 3 during training, and Customer 4 during testing (Fig. 7)",
            filename_prefix="fig7_customer_transfer",
            y_lim_train=(0, max(30.0, float(max_tr) * 1.15)),
            y_lim_test=(0, max(30.0, float(max_te) * 1.15)),
        )

    return {
        "meta_train": meta_train_mean,
        "meta_test": meta_test_mean,
        "a2c_train": a2c_train_mean,
        "a2c_test": a2c_test_mean,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--test_episodes", type=int, default=100, help="Number of testing episodes")
    args = parser.parse_args()

    run_customer_transfer_experiment(train_episodes=args.episodes, test_episodes=args.test_episodes)
