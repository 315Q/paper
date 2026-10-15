"""
Experiment 3: Comfort Level vs. Electricity Cost Trade-Off.
Reproduces Table IV and Fig. 10 from Section IV-D of IEEE TCSI 2023 paper:
- Comfort parameter alpha in {2.0, 1.0, 0.2}
"""

import os
import argparse
import copy
import numpy as np
from typing import Optional
from config import HEMSConfig, default_cfg
from data.task_distribution import TaskDistribution
from algorithms.meta_rl import MetaRLTrainer
from visualization.plotter import HEMSVisualizer


def run_comfort_tradeoff_experiment(
    train_episodes: int = 1000,
    test_episodes: int = 100,
    save_fig: bool = True,
):
    print("=" * 70)
    print(">>> RUNNING EXPERIMENT 3: COMFORT LEVEL TRADE-OFF (TABLE IV & FIG. 10) <<<")
    print("=" * 70)

    alpha_values = [2.0, 1.0, 0.2]
    train_curves_dict = {}
    test_curves_dict = {}
    results_summary = []

    for alpha in alpha_values:
        print(f"\n--- Running Meta-RL with comfort parameter alpha = {alpha} ---")
        cfg = copy.deepcopy(default_cfg)
        cfg.alpha_td_max = alpha

        tasks = TaskDistribution(cfg=cfg, seed=cfg.seed)
        train_tasks, test_tasks = tasks.get_tasks_for_customer_transfer()

        trainer = MetaRLTrainer(cfg=cfg)
        train_costs = trainer.train(
            train_tasks=train_tasks,
            num_episodes=train_episodes,
            batch_size=cfg.meta_batch_size,
            log_interval=max(1, train_episodes // 5),
        )
        test_costs = trainer.evaluate(test_tasks=test_tasks, num_episodes=test_episodes)

        label = rf"$\alpha={alpha}$"
        train_curves_dict[label] = train_costs
        test_curves_dict[label] = test_costs

        tr_mean = float(np.mean(train_costs[-100:]))
        te_mean = float(np.mean(test_costs))
        results_summary.append((alpha, tr_mean, te_mean))

    # Paper Table IV reference:
    # alpha=2: 22.76, 24.44
    # alpha=1: 15.15, 11.81
    # alpha=0.2: 7.68, 7.65
    ref_table = {
        2.0: ("22.76", "24.44"),
        1.0: ("15.15", "11.81"),
        0.2: ("7.68", "7.65"),
    }

    cols = ["alpha", "Cost During Training (A$)", "Cost During Testing (A$)"]
    rows = []
    for alpha, tr, te in results_summary:
        ref_tr, ref_te = ref_table.get(alpha, ("-", "-"))
        rows.append([str(alpha), f"{tr:.2f} (Paper: {ref_tr})", f"{te:.2f} (Paper: {ref_te})"])

    visualizer = HEMSVisualizer()
    visualizer.print_benchmark_table("TABLE IV: HOME ENERGY COST OF THE CUSTOMERS WITH DIFFERENT COMFORT LEVELS", cols, rows)

    if save_fig:
        visualizer.plot_comfort_levels(
            train_data=train_curves_dict,
            test_data=test_curves_dict,
            filename_prefix="fig10_comfort_tradeoff",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--test_episodes", type=int, default=100)
    args = parser.parse_args()

    run_comfort_tradeoff_experiment(train_episodes=args.episodes, test_episodes=args.test_episodes)
