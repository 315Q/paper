"""
Experiment 2: Cross-Season Generalization & Transferability.
Reproduces Tables II & III, and Figs. 8 & 9 from Section IV-C of IEEE TCSI 2023 paper:
- Case 1: Train on Spring, Summer, Autumn -> Test on Winter (Table II, Fig. 8)
- Case 2: Train on Spring, Autumn, Winter -> Test on Summer (Table III, Fig. 9)
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


def run_season_transfer_experiment(
    train_episodes: int = 1000,
    test_episodes: int = 100,
    cfg: Optional[HEMSConfig] = None,
    save_fig: bool = True,
):
    print("=" * 70)
    print(">>> RUNNING EXPERIMENT 2: CROSS-SEASON TRANSFER (TABLES II & III) <<<")
    print("=" * 70)

    cfg = cfg or default_cfg
    tasks = TaskDistribution(cfg=cfg, seed=cfg.seed)
    visualizer = HEMSVisualizer()
    rng = np.random.default_rng(cfg.seed)

    # -------------------------------------------------------------------------
    # CASE 1: Transfer from Spring, Summer, Autumn -> Winter
    # -------------------------------------------------------------------------
    print("\n" + "#" * 60)
    print(">>> [CASE 1] Train on Spring, Summer, Autumn -> Test on Winter")
    print("#" * 60)
    train_tasks_w, test_tasks_w = tasks.get_tasks_for_season_transfer(test_season="winter")

    meta_trainer_w = MetaRLTrainer(cfg=cfg)
    meta_train_costs_w = meta_trainer_w.train(
        train_tasks=train_tasks_w,
        num_episodes=train_episodes,
        batch_size=cfg.meta_batch_size,
        log_interval=max(1, train_episodes // 5),
    )
    meta_test_costs_w = meta_trainer_w.evaluate(test_tasks=test_tasks_w, num_episodes=test_episodes)

    a2c_w = StandardA2CBaseline(cfg=cfg)
    a2c_train_costs_w = a2c_w.train(tasks=train_tasks_w, num_episodes=train_episodes)
    a2c_test_costs_w = a2c_w.evaluate(test_tasks=test_tasks_w, num_episodes=test_episodes)

    meta_train_mean_w = float(np.mean(meta_train_costs_w[-100:])) if len(meta_train_costs_w) >= 100 else float(np.mean(meta_train_costs_w))
    meta_test_mean_w = float(np.mean(meta_test_costs_w))
    a2c_train_mean_w = float(np.mean(a2c_train_costs_w[-100:])) if len(a2c_train_costs_w) >= 100 else float(np.mean(a2c_train_costs_w))
    a2c_test_mean_w = float(np.mean(a2c_test_costs_w))

    cols = ["Algorithm", "Cost During Training (A$) [Ours / Paper]", "Cost During Testing (A$) [Ours / Paper]"]
    rows_w = [
        ["Meta-RL (Proposed)", f"**{meta_train_mean_w:.2f}** / 11.72", f"**{meta_test_mean_w:.2f}** / 9.29"],
        ["Standard AC [35]", f"{a2c_train_mean_w:.2f} / 18.42", f"{a2c_test_mean_w:.2f} / 16.29"],
        ["TRPO [29] (Paper Ref)", "- / 17.84", "- / 19.10"],
        ["PPO [41] (Paper Ref)", "- / 17.27", "- / 11.74"],
        ["DDPG [42] (Paper Ref)", "- / 18.23", "- / 16.10"],
    ]
    visualizer.print_benchmark_table("TABLE II: HOME ENERGY COST WITHIN 24 HOURS IN WINTER (TESTING)", cols, rows_w)

    if save_fig:
        max_tr_w = max(meta_train_costs_w + a2c_train_costs_w) if (meta_train_costs_w and a2c_train_costs_w) else 30.0
        max_te_w = max(meta_test_costs_w + a2c_test_costs_w) if (meta_test_costs_w and a2c_test_costs_w) else 25.0
        visualizer.plot_transfer_comparison(
            train_curves={"Meta-RL": meta_train_costs_w, "AC": a2c_train_costs_w},
            test_curves={"Meta-RL": meta_test_costs_w, "AC": a2c_test_costs_w},
            title="Customers' home energy cost in spring, summer, and autumn during training, and in winter during testing (Fig. 8)",
            filename_prefix="fig8_season_transfer_winter",
            y_lim_train=(0, max(25.0, float(max_tr_w) * 1.15)),
            y_lim_test=(0, max(20.0, float(max_te_w) * 1.15)),
        )

    # -------------------------------------------------------------------------
    # CASE 2: Transfer from Spring, Autumn, Winter -> Summer
    # -------------------------------------------------------------------------
    print("\n" + "#" * 60)
    print(">>> [CASE 2] Train on Spring, Autumn, Winter -> Test on Summer")
    print("#" * 60)
    train_tasks_s, test_tasks_s = tasks.get_tasks_for_season_transfer(test_season="summer")

    meta_trainer_s = MetaRLTrainer(cfg=cfg)
    meta_train_costs_s = meta_trainer_s.train(
        train_tasks=train_tasks_s,
        num_episodes=train_episodes,
        batch_size=cfg.meta_batch_size,
        log_interval=max(1, train_episodes // 5),
    )
    meta_test_costs_s = meta_trainer_s.evaluate(test_tasks=test_tasks_s, num_episodes=test_episodes)

    a2c_s = StandardA2CBaseline(cfg=cfg)
    a2c_train_costs_s = a2c_s.train(tasks=train_tasks_s, num_episodes=train_episodes)
    a2c_test_costs_s = a2c_s.evaluate(test_tasks=test_tasks_s, num_episodes=test_episodes)

    meta_train_mean_s = float(np.mean(meta_train_costs_s[-100:])) if len(meta_train_costs_s) >= 100 else float(np.mean(meta_train_costs_s))
    meta_test_mean_s = float(np.mean(meta_test_costs_s))
    a2c_train_mean_s = float(np.mean(a2c_train_costs_s[-100:])) if len(a2c_train_costs_s) >= 100 else float(np.mean(a2c_train_costs_s))
    a2c_test_mean_s = float(np.mean(a2c_test_costs_s))

    rows_s = [
        ["Meta-RL (Proposed)", f"**{meta_train_mean_s:.2f}** / 11.72", f"**{meta_test_mean_s:.2f}** / 8.79"],
        ["Standard AC [35]", f"{a2c_train_mean_s:.2f} / 18.42", f"{a2c_test_mean_s:.2f} / 14.54"],
        ["TRPO [29] (Paper Ref)", "- / 10.07", "- / 18.89"],
        ["PPO [41] (Paper Ref)", "- / 11.90", "- / 29.05"],
        ["DDPG [42] (Paper Ref)", "- / 10.96", "- / 9.95"],
    ]
    visualizer.print_benchmark_table("TABLE III: HOME ENERGY COST WITHIN 24 HOURS IN SUMMER (TESTING)", cols, rows_s)

    if save_fig:
        max_tr_s = max(meta_train_costs_s + a2c_train_costs_s) if (meta_train_costs_s and a2c_train_costs_s) else 30.0
        max_te_s = max(meta_test_costs_s + a2c_test_costs_s) if (meta_test_costs_s and a2c_test_costs_s) else 25.0
        visualizer.plot_transfer_comparison(
            train_curves={"Meta-RL": meta_train_costs_s, "AC": a2c_train_costs_s},
            test_curves={"Meta-RL": meta_test_costs_s, "AC": a2c_test_costs_s},
            title="Customers' home energy cost in spring, autumn, and winter during training, in summer during testing (Fig. 9)",
            filename_prefix="fig9_season_transfer_summer",
            y_lim_train=(0, max(25.0, float(max_tr_s) * 1.15)),
            y_lim_test=(0, max(20.0, float(max_te_s) * 1.15)),
        )

    return {
        "winter_case1": {
            "meta_train": meta_train_mean_w,
            "meta_test": meta_test_mean_w,
            "a2c_train": a2c_train_mean_w,
            "a2c_test": a2c_test_mean_w,
        },
        "summer_case2": {
            "meta_train": meta_train_mean_s,
            "meta_test": meta_test_mean_s,
            "a2c_train": a2c_train_mean_s,
            "a2c_test": a2c_test_mean_s,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--test_episodes", type=int, default=100)
    args = parser.parse_args()

    run_season_transfer_experiment(train_episodes=args.episodes, test_episodes=args.test_episodes)
