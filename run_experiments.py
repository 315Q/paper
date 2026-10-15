"""
Master Execution Pipeline for Meta-RL HEMS Reproduction.
Paper: Meta-Reinforcement Learning-Based Transferable Scheduling Strategy for Energy Management
       (IEEE TCSI 2023, Luolin Xiong et al.)
"""

import sys
import argparse
from experiments.exp1_customer_transfer import run_customer_transfer_experiment
from experiments.exp2_season_transfer import run_season_transfer_experiment
from experiments.exp3_comfort_tradeoff import run_comfort_tradeoff_experiment


def main():
    parser = argparse.ArgumentParser(description="Reproduce IEEE TCSI 2023 Meta-RL HEMS Paper")
    parser.add_argument(
        "--exp",
        type=str,
        default="1",
        choices=["1", "2", "3", "all"],
        help="Experiment to run: 1 (Customer Transfer), 2 (Season Transfer), 3 (Comfort Trade-off), or all",
    )
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes (default: 1000)")
    parser.add_argument("--test_episodes", type=int, default=100, help="Number of testing episodes (default: 100)")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test mode (50 train, 10 test)")

    args = parser.parse_args()

    train_ep = 50 if args.quick else args.episodes
    test_ep = 10 if args.quick else args.test_episodes

    print("*" * 75)
    print("   IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS—I (TCSI 2023) REPRODUCTION")
    print("   Meta-Reinforcement Learning-Based Transferable Scheduling Strategy")
    print("   Author: Luolin Xiong, Yang Tang, Chensheng Liu, et al.")
    print(f"   Mode: {'Quick Smoke Test' if args.quick else 'Full Benchmark Reproduction'}")
    print(f"   Train Episodes: {train_ep} | Test Episodes: {test_ep}")
    print("*" * 75)

    if args.exp in ["1", "all"]:
        run_customer_transfer_experiment(train_episodes=train_ep, test_episodes=test_ep)

    if args.exp in ["2", "all"]:
        run_season_transfer_experiment(train_episodes=train_ep, test_episodes=test_ep)

    if args.exp in ["3", "all"]:
        run_comfort_tradeoff_experiment(train_episodes=train_ep, test_episodes=test_ep)

    print("\n" + "=" * 75)
    print(">>> ALL SPECIFIED EXPERIMENTS SUCCESSFULLY EXECUTED <<<")
    print("Check the 'results/' directory for publication-grade figures and logs:")
    print("  - results/fig7_customer_transfer.png (.pdf)")
    print("  - results/fig8_season_transfer_winter.png (.pdf)")
    print("  - results/fig9_season_transfer_summer.png (.pdf)")
    print("  - results/fig10_comfort_tradeoff.png")
    print("=" * 75)


if __name__ == "__main__":
    main()
