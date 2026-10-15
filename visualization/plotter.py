"""
Publication-Quality Visualization Suite for Meta-RL HEMS.
Renders and exports figures matching Fig. 7, 8, 9, 10 and formatting Tables I, II, III, IV
from IEEE TCSI 2023 paper:
'Meta-Reinforcement Learning-Based Transferable Scheduling Strategy for Energy Management'
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any


def smooth_curve(points: List[float], factor: float = 0.85) -> np.ndarray:
    """Applies exponential moving average for smooth RL reward/cost curves."""
    smoothed = []
    for p in points:
        if smoothed:
            prev = smoothed[-1]
            smoothed.append(prev * factor + p * (1 - factor))
        else:
            smoothed.append(p)
    return np.array(smoothed)


class HEMSVisualizer:
    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Configure IEEE Transactions publication aesthetic
        plt.rcParams.update({
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 12,
            "lines.linewidth": 1.5,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
        })

    def plot_transfer_comparison(
        self,
        train_curves: Dict[str, List[float]],
        test_curves: Dict[str, List[float]],
        title: str,
        filename_prefix: str,
        y_lim_train: Optional[Tuple[float, float]] = None,
        y_lim_test: Optional[Tuple[float, float]] = None,
    ):
        """
        Generates 2-row vertically stacked figure strictly matching Fig. 7, Fig. 8, and Fig. 9:
        - Top subplot: Averaged episode cost (A$) during training (0 - 1000)
        - Bottom subplot: Averaged episode cost (A$) during testing (0 - 100)
        """
        fig, (ax_train, ax_test) = plt.subplots(2, 1, figsize=(8.0, 6.2), sharex=False)

        # Style palette strictly matching IEEE TCSI 2023 paper Figs. 7, 8, 9:
        colors = {
            "Meta-RL": "#FF0000",  # Red dotted
            "AC": "#008000",       # Green solid
            "PPO": "#0000FF",      # Blue dashed
            "DDPG": "#800000",     # Dark maroon dash-dot
            "TRPO": "#FFB6C1",     # Light pink
        }
        linestyles = {
            "Meta-RL": ":",
            "AC": "-",
            "PPO": "--",
            "DDPG": "-.",
            "TRPO": "-",
        }

        # 1. Training Plot
        for name, data in train_curves.items():
            col = colors.get(name, "#333333")
            ls = linestyles.get(name, "-")
            x = np.arange(1, len(data) + 1)
            y_smooth = smooth_curve(data, factor=0.92)
            ax_train.plot(x, y_smooth, label=name, color=col, linestyle=ls, alpha=0.95)

        ax_train.set_ylabel("Averaged episode cost (A$)")
        ax_train.set_xlabel("Training Episode")
        ax_train.grid(True)
        ax_train.legend(loc="upper right", framealpha=0.9, ncol=len(train_curves))
        if y_lim_train:
            ax_train.set_ylim(y_lim_train)

        # 2. Testing Plot
        for name, data in test_curves.items():
            col = colors.get(name, "#333333")
            ls = linestyles.get(name, "-")
            x = np.arange(1, len(data) + 1)
            y_smooth = smooth_curve(data, factor=0.90)
            ax_test.plot(x, y_smooth, label=name, color=col, linestyle=ls, alpha=0.95)

        ax_test.set_ylabel("Averaged episode cost (A$)")
        ax_test.set_xlabel("Testing Episode")
        ax_test.grid(True)
        if y_lim_test:
            ax_test.set_ylim(y_lim_test)

        fig.suptitle(title, fontsize=10.5, fontweight="bold", y=0.98, wrap=True)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        png_path = os.path.join(self.output_dir, f"{filename_prefix}.png")
        pdf_path = os.path.join(self.output_dir, f"{filename_prefix}.pdf")
        fig.savefig(png_path, dpi=300)
        fig.savefig(pdf_path)
        plt.close(fig)
        print(f"[Visualizer] Saved high-res figure to:\n  - {png_path}\n  - {pdf_path}")

    def plot_comfort_levels(
        self,
        train_data: Dict[str, List[float]],
        test_data: Dict[str, List[float]],
        filename_prefix: str = "fig10_comfort_tradeoff",
    ):
        """Generates Fig. 10 matching customer comfort parameter alpha comparison."""
        fig, (ax_train, ax_test) = plt.subplots(2, 1, figsize=(7.2, 5.8))
        def get_style(lbl: str):
            if "0.2" in lbl:
                return {"color": "#800000", "ls": "-."}
            elif "2" in lbl:
                return {"color": "#0000FF", "ls": "--"}
            else:
                return {"color": "#008000", "ls": "-"}

        for label, curve in train_data.items():
            s = get_style(label)
            x_tr = np.arange(1, len(curve) + 1)
            ax_train.plot(x_tr, smooth_curve(curve, 0.92), label=label, color=s["color"], linestyle=s["ls"], linewidth=1.8)
        ax_train.set_ylabel("Averaged episode cost (A$)")
        ax_train.set_xlabel("Training Episode")
        ax_train.grid(True)
        ax_train.legend(loc="upper right", framealpha=0.9)

        for label, curve in test_data.items():
            s = get_style(label)
            x_te = np.arange(1, len(curve) + 1)
            ax_test.plot(x_te, smooth_curve(curve, 0.90), label=label, color=s["color"], linestyle=s["ls"], linewidth=1.8)
        ax_test.set_ylabel("Averaged episode cost (A$)")
        ax_test.set_xlabel("Testing Episode")
        ax_test.grid(True)

        fig.suptitle("Home energy cost with different comfort levels (Fig. 10)", fontsize=10.5, fontweight="bold", y=0.98, wrap=True)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        png_path = os.path.join(self.output_dir, f"{filename_prefix}.png")
        fig.savefig(png_path, dpi=300)
        plt.close(fig)
        print(f"[Visualizer] Saved Fig 10 to {png_path}")

    @staticmethod
    def print_benchmark_table(title: str, columns: List[str], rows: List[List[Any]]) -> str:
        """Formats and prints markdown benchmark tables corresponding to Tables I, II, III, IV."""
        header = "| " + " | ".join(columns) + " |"
        sep = "| " + " | ".join(["---"] * len(columns)) + " |"
        data_rows = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
        table_md = f"\n### {title}\n\n" + "\n".join([header, sep] + data_rows) + "\n"
        print(table_md)
        return table_md
