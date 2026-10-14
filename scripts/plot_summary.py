"""One-glance summary chart of every agent, built from the JSON files in results/.

Bars are the mean over seeds of the 100-episode mean return; dots are individual seeds.

Usage:
    python -m scripts.plot_summary --out figures/summary.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from scripts.plot import INK, INK_MUTED, SERIES_COLORS, SOLVED, SURFACE, style_axis  # noqa: E402

PPO_COLOR, DQN_COLOR, BASELINE_COLOR = SERIES_COLORS[0], SERIES_COLORS[1], "#a3a29c"
SEEDS = (0, 1, 2)


def load_results(results_dir: Path) -> dict[str, float]:
    """Map checkpoint path -> 100-episode mean return, across all result files."""
    returns: dict[str, float] = {}
    for path in results_dir.glob("*.json"):
        data = json.loads(path.read_text())
        if "mean_return" in data:  # random baseline: a single flat summary
            returns["random"] = data["mean_return"]
        else:
            returns.update({k: v["mean_return"] for k, v in data.items()})
    return returns


def final(run: str) -> str:
    return f"runs/{run}/checkpoints/final.zip"


def best_checkpoint(returns: dict[str, float], run: str) -> float:
    prefix = f"runs/{run}/checkpoints/"
    return max(v for k, v in returns.items() if k.startswith(prefix))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("figures/summary.png"))
    args = parser.parse_args()

    r = load_results(args.results)
    rows: list[tuple[str, list[float], str]] = [
        ("PPO, fuel penalty 3x", [r[final(f"ppo_fuel3_s{s}")] for s in SEEDS], PPO_COLOR),
        ("PPO", [r[final(f"ppo_s{s}")] for s in SEEDS], PPO_COLOR),
        ("DQN, best checkpoint", [best_checkpoint(r, f"dqn_s{s}") for s in SEEDS], DQN_COLOR),
        ("PPO, free fuel (0x)", [r[final(f"ppo_fuel0_s{s}")] for s in SEEDS], PPO_COLOR),
        ("PPO, fuel penalty 10x", [r[final(f"ppo_fuel10_s{s}")] for s in SEEDS], PPO_COLOR),
        ("Random agent", [r["random"]], BASELINE_COLOR),
        ("DQN, final model (1M)", [r[final(f"dqn_s{s}")] for s in SEEDS], DQN_COLOR),
    ]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    fig.patch.set_facecolor(SURFACE)
    y = np.arange(len(rows))[::-1]
    for yi, (_, values, color) in zip(y, rows, strict=True):
        mean = float(np.mean(values))
        ax.barh(yi, mean, height=0.6, color=color, alpha=0.85)
        if len(values) > 1:
            ax.scatter(values, [yi] * len(values), s=28, color=INK, zorder=3, linewidths=0)
        offset = 12 if mean >= 0 else -12
        ax.annotate(
            f"{mean:+.0f}",
            (max(mean, max(values)) if mean >= 0 else min(mean, min(values)), yi),
            xytext=(offset, 0),
            textcoords="offset points",
            va="center",
            ha="left" if mean >= 0 else "right",
            color=INK,
            fontsize=10,
            fontweight="bold",
        )
    ax.axvline(SOLVED, color=INK_MUTED, linewidth=1, linestyle="--")
    ax.annotate(
        "solved (200)",
        (SOLVED, len(rows) - 0.45),
        xytext=(4, 0),
        textcoords="offset points",
        color=INK_MUTED,
        fontsize=8,
    )
    ax.axvline(0, color=INK_MUTED, linewidth=0.8)
    ax.set_yticks(y, [label for label, _, _ in rows], color=INK, fontsize=10)
    ax.set_xlim(-720, 380)
    ax.set_ylim(-0.6, len(rows) - 0.2)
    style_axis(ax, "mean return over 100 episodes (bar: mean of 3 seeds, dots: seeds)", "")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("LunarLander-v3: every agent at a glance", color=INK, fontsize=12, loc="left")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
