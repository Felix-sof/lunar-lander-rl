"""Plot evaluation curves: mean over seeds (line) and min-max range (band).

Usage:
    python -m scripts.plot --group "PPO=runs/ppo_s*" --group "DQN=runs/dqn_s*" \
        --out figures/ppo_vs_dqn.png --wall-time
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Validated categorical palette (fixed order, never cycled) and neutral inks.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
SOLVED = 200.0


def load_eval(run_dir: Path) -> dict[str, np.ndarray]:
    with (run_dir / "eval.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {key: np.array([float(r[key]) for r in rows]) for key in rows[0]}


def aggregate(
    runs: list[dict[str, np.ndarray]], x_key: str, y_key: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate every seed onto the shortest run's x grid; return x, mean, min, max."""
    grid = min((r[x_key] for r in runs), key=lambda x: x[-1])
    ys = np.stack([np.interp(grid, r[x_key], r[y_key]) for r in runs])
    return grid, ys.mean(axis=0), ys.min(axis=0), ys.max(axis=0)


def style_axis(ax: plt.Axes, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", action="append", required=True, help='"Label=glob"')
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="LunarLander-v3: evaluation return")
    parser.add_argument("--wall-time", action="store_true", help="add a return vs time panel")
    parser.add_argument("--baseline", type=Path, default=Path("runs/random_baseline.json"))
    args = parser.parse_args()

    groups: list[tuple[str, list[dict[str, np.ndarray]]]] = []
    for spec in args.group:
        label, pattern = spec.split("=", 1)
        dirs = sorted(Path(p) for p in glob.glob(pattern) if (Path(p) / "eval.csv").exists())
        if not dirs:
            raise SystemExit(f"no runs match {pattern}")
        groups.append((f"{label} (n={len(dirs)})", [load_eval(d) for d in dirs]))

    panels = [("timesteps", "mean_return", "environment steps", "mean return (10 episodes)")]
    panels.append(("timesteps", "landed_rate", "environment steps", "landed rate"))
    if args.wall_time:
        panels.append(("wall_time_s", "mean_return", "wall-clock minutes", "mean return"))

    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.2), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    baseline = None
    if args.baseline.exists():
        baseline = json.loads(args.baseline.read_text())["mean_return"]

    for ax, (x_key, y_key, xlabel, ylabel) in zip(axes[0], panels, strict=True):
        scale = 1 / 60 if x_key == "wall_time_s" else 1.0
        for color, (label, runs) in zip(SERIES_COLORS, groups, strict=False):
            x, mean, lo, hi = aggregate(runs, x_key, y_key)
            ax.fill_between(x * scale, lo, hi, color=color, alpha=0.18, linewidth=0)
            ax.plot(x * scale, mean, color=color, linewidth=2, label=label)
        if y_key == "mean_return":
            ax.axhline(SOLVED, color=INK_MUTED, linewidth=1, linestyle="--")
            ax.annotate(
                "solved (200)",
                (0.01, SOLVED),
                xycoords=("axes fraction", "data"),
                xytext=(0, 4),
                textcoords="offset points",
                color=INK_MUTED,
                fontsize=8,
            )
            if baseline is not None:
                ax.axhline(baseline, color=INK_MUTED, linewidth=1, linestyle=":")
                ax.annotate(
                    f"random agent ({baseline:.0f})",
                    (0.01, baseline),
                    xycoords=("axes fraction", "data"),
                    xytext=(0, 4),
                    textcoords="offset points",
                    color=INK_MUTED,
                    fontsize=8,
                )
        if y_key == "landed_rate":
            ax.set_ylim(-0.03, 1.03)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        if x_key == "timesteps":
            ax.xaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:g}M" if v else "0")
            )
        style_axis(ax, xlabel, ylabel)

    axes[0][0].legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    fig.suptitle(args.title + "  (line: seed mean, band: min-max)", color=INK, fontsize=12)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, facecolor=SURFACE)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
