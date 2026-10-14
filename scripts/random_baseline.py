"""Measure a uniformly random agent: the zero point every trained agent must beat.

Usage:
    python -m scripts.random_baseline --episodes 100
    python -m scripts.random_baseline --episodes 3 --watch
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lunar.envs import make_env
from lunar.evaluation import format_summary, run_episode, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--watch", action="store_true", help="render in a window")
    parser.add_argument("--out", type=Path, default=Path("runs/random_baseline.json"))
    args = parser.parse_args()

    env = make_env(render_mode="human" if args.watch else None)
    env.action_space.seed(args.seed)
    stats = [
        run_episode(env, lambda _obs: env.action_space.sample(), seed=args.seed + i)
        for i in range(args.episodes)
    ]
    env.close()

    summary = summarize(stats)
    print(format_summary(summary))
    if not args.watch:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2))
        print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
