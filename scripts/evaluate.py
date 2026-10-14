"""Evaluate trained models over many episodes on the original reward.

Usage:
    python -m scripts.evaluate runs/ppo_s0 runs/dqn_s0 --episodes 100
    python -m scripts.evaluate runs/ppo_s0/checkpoints/step_100000.zip
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lunar.callbacks import evaluate_model
from lunar.evaluation import format_summary
from lunar.training import load_model, resolve_model_path

# Distinct from the seeds used for the eval curve during training.
FINAL_EVAL_SEED = 50_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", type=Path, help="run dirs or checkpoint files")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--out", type=Path, default=None, help="write all summaries as JSON")
    args = parser.parse_args()

    results: dict[str, dict[str, float]] = {}
    for path in args.models:
        summary = evaluate_model(load_model(path), args.episodes, seed=FINAL_EVAL_SEED)
        name = resolve_model_path(path).as_posix()
        results[name] = summary
        print(f"== {name}\n{format_summary(summary)}\n", flush=True)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
