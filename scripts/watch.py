"""Watch a trained agent land, live in a window.

Usage:
    python -m scripts.watch runs/ppo_s0
    python -m scripts.watch runs/ppo_s0/checkpoints/step_100000.zip --episodes 3
    python -m scripts.watch runs/ppo_fuel10_s0 --seed 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lunar.envs import make_env
from lunar.evaluation import run_episode
from lunar.training import load_model, resolve_model_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="run dir or checkpoint file")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true", help="sample actions")
    args = parser.parse_args()

    model = load_model(args.model)
    print(f"model: {resolve_model_path(args.model).as_posix()}")
    env = make_env(render_mode="human")
    deterministic = not args.stochastic
    try:
        for i in range(args.episodes):
            stats = run_episode(
                env,
                lambda obs: int(model.predict(obs, deterministic=deterministic)[0]),
                seed=args.seed + i,
            )
            print(
                f"episode {i + 1}: {stats.outcome.value:<8} return {stats.episode_return:7.1f}  "
                f"steps {stats.length:4d}  main engine {stats.main_engine_steps:4d}  "
                f"side engines {stats.side_engine_steps:4d}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
