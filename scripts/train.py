"""Train PPO or DQN on LunarLander.

Examples:
    python -m scripts.train --algo ppo --seed 0 --timesteps 1000000
    python -m scripts.train --algo dqn --seed 0 --timesteps 500000
    python -m scripts.train --algo ppo --fuel-main-scale 0 --run-name ppo_fuel0_s0
    python -m scripts.train --algo ppo --seed 0 --timesteps 2000000 --resume

--timesteps is always the absolute total; --resume trains only the missing steps.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from lunar.training import TrainConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", choices=["ppo", "dqn"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fuel-main-scale", type=float, default=1.0)
    parser.add_argument("--fuel-side-scale", type=float, default=1.0)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--run-name", default=None, help="default: <algo>_s<seed>")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = TrainConfig(
        algo=args.algo,
        run_dir=args.runs_dir / (args.run_name or f"{args.algo}_s{args.seed}"),
        total_timesteps=args.timesteps,
        seed=args.seed,
        n_envs=args.n_envs,
        device=args.device,
        fuel_main_scale=args.fuel_main_scale,
        fuel_side_scale=args.fuel_side_scale,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        torch_threads=args.torch_threads,
    )
    start = time.perf_counter()
    model = train(config, resume=args.resume)
    minutes = (time.perf_counter() - start) / 60
    print(f"done: {model.num_timesteps} timesteps in {minutes:.1f} min -> {config.run_dir}")


if __name__ == "__main__":
    main()
