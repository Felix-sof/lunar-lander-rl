from __future__ import annotations

import csv
from pathlib import Path

from lunar.training import TrainConfig, build_model, remaining_timesteps, train

TINY_PPO = {"n_steps": 64, "batch_size": 32, "n_epochs": 1}


def _config(tmp_path: Path, algo: str, total: int, **overrides: object) -> TrainConfig:
    return TrainConfig(
        algo=algo,
        run_dir=tmp_path / algo,
        total_timesteps=total,
        n_envs=1,
        eval_freq=128,
        eval_episodes=1,
        milestones=[128],
        latest_every=64,
        **overrides,  # type: ignore[arg-type]
    )


def _eval_timesteps(run_dir: Path) -> list[int]:
    with (run_dir / "eval.csv").open(newline="") as f:
        return [int(row["timesteps"]) for row in csv.DictReader(f)]


def test_remaining_timesteps() -> None:
    assert remaining_timesteps(1000, 0) == 1000
    assert remaining_timesteps(1000, 600) == 400
    assert remaining_timesteps(1000, 1200) == 0


def test_ppo_resume_reaches_absolute_total(tmp_path: Path) -> None:
    first = train(_config(tmp_path, "ppo", 128, hyperparams_override=TINY_PPO))
    assert first.num_timesteps == 128

    # --timesteps is the absolute target: resuming to 256 trains 128 more, not 256 more.
    resumed = train(_config(tmp_path, "ppo", 256, hyperparams_override=TINY_PPO), resume=True)
    assert resumed.num_timesteps == 256

    run_dir = tmp_path / "ppo"
    assert (run_dir / "checkpoints" / "step_128.zip").exists()
    assert (run_dir / "checkpoints" / "final.zip").exists()
    steps = _eval_timesteps(run_dir)
    assert steps == sorted(set(steps)), "eval rows must be unique and increasing"
    assert steps[0] == 0 and steps[-1] == 256


def test_dqn_resume_restores_replay_buffer(tmp_path: Path) -> None:
    override = {"buffer_size": 1000, "learning_starts": 50}
    train(_config(tmp_path, "dqn", 200, hyperparams_override=override))
    model = build_model(_config(tmp_path, "dqn", 400, hyperparams_override=override), True)
    assert model.num_timesteps == 200
    assert model.replay_buffer.size() == 200
    model.get_env().close()
