"""Training entry point shared by the CLI and the tests."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.logger import configure

from lunar.callbacks import CheckpointCallback, EvalCSVCallback
from lunar.envs import make_train_vec_env
from lunar.power import prevent_sleep

ALGOS: dict[str, type[BaseAlgorithm]] = {"ppo": PPO, "dqn": DQN}

# Tuned hyperparameters from RL Baselines3 Zoo (LunarLander).
HYPERPARAMS: dict[str, dict[str, Any]] = {
    "ppo": {
        "n_steps": 1024,
        "batch_size": 64,
        "gae_lambda": 0.98,
        "gamma": 0.999,
        "n_epochs": 4,
        "ent_coef": 0.01,
    },
    "dqn": {
        "learning_rate": 6.3e-4,
        "batch_size": 128,
        "buffer_size": 50_000,
        "learning_starts": 0,
        "gamma": 0.99,
        "target_update_interval": 250,
        "train_freq": 4,
        "gradient_steps": -1,
        "exploration_fraction": 0.12,
        "exploration_final_eps": 0.1,
        "policy_kwargs": {"net_arch": [256, 256]},
    },
}
DEFAULT_N_ENVS = {"ppo": 16, "dqn": 1}


@dataclass
class TrainConfig:
    algo: str
    run_dir: Path
    total_timesteps: int
    seed: int = 0
    n_envs: int | None = None
    device: str = "cpu"
    fuel_main_scale: float = 1.0
    fuel_side_scale: float = 1.0
    eval_freq: int = 10_000
    eval_episodes: int = 10
    milestones: list[int] = field(
        default_factory=lambda: [10_000, 50_000, 100_000, 250_000, 500_000]
    )
    latest_every: int = 50_000
    hyperparams_override: dict[str, Any] = field(default_factory=dict)
    torch_threads: int | None = None  # cap when several runs share the CPU
    verbose: int = 0

    def resolved_n_envs(self) -> int:
        return self.n_envs if self.n_envs is not None else DEFAULT_N_ENVS[self.algo]


def remaining_timesteps(target_total: int, done: int) -> int:
    """Steps still to train so the model ends at `target_total` overall.

    SB3's `learn(total_timesteps=n, reset_num_timesteps=False)` trains for n *additional*
    steps, so resuming must pass the difference, not the absolute target.
    """
    return max(0, target_total - done)


def resolve_model_path(path: Path) -> Path:
    """Accept a run directory (-> its final checkpoint) or a checkpoint file."""
    if path.is_dir():
        return path / "checkpoints" / "final.zip"
    return path if path.suffix == ".zip" else path.with_suffix(".zip")


def load_model(path: Path, device: str = "cpu") -> BaseAlgorithm:
    """Load a checkpoint; the algorithm is read from the run's config.json."""
    path = resolve_model_path(path)
    config_file = path.parent.parent / "config.json"
    algo = json.loads(config_file.read_text())["algo"]
    return ALGOS[algo].load(path, device=device)


def _latest_paths(run_dir: Path) -> tuple[Path, Path]:
    ckpt = run_dir / "checkpoints"
    return ckpt / "latest.zip", ckpt / "latest_replay_buffer.pkl"


def build_model(config: TrainConfig, resume: bool) -> BaseAlgorithm:
    """Create a fresh model, or load the latest checkpoint when resuming."""
    algo_cls = ALGOS[config.algo]
    env = make_train_vec_env(
        config.resolved_n_envs(), config.seed, config.fuel_main_scale, config.fuel_side_scale
    )
    latest, buffer = _latest_paths(config.run_dir)
    if resume:
        if not latest.exists():
            raise FileNotFoundError(f"cannot resume, no checkpoint at {latest}")
        model = algo_cls.load(latest, env=env, device=config.device)
        if buffer.exists() and hasattr(model, "load_replay_buffer"):
            model.load_replay_buffer(buffer)
        return model
    params = {**HYPERPARAMS[config.algo], **config.hyperparams_override}
    return algo_cls(
        "MlpPolicy", env, seed=config.seed, device=config.device, verbose=config.verbose, **params
    )


def train(config: TrainConfig, resume: bool = False) -> BaseAlgorithm:
    """Train (or resume) until the model has seen `config.total_timesteps` steps in total."""
    config.run_dir.mkdir(parents=True, exist_ok=True)
    if config.torch_threads is not None:
        torch.set_num_threads(config.torch_threads)
    if not resume:
        config_dict = {**asdict(config), "run_dir": str(config.run_dir)}
        (config.run_dir / "config.json").write_text(json.dumps(config_dict, indent=2))

    model = build_model(config, resume)
    # Plain CSV/stdout logging; TensorBoard is avoided because of the non-ASCII project path.
    model.set_logger(configure(str(config.run_dir / "sb3_log"), ["csv"]))
    callbacks = [
        EvalCSVCallback(config.run_dir / "eval.csv", config.eval_freq, config.eval_episodes),
        CheckpointCallback(config.run_dir / "checkpoints", config.milestones, config.latest_every),
    ]
    steps = remaining_timesteps(config.total_timesteps, model.num_timesteps)
    with prevent_sleep():
        model.learn(total_timesteps=steps, callback=callbacks, reset_num_timesteps=not resume)
    model.get_env().close()
    return model
