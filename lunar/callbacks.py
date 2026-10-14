"""Training callbacks: periodic evaluation to CSV and milestone checkpoints."""

from __future__ import annotations

import csv
import time
from pathlib import Path

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

from lunar.envs import make_env
from lunar.evaluation import run_episode, summarize

EVAL_COLUMNS = [
    "timesteps",
    "wall_time_s",
    "mean_return",
    "std_return",
    "landed_rate",
    "crashed_rate",
    "timeout_rate",
    "mean_length",
    "mean_main_engine_steps",
    "mean_side_engine_steps",
]
EVAL_SEED_OFFSET = 10_000  # eval seeds never overlap typical training seeds


def evaluate_model(model: BaseAlgorithm, episodes: int, seed: int = EVAL_SEED_OFFSET) -> dict:
    """Deterministic evaluation on the *original* reward, with behaviour statistics."""
    env = make_env()
    stats = [
        run_episode(env, lambda obs: int(model.predict(obs, deterministic=True)[0]), seed + i)
        for i in range(episodes)
    ]
    env.close()
    return summarize(stats)


class EvalCSVCallback(BaseCallback):
    """Evaluate every `eval_freq` timesteps and append a row to a CSV file.

    The same eval seeds are used every time, so changes between rows come from the policy,
    not from different starting conditions.
    """

    def __init__(self, csv_path: Path, eval_freq: int, episodes: int, verbose: int = 1) -> None:
        super().__init__(verbose)
        self.csv_path = csv_path
        self.eval_freq = eval_freq
        self.episodes = episodes
        self._next_eval = 0
        self._last_eval_t = -1
        self._wall_offset = 0.0
        self._start = 0.0

    def _on_training_start(self) -> None:
        self._start = time.perf_counter()
        rows: list[dict[str, str]] = []
        resumed = self.csv_path.exists() and self.num_timesteps > 0
        if resumed:
            # Drop rows logged after the checkpoint we resume from (the crashed tail).
            with self.csv_path.open(newline="") as f:
                rows = [r for r in csv.DictReader(f) if int(r["timesteps"]) <= self.num_timesteps]
            self._wall_offset = float(rows[-1]["wall_time_s"]) if rows else 0.0
        with self.csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=EVAL_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        self._last_eval_t = int(rows[-1]["timesteps"]) if rows else -1
        # First eval at the next multiple of eval_freq (0 on a fresh run: the untrained policy).
        self._next_eval = -(-self.num_timesteps // self.eval_freq) * self.eval_freq
        self._maybe_evaluate()

    def _on_step(self) -> bool:
        self._maybe_evaluate()
        return True

    def _on_training_end(self) -> None:
        self._evaluate()

    def _maybe_evaluate(self) -> None:
        if self.num_timesteps >= self._next_eval:
            self._evaluate()
            while self._next_eval <= self.num_timesteps:
                self._next_eval += self.eval_freq

    def _evaluate(self) -> None:
        if self.num_timesteps == self._last_eval_t:
            return
        self._last_eval_t = self.num_timesteps
        summary = evaluate_model(self.model, self.episodes)
        row = {
            "timesteps": self.num_timesteps,
            "wall_time_s": self._wall_offset + time.perf_counter() - self._start,
            **summary,
        }
        with self.csv_path.open("a", newline="") as f:
            csv.writer(f).writerow([row[c] for c in EVAL_COLUMNS])
        for key in ("mean_return", "landed_rate", "timeout_rate"):
            self.logger.record(f"eval/{key}", summary[key])
        if self.verbose:
            print(
                f"[eval] t={self.num_timesteps:>8d}  return={summary['mean_return']:7.1f}  "
                f"landed={summary['landed_rate']:.0%}  crashed={summary['crashed_rate']:.0%}  "
                f"timeout={summary['timeout_rate']:.0%}",
                flush=True,
            )


class CheckpointCallback(BaseCallback):
    """Save named milestone checkpoints plus a rolling `latest` checkpoint for resuming.

    Milestones are saved at the first step at or past each value (vectorised envs advance
    `num_timesteps` by n_envs per step). Off-policy models also save their replay buffer with
    `latest`, otherwise a resumed DQN would restart from an empty buffer.
    """

    def __init__(self, save_dir: Path, milestones: list[int], latest_every: int) -> None:
        super().__init__()
        self.save_dir = save_dir
        self.milestones = sorted(milestones)
        self.latest_every = latest_every
        self._pending: list[int] = []
        self._last_latest = 0

    def _on_training_start(self) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        # On resume, milestones already passed were saved by the earlier run.
        self._pending = [m for m in self.milestones if m > self.num_timesteps]
        self._last_latest = self.num_timesteps

    def _on_step(self) -> bool:
        while self._pending and self.num_timesteps >= self._pending[0]:
            self.model.save(self.save_dir / f"step_{self._pending.pop(0)}")
        if self.num_timesteps - self._last_latest >= self.latest_every:
            self.save_latest()
        return True

    def _on_training_end(self) -> None:
        self.save_latest()
        self.model.save(self.save_dir / "final")

    def save_latest(self) -> None:
        self.model.save(self.save_dir / "latest")
        if isinstance(self.model, OffPolicyAlgorithm):
            self.model.save_replay_buffer(self.save_dir / "latest_replay_buffer")
        self._last_latest = self.num_timesteps
