"""Episode rollouts with behaviour statistics, not just the return.

Mean return alone can hide degenerate behaviour (e.g. hovering forever), so every
episode also records how it ended and how much fuel it used.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import gymnasium as gym
import numpy as np

from lunar.envs import FIRE_LEFT, FIRE_MAIN, FIRE_RIGHT

Policy = Callable[[np.ndarray], int]

# LunarLander overwrites the final step reward with exactly these values.
CRASH_REWARD = -100.0
LANDED_REWARD = 100.0


class Outcome(str, Enum):
    LANDED = "landed"  # came to rest (lander went to sleep)
    CRASHED = "crashed"  # body touched the ground or flew off-screen
    TIMEOUT = "timeout"  # hit the time limit, typically hovering


@dataclass(frozen=True)
class EpisodeStats:
    episode_return: float
    length: int
    outcome: Outcome
    main_engine_steps: int
    side_engine_steps: int


def classify_outcome(terminated: bool, truncated: bool, last_reward: float) -> Outcome:
    """Infer how an episode ended from its final transition."""
    if truncated and not terminated:
        return Outcome.TIMEOUT
    if last_reward == LANDED_REWARD:
        return Outcome.LANDED
    return Outcome.CRASHED


def run_episode(env: gym.Env, policy: Policy, seed: int | None = None) -> EpisodeStats:
    """Play one episode with `policy` and collect its statistics."""
    obs, _ = env.reset(seed=seed)
    total, length, main_steps, side_steps = 0.0, 0, 0, 0
    while True:
        action = int(policy(obs))
        main_steps += action == FIRE_MAIN
        side_steps += action in (FIRE_LEFT, FIRE_RIGHT)
        obs, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
        length += 1
        if terminated or truncated:
            return EpisodeStats(
                episode_return=total,
                length=length,
                outcome=classify_outcome(terminated, truncated, float(reward)),
                main_engine_steps=main_steps,
                side_engine_steps=side_steps,
            )


def summarize(stats: list[EpisodeStats]) -> dict[str, float]:
    """Aggregate episode statistics into a flat dict of numbers."""
    returns = np.array([s.episode_return for s in stats])
    summary = {
        "episodes": float(len(stats)),
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "min_return": float(returns.min()),
        "max_return": float(returns.max()),
        "mean_length": float(np.mean([s.length for s in stats])),
        "mean_main_engine_steps": float(np.mean([s.main_engine_steps for s in stats])),
        "mean_side_engine_steps": float(np.mean([s.side_engine_steps for s in stats])),
    }
    for outcome in Outcome:
        summary[f"{outcome.value}_rate"] = float(np.mean([s.outcome == outcome for s in stats]))
    return summary


def format_summary(summary: dict[str, float]) -> str:
    """Human-readable, ASCII-only one-block summary (safe for the Windows console)."""
    return (
        f"episodes: {int(summary['episodes'])}\n"
        f"return:   {summary['mean_return']:.1f} +/- {summary['std_return']:.1f} "
        f"(min {summary['min_return']:.1f}, max {summary['max_return']:.1f})\n"
        f"length:   {summary['mean_length']:.0f} steps\n"
        f"outcomes: landed {summary['landed_rate']:.0%}, crashed {summary['crashed_rate']:.0%}, "
        f"timeout {summary['timeout_rate']:.0%}\n"
        f"engines:  main {summary['mean_main_engine_steps']:.0f}, "
        f"side {summary['mean_side_engine_steps']:.0f} steps/episode"
    )
