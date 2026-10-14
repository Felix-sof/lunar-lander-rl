"""Environment construction helpers."""

from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

ENV_ID = "LunarLander-v3"

# Discrete actions of LunarLander.
NOOP, FIRE_LEFT, FIRE_MAIN, FIRE_RIGHT = 0, 1, 2, 3


def make_env(render_mode: str | None = None) -> gym.Env:
    """Create a single LunarLander environment with the original reward."""
    return gym.make(ENV_ID, render_mode=render_mode)


def make_train_env_fn(
    main_scale: float = 1.0, side_scale: float = 1.0, monitor_file: str | None = None
) -> Callable[[], gym.Env]:
    """Factory for one training env.

    Monitor wraps the raw env *before* reward shaping, so logged episode returns are always
    the original LunarLander return and runs with different shaping stay comparable.
    """

    def _init() -> gym.Env:
        # Local import avoids a cycle: lunar.wrappers imports the action constants above.
        from lunar.wrappers import FuelCostWrapper

        env: gym.Env = Monitor(make_env(), filename=monitor_file)
        if main_scale != 1.0 or side_scale != 1.0:
            env = FuelCostWrapper(env, main_scale=main_scale, side_scale=side_scale)
        return env

    return _init


def make_train_vec_env(
    n_envs: int, seed: int, main_scale: float = 1.0, side_scale: float = 1.0
) -> VecEnv:
    """Vectorised training envs; subprocesses when n_envs > 1."""
    fns = [make_train_env_fn(main_scale, side_scale) for _ in range(n_envs)]
    vec_env: VecEnv = SubprocVecEnv(fns) if n_envs > 1 else DummyVecEnv(fns)
    vec_env.seed(seed)
    vec_env.action_space.seed(seed)
    return vec_env
