"""Reward shaping wrappers."""

from __future__ import annotations

from typing import Any, SupportsFloat

import gymnasium as gym

from lunar.envs import FIRE_LEFT, FIRE_MAIN, FIRE_RIGHT

# Per-frame fuel penalties hard-coded in gymnasium's LunarLander (discrete actions).
MAIN_ENGINE_COST = 0.30
SIDE_ENGINE_COST = 0.03


class FuelCostWrapper(gym.Wrapper):
    """Rescale LunarLander's fuel penalty.

    `main_scale=0` makes fuel free, `1` is the original reward, `10` makes fuel ten times as
    expensive. The original reward is kept in `info["original_reward"]`.

    On terminal steps LunarLander overwrites the reward with exactly -100 (crash) or +100
    (landed), so no fuel penalty was charged there and none is rescaled.
    """

    def __init__(self, env: gym.Env, main_scale: float = 1.0, side_scale: float = 1.0) -> None:
        super().__init__(env)
        self.main_scale = main_scale
        self.side_scale = side_scale

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["original_reward"] = float(reward)
        if not terminated:
            reward = float(reward) - self.fuel_delta(int(action))
        return obs, reward, terminated, truncated, info

    def fuel_delta(self, action: int) -> float:
        """Extra penalty on top of the original one for taking `action`."""
        if action == FIRE_MAIN:
            return (self.main_scale - 1.0) * MAIN_ENGINE_COST
        if action in (FIRE_LEFT, FIRE_RIGHT):
            return (self.side_scale - 1.0) * SIDE_ENGINE_COST
        return 0.0
