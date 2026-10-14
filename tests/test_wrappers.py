from __future__ import annotations

import numpy as np
import pytest

from lunar.envs import FIRE_LEFT, FIRE_MAIN, NOOP, make_env
from lunar.wrappers import MAIN_ENGINE_COST, SIDE_ENGINE_COST, FuelCostWrapper


@pytest.mark.parametrize("main_scale", [0.0, 1.0, 10.0])
def test_fuel_delta(main_scale: float) -> None:
    wrapper = FuelCostWrapper(make_env(), main_scale=main_scale, side_scale=2.0)
    assert wrapper.fuel_delta(FIRE_MAIN) == pytest.approx((main_scale - 1) * MAIN_ENGINE_COST)
    assert wrapper.fuel_delta(FIRE_LEFT) == pytest.approx(SIDE_ENGINE_COST)
    assert wrapper.fuel_delta(NOOP) == 0.0
    wrapper.close()


def test_shaped_reward_matches_original_minus_delta() -> None:
    """Same seed and actions: shaped reward differs from the original by exactly the delta."""
    rng = np.random.default_rng(0)
    actions = rng.integers(0, 4, size=500)
    plain = make_env()
    shaped = FuelCostWrapper(make_env(), main_scale=10.0, side_scale=0.0)
    plain.reset(seed=3)
    shaped.reset(seed=3)
    for action in actions:
        _, r_plain, term, trunc, _ = plain.step(int(action))
        _, r_shaped, term_s, _, info = shaped.step(int(action))
        assert term == term_s
        assert info["original_reward"] == pytest.approx(r_plain)
        expected = r_plain if term else r_plain - shaped.fuel_delta(int(action))
        assert r_shaped == pytest.approx(expected)
        if term or trunc:
            break
    plain.close()
    shaped.close()
