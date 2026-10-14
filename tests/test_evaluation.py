from __future__ import annotations

import pytest

from lunar.envs import NOOP, make_env
from lunar.evaluation import EpisodeStats, Outcome, classify_outcome, run_episode, summarize


@pytest.mark.parametrize(
    ("terminated", "truncated", "last_reward", "expected"),
    [
        (True, False, 100.0, Outcome.LANDED),
        (True, False, -100.0, Outcome.CRASHED),
        (False, True, -0.5, Outcome.TIMEOUT),
    ],
)
def test_classify_outcome(
    terminated: bool, truncated: bool, last_reward: float, expected: Outcome
) -> None:
    assert classify_outcome(terminated, truncated, last_reward) is expected


def test_noop_policy_free_falls_and_crashes() -> None:
    env = make_env()
    stats = run_episode(env, lambda _obs: NOOP, seed=0)
    env.close()
    assert stats.outcome is Outcome.CRASHED
    assert stats.main_engine_steps == 0
    assert stats.side_engine_steps == 0
    assert 0 < stats.length < 1000


def test_summarize_rates_sum_to_one() -> None:
    stats = [
        EpisodeStats(200.0, 300, Outcome.LANDED, 50, 10),
        EpisodeStats(-100.0, 100, Outcome.CRASHED, 0, 0),
        EpisodeStats(-50.0, 1000, Outcome.TIMEOUT, 900, 20),
        EpisodeStats(250.0, 250, Outcome.LANDED, 40, 30),
    ]
    summary = summarize(stats)
    assert summary["mean_return"] == pytest.approx(75.0)
    assert summary["landed_rate"] == pytest.approx(0.5)
    rates = summary["landed_rate"] + summary["crashed_rate"] + summary["timeout_rate"]
    assert rates == pytest.approx(1.0)
