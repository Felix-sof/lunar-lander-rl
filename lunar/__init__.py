"""Reinforcement learning warm-up on Gymnasium LunarLander."""

import sys

# SB3's logger imports tensorboard, which imports TensorFlow when it is installed. That
# costs ~320 MB and ~4 s per process, and every SubprocVecEnv worker on Windows (spawn)
# pays it again: 48 workers starting at once exhausted RAM and crashed mid-import.
# Marking TensorFlow as unavailable makes tensorboard fall back to its lightweight stub.
# This package is imported before SB3 both in the main process and in every worker.
sys.modules.setdefault("tensorflow", None)  # type: ignore[arg-type]
