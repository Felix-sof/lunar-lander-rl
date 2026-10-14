"""Keep Windows awake during long training runs.

Sleep can break the pipes between the main process and SubprocVecEnv workers, which then
hangs training silently. SetThreadExecutionState blocks idle sleep while the flag is held.
"""

from __future__ import annotations

import ctypes
import sys
from collections.abc import Iterator
from contextlib import contextmanager

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


@contextmanager
def prevent_sleep() -> Iterator[None]:
    """Block system idle sleep for the duration of the context (no-op off Windows)."""
    if sys.platform != "win32":
        yield
        return
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    try:
        yield
    finally:
        kernel32.SetThreadExecutionState(ES_CONTINUOUS)
