"""Shared test setup for xr_teleoperate/tests.

Stubs ``logging_mp`` (a teleop-env-only dep) and puts ``teleop/`` on sys.path
so tests can import ``utils.lerobot_episode_writer`` without needing the full
teleop conda environment.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path


# ---- stub logging_mp (only available in the `tv` conda env) ----
if "logging_mp" not in sys.modules:

    class _StubLogger:
        def info(self, *a, **k): pass
        def warning(self, *a, **k): pass
        def error(self, *a, **k): pass
        def debug(self, *a, **k): pass

    _stub = types.ModuleType("logging_mp")
    _stub.getLogger = lambda *a, **k: _StubLogger()  # type: ignore[attr-defined]
    sys.modules["logging_mp"] = _stub


# ---- make `utils.lerobot_episode_writer` importable ----
_TELEOP_DIR = Path(__file__).resolve().parent.parent / "teleop"
if str(_TELEOP_DIR) not in sys.path:
    sys.path.insert(0, str(_TELEOP_DIR))


# ---- common fixtures ----
import numpy as np
import pytest


@pytest.fixture
def image_size() -> tuple[int, int]:
    """(W, H) — keep small for speed."""
    return (640, 480)


@pytest.fixture
def dummy_frame(image_size: tuple[int, int]) -> np.ndarray:
    """A deterministic BGR frame matching image_size."""
    W, H = image_size
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[:, :, 0] = 32   # B
    frame[:, :, 1] = 64   # G
    frame[:, :, 2] = 128  # R
    return frame


@pytest.fixture
def dummy_state() -> dict:
    """Nested-dict state matching Unitree_G1_Inspire_HeadOnly_Mono 26D layout."""
    return {
        "left_arm":  {"qpos": [0.1] * 7, "qvel": [0.0] * 7, "torque": [0.0] * 7},
        "right_arm": {"qpos": [0.2] * 7, "qvel": [0.0] * 7, "torque": [0.0] * 7},
        "left_ee":   {"qpos": [0.3] * 6, "qvel": [0.0] * 6, "torque": [0.0] * 6},
        "right_ee":  {"qpos": [0.4] * 6, "qvel": [0.0] * 6, "torque": [0.0] * 6},
    }


@pytest.fixture
def dummy_action(dummy_state: dict) -> dict:
    """Same shape as state; different magnitudes so tests can tell them apart."""
    return {
        "left_arm":  {"qpos": [0.5] * 7},
        "right_arm": {"qpos": [0.6] * 7},
        "left_ee":   {"qpos": [0.7] * 6},
        "right_ee":  {"qpos": [0.8] * 6},
    }


@pytest.fixture
def dummy_inputs(dummy_frame, dummy_state, dummy_action):
    """Convenience: (colors, states, actions) triple for add_item()."""
    return ({"color_0": dummy_frame}, dummy_state, dummy_action)
