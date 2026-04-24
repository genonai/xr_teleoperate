"""Unit tests for base-velocity math module.

Spec: docs/superpowers/specs/2026-04-24-teleop-base-velocity-recording-design.md §4.

Import uses ``utils.base_velocity`` (not ``teleop.utils.base_velocity``) because
conftest.py inserts teleop/ into sys.path, matching the rest of the test suite
(see test_lerobot_episode_writer.py for the same pattern).
"""
import pytest
from utils.base_velocity import r3_stick_to_cmd_vel, DEADBAND, SCALE_VX, SCALE_VY, SCALE_VYAW  # type: ignore[import-not-found]


def test_zero_stick_gives_zero_velocity():
    assert r3_stick_to_cmd_vel(0.0, 0.0, 0.0) == (0.0, 0.0, 0.0)


def test_forward_stick_gives_positive_vx():
    # Forward on left stick = ly negative per Quest convention (spec §4).
    vx, vy, vyaw = r3_stick_to_cmd_vel(lx=0.0, ly=-1.0, rx=0.0)
    assert vx == pytest.approx(SCALE_VX)
    assert vy == 0.0
    assert vyaw == 0.0


def test_right_stick_gives_negative_vy():
    vx, vy, vyaw = r3_stick_to_cmd_vel(lx=1.0, ly=0.0, rx=0.0)
    assert vx == 0.0
    assert vy == pytest.approx(-SCALE_VY)
    assert vyaw == 0.0


def test_rx_gives_negative_vyaw():
    vx, vy, vyaw = r3_stick_to_cmd_vel(lx=0.0, ly=0.0, rx=1.0)
    assert vyaw == pytest.approx(-SCALE_VYAW)


def test_deadband_zeros_small_input():
    v = r3_stick_to_cmd_vel(lx=DEADBAND - 1e-6, ly=-(DEADBAND - 1e-6), rx=DEADBAND - 1e-6)
    assert v == (0.0, 0.0, 0.0)


def test_deadband_boundary_passes_through():
    # At exactly DEADBAND the input is NOT zeroed (strict <).
    vx, _, _ = r3_stick_to_cmd_vel(lx=0.0, ly=-DEADBAND, rx=0.0)
    assert vx == pytest.approx(DEADBAND * SCALE_VX)


def test_full_deflection_caps_at_scale():
    vx, vy, vyaw = r3_stick_to_cmd_vel(lx=-1.0, ly=-1.0, rx=-1.0)
    assert vx == pytest.approx(SCALE_VX)
    assert vy == pytest.approx(SCALE_VY)
    assert vyaw == pytest.approx(SCALE_VYAW)
