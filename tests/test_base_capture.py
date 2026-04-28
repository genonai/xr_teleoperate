"""Snapshot-helper tests for base-velocity capture (spec §6).

Tests the snapshot-read helpers that the teleop record loop uses to
pull the latest DDS sample atomically. No real DDS — the helpers
operate on a plain single-element list ('box') so they're testable
without a publisher.
"""
from __future__ import annotations

import pytest

from utils.base_velocity import (
    DEADBAND,
    SCALE_VX,
    SCALE_VY,
    SCALE_VYAW,
    fsm_mode_to_enum,
    read_sport_snapshot,
    read_wireless_snapshot,
)


# B4: Sport snapshot None-fallback.
def test_sport_snapshot_none_fallback():
    base, fsm = read_sport_snapshot([None])
    assert base == [0.0, 0.0, 0.0, 0.0]
    assert fsm == 0


# B5: Wireless snapshot None-fallback.
def test_wireless_snapshot_none_fallback():
    cmd = read_wireless_snapshot([None])
    assert cmd == (0.0, 0.0, 0.0)


# B6: Sport snapshot non-None happy path — components flow into base_achieved
# in canonical (vx, vy, vz, yaw_speed) order, mode flows through fsm_mode_to_enum.
def test_sport_snapshot_happy_path():
    box = [(200, 0.5, -0.1, 0.0, 0.3)]
    base, fsm = read_sport_snapshot(box)
    assert base == [0.5, -0.1, 0.0, 0.3]
    assert fsm == fsm_mode_to_enum(200)


# B7: Stick-mapping composition with outside-deadband values.
# Pins: callback stored RAW (not pre-scaled, not pre-deadbanded), and
# read pipes through r3_stick_to_cmd_vel (sign-flip + scale-after-deadband).
# Outside-deadband single-axis input. Mapping: lx→vy, ly→vx, rx→vyaw,
# all sign-flipped. With (lx=0.5, ly=0, rx=0): vy = -0.5*SCALE_VY = -0.15;
# vx = vyaw = 0.
def test_wireless_snapshot_outside_deadband_composition():
    box = [(0.5, 0.0, 0.0)]
    cmd = read_wireless_snapshot(box)
    assert cmd == pytest.approx((0.0, -0.5 * SCALE_VY, 0.0))
    # Sanity: the magnitude is non-trivial (not within deadband-zero region).
    assert abs(cmd[1]) > 0.1


# B8: Frozen-stale invariant — between callback fires, the read returns
# the last-seen value, NOT the None-fallback. Load-bearing for training data:
# frozen-stale is a different signal than zero-fallback.
def test_sport_snapshot_freezes_at_last_value_between_callbacks():
    box = [None]
    # Simulate one callback fire.
    box[0] = (200, 0.5, -0.1, 0.0, 0.3)
    base1, fsm1 = read_sport_snapshot(box)
    # No callback in between — second read with no new data.
    base2, fsm2 = read_sport_snapshot(box)
    assert base1 == base2 == [0.5, -0.1, 0.0, 0.3]
    assert fsm1 == fsm2 == fsm_mode_to_enum(200)


# B8 (mirror): Wireless snapshot freezes at last value too.
def test_wireless_snapshot_freezes_at_last_value_between_callbacks():
    box = [None]
    box[0] = (0.5, 0.0, 0.0)
    cmd1 = read_wireless_snapshot(box)
    cmd2 = read_wireless_snapshot(box)
    assert cmd1 == cmd2 == pytest.approx((0.0, -0.5 * SCALE_VY, 0.0))
