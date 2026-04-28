"""Pure-function helpers for base-velocity capture.

Split out from teleop_hand_and_arm.py so the math (stick→velocity map,
FSM enum mapping) is unit-testable without running DDS or the teleop loop.

Spec: docs/superpowers/specs/2026-04-24-teleop-base-velocity-recording-design.md
"""
from __future__ import annotations

# Linear map gain — matches the existing VR-thumbstick → loco_wrapper.Move()
# call in teleop_hand_and_arm.py:539-541. Signs to be bench-calibrated on PC2
# (see scripts/real_eval/print_base_vel.py).
SCALE_VX: float = 0.3
SCALE_VY: float = 0.3
SCALE_VYAW: float = 0.3
DEADBAND: float = 0.05  # |axis| < DEADBAND → 0 (see spec §4)


def _apply_deadband(v: float) -> float:
    return 0.0 if abs(v) < DEADBAND else v


def r3_stick_to_cmd_vel(lx: float, ly: float, rx: float) -> tuple[float, float, float]:
    """Map R3 stick axes to (vx, vy, vyaw) base velocity in robot body frame.

    Sign convention mirrors teleop_hand_and_arm.py:539-541 (Quest controller
    path). Bench-calibrate on PC2 before first real recording.
    """
    vx = -_apply_deadband(ly) * SCALE_VX
    vy = -_apply_deadband(lx) * SCALE_VY
    vyaw = -_apply_deadband(rx) * SCALE_VYAW
    return (vx, vy, vyaw)


# FSM enum — dense 0..6 int8 encoding for the observation.fsm_mode column.
# Separated from raw SportModeState.mode IDs to dodge the ML normalization
# footgun (a naive pipeline treating `706 > 200` as quantitative).
#
# Spec §3.1. Slot 6 (TRANSITIONING) is reserved; raw-ID populated during
# bench calibration once we observe which intermediate IDs the sport service
# emits during 706 squat↔stand transitions.
FSM_ENUM: dict[int, int] = {
    200: 1,  # STAND
    706: 2,  # SQUAT (steady)
    1: 3,    # DAMP
    3: 4,    # SIT
    0: 5,    # ZERO_TORQUE
    # Add intermediate transition IDs → 6 here after calibration.
}


def fsm_mode_to_enum(raw_mode: int) -> int:
    """Map raw SportModeState.mode (uint8) to dense enum 0..6 int8.

    Returns 0 (UNKNOWN) for any unmapped mode. After bench calibration,
    intermediate squat-transition IDs should be added to FSM_ENUM → 6.
    """
    return FSM_ENUM.get(int(raw_mode), 0)


def read_sport_snapshot(box: list) -> tuple[list[float], int]:
    """Read latest SportModeState snapshot from a single-element box.

    box[0] is either None (no message yet) or a 5-tuple
    (mode:int, vx, vy, vz, yaw_speed:float).
    Returns (base_achieved [vx, vy, vz, yaw_speed], fsm_enum).
    Single attribute load — relies on CPython GIL atomicity of list-index
    assignment, no lock required.
    """
    snap = box[0]
    if snap is None:
        return [0.0, 0.0, 0.0, 0.0], 0
    mode, vx, vy, vz, yaw_speed = snap
    return [vx, vy, vz, yaw_speed], fsm_mode_to_enum(mode)


def read_wireless_snapshot(box: list) -> list[float]:
    """Read latest WirelessController snapshot, pipe through r3_stick_to_cmd_vel.

    box[0] is either None or a 3-tuple (lx, ly, rx) with raw stick values
    (deadband + scaling NOT pre-applied — both are r3_stick_to_cmd_vel's job).
    """
    snap = box[0]
    if snap is None:
        return [0.0, 0.0, 0.0]
    lx, ly, rx = snap
    return list(r3_stick_to_cmd_vel(lx, ly, rx))
