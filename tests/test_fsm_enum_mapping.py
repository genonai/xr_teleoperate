from utils.base_velocity import fsm_mode_to_enum, FSM_ENUM


def test_stand_maps_to_1():
    assert fsm_mode_to_enum(200) == 1


def test_squat_maps_to_2():
    assert fsm_mode_to_enum(706) == 2


def test_damp_maps_to_3():
    assert fsm_mode_to_enum(1) == 3


def test_sit_maps_to_4():
    assert fsm_mode_to_enum(3) == 4


def test_zero_torque_maps_to_5():
    assert fsm_mode_to_enum(0) == 5


def test_unknown_id_maps_to_0():
    assert fsm_mode_to_enum(999) == 0


def test_no_collisions_in_enum_table():
    # Every mapped mode should map to a distinct enum value in 1..6.
    values = list(FSM_ENUM.values())
    assert len(values) == len(set(values)), "duplicate enum values in FSM_ENUM"
    assert all(1 <= v <= 6 for v in values), "enum values must be 1..6 (0=UNKNOWN, 6=TRANSITIONING)"


def test_returns_int_for_numpy_input():
    # msg.mode comes from DDS as uint8; cast must still work.
    import numpy as np
    assert fsm_mode_to_enum(np.uint8(200)) == 1
