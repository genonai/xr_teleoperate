"""Unit tests for teleop.fsm.

Acceptance criteria from genonai/IL_PhysicalAI#31: tests must run on a
developer Mac with pytest only — no DDS / Vuer / Isaac Sim / multiprocessing
daemon required. teleop.fsm is the module that was extracted from
teleop_hand_and_arm.py specifically to clear that bar; importing
teleop_hand_and_arm.py itself triggers DDS / vuer / televuer at module load
and is impossible on a clean Mac.

Run::

    cd repos/xr_teleoperate && python -m pytest teleop/tests/test_fsm.py -v
"""
from __future__ import annotations

import logging

import pytest

from teleop import fsm


# Reset every FSM global before each test. Using autouse so the test body
# never has to remember the boilerplate. Pytest re-runs this for every test;
# state never leaks between cases even when the module's globals are mutated.
#
# IMPORTANT: keep this list in sync with the module-level state globals in
# teleop/fsm.py. If a new global is added there (a future base-vel state,
# e.g.), forgetting to reset it here will cause silent leaks across tests.
@pytest.fixture(autouse=True)
def _reset_fsm_state():
    fsm.START          = False
    fsm.STOP           = False
    fsm.READY          = False
    fsm.RECORD_RUNNING = False
    fsm.RECORD_TOGGLE  = False
    fsm.PAUSED         = False
    yield


# Always run with caplog at INFO level so on_press's info-line for the
# PAUSED toggle is captured.
@pytest.fixture(autouse=True)
def _caplog_info(caplog):
    caplog.set_level(logging.INFO, logger="teleop.fsm")
    return caplog


# ---------------------------------------------------------------------------
# Single-key behavior
# ---------------------------------------------------------------------------

def test_r_sets_start_true():
    fsm.on_press('r')
    assert fsm.START is True
    assert fsm.STOP is False  # untouched


def test_q_sets_stop_true_and_clears_start():
    fsm.START = True  # simulate an in-progress session
    fsm.on_press('q')
    assert fsm.START is False
    assert fsm.STOP is True


def test_q_during_wait_loop_sets_stop_without_start():
    """Quit-before-start path. Pins the property the entry-point's
    STOP-gate (added in commit 4926b5c) relies on: pressing `q` while
    we're still waiting for `r` must produce STOP=True without ever
    setting START=True. If this regresses, the gate would silently
    re-spin the motors on quit-before-start."""
    assert fsm.START is False    # initial state from fixture
    fsm.on_press('q')
    assert fsm.STOP is True
    assert fsm.START is False    # never went True


def test_r_after_q_still_sets_start():
    """Pin an undocumented FSM transition: after `q`, `r` is ungated and
    still sets START=True. Operationally inert because the main loop has
    already exited on STOP=True, but pinned here so a future code change
    that adds a `not STOP` gate to `r` (or doesn't) is a deliberate decision,
    not a silent drift."""
    fsm.on_press('q')
    assert fsm.STOP is True
    fsm.on_press('r')
    assert fsm.START is True
    assert fsm.STOP is True      # STOP stays — `r` does not clear it


# ---------------------------------------------------------------------------
# `s` key — start/stop recording toggle
# ---------------------------------------------------------------------------

def test_s_accepted_when_started_and_not_paused():
    fsm.START = True
    fsm.on_press('s')
    assert fsm.RECORD_TOGGLE is True


def test_s_rejected_when_not_started(_caplog_info):
    fsm.on_press('s')
    assert fsm.RECORD_TOGGLE is False
    assert any(
        "s rejected" in record.message and "guard not satisfied" in record.message
        for record in _caplog_info.records
    )


def test_s_rejected_when_paused(_caplog_info):
    fsm.START = True
    fsm.PAUSED = True
    fsm.on_press('s')
    assert fsm.RECORD_TOGGLE is False
    assert any("s rejected" in r.message for r in _caplog_info.records)


# ---------------------------------------------------------------------------
# `p` key — pause/resume toggle
# ---------------------------------------------------------------------------

def test_p_accepted_toggles_paused_and_logs(_caplog_info):
    fsm.START = True
    fsm.on_press('p')
    assert fsm.PAUSED is True
    # synchronous log line for press feedback
    assert any(
        "PAUSED → True" in r.message for r in _caplog_info.records
    ), "expected '[on_press] PAUSED → True' on first press"

    _caplog_info.clear()
    fsm.on_press('p')
    assert fsm.PAUSED is False
    assert any(
        "PAUSED → False" in r.message for r in _caplog_info.records
    ), "expected '[on_press] PAUSED → False' on second press"


def test_p_rejected_when_recording(_caplog_info):
    fsm.START = True
    fsm.RECORD_RUNNING = True
    fsm.on_press('p')
    assert fsm.PAUSED is False  # unchanged
    assert any("p rejected" in r.message for r in _caplog_info.records)


def test_p_rejected_when_record_toggle_pending(_caplog_info):
    """The `not RECORD_TOGGLE` guard closes the s→p sub-tick race.

    User presses `s` (sets RECORD_TOGGLE=True), then presses `p` before the
    main loop has had a chance to promote RECORD_TOGGLE → RECORD_RUNNING.
    Without the guard, `p` would be accepted because RECORD_RUNNING is still
    False, and the next tick would start an episode whose first frame is a
    frozen pose.
    """
    fsm.START = True
    fsm.RECORD_RUNNING = False     # main loop hasn't promoted yet
    fsm.RECORD_TOGGLE = True       # `s` was just pressed
    fsm.on_press('p')
    assert fsm.PAUSED is False
    assert any("p rejected" in r.message for r in _caplog_info.records)


def test_p_rejected_when_not_started(_caplog_info):
    fsm.on_press('p')
    assert fsm.PAUSED is False
    assert any("p rejected" in r.message for r in _caplog_info.records)


# ---------------------------------------------------------------------------
# Unmapped keys
# ---------------------------------------------------------------------------

def test_unmapped_key_logs_no_action(_caplog_info):
    fsm.on_press('x')
    assert fsm.START is False
    assert fsm.STOP is False
    assert fsm.PAUSED is False
    assert any(
        "x" in r.message and "no action defined" in r.message
        for r in _caplog_info.records
    )


# ---------------------------------------------------------------------------
# get_state heartbeat
# ---------------------------------------------------------------------------

def test_get_state_initial_snapshot():
    snap = fsm.get_state()
    assert snap == {
        "START": False,
        "STOP": False,
        "READY": False,
        "RECORD_RUNNING": False,
        "PAUSED": False,
    }


def test_get_state_reflects_mutations():
    fsm.on_press('r')
    fsm.READY = True
    snap = fsm.get_state()
    assert snap["START"] is True
    assert snap["READY"] is True
    assert snap["STOP"] is False
