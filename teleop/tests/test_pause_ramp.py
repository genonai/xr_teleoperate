"""Unit tests for teleop.pause_ramp.apply_pause_ramp.

Exercises every code path of the helper that replaced the 5 duplicated
per-EE hand-write branches in teleop_hand_and_arm.py. Tests run on a
clean Mac without robotics deps (per genonai/IL_PhysicalAI#30 acceptance
criteria).

Run::

    cd repos/xr_teleoperate && python -m pytest teleop/tests/test_pause_ramp.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from teleop.pause_ramp import apply_pause_ramp


# ---------------------------------------------------------------------------
# Helpers — record every write_fn invocation so tests can assert what reached
# the (mock) shared memory.
# ---------------------------------------------------------------------------

class WriteRecorder:
    """Captures (out_left, out_right) tuples passed to write_fn."""

    def __init__(self):
        self.calls = []

    def __call__(self, out_left, out_right):
        self.calls.append((out_left, out_right))


@pytest.fixture
def writer():
    return WriteRecorder()


# ---------------------------------------------------------------------------
# Array path (numpy hand-pose vectors)
# ---------------------------------------------------------------------------

def test_array_normal_tick_writes_fresh_input(writer):
    """alpha=1.0, not paused → write fresh input verbatim, update last_commanded.

    Pins the defensive .copy() at the end-of-tick update: last_commanded
    must NOT alias new_l/new_r so future caller mutations cannot leak in.
    """
    new_l = np.array([1.0, 2.0, 3.0])
    new_r = np.array([4.0, 5.0, 6.0])
    frozen = {}
    last = {}

    apply_pause_ramp(
        new_l, new_r, writer, is_array=True,
        entering_pause=False, current_tick_paused=False, alpha=1.0,
        frozen_snapshot=frozen, last_commanded=last,
    )

    assert len(writer.calls) == 1
    out_l, out_r = writer.calls[0]
    np.testing.assert_array_equal(out_l, new_l)
    np.testing.assert_array_equal(out_r, new_r)
    np.testing.assert_array_equal(last['left'], new_l)
    np.testing.assert_array_equal(last['right'], new_r)
    # last_commanded must be a copy, not aliased to new_l/new_r.
    assert last['left'] is not new_l
    assert last['right'] is not new_r


def test_array_entering_pause_snapshots_from_last_commanded(writer):
    """First paused tick: snapshot from last_commanded (C0 continuity).

    Also pins the defensive .copy() at the snapshot site — mutating
    last_left after the call must NOT affect frozen['left'].
    """
    last_left = np.array([10.0, 11.0, 12.0])
    last_right = np.array([13.0, 14.0, 15.0])
    new_l = np.array([99.0, 99.0, 99.0])  # fresh input — should NOT enter snapshot
    new_r = np.array([88.0, 88.0, 88.0])
    frozen = {}
    last = {'left': last_left, 'right': last_right}

    apply_pause_ramp(
        new_l, new_r, writer, is_array=True,
        entering_pause=True, current_tick_paused=True, alpha=0.0,
        frozen_snapshot=frozen, last_commanded=last,
    )

    # Snapshot captured from last_commanded, NOT from fresh input.
    np.testing.assert_array_equal(frozen['left'], last_left)
    np.testing.assert_array_equal(frozen['right'], last_right)
    # And it must be a copy — mutating last_left after the call must not
    # bleed into the snapshot. This pins the defensive copy.
    last_left[0] = 999.0
    assert frozen['left'][0] == 10.0
    # No write while paused.
    assert writer.calls == []


def test_array_paused_tick_skips_write(writer):
    """Subsequent paused ticks: skip write, leave frozen AND last_commanded
    untouched. Pins both contracts (round-2 review pointed out the
    last_commanded check was missing)."""
    frozen = {'left': np.array([1.0]), 'right': np.array([2.0])}
    frozen_left_before = frozen['left'].copy()
    last = {'left': np.array([0.0]), 'right': np.array([0.0])}
    last_left_before = last['left'].copy()
    last_right_before = last['right'].copy()

    apply_pause_ramp(
        np.array([99.0]), np.array([99.0]), writer, is_array=True,
        entering_pause=False, current_tick_paused=True, alpha=0.0,
        frozen_snapshot=frozen, last_commanded=last,
    )

    assert writer.calls == []
    np.testing.assert_array_equal(frozen['left'], frozen_left_before)
    # last_commanded must be untouched on a paused tick — the early return
    # before write_fn skips both the write AND the last_commanded update,
    # so a re-pause N ticks later still snapshots from the same anchor.
    np.testing.assert_array_equal(last['left'], last_left_before)
    np.testing.assert_array_equal(last['right'], last_right_before)


def test_array_ramp_blend_is_linear(writer):
    """During resume ramp (alpha=0.5), out = 0.5*frozen + 0.5*new."""
    frozen_l = np.array([0.0, 0.0])
    frozen_r = np.array([10.0, 10.0])
    new_l = np.array([100.0, 200.0])
    new_r = np.array([20.0, 30.0])
    frozen = {'left': frozen_l, 'right': frozen_r}
    last = {}

    apply_pause_ramp(
        new_l, new_r, writer, is_array=True,
        entering_pause=False, current_tick_paused=False, alpha=0.5,
        frozen_snapshot=frozen, last_commanded=last,
    )

    expected_l = np.array([50.0, 100.0])   # 0 + 0.5 * (100-0), 0 + 0.5 * (200-0)
    expected_r = np.array([15.0, 20.0])    # 10 + 0.5 * (20-10), 10 + 0.5 * (30-10)
    np.testing.assert_allclose(writer.calls[0][0], expected_l)
    np.testing.assert_allclose(writer.calls[0][1], expected_r)
    # last_commanded reflects what was actually written (the blended intermediate)
    np.testing.assert_allclose(last['left'], expected_l)
    np.testing.assert_allclose(last['right'], expected_r)


# ---------------------------------------------------------------------------
# Scalar path (gripper trigger floats)
# ---------------------------------------------------------------------------

def test_scalar_normal_tick(writer):
    apply_pause_ramp(
        0.7, 0.3, writer, is_array=False,
        entering_pause=False, current_tick_paused=False, alpha=1.0,
        frozen_snapshot={}, last_commanded={},
    )
    assert writer.calls == [(0.7, 0.3)]


def test_scalar_entering_pause_snapshots_from_last_commanded(writer):
    frozen = {}
    last = {'left': 0.42, 'right': 0.99}

    apply_pause_ramp(
        0.0, 0.0, writer, is_array=False,
        entering_pause=True, current_tick_paused=True, alpha=0.0,
        frozen_snapshot=frozen, last_commanded=last,
    )

    assert frozen['left'] == 0.42   # from last_commanded, not from new (0.0)
    assert frozen['right'] == 0.99
    assert writer.calls == []


def test_scalar_ramp_blend(writer):
    frozen = {'left': 0.0, 'right': 10.0}
    apply_pause_ramp(
        100.0, 30.0, writer, is_array=False,
        entering_pause=False, current_tick_paused=False, alpha=0.25,
        frozen_snapshot=frozen, last_commanded={},
    )
    # 0 + 0.25*(100-0) = 25.0;  10 + 0.25*(30-10) = 15.0
    assert writer.calls == [(pytest.approx(25.0), pytest.approx(15.0))]


# ---------------------------------------------------------------------------
# Bootstrap: first pause ever, last_commanded is empty
# ---------------------------------------------------------------------------

def test_array_bootstrap_falls_back_to_new_when_last_empty(writer):
    """When last_commanded is {} (first pause ever), snapshot from new input.

    The bootstrap value is read but never observable: paused branches skip
    writes, so the snapshot is only used as the blend-from anchor when
    resume happens, by which time last_commanded has been updated by at
    least one non-paused tick.

    Also pins the defensive copy on the bootstrap path.
    """
    new_l = np.array([1.0, 2.0])
    new_r = np.array([3.0, 4.0])
    frozen = {}
    last = {}

    apply_pause_ramp(
        new_l, new_r, writer, is_array=True,
        entering_pause=True, current_tick_paused=True, alpha=0.0,
        frozen_snapshot=frozen, last_commanded=last,
    )

    np.testing.assert_array_equal(frozen['left'], new_l)
    np.testing.assert_array_equal(frozen['right'], new_r)
    # Defensive copy: mutating new_l after the call must not affect frozen.
    new_l[0] = 999.0
    assert frozen['left'][0] == 1.0


def test_scalar_bootstrap_falls_back_to_new_when_last_empty(writer):
    """Round-2 review fix: previous version of this test had ZERO assertions
    and always passed. Now pins the actual scalar bootstrap behavior."""
    frozen = {}
    last = {}

    apply_pause_ramp(
        0.5, 0.7, writer, is_array=False,
        entering_pause=True, current_tick_paused=True, alpha=0.0,
        frozen_snapshot=frozen, last_commanded=last,
    )

    # Bootstrap from new (since last is empty).
    assert frozen['left'] == 0.5
    assert frozen['right'] == 0.7
    # No write while paused.
    assert writer.calls == []


# ---------------------------------------------------------------------------
# C0 continuity: re-pause mid-ramp captures the blended intermediate
# ---------------------------------------------------------------------------

def test_array_re_pause_mid_ramp_captures_intermediate(writer):
    """The full Scenario 7 path: pause at A, resume to B, re-pause mid-ramp.

    Walk through several ticks with the same frozen/last dicts to verify
    the snapshot on re-pause captures the blended intermediate written
    during the ramp, not the raw fresh input.
    """
    frozen = {}
    last = {}
    pos_a = np.array([0.0, 0.0])
    pos_b = np.array([100.0, 100.0])
    pos_c = np.array([-50.0, -50.0])

    # Tick 0: not paused, write A — last is now A
    apply_pause_ramp(
        pos_a, pos_a, writer, is_array=True,
        entering_pause=False, current_tick_paused=False, alpha=1.0,
        frozen_snapshot=frozen, last_commanded=last,
    )
    np.testing.assert_array_equal(last['left'], pos_a)

    # Tick 1: entering pause → snapshot from last (= A). VR moves to B.
    # Even though new_left=pos_b, frozen captures pos_a.
    apply_pause_ramp(
        pos_b, pos_b, writer, is_array=True,
        entering_pause=True, current_tick_paused=True, alpha=0.0,
        frozen_snapshot=frozen, last_commanded=last,
    )
    np.testing.assert_array_equal(frozen['left'], pos_a)

    # Tick 2: leaving pause → start ramp at alpha=0.5 toward B
    apply_pause_ramp(
        pos_b, pos_b, writer, is_array=True,
        entering_pause=False, current_tick_paused=False, alpha=0.5,
        frozen_snapshot=frozen, last_commanded=last,
    )
    # last is now (A + 0.5*(B-A)) = (50, 50) — the blended intermediate
    np.testing.assert_array_equal(last['left'], np.array([50.0, 50.0]))

    # Tick 3: re-pause mid-ramp → entering_pause snapshots from last (= 50,50)
    # NOT from new_left (which would be pos_c if VR has moved again).
    apply_pause_ramp(
        pos_c, pos_c, writer, is_array=True,
        entering_pause=True, current_tick_paused=True, alpha=0.0,
        frozen_snapshot=frozen, last_commanded=last,
    )
    np.testing.assert_array_equal(frozen['left'], np.array([50.0, 50.0]))
    # ^ This is the C0 continuity guarantee. Without it (snapshotting from
    # new_left), frozen would become pos_c and the next resume would jump.


# ---------------------------------------------------------------------------
# Sanity: signature is keyword-only beyond positional new_l, new_r, write_fn
# ---------------------------------------------------------------------------

def test_keyword_only_args_enforced(writer):
    """is_array, entering_pause, current_tick_paused, alpha, frozen_snapshot,
    and last_commanded must all be passed as kwargs.

    Guards against a future caller silently passing them positionally and
    getting them confused with each other (e.g. swapping is_array and
    current_tick_paused, which would silently break the array path).
    """
    with pytest.raises(TypeError):
        apply_pause_ramp(
            0.5, 0.5, writer,
            True,  # this is `is_array=` but passed positionally — should fail
            False, False, 1.0,
            {}, {},
        )


# ---------------------------------------------------------------------------
# Round-2 review additions: untested contracts
# ---------------------------------------------------------------------------

def test_write_fn_exception_leaves_last_commanded_clean(writer):
    """If write_fn raises, last_commanded must NOT be updated.

    Pins the contract that we don't track values we never successfully
    wrote. Currently enforced by the assignment lines being AFTER the
    write_fn call — if write_fn raises, the assignment never executes.
    """
    def failing_write_fn(out_left, out_right):
        raise RuntimeError("simulated shared-memory lock failure")

    last = {}
    new_l = np.array([1.0, 2.0])
    new_r = np.array([3.0, 4.0])

    with pytest.raises(RuntimeError, match="simulated shared-memory"):
        apply_pause_ramp(
            new_l, new_r, failing_write_fn, is_array=True,
            entering_pause=False, current_tick_paused=False, alpha=1.0,
            frozen_snapshot={}, last_commanded=last,
        )

    # last_commanded must still be empty — no value reached shared memory,
    # so we have nothing to track.
    assert last == {}


def test_scalar_paused_tick_skips_write_and_last_commanded(writer):
    """Scalar variant of the paused-tick contract — pins last_commanded
    is not updated even though the early return is shared with the array
    path. Round-2 review pointed out this contract was untested."""
    frozen = {'left': 0.1, 'right': 0.2}
    last = {'left': 0.3, 'right': 0.4}

    apply_pause_ramp(
        0.99, 0.88, writer, is_array=False,
        entering_pause=False, current_tick_paused=True, alpha=0.0,
        frozen_snapshot=frozen, last_commanded=last,
    )

    assert writer.calls == []
    assert frozen == {'left': 0.1, 'right': 0.2}    # unchanged
    assert last   == {'left': 0.3, 'right': 0.4}    # unchanged
