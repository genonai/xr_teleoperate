"""teleop.pause_ramp — per-EE hand/gripper write helper for pause/ramp.

Extracts the snapshot/skip/blend/write/update logic that was duplicated
across 5 per-EE branches in ``teleop/teleop_hand_and_arm.py`` (dex3/brainco
+ hand, dex1 + controller, dex1 + hand, inspire_dfx/ftp + hand, inspire_dfx
/ftp + controller). Each call site now passes the per-EE-specific bits
(value source, write target, array-vs-scalar flag) and the helper handles
the pause-aware blend uniformly.

The function imports only Python stdlib + duck-typed numpy operations
(``+``, ``*``, ``-``, ``.copy()``), so it is Mac-importable and unit-testable
without DDS / Vuer / Isaac / multiprocessing.

The pause/ramp invariant the helper preserves (matches PR genonai/xr_teleoperate#5
spec): snapshot at pause entry from ``last_commanded`` (not from fresh input),
skip writes while paused, blend ``alpha * fresh + (1 - alpha) * frozen``
during resume ramp, update ``last_commanded`` at every non-paused tick.
"""
from __future__ import annotations

from typing import Any, Callable, MutableMapping


def apply_pause_ramp(
    new_left: Any,
    new_right: Any,
    write_fn: Callable[[Any, Any], None],
    *,
    entering_pause: bool,
    current_tick_paused: bool,
    alpha: float,
    frozen_snapshot: MutableMapping[str, Any],
    last_commanded: MutableMapping[str, Any],
) -> None:
    """Apply the pause/ramp blend to one EE's left/right values.

    Contract: ``new_left`` / ``new_right`` MUST be fresh, non-aliased values
    each call. At every existing call site they are either freshly-flattened
    numpy arrays (``tele_data.left_hand_pos.flatten()``) or immutable Python
    floats — neither aliases anything the caller mutates later. The helper
    relies on this so it can store references directly without defensive
    ``.copy()``: every value it persists into ``frozen_snapshot`` or
    ``last_commanded`` is already a fresh allocation. If a future caller
    passes a non-fresh array, the saved reference would alias the caller's
    buffer and silently track later mutations — bugs would manifest as
    ramp/pause behavior drifting with whatever the caller does to the
    "fresh" input. Document new call sites accordingly.

    Args:
        new_left, new_right: fresh values from this tick's tele_data.
        write_fn: closure ``(out_left, out_right) -> None`` that handles
            the actual shared-memory write (with whatever locks the target
            requires). Called only when not paused.
        entering_pause: top-of-tick edge; True only on the first paused tick.
            Triggers the snapshot from ``last_commanded`` so the next resume
            blends from the actual last-commanded value (C0 continuity), not
            from fresh input.
        current_tick_paused: per-tick local read of the FSM's PAUSED flag.
            When True, skip the write and leave shared memory holding the
            previous values.
        alpha: blend factor in [0.0, 1.0]. 0.0 = fully frozen, 1.0 = fully
            tracking fresh input. Computed by the caller's top-of-tick block.
        frozen_snapshot: dict keyed ``'left'`` / ``'right'``. Mutated in-place
            on ``entering_pause`` to capture the blend-from anchor; read on
            every subsequent paused tick and during the resume ramp.
        last_commanded: dict keyed ``'left'`` / ``'right'``. Mutated in-place
            at the end of every non-paused tick with what was just written,
            so the next ``entering_pause`` snapshot has C0-continuous data.

    Returns: ``None``. All side effects flow through ``write_fn`` and the
    two mutable mapping arguments.
    """
    # Snapshot at pause entry — from last_commanded (C0-continuous), not
    # from fresh input. Bootstraps with new_* on the very first tick when
    # last_commanded hasn't been written yet; paused branches skip writes,
    # so the bootstrap value is read but never observable.
    #
    # No defensive .copy(): both `last_commanded.get(...)` (set by a prior
    # tick's `last_commanded[...] = out_left`) and the bootstrap fallback
    # (`new_left`, fresh per the contract above) are already non-aliased.
    if entering_pause:
        frozen_snapshot['left']  = last_commanded.get('left',  new_left)
        frozen_snapshot['right'] = last_commanded.get('right', new_right)

    if current_tick_paused:
        # Hand controller continues reading the last value we wrote — leave
        # shared memory alone. last_commanded is also intentionally NOT
        # updated here, so a subsequent re-pause snapshots the same anchor.
        return

    # Compute output: fully fresh at alpha=1.0, blended during resume ramp.
    if alpha < 1.0:
        f_left  = frozen_snapshot['left']
        f_right = frozen_snapshot['right']
        out_left  = f_left  + alpha * (new_left  - f_left)
        out_right = f_right + alpha * (new_right - f_right)
    else:
        out_left, out_right = new_left, new_right

    write_fn(out_left, out_right)

    # Track the actually-commanded value for the next pause's snapshot.
    # No defensive .copy(): out_left/out_right are either fresh from the
    # call site (alpha=1.0) or freshly allocated by the lerp expression
    # above (alpha<1.0). Neither aliases anything observed by the caller.
    last_commanded['left']  = out_left
    last_commanded['right'] = out_right
