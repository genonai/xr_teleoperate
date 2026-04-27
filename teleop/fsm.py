"""teleop.fsm — finite-state machine for keyboard-driven teleop control.

Lives in its own module specifically so it can be imported on developer
machines that lack DDS / Vuer / Isaac Sim / multiprocessing-aware logging
(``logging_mp``). Importing :mod:`teleop.teleop_hand_and_arm` triggers all of
those at module load time and is impossible on a clean Mac. Importing this
module triggers only the standard library, so the FSM is unit-testable from
``pytest`` without any robotics dependencies installed.

Public surface:
    State globals (read by the main loop, mutated by :func:`on_press`)::

        START, STOP, READY, RECORD_RUNNING, RECORD_TOGGLE, PAUSED

    Constants (recomputed from ``--frequency`` at startup)::

        RAMP_TICKS, RAMP_DURATION_SEC

    Callables::

        on_press(key)        — sshkeyboard / IPC handler
        get_state() -> dict  — heartbeat snapshot

The teleop entry point (:mod:`teleop.teleop_hand_and_arm`) imports this
module as ``from teleop import fsm`` and references state via attribute
access (``fsm.START``, ``fsm.STOP``, ...) — required so that mutations
performed by :func:`on_press` are visible to the main loop running in the
same process.

FSM diagram::

      state          [Ready]      ==>        [Recording]     ==>         [AutoSave]     -->     [Ready]
      START           True                      True                        True                 True
      READY           True                      False                       False                True
      RECORD_RUNNING  False                     True                        False                False
      RECORD_TOGGLE   False         True        False           True        False                False

  ==> manual: when READY is True, set RECORD_TOGGLE=True to transition.
  --> auto  : Auto-transition after saving data.

  [p] Pause branch (only from [Ready], not from [Recording]):
     [Ready]  --p-->  [Paused]  --p-->  [Resuming (ramp, RAMP_TICKS ticks)]  -->  [Ready]
  During [Paused]: arm re-sends frozen target, hand writes skipped, [s] rejected.
  During [Resuming]: alpha sweeps 1/RAMP_TICKS → 1.0, blending frozen → fresh.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# --- State transitions -------------------------------------------------------

START          = False  # Enable to start robot following VR user motion
STOP           = False  # Enable to begin system exit procedure
READY          = False  # Ready to (1) enter START state, (2) enter RECORD_RUNNING state
RECORD_RUNNING = False  # True if [Recording]
RECORD_TOGGLE  = False  # Toggle recording state
PAUSED         = False  # True while [p] paused — robot holds last commanded pose

# Ramp timing — RAMP_TICKS is recomputed from args.frequency at startup
# in teleop_hand_and_arm.py so the wall-clock duration stays at
# RAMP_DURATION_SEC regardless of control rate. Placeholder = 30 ticks @ 30 Hz.
RAMP_DURATION_SEC = 1.0
RAMP_TICKS        = 30


def on_press(key):
    """Keyboard callback for the teleop FSM.

    Pure logic — mutates module globals only. Safe to call from the
    sshkeyboard listener thread (Python's GIL serializes the simple-bool
    reassignments) or from the IPC server callback.

    The branch order matters: each ``elif`` is the gate that the prior
    branches did not match. The two ``elif`` branches at the end split
    "known key whose guard rejected it" from "key not bound to anything"
    so the warning text can name the actual failure mode.
    """
    global STOP, START, RECORD_TOGGLE, PAUSED
    if key == 'r':
        START = True
    elif key == 'q':
        START = False
        STOP = True
    elif key == 's' and START and not PAUSED:
        RECORD_TOGGLE = True
    elif key == 'p' and START and not RECORD_RUNNING and not RECORD_TOGGLE:
        # Also gate on RECORD_TOGGLE: closes the sub-tick race where `s` flips
        # RECORD_TOGGLE=True but the main loop has not yet promoted it to
        # RECORD_RUNNING=True — otherwise a `p` pressed between those two
        # events would sneak paused frames into the new episode.
        PAUSED = not PAUSED
        # Synchronous feedback so a fast double-press (where both flips happen
        # between two main-loop ticks → no edge detected → no ⏸/▶ log) still
        # tells the operator that each press registered.
        logger.info(f"[on_press] PAUSED → {PAUSED}")
    elif key in ('s', 'p'):
        # Key is known but its guard rejected it (not tracking, wrong state, etc.)
        logger.warning(f"[on_press] {key} rejected — guard not satisfied for current state.")
    else:
        # Key is not bound to any action in this FSM.
        logger.warning(f"[on_press] {key} — no action defined for this key.")


def get_state() -> dict:
    """Return current heartbeat state as a JSON-serializable dict.

    Used by the IPC server's heartbeat endpoint to broadcast FSM state
    to remote dashboards. Read-only — does not mutate.

    Note: ``RECORD_TOGGLE`` is intentionally omitted. It is a transient
    edge-triggered command flag (set by :func:`on_press`, cleared by the
    main loop within one tick), not a steady-state value, so sampling it
    via the heartbeat would race the consumer that handles the toggle.
    """
    return {
        "START": START,
        "STOP": STOP,
        "READY": READY,
        "RECORD_RUNNING": RECORD_RUNNING,
        "PAUSED": PAUSED,
    }
