# GENERATED — do not edit. Sync: tools/sync_homie_client.sh
# Source of truth: scripts/real_eval/homie_io.py
"""HOMIE-release-mode I/O wrappers for eval_remote.py.

Drop-in replacements for G1_29_ArmController and InspireFTP*Controller that
route arm commands through LCM ``arm_action`` (g1_control merges with HOMIE
legs) and hand commands through DDS ``rt/inspire_hand/ctrl/{l,r}``
(``Headless_driver_double`` subscribes and owns Modbus).

This is the canonical HOMIE-release-mode path per
``repos/physical_ai_expo/physical_ai_expo/pickhold/runner.py``. It avoids:

  * the ``rt/arm_sdk`` grinding (no AI arbiter under HOMIE release mode →
    our writes fight HOMIE's ``rt/lowcmd`` output), and
  * the direct-Modbus grinding (two RS485 masters: us via
    ``InspireFTP*Controller`` and ``Headless_driver_double``).

These wrappers expose the small subset of the original APIs that
``scripts/remote_inference/eval_remote.py`` actually touches. They are NOT
full reimplementations — anything not used by eval_remote is intentionally
omitted.

Constructor argument order MIRRORS the original FTP controllers
(``left_in, right_in, data_lock, state_out, action_out, ..., tau_est_out``)
so the call sites in eval_remote remain identical apart from the class name.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

INSPIRE_NUM_MOTORS = 6


# ── Internals ────────────────────────────────────────────────────────────


class _LowstateBufferShim:
    """Mimics ``G1_29_ArmController.lowstate_buffer.GetData()`` for callers
    that touch ``arm_ctrl.lowstate_buffer`` directly (e.g. SafetyHeartbeat).

    Under ``--use-homie`` we skip the heartbeat anyway, but we keep this
    shim for safety in case something else reaches in.
    """

    def __init__(self, get_lowstate_callable):
        self._get = get_lowstate_callable

    def GetData(self):
        return self._get()


# ── Arm controller wrapper ───────────────────────────────────────────────


class HomieArmController:
    """Drop-in replacement for ``G1_29_ArmController`` under HOMIE release mode.

    Routes arm commands through LCM ``arm_action`` (``ArmActionPublisher``)
    instead of DDS ``rt/arm_sdk``. Reads arm state from ``rt/lowstate``
    directly. Waist is owned by ``g1_control``, not by us — the
    waist-upright monkey-patch in eval_remote must be skipped under
    ``--use-homie``.

    API surface (only what eval_remote uses):
        ctrl_dual_arm(target_q, target_tauff=None)
        ctrl_dual_arm_go_home()
        get_current_motor_q() -> 29D ndarray
        get_current_dual_arm_q() -> 14D ndarray
        get_current_dual_arm_dq() -> 14D ndarray
        enable_publishing() -> no-op
        stop()
        lowstate_buffer.GetData() -> raw LowState_ or None
        _publishing_enabled (attribute, default True)
    """

    def __init__(self, lc, arm_pub_hz: Optional[float] = None,
                 lerp_sec: float = 0.15):
        from physical_ai_expo.motion.capture_playback import (
            ArmActionPublisher, DEFAULT_ARM_PUB_HZ,
        )
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as hg_LowState

        if arm_pub_hz is None:
            arm_pub_hz = DEFAULT_ARM_PUB_HZ

        # Subscribe FIRST so we can snapshot current arm pose before we
        # start the publisher (avoids a first-tick jump to ARM_NOMINAL_POSE).
        self._lowstate_sub = ChannelSubscriber("rt/lowstate", hg_LowState)
        self._lowstate_sub.Init()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if self._lowstate_sub.Read() is not None:
                break
            time.sleep(0.02)

        cur_q = self._read_lowstate_q_29d()
        if cur_q is None:
            initial_arm = None
            logger.warning("HomieArmController: no rt/lowstate within 2 s; "
                           "ArmActionPublisher will start at ARM_NOMINAL_POSE")
        else:
            initial_arm = cur_q[15:29].copy()

        self._arm_pub = ArmActionPublisher(
            lc, hz=arm_pub_hz, idle_pose=initial_arm, lerp_sec=lerp_sec,
        )
        self._arm_pub.start()

        # Back-compat attributes / shim
        self._publishing_enabled = True
        self.lowstate_buffer = _LowstateBufferShim(self._read_raw_lowstate)

        logger.info(
            "HomieArmController ready: ArmActionPublisher started "
            "(lcm arm_action @ %.0f Hz, lerp=%.3fs, initial_arm=%s)",
            arm_pub_hz, lerp_sec,
            "current" if initial_arm is not None else "ARM_NOMINAL_POSE",
        )

    # ── lowstate helpers ────────────────────────────────────────────────

    def _read_raw_lowstate(self):
        """Returns the raw rt/lowstate IDL object (LowState_) or None."""
        return self._lowstate_sub.Read()

    def _read_lowstate_q_29d(self) -> Optional[np.ndarray]:
        ls = self._lowstate_sub.Read()
        if ls is None:
            return None
        return np.array([ls.motor_state[i].q for i in range(29)],
                        dtype=np.float64)

    def _read_lowstate_dq_29d(self) -> Optional[np.ndarray]:
        ls = self._lowstate_sub.Read()
        if ls is None:
            return None
        return np.array([ls.motor_state[i].dq for i in range(29)],
                        dtype=np.float64)

    # ── G1_29_ArmController API surface ─────────────────────────────────

    def ctrl_dual_arm(self, target_q, target_tauff=None):
        """Set 14D arm target via LCM arm_action.

        Args:
            target_q: 14D (arm-only) or 29D (full body — we slice [15:29]).
            target_tauff: ignored. ArmActionPublisher LCM path is positions-
                only; ``g1_control`` handles arm torque internally on its
                merged ``rt/lowcmd`` output.
        """
        target_q = np.asarray(target_q, dtype=np.float64)
        if target_q.size == 29:
            arm_14 = target_q[15:29]
        elif target_q.size == 14:
            arm_14 = target_q
        else:
            raise ValueError(
                f"target_q size {target_q.size} (expected 14 or 29)")
        self._arm_pub.set_target(arm_14)

    def ctrl_dual_arm_go_home(self):
        """Ramp arm targets from current → ARM_NOMINAL_POSE over 2 s."""
        from physical_ai_expo.motion.capture_playback import ARM_NOMINAL_POSE
        self.ramp_to(ARM_NOMINAL_POSE, duration_s=2.0)

    def ramp_to(self, target_14d, duration_s: float = 2.0, hz: float = 30.0):
        """Linear ramp arm targets from current → target_14d over duration_s.

        Useful for safe transitions; mirrors the upstream
        ``ctrl_dual_arm_go_home`` pattern but exposed for external rampers.
        """
        target = np.asarray(target_14d, dtype=np.float64).reshape(-1)
        if target.size != 14:
            raise ValueError(f"ramp target size {target.size} (expected 14)")
        cur = self._arm_pub.get_target().astype(np.float64)
        ticks = max(1, int(round(duration_s * hz)))
        dt = 1.0 / hz
        for i in range(ticks + 1):
            alpha = i / ticks
            interp = (1.0 - alpha) * cur + alpha * target
            self._arm_pub.set_target(interp)
            time.sleep(dt)

    def get_current_motor_q(self):
        """Returns 29D motor q from rt/lowstate; zeros if not yet ready."""
        q = self._read_lowstate_q_29d()
        return q if q is not None else np.zeros(29)

    def get_current_dual_arm_q(self):
        """14D arm-only q (joints 15..28)."""
        return self.get_current_motor_q()[15:29]

    def get_current_dual_arm_dq(self):
        """14D arm-only dq (joints 15..28); zeros if not yet ready."""
        dq = self._read_lowstate_dq_29d()
        if dq is None:
            return np.zeros(14)
        return dq[15:29]

    def enable_publishing(self):
        """No-op: ArmActionPublisher is always running once started."""
        pass

    def speed_gradual_max(self, t: float = 5.0):
        """No-op shim for ``G1_29_ArmController.speed_gradual_max``.

        Upstream this ramps the ``rt/arm_sdk`` weight from 0→1 over ``t``
        seconds. Under HOMIE we don't write ``rt/arm_sdk`` at all
        (g1_control merges arm_action into rt/lowcmd directly), so there's
        no weight to ramp. Kept for API parity with G1_29_ArmController.
        """
        del t  # unused — accepted for signature compatibility
        return

    def stop(self):
        if self._arm_pub is not None:
            self._arm_pub.stop()
            self._arm_pub = None


# ── Hand controller wrapper ──────────────────────────────────────────────


class HomieHandController:
    """Drop-in replacement for ``InspireFTP{1,6}DOFController`` under HOMIE.

    Reads commanded grip from shared mp arrays/values (eval_remote populates
    them from policy output). Publishes via DDS ``rt/inspire_hand/ctrl/{l,r}``
    using ``InspireHandPublisher``. Reads state from DDS
    ``rt/inspire_hand/state/{l,r}``. ``Headless_driver_double`` (started by
    ``run_homie.sh``) subscribes to the ctrl topics and owns Modbus.

    Constructor signature mirrors ``InspireFTP{1,6}DOFController`` so the
    call site in eval_remote is identical apart from the class name.
    """

    def __init__(
        self,
        left_in,            # mp.Value("d") (1dof) or mp.Array("d", 6) (6dof)
        right_in,           # mp.Value("d") (1dof) or mp.Array("d", 6) (6dof)
        data_lock,          # mp.Lock
        state_out,          # mp.Array("d", 2) (1dof) or mp.Array("d", 12) (6dof)
        action_out,         # mp.Array("d", 2) (1dof) or mp.Array("d", 12) (6dof)
        simulation_mode: bool = False,
        fps: int = 30,
        tau_est_out=None,   # mp.Array("d", 12) or None — kept zeros under --use-homie
        mode: Optional[str] = None,
    ):
        # Auto-detect mode from state_out length if not explicitly given.
        if mode is None:
            try:
                state_len = len(state_out)
            except TypeError:
                state_len = 0
            mode = "6dof" if state_len >= 12 else "1dof"
        if mode not in ("1dof", "6dof"):
            raise ValueError(f"mode must be '1dof' or '6dof', got {mode!r}")
        self._mode = mode

        self._left_in = left_in
        self._right_in = right_in
        self._data_lock = data_lock
        self._state_out = state_out
        self._action_out = action_out
        self._tau_est_out = tau_est_out
        self._publish_hz = float(fps) if fps else 30.0
        self.simulation_mode = simulation_mode

        # Lazy imports so this file is Mac-importable.
        from physical_ai_expo.motion.capture_playback import (
            InspireHandPublisher,
        )
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from inspire_sdkpy import inspire_dds

        self._hand_pub = InspireHandPublisher()
        self._left_state_sub = ChannelSubscriber(
            "rt/inspire_hand/state/l", inspire_dds.inspire_hand_state)
        self._left_state_sub.Init()
        self._right_state_sub = ChannelSubscriber(
            "rt/inspire_hand/state/r", inspire_dds.inspire_hand_state)
        self._right_state_sub.Init()

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.running = False

        # Auto-start (matches InspireFTP*Controller behaviour: spawn control
        # thread inside __init__, no separate .start() call).
        self._start_thread()

        logger.info(
            "HomieHandController initialized (mode=%s, fps=%.0f, sim=%s, "
            "publishes DDS rt/inspire_hand/ctrl/{l,r})",
            self._mode, self._publish_hz, simulation_mode,
        )

    def _start_thread(self):
        if self._thread is not None:
            return
        self._stop_event.clear()
        self.running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="homie-hand-ctrl",
        )
        self._thread.start()

    def stop(self):
        self.running = False
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("HomieHandController stopped")

    def _read_command_6d(self):
        """Read current commanded values into (left_6d, right_6d).

        For 1dof: broadcasts the scalar to all 6 motors.
        For 6dof: copies the 6 per-finger values directly.
        """
        if self._mode == "1dof":
            l = float(np.clip(self._left_in.value, 0.0, 1.0))
            r = float(np.clip(self._right_in.value, 0.0, 1.0))
            left_6d = np.full(INSPIRE_NUM_MOTORS, l, dtype=np.float64)
            right_6d = np.full(INSPIRE_NUM_MOTORS, r, dtype=np.float64)
        else:
            left_6d = np.array(
                [float(np.clip(self._left_in[i], 0.0, 1.0))
                 for i in range(INSPIRE_NUM_MOTORS)],
                dtype=np.float64)
            right_6d = np.array(
                [float(np.clip(self._right_in[i], 0.0, 1.0))
                 for i in range(INSPIRE_NUM_MOTORS)],
                dtype=np.float64)
        return left_6d, right_6d

    def _read_state_6d(self):
        """Read DDS state into (left_6d, right_6d), normalized [0,1].

        Returns previous-known values (zeros initially) if no message
        is available — matches the stale-data tolerance of the FTP
        subscribe loop.
        """
        left_6d = np.zeros(INSPIRE_NUM_MOTORS, dtype=np.float64)
        right_6d = np.zeros(INSPIRE_NUM_MOTORS, dtype=np.float64)
        lh = self._left_state_sub.Read()
        if (lh is not None and hasattr(lh, "angle_act")
                and len(lh.angle_act) >= INSPIRE_NUM_MOTORS):
            for i in range(INSPIRE_NUM_MOTORS):
                left_6d[i] = lh.angle_act[i] / 1000.0
        rh = self._right_state_sub.Read()
        if (rh is not None and hasattr(rh, "angle_act")
                and len(rh.angle_act) >= INSPIRE_NUM_MOTORS):
            for i in range(INSPIRE_NUM_MOTORS):
                right_6d[i] = rh.angle_act[i] / 1000.0
        return left_6d, right_6d

    def _write_state_action(self, left_6d_cmd, right_6d_cmd,
                            left_6d_state, right_6d_state):
        """Mirror the FTP controller's state_out / action_out semantics.

        1dof shape (2,):
          state_out[0/1]  = mean(left_state) / mean(right_state)
          action_out[0/1] = left_grip_in.value / right_grip_in.value
        6dof shape (12,):
          state_out[0..5] = left_state, [6..11] = right_state
          action_out[0..5] = left_cmd,  [6..11] = right_cmd
        """
        if self._mode == "1dof":
            left_state_avg = float(np.mean(left_6d_state))
            right_state_avg = float(np.mean(right_6d_state))
            left_cmd_scalar = float(self._left_in.value)
            right_cmd_scalar = float(self._right_in.value)
            with self._data_lock:
                self._state_out[0] = left_state_avg
                self._state_out[1] = right_state_avg
                self._action_out[0] = left_cmd_scalar
                self._action_out[1] = right_cmd_scalar
        else:
            with self._data_lock:
                for i in range(INSPIRE_NUM_MOTORS):
                    self._state_out[i] = float(left_6d_state[i])
                    self._state_out[INSPIRE_NUM_MOTORS + i] = float(right_6d_state[i])
                    self._action_out[i] = float(left_6d_cmd[i])
                    self._action_out[INSPIRE_NUM_MOTORS + i] = float(right_6d_cmd[i])

        # tau_est_out stays at zeros — DDS hand state has no torque field.
        # We deliberately do NOT touch self._tau_est_out under --use-homie.

    def _run(self):
        period = 1.0 / self._publish_hz
        while not self._stop_event.is_set():
            t_start = time.monotonic()
            try:
                left_6d_cmd, right_6d_cmd = self._read_command_6d()
                self._hand_pub.publish(left_6d_cmd, right_6d_cmd)

                if self.simulation_mode:
                    left_6d_state, right_6d_state = left_6d_cmd, right_6d_cmd
                else:
                    left_6d_state, right_6d_state = self._read_state_6d()

                self._write_state_action(
                    left_6d_cmd, right_6d_cmd,
                    left_6d_state, right_6d_state,
                )
            except Exception as e:  # noqa: BLE001 — daemon thread soft-fail
                logger.warning("HomieHandController loop error: %s", e)

            elapsed = time.monotonic() - t_start
            sleep_s = max(0.0, period - elapsed)
            if self._stop_event.wait(sleep_s):
                return
