# GENERATED — do not edit. Sync: tools/sync_homie_client.sh
# Source of truth: scripts/real_eval/homie_eval_gate.py
"""HomieGate — TCP-JSON client for `homie_process_test.py` with pre-flight
liveness, posture-patch verification, conditional calibration, and a
poll-based watchdog. Stdlib only.

Spec: docs/superpowers/specs/2026-04-28-homie-eval-teleop-design.md
"""
from __future__ import annotations
import json
import logging
import re
import socket
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class HomieState:
    """State retrieved from `homie_process_test.py`'s `status` reply.

    `calibrated`, `controller_running`, `nominal_height`, `final_goal` are
    populated by the post-`feat/status-rpc-fields` HOMIE; on older HOMIEs
    they are None and the gate falls back to regex source-file checks +
    assume-not-calibrated.
    """
    vx: float = 0.0
    vy: float = 0.0
    yaw: float = 0.0
    height: Optional[float] = None
    expires_in: float = 0.0
    calibrated: Optional[bool] = None
    controller_running: Optional[bool] = None
    nominal_height: Optional[float] = None
    final_goal: Optional[list] = None

    @classmethod
    def from_status_reply(cls, reply: dict) -> "HomieState":
        return cls(
            vx=float(reply.get("vx", 0.0)),
            vy=float(reply.get("vy", 0.0)),
            yaw=float(reply.get("yaw", 0.0)),
            height=reply.get("height"),
            expires_in=float(reply.get("expires_in", 0.0)),
            calibrated=reply.get("calibrated"),
            controller_running=reply.get("controller_running"),
            nominal_height=reply.get("nominal_height"),
            final_goal=reply.get("final_goal"),
        )

    def has_enriched_fields(self) -> bool:
        return self.calibrated is not None and self.nominal_height is not None


class HomieGateError(Exception):
    """Raised on connect/probe/calibrate failures."""


class HomieGate:
    """TCP-JSON client for `homie_process_test.py` with retry, posture
    verification, conditional calibration, and a poll-based watchdog."""

    def __init__(self, host: str, port: int,
                 connect_deadline_s: float = 30.0,
                 connect_interval_s: float = 1.0,
                 socket_timeout_s: float = 0.5):
        self.host = host
        self.port = port
        self.connect_deadline_s = connect_deadline_s
        self.connect_interval_s = connect_interval_s
        self.socket_timeout_s = socket_timeout_s
        self._sock: Optional[socket.socket] = None
        self._buf = b""
        self._sock_lock = threading.Lock()
        self._wd_thread: Optional[threading.Thread] = None
        self._wd_stop = threading.Event()
        self._wd_fired = False

    def connect_with_retry(self) -> None:
        deadline = time.monotonic() + self.connect_deadline_s
        last_err: Optional[Exception] = None
        attempts = 0
        logged_waiting = False
        while time.monotonic() < deadline:
            try:
                sock = socket.create_connection(
                    (self.host, self.port), timeout=self.socket_timeout_s)
                sock.settimeout(self.socket_timeout_s)
                self._sock = sock
                self._buf = b""
                if attempts > 0:
                    logger.info("HOMIE connection ready after %d attempt(s)",
                                attempts + 1)
                return
            except (ConnectionRefusedError, socket.timeout, OSError) as e:
                last_err = e
                attempts += 1
                if not logged_waiting and attempts >= 2:
                    logger.info("waiting for HOMIE on %s:%d ...",
                                self.host, self.port)
                    logged_waiting = True
                time.sleep(self.connect_interval_s)
        raise HomieGateError(
            f"Could not connect to HOMIE on {self.host}:{self.port} "
            f"within {self.connect_deadline_s}s deadline "
            f"(last error: {last_err}). "
            f"Is `run_homie.sh` running in Terminal A on PC2?")

    def _send_recv(self, msg: dict) -> dict:
        """Send a JSON command, read one line back. Caller holds the lock."""
        if self._sock is None:
            raise HomieGateError("not connected")
        line = (json.dumps(msg) + "\n").encode()
        try:
            self._sock.sendall(line)
            while b"\n" not in self._buf:
                chunk = self._sock.recv(4096)
                if not chunk:
                    raise HomieGateError("HOMIE closed connection")
                self._buf += chunk
            payload, self._buf = self._buf.split(b"\n", 1)
            return json.loads(payload.decode())
        except (socket.timeout, OSError, ValueError) as e:
            raise HomieGateError(f"HOMIE I/O error: {e}") from e

    def probe_state(self) -> HomieState:
        """Send {"cmd":"status"}, parse into HomieState. Strict 0.5 s timeout
        per spec §1.4 invariant 7."""
        with self._sock_lock:
            reply = self._send_recv({"cmd": "status"})
        if not reply.get("ok"):
            raise HomieGateError(f"HOMIE status returned not-ok: {reply}")
        return HomieState.from_status_reply(reply)

    # Regex matchers — load-bearing literals from apply_homie_r1x_patch.py
    _RE_PATCHED_HEIGHT_INIT = re.compile(r"command\[3\]\s*=\s*0\.68\b")
    _RE_PATCHED_HEIGHT_GET = re.compile(r"cmd_height\s*=\s*0\.68\s*-\s*0\.40")
    _RE_PATCHED_GOAL_FIRST = re.compile(r"-?0\.2616\b")
    _RE_PATCHED_GOAL_KNEE = re.compile(r"\b0\.5027\b")

    def verify_posture_patch(self, expo_root: str, state: HomieState,
                              skip_check: bool = False) -> None:
        """Verify the R1+X posture patch is applied. Prefer enriched status
        fields (post upstream PR); fall back to regex source-file checks.

        Raises HomieGateError with the apply-script hint on failure.

        If skip_check=True, logs a warning and returns without checking —
        operator escape hatch for the rare case where they accept the risk.
        """
        if skip_check:
            logger.warning("posture patch verification SKIPPED via "
                           "--skip-posture-check — operator accepts risk of "
                           "table scrape if patch is missing.")
            return
        if state.has_enriched_fields():
            self._verify_via_status(state)
            return
        if not expo_root:
            raise HomieGateError(
                "physical_ai_expo root not provided — pass --physical-ai-expo-root "
                "or set PHYSICAL_AI_EXPO_ROOT, or use --skip-posture-check.")
        self._verify_via_regex(expo_root)

    def _verify_via_regex(self, expo_root: str) -> None:
        utils = (Path(expo_root) / "physical_ai_expo" / "third_party"
                 / "g1_gym_deploy" / "utils")
        cheetah = utils / "cheetah_state_estimator.py"
        deploy = utils / "deployment_runner.py"
        problems = []
        if not cheetah.exists():
            problems.append(f"missing {cheetah}")
        else:
            txt = cheetah.read_text()
            if not self._RE_PATCHED_HEIGHT_INIT.search(txt):
                problems.append(f"cheetah_state_estimator.py: command[3]=0.68 not found")
            if not self._RE_PATCHED_HEIGHT_GET.search(txt):
                problems.append(f"cheetah_state_estimator.py: cmd_height=0.68 not found")
        if not deploy.exists():
            problems.append(f"missing {deploy}")
        else:
            txt = deploy.read_text()
            if not self._RE_PATCHED_GOAL_FIRST.search(txt):
                problems.append(f"deployment_runner.py: R1+X final_goal[0]=-0.2616 not found")
            if not self._RE_PATCHED_GOAL_KNEE.search(txt):
                problems.append(f"deployment_runner.py: R1+X final_goal knee=0.5027 not found")
        if problems:
            raise HomieGateError(
                "HOMIE posture patch missing or out of date:\n  - "
                + "\n  - ".join(problems)
                + f"\n\nFix:\n  python scripts/real_eval/apply_homie_r1x_patch.py "
                + f"{expo_root} --apply\n\nThen restart `run_homie.sh` in Terminal A.")

    def _verify_via_status(self, state: HomieState) -> None:
        problems = []
        if state.nominal_height is not None and abs(state.nominal_height - 0.68) > 1e-3:
            problems.append(f"nominal_height={state.nominal_height} (expected 0.68)")
        if state.final_goal is not None and len(state.final_goal) >= 4:
            if abs(state.final_goal[0] - (-0.2616)) > 1e-3:
                problems.append(f"final_goal[0]={state.final_goal[0]} (expected -0.2616)")
            if abs(state.final_goal[3] - 0.5027) > 1e-3:
                problems.append(f"final_goal[3]={state.final_goal[3]} (expected 0.5027)")
        if problems:
            raise HomieGateError(
                "HOMIE running with un-patched posture (per status RPC):\n  - "
                + "\n  - ".join(problems)
                + "\n\nApply the patch in physical_ai_expo and restart run_homie.sh.")

    CALIBRATE_GAP_S = 2.0  # gap between the two calibrate pulses (per HomieClient default)

    def calibrate_if_needed(self, state: HomieState) -> None:
        """If HOMIE reports already-calibrated (via enriched status), no-op.
        Else send 2 calibrate pulses with a CALIBRATE_GAP_S gap and re-probe.
        """
        if state.calibrated is True and state.controller_running is True:
            logger.info("HOMIE already calibrated and running — skipping calibrate")
            return
        # Either explicitly false or unknown (older HOMIE) → send pulses.
        logger.info("HOMIE not yet calibrated — sending 2 calibrate pulses")
        with self._sock_lock:
            self._send_recv({"cmd": "calibrate"})
        time.sleep(self.CALIBRATE_GAP_S)
        with self._sock_lock:
            self._send_recv({"cmd": "calibrate"})
        time.sleep(self.CALIBRATE_GAP_S)
        # Best-effort re-probe; on enriched HOMIE this confirms; on legacy
        # we can't tell, so we proceed.
        try:
            new_state = self.probe_state()
            if new_state.calibrated is False:
                raise HomieGateError(
                    "HOMIE rejected calibration (status.calibrated still false). "
                    "Inspect Terminal A logs for DeploymentRunner errors.")
        except HomieGateError:
            raise

    def start_watchdog(self, period_s: float = 1.0, fail_threshold: int = 3,
                       on_abort=None) -> None:
        """Spawn a daemon thread that polls probe_state() every period_s; if
        fail_threshold consecutive polls fail, calls on_abort() exactly once
        and exits. Period drift is corrected per §1.4 invariant 7."""
        if self._wd_thread is not None:
            raise HomieGateError("watchdog already started")
        self._wd_stop.clear()
        self._wd_fired = False

        def loop():
            consecutive_fails = 0
            while not self._wd_stop.is_set():
                t_start = time.monotonic()
                try:
                    self.probe_state()
                    consecutive_fails = 0
                except HomieGateError:
                    consecutive_fails += 1
                    if consecutive_fails >= fail_threshold and not self._wd_fired:
                        self._wd_fired = True
                        if on_abort is not None:
                            try:
                                on_abort()
                            except Exception as e:
                                logger.error("on_abort raised: %s", e)
                        return
                latency = time.monotonic() - t_start
                sleep_s = max(0.0, period_s - latency)
                # Wake on stop event without blocking the full sleep
                if self._wd_stop.wait(sleep_s):
                    return

        self._wd_thread = threading.Thread(
            target=loop, daemon=True, name="HomieGate-watchdog")
        self._wd_thread.start()

    def stop(self) -> None:
        """Best-effort {"cmd":"stop"} (zero velocity to HOMIE). Swallows
        socket errors — typically called from a `finally:` block when HOMIE
        may already be dead."""
        try:
            with self._sock_lock:
                self._send_recv({"cmd": "stop"})
        except HomieGateError as e:
            logger.warning("HomieGate.stop() best-effort failed: %s", e)
        except Exception as e:
            logger.warning("HomieGate.stop() unexpected error: %s", e)

    def close(self) -> None:
        # Stop watchdog first so it doesn't race with socket close
        self._wd_stop.set()
        if self._wd_thread is not None:
            self._wd_thread.join(timeout=2.0)
            self._wd_thread = None
        with self._sock_lock:
            if self._sock is not None:
                try:
                    self._sock.close()
                except OSError:
                    pass
                self._sock = None


def _cli_main() -> int:
    import argparse
    p = argparse.ArgumentParser(description="HomieGate probe CLI (for L2 drill)")
    p.add_argument("--homie-host", default="127.0.0.1")
    p.add_argument("--homie-port", type=int, default=7701)
    p.add_argument("--probe", action="store_true",
                   help="Connect, probe state, print, exit.")
    p.add_argument("--once", action="store_true",
                   help="Single probe (default: poll every 1 s until Ctrl+C).")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if not args.probe:
        p.error("--probe is required (only mode supported in CLI)")

    gate = HomieGate(args.homie_host, args.homie_port,
                     connect_deadline_s=10.0, connect_interval_s=0.5)
    try:
        gate.connect_with_retry()
        while True:
            state = gate.probe_state()
            print(f"calibrated={state.calibrated} "
                  f"controller_running={state.controller_running} "
                  f"nominal_height={state.nominal_height} "
                  f"vx={state.vx:.3f} vy={state.vy:.3f} yaw={state.yaw:.3f}")
            if args.once:
                return 0
            time.sleep(1.0)
    except HomieGateError as e:
        print(f"HOMIE error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        gate.close()


if __name__ == "__main__":
    sys.exit(_cli_main())
