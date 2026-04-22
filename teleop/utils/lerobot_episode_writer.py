"""LeRobotEpisodeWriter — direct-to-LeRobot v3 recorder for xr_teleoperate.

Replaces the JSON-intermediate recording path with per-episode parquet + MP4
artifacts written to a staging directory. A session-end consolidation script
(`finalize_lerobot_dataset.py`) repacks staging into the LeRobot v3 chunked
layout.

Design contract — see docs/superpowers/specs/2026-04-21-direct-lerobot-recording-design.md

Schema is locked on Unitree_G1_Inspire_HeadOnly_Mono:
    observation.state / action: float32[26]
        left_arm.qpos[7] + right_arm.qpos[7] + left_ee.qpos[6] + right_ee.qpos[6]
    observation.images.cam_head: 640x480 BGR @ 30 fps (incoming key 'color_0')

Abort semantics — IL temporal-sync requirement:
    On video-queue overflow, ffmpeg failure, or max-duration breach, the
    entire episode is discarded. Silently dropping frames while keeping the
    corresponding state/action rows desynchronizes modalities and produces
    silently-corrupted training data.
"""

from __future__ import annotations

import datetime
import json
import os
import queue
import shutil
import subprocess
import time
from pathlib import Path
from threading import Thread

import numpy as np

import logging_mp

try:
    logger_mp = logging_mp.getLogger(__name__)
except AttributeError:
    # PC2 / .223 ships a logging_mp variant that exposes snake_case
    # get_logger instead of camelCase getLogger. Same call semantics.
    # See feedback_teleop_223_setup.md.
    logger_mp = logging_mp.get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants — see spec §2 (locked schema) and §4 (hot-path mechanics)
# ---------------------------------------------------------------------------

ACTION_DIM = 26
STATE_DIM = 26
# 26D field order: left_arm.qpos (7) + right_arm.qpos (7)
#                + left_ee.qpos  (6) + right_ee.qpos  (6)
FLATTEN_PARTS: tuple[tuple[str, int], ...] = (
    ("left_arm", 7),
    ("right_arm", 7),
    ("left_ee", 6),
    ("right_ee", 6),
)

DEFAULT_FPS = 30
DEFAULT_IMAGE_SIZE = (640, 480)  # (W, H)
DEFAULT_MAX_EPISODE_SEC = 60
DEFAULT_QUEUE_HEADROOM_SEC = 2  # bounded queue size = fps * this
DEFAULT_ROBOT_TYPE = "Unitree_G1_Inspire_HeadOnly_Mono"
DEFAULT_CAMERA_INCOMING_KEY = "color_0"  # per ROBOT_CONFIGS camera_to_image_key
CAM_FEATURE_KEY = "observation.images.cam_head"


class LeRobotEpisodeWriter:
    """Direct-to-LeRobot episode writer. Drop-in for EpisodeWriter.

    Public interface matches ``teleop.utils.episode_writer.EpisodeWriter``:
        is_ready() -> bool
        create_episode() -> bool
        add_item(colors, depths, states, actions, tactiles, audios, sim_state) -> None
        save_episode() -> bool
        close() -> None

    Additional read-only attribute used by the main loop to detect aborts:
        episode_corrupted: bool
    """

    def __init__(
        self,
        task_dir: str,
        task_goal: str | None = None,
        task_desc: str | None = None,
        task_steps: str | None = None,
        frequency: int = DEFAULT_FPS,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
        rerun_log: bool = True,
        robot_type: str = DEFAULT_ROBOT_TYPE,
        max_episode_sec: int = DEFAULT_MAX_EPISODE_SEC,
    ) -> None:
        if frequency != DEFAULT_FPS:
            # Spec §2 locks fps at 30. Permit other values but warn — the
            # LeRobot video feature is fps-tagged at consolidation time.
            logger_mp.warning(
                f"LeRobotEpisodeWriter instantiated at {frequency} Hz; "
                f"spec locks at {DEFAULT_FPS}. MP4 timestamps may desync."
            )

        self.task_dir = Path(task_dir)
        self.staging_dir = self.task_dir / "_staging"
        self.staging_dir.mkdir(parents=True, exist_ok=True)

        self.fps = int(frequency)
        self.image_size = tuple(image_size)  # (W, H)
        self.robot_type = robot_type
        self.task_goal = task_goal or ""
        self.task_desc = task_desc or ""
        self.task_steps = task_steps or ""

        self.max_frames = self.fps * int(max_episode_sec)
        self.queue_size = self.fps * DEFAULT_QUEUE_HEADROOM_SEC

        # Preallocated per-session buffers (reused across episodes).
        self._state_buf = np.zeros((self.max_frames, STATE_DIM), dtype=np.float32)
        self._action_buf = np.zeros((self.max_frames, ACTION_DIM), dtype=np.float32)
        self._ts_buf = np.zeros((self.max_frames,), dtype=np.float32)
        self._n_frames = 0

        # Per-session state.
        self.episode_id = self._scan_latest_episode_id()
        self.item_id = -1
        self.is_available = True
        self.episode_corrupted = False

        # Per-episode handles, allocated in create_episode().
        self._ep_tmp_dir: Path | None = None
        self._ep_final_dir: Path | None = None
        self._frame_queue: queue.Queue | None = None
        self._worker_thread: Thread | None = None
        self._worker_error: BaseException | None = None
        self._ffmpeg_proc: subprocess.Popen | None = None
        self._ep_start_time: float = 0.0
        self._stop_worker = False

        # Rerun is optional; delay import so headless setups don't pay for it.
        self.rerun_log = rerun_log
        self._rerun_logger = None
        if rerun_log:
            try:
                from .rerun_visualizer import RerunLogger

                self._rerun_logger = RerunLogger(
                    prefix="online/", IdxRangeBoundary=60, memory_limit="300MB"
                )
            except Exception as e:
                logger_mp.warning(f"RerunLogger unavailable: {e}")
                self.rerun_log = False

        logger_mp.info(
            f"LeRobotEpisodeWriter initialized at {self.staging_dir} "
            f"(next episode_id={self.episode_id + 1}, robot_type={robot_type})"
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        return self.is_available

    def create_episode(self) -> bool:
        if not self.is_available:
            logger_mp.info("create_episode: still finalizing prior episode")
            return False

        self.episode_id += 1
        self.item_id = -1
        self._n_frames = 0
        self.episode_corrupted = False
        self._worker_error = None
        self._ep_start_time = time.monotonic()

        name = f"episode_{self.episode_id:06d}"
        self._ep_tmp_dir = self.staging_dir / f"{name}.tmp"
        self._ep_final_dir = self.staging_dir / name

        # Clean any stale .tmp from a prior crash.
        if self._ep_tmp_dir.exists():
            logger_mp.warning(f"removing stale staging dir {self._ep_tmp_dir}")
            shutil.rmtree(self._ep_tmp_dir, ignore_errors=True)
        self._ep_tmp_dir.mkdir(parents=True)

        if not self._spawn_ffmpeg():
            # ffmpeg failed to start — treat as abort.
            self._teardown_episode(purge=True)
            return False

        self._frame_queue = queue.Queue(maxsize=self.queue_size)
        self._stop_worker = False
        self._worker_thread = Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

        self.is_available = False
        self.episode_dir = str(self._ep_final_dir)  # kept for API parity
        logger_mp.info(f"episode created: {self._ep_tmp_dir}")
        return True

    def add_item(
        self,
        colors: dict,
        depths: dict | None = None,
        states: dict | None = None,
        actions: dict | None = None,
        tactiles=None,
        audios=None,
        sim_state=None,
    ) -> None:
        if self.is_available or self.episode_corrupted:
            # No active episode, or we've already aborted this one.
            return

        # Surface any async worker failure as an abort on the next call.
        if self._worker_error is not None:
            self._abort_episode(f"encoder worker died: {self._worker_error!r}")
            return

        self.item_id += 1

        # Max-duration guard.
        if self._n_frames >= self.max_frames:
            self._abort_episode(
                f"episode exceeded max duration "
                f"({self.max_frames // self.fps}s / {self.max_frames} frames)"
            )
            return

        # Commit state/action row — MUST be in lockstep with the frame push
        # below. If put_nowait raises, we abort before the frame is queued
        # and roll back n_frames so the (state, action, ts) buffers stay
        # consistent with what's been sent to ffmpeg.
        idx = self._n_frames

        # Fill 26D state + action directly into preallocated buffer slices.
        # The prior _flatten_26d() helper allocated a fresh per-frame list +
        # numpy scratch array, contradicting the "zero hot-path allocation"
        # invariant advertised in the spec. Writing straight into the buffer
        # keeps the hot path free of Python-level allocations per frame.
        try:
            self._fill_26d_row(self._state_buf, idx, states, kind="state")
            self._fill_26d_row(self._action_buf, idx, actions, kind="action")
        except (KeyError, ValueError, TypeError) as e:
            self._abort_episode(f"state/action flattening failed: {e}")
            return

        # Extract head-cam BGR frame.
        frame = None if colors is None else colors.get(DEFAULT_CAMERA_INCOMING_KEY)
        if frame is None:
            self._abort_episode(
                f"missing '{DEFAULT_CAMERA_INCOMING_KEY}' frame in colors"
            )
            return
        if frame.shape[:2] != (self.image_size[1], self.image_size[0]):
            self._abort_episode(
                f"frame shape {frame.shape[:2]} != expected "
                f"{(self.image_size[1], self.image_size[0])}"
            )
            return

        self._ts_buf[idx] = time.monotonic() - self._ep_start_time
        self._n_frames = idx + 1

        # Pack a 4-tuple so the worker thread can handle both the ffmpeg pipe
        # and Rerun logging off the 60 Hz hot path. Rerun IPC can spike
        # unpredictably (shared-memory ring buffer or not) — keeping it out
        # of add_item is what preserves the 16.67 ms budget.
        #
        # Caller contract (matches existing EpisodeWriter): do not mutate
        # `states` / `actions` between this call and the next teleop
        # iteration; we queue references, not deep copies.
        try:
            self._frame_queue.put_nowait(
                (frame, states, actions, self.item_id)
            )
        except queue.Full:
            # Roll back the row we just wrote — the abort will discard the
            # buffers anyway, but this preserves the invariant for unit tests.
            self._n_frames = idx
            self._abort_episode(
                f"encoder queue overflow (maxsize={self.queue_size}) — "
                f"ffmpeg fell >{DEFAULT_QUEUE_HEADROOM_SEC}s behind"
            )
            return

    def save_episode(self) -> bool:
        """Finalize current episode. Returns True on success, False on abort."""
        if self.is_available:
            logger_mp.warning("save_episode called with no active episode")
            return False
        if self.episode_corrupted:
            logger_mp.info("save_episode: episode already aborted; ignoring")
            return False

        if self._n_frames == 0:
            self._abort_episode("episode has zero frames")
            return False

        # Signal worker to drain, then close ffmpeg cleanly.
        self._stop_worker = True
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=5.0)
            if self._worker_thread.is_alive():
                self._abort_episode("encoder worker failed to drain in 5s")
                return False

        if self._worker_error is not None:
            self._abort_episode(f"encoder worker died: {self._worker_error!r}")
            return False

        # Close ffmpeg stdin and wait for MP4 finalization.
        assert self._ffmpeg_proc is not None
        try:
            if self._ffmpeg_proc.stdin is not None:
                self._ffmpeg_proc.stdin.close()
        except Exception:
            pass
        try:
            rc = self._ffmpeg_proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            self._ffmpeg_proc.kill()
            self._abort_episode("ffmpeg failed to finalize within 10s")
            return False
        if rc != 0:
            self._abort_episode(f"ffmpeg exited with code {rc}")
            return False

        # Write parquet + meta.json into the .tmp dir.
        assert self._ep_tmp_dir is not None
        try:
            self._write_parquet(self._ep_tmp_dir / "data.parquet")
            self._write_meta(self._ep_tmp_dir / "meta.json")
        except Exception as e:
            self._abort_episode(f"parquet/meta write failed: {e}")
            return False

        # Atomic parent-dir rename.
        assert self._ep_final_dir is not None
        try:
            os.rename(self._ep_tmp_dir, self._ep_final_dir)
        except OSError as e:
            self._abort_episode(f"atomic rename failed: {e}")
            return False

        logger_mp.info(
            f"episode {self.episode_id:06d} saved ({self._n_frames} frames) → "
            f"{self._ep_final_dir}"
        )
        self._teardown_episode(purge=False)
        return True

    def close(self) -> None:
        """Ensure any in-flight episode is either saved or cleaned up."""
        if not self.is_available:
            if self.episode_corrupted:
                self._teardown_episode(purge=True)
            else:
                # Best-effort save of partial data; caller shouldn't rely on
                # this path — explicit save_episode() is preferred.
                saved = False
                try:
                    saved = self.save_episode()
                except Exception as e:
                    logger_mp.error(f"close: save_episode failed: {e}")
                if not saved:
                    self._teardown_episode(purge=True)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _scan_latest_episode_id(self) -> int:
        """Return the highest existing episode_id in staging (or -1)."""
        latest = -1
        if not self.staging_dir.exists():
            return latest
        for p in self.staging_dir.iterdir():
            if not p.is_dir():
                continue
            name = p.name
            if name.endswith(".tmp"):
                continue
            if not name.startswith("episode_"):
                continue
            try:
                n = int(name.split("_")[-1])
            except ValueError:
                continue
            latest = max(latest, n)
        return latest

    def _spawn_ffmpeg(self) -> bool:
        assert self._ep_tmp_dir is not None
        W, H = self.image_size
        out_path = self._ep_tmp_dir / "cam_head.mp4"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{W}x{H}",
            "-r", str(self.fps),
            "-i", "pipe:",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            "-g", "10",  # torchcodec-friendly keyframe interval (spec §4.2)
            "-an",
            str(out_path),
        ]
        try:
            # stderr=DEVNULL: we already run ffmpeg with `-loglevel error`, and
            # we never read from the stderr PIPE. An un-drained PIPE deadlocks
            # ffmpeg once its stderr buffer (~64KB on Linux) fills — see
            # review feedback on PR #2. Dropping to DEVNULL removes the
            # deadlock vector at the cost of losing error text (we already
            # surface ffmpeg exit code in save_episode's abort message).
            self._ffmpeg_proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            logger_mp.error("ffmpeg not found on PATH — cannot record video")
            return False
        except Exception as e:
            logger_mp.error(f"failed to spawn ffmpeg: {e}")
            return False
        return True

    def _worker_loop(self) -> None:
        """Drain the bounded queue; pipe frames to ffmpeg and log to Rerun.

        Both ffmpeg piping and Rerun logging run here — off the 60 Hz hot
        path — to keep ``add_item`` to pure memory writes + ``put_nowait``.

        Sets ``self._worker_error`` on any ffmpeg I/O failure and exits.
        Rerun failures are swallowed (non-fatal: observability only). The
        main thread is responsible for teardown — worker never mutates
        dirs or spawns/joins other threads.
        """
        assert self._ffmpeg_proc is not None
        stdin = self._ffmpeg_proc.stdin
        assert stdin is not None
        try:
            while True:
                try:
                    item = self._frame_queue.get(timeout=0.1)
                except queue.Empty:
                    if self._stop_worker and self._frame_queue.empty():
                        return
                    continue

                frame, states, actions, item_id = item

                # Primary responsibility: pipe frame to ffmpeg. Failure here
                # corrupts the episode.
                try:
                    stdin.write(frame.tobytes())
                except (BrokenPipeError, OSError) as e:
                    self._worker_error = e
                    return

                # Off-hot-path observability: Rerun logging. Never allowed
                # to fail the episode — catch and log at DEBUG only.
                if self.rerun_log and self._rerun_logger is not None:
                    try:
                        self._rerun_logger.log_item_data(
                            {
                                "idx": item_id,
                                "colors": {DEFAULT_CAMERA_INCOMING_KEY: frame},
                                "states": states,
                                "actions": actions,
                            }
                        )
                    except Exception as e:
                        logger_mp.debug(
                            f"rerun log_item_data failed (non-fatal): {e}"
                        )
        except Exception as e:
            self._worker_error = e

    def _fill_26d_row(
        self, buf: np.ndarray, idx: int, data: dict | None, kind: str
    ) -> None:
        """Write 26D qpos fields directly into ``buf[idx, :]``.

        Validates per-part presence and dim; on failure raises KeyError /
        ValueError / TypeError which the caller turns into an episode abort.
        Intentionally zero-alloc on the happy path: numpy's ``buf[idx, a:b] =
        seq`` assignment copies element-wise into the existing buffer row,
        without materializing an intermediate concatenated array.
        """
        if data is None:
            raise ValueError(f"{kind} is None")
        offset = 0
        for part_key, expected_dim in FLATTEN_PARTS:
            part = data.get(part_key)
            if part is None:
                raise KeyError(f"{kind}.{part_key} missing")
            qpos = part.get("qpos") if isinstance(part, dict) else None
            if qpos is None:
                raise KeyError(f"{kind}.{part_key}.qpos missing")
            # Length check — lists, numpy arrays, and other sized iterables.
            try:
                n = len(qpos)
            except TypeError as e:
                raise TypeError(
                    f"{kind}.{part_key}.qpos is not sized: {e}"
                ) from e
            if n != expected_dim:
                raise ValueError(
                    f"{kind}.{part_key}.qpos has length {n}, "
                    f"expected {expected_dim}"
                )
            # Direct slice assignment: numpy will cast elements to float32
            # during the in-place copy. No Python-level array allocation.
            buf[idx, offset : offset + expected_dim] = qpos
            offset += expected_dim
        if offset != 26:
            raise ValueError(f"filled {kind} offset {offset} != 26")

    def _write_parquet(self, path: Path) -> None:
        """Write per-episode parquet. Consolidator repacks into v3 chunked layout."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        n = self._n_frames
        # Store observation.state / action as fixed-size list<float32>. The
        # consolidator re-emits in LeRobot v3 canonical column types.
        state_col = [self._state_buf[i].tolist() for i in range(n)]
        action_col = [self._action_buf[i].tolist() for i in range(n)]

        table = pa.table(
            {
                "observation.state": pa.array(
                    state_col, type=pa.list_(pa.float32(), STATE_DIM)
                ),
                "action": pa.array(
                    action_col, type=pa.list_(pa.float32(), ACTION_DIM)
                ),
                "timestamp": pa.array(self._ts_buf[:n].tolist(), type=pa.float32()),
                "frame_index": pa.array(list(range(n)), type=pa.int64()),
                "episode_index": pa.array([self.episode_id] * n, type=pa.int64()),
                # `index` (global) and `task_index` are filled by the consolidator.
            }
        )
        pq.write_table(table, path)

    def _write_meta(self, path: Path) -> None:
        meta = {
            "episode_id": self.episode_id,
            "n_frames": self._n_frames,
            "fps": self.fps,
            "image_size": list(self.image_size),
            "robot_type": self.robot_type,
            "task": {
                "goal": self.task_goal,
                "desc": self.task_desc,
                "steps": self.task_steps,
            },
            # datetime.utcnow() is deprecated in Python 3.12. Use timezone-aware
            # now(UTC) and strip the +00:00 suffix so the wire format stays
            # "...Z" (ISO-8601 Zulu), matching the prior emission.
            "created_utc": datetime.datetime.now(datetime.timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
            "writer_version": "1.0.0",
        }
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    def _abort_episode(self, reason: str) -> None:
        """Mark episode corrupted, tear down ffmpeg/worker, purge staging dir.

        Safe to call from add_item / save_episode (main thread only).
        Worker threads must set ``_worker_error`` and return; the next main-
        thread call surfaces the error and triggers this routine.
        """
        if self.episode_corrupted:
            return
        self.episode_corrupted = True
        logger_mp.error(
            f"!!! EPISODE ABORTED (episode_id={self.episode_id}, "
            f"frames={self._n_frames}): {reason}"
        )
        # Audible alert via BEL; xr_teleoperate runs with a VR headset where
        # terminal output is often not visible.
        try:
            print("\a", end="", flush=True)
        except Exception:
            pass
        self._teardown_episode(purge=True)
        # Roll back episode_id so operator re-records as the same ID.
        self.episode_id -= 1

    def _teardown_episode(self, purge: bool) -> None:
        """Release per-episode resources. Idempotent."""
        # Stop the worker.
        self._stop_worker = True
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        self._worker_thread = None
        self._frame_queue = None

        # Terminate ffmpeg if still running.
        if self._ffmpeg_proc is not None:
            if self._ffmpeg_proc.poll() is None:
                self._ffmpeg_proc.terminate()
                try:
                    self._ffmpeg_proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self._ffmpeg_proc.kill()
            self._ffmpeg_proc = None

        # Purge the .tmp staging dir on abort; keep it on clean save (the
        # atomic rename has already moved it to its final name).
        if purge and self._ep_tmp_dir is not None and self._ep_tmp_dir.exists():
            shutil.rmtree(self._ep_tmp_dir, ignore_errors=True)

        self._ep_tmp_dir = None
        self._ep_final_dir = None
        self.is_available = True
