"""Unit tests for LeRobotEpisodeWriter.

Covers the invariants from spec §7.1:
    - Clean-save emits valid parquet + MP4 with matching frame counts.
    - Atomic parent-dir rename leaves no `.tmp/` residue.
    - Queue overflow aborts the episode and purges staging (IL temporal-sync
      requirement — no silent frame drops).
    - Max-duration breach aborts.
    - Aborts don't touch prior completed episodes.
    - Hot-path buffers are preallocated once and not reallocated per item.
    - Malformed inputs (missing cam key, wrong shapes) abort cleanly.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from utils.lerobot_episode_writer import (  # type: ignore[import-not-found]
    ACTION_DIM,
    CAM_FEATURE_KEY,
    DEFAULT_CAMERA_INCOMING_KEY,
    LeRobotEpisodeWriter,
    STATE_DIM,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _count_mp4_frames(path: Path) -> int:
    """Count frames in an MP4 via ffprobe."""
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-count_frames",
            "-show_entries", "stream=nb_read_frames",
            "-of", "default=nk=1:nw=1",
            str(path),
        ],
        text=True,
    ).strip()
    return int(out)


def _make_writer(
    tmp_path: Path,
    *,
    max_episode_sec: int = 60,
    rerun_log: bool = False,
) -> LeRobotEpisodeWriter:
    return LeRobotEpisodeWriter(
        task_dir=str(tmp_path / "task"),
        task_goal="pick up test object",
        frequency=30,
        image_size=(640, 480),
        rerun_log=rerun_log,
        max_episode_sec=max_episode_sec,
    )


class _BlockingWriter(LeRobotEpisodeWriter):
    """Subclass whose worker never drains the queue — forces Full on overflow."""

    def _worker_loop(self) -> None:  # type: ignore[override]
        while not self._stop_worker:
            time.sleep(0.01)


# ---------------------------------------------------------------------------
# happy-path tests
# ---------------------------------------------------------------------------


def test_create_then_save_produces_valid_parquet_and_mp4(
    tmp_path: Path, dummy_inputs
):
    colors, states, actions = dummy_inputs
    w = _make_writer(tmp_path)

    assert w.is_ready()
    assert w.create_episode() is True
    assert not w.is_ready()  # writer locked while episode in flight

    N = 10
    for _ in range(N):
        w.add_item(colors=colors, states=states, actions=actions)

    assert w.save_episode() is True
    assert w.is_ready()
    assert not w.episode_corrupted

    ep_dir = Path(w.staging_dir) / "episode_000000"
    assert ep_dir.is_dir(), "final episode dir missing"
    assert (ep_dir / "data.parquet").is_file()
    assert (ep_dir / "cam_head.mp4").is_file()
    assert (ep_dir / "meta.json").is_file()

    # Parquet row count matches N.
    table = pq.read_table(ep_dir / "data.parquet")
    assert table.num_rows == N
    # Required columns present.
    cols = set(table.column_names)
    assert {"observation.state", "action", "timestamp", "frame_index", "episode_index"} <= cols
    # 26D fixed-size list entries.
    first_state = table.column("observation.state")[0].as_py()
    first_action = table.column("action")[0].as_py()
    assert len(first_state) == STATE_DIM == 26
    assert len(first_action) == ACTION_DIM == 26
    # State values match what dummy_state flattens to:
    #   left_arm (0.1)*7 + right_arm (0.2)*7 + left_ee (0.3)*6 + right_ee (0.4)*6
    np.testing.assert_allclose(first_state[:7],    [0.1] * 7, atol=1e-6)
    np.testing.assert_allclose(first_state[7:14],  [0.2] * 7, atol=1e-6)
    np.testing.assert_allclose(first_state[14:20], [0.3] * 6, atol=1e-6)
    np.testing.assert_allclose(first_state[20:26], [0.4] * 6, atol=1e-6)
    # Monotonic timestamps.
    ts = table.column("timestamp").to_pylist()
    assert all(t1 >= t0 for t0, t1 in zip(ts, ts[1:]))
    # frame_index is 0..N-1.
    assert table.column("frame_index").to_pylist() == list(range(N))

    # MP4 has exactly N frames.
    assert _count_mp4_frames(ep_dir / "cam_head.mp4") == N

    w.close()


def test_save_is_atomic_no_tmp_left(tmp_path: Path, dummy_inputs):
    """Clean save must leave no `.tmp/` directory behind."""
    colors, states, actions = dummy_inputs
    w = _make_writer(tmp_path)
    w.create_episode()
    for _ in range(5):
        w.add_item(colors=colors, states=states, actions=actions)
    assert w.save_episode() is True

    staging = Path(w.staging_dir)
    tmp_dirs = [p for p in staging.iterdir() if p.name.endswith(".tmp")]
    assert tmp_dirs == [], f"stale .tmp dirs after clean save: {tmp_dirs}"
    assert (staging / "episode_000000").is_dir()
    w.close()


def test_episode_ids_increment_across_saves(tmp_path: Path, dummy_inputs):
    colors, states, actions = dummy_inputs
    w = _make_writer(tmp_path)
    for expected_id in range(3):
        w.create_episode()
        for _ in range(3):
            w.add_item(colors=colors, states=states, actions=actions)
        assert w.save_episode() is True
        assert (Path(w.staging_dir) / f"episode_{expected_id:06d}").is_dir()
    w.close()


# ---------------------------------------------------------------------------
# abort-path tests (IL temporal-sync requirement)
# ---------------------------------------------------------------------------


def test_queue_overflow_triggers_episode_abort(tmp_path: Path, dummy_inputs):
    """Critical: overflowing the encoder queue must abort, not silently drop."""
    colors, states, actions = dummy_inputs

    w = _BlockingWriter(
        task_dir=str(tmp_path / "task"),
        frequency=30,
        image_size=(640, 480),
        rerun_log=False,
    )
    w.create_episode()

    ep_tmp_dir = Path(w.staging_dir) / "episode_000000.tmp"
    assert ep_tmp_dir.is_dir()

    # Feed queue_size + 1 frames. Worker never drains → put_nowait raises Full.
    for _ in range(w.queue_size + 5):
        w.add_item(colors=colors, states=states, actions=actions)
        if w.episode_corrupted:
            break

    assert w.episode_corrupted, "queue overflow did not trigger abort"
    assert w.is_ready(), "writer must be available for next take after abort"
    # Staging dir was purged.
    assert not ep_tmp_dir.exists(), "tmp dir must be purged on abort"
    # No final dir was written.
    assert not (Path(w.staging_dir) / "episode_000000").exists()
    # episode_id rolled back (-1 since nothing has been saved).
    assert w.episode_id == -1, f"expected rollback to -1, got {w.episode_id}"

    w.close()


def test_max_duration_triggers_episode_abort(tmp_path: Path, dummy_inputs):
    """Exceeding max_episode_sec must abort (no dynamic buffer resize)."""
    colors, states, actions = dummy_inputs
    # max_episode_sec=1 → max_frames = 30 at 30 fps.
    w = _make_writer(tmp_path, max_episode_sec=1)
    w.create_episode()

    max_frames = w.max_frames
    assert max_frames == 30
    for _ in range(max_frames + 5):
        w.add_item(colors=colors, states=states, actions=actions)
        if w.episode_corrupted:
            break

    assert w.episode_corrupted
    assert w.is_ready()
    assert not (Path(w.staging_dir) / "episode_000000.tmp").exists()
    assert not (Path(w.staging_dir) / "episode_000000").exists()
    w.close()


def test_abort_does_not_touch_prior_completed_episodes(
    tmp_path: Path, dummy_inputs
):
    """Aborting episode N must leave episodes 0..N-1 intact on disk."""
    colors, states, actions = dummy_inputs
    w = _make_writer(tmp_path)

    # Save episode 0 cleanly.
    w.create_episode()
    for _ in range(5):
        w.add_item(colors=colors, states=states, actions=actions)
    assert w.save_episode() is True
    ep0 = Path(w.staging_dir) / "episode_000000"
    ep0_files_before = {p.name: p.stat().st_size for p in ep0.iterdir()}

    # Start episode 1 and force an abort (inject a bad frame shape).
    w.create_episode()
    w.add_item(colors=colors, states=states, actions=actions)  # 1 good frame
    bad_colors = {"color_0": np.zeros((100, 100, 3), dtype=np.uint8)}  # wrong shape
    w.add_item(colors=bad_colors, states=states, actions=actions)

    assert w.episode_corrupted
    # Episode 1's .tmp purged, final dir never created.
    assert not (Path(w.staging_dir) / "episode_000001.tmp").exists()
    assert not (Path(w.staging_dir) / "episode_000001").exists()
    # Episode 0's artifacts unchanged.
    ep0_files_after = {p.name: p.stat().st_size for p in ep0.iterdir()}
    assert ep0_files_before == ep0_files_after, (
        f"episode 0 was modified during episode 1 abort: "
        f"before={ep0_files_before} after={ep0_files_after}"
    )

    w.close()


def test_abort_after_rollback_recreate_reuses_same_id(
    tmp_path: Path, dummy_inputs
):
    """After abort, the operator's next create_episode() reuses the aborted ID."""
    colors, states, actions = dummy_inputs
    w = _make_writer(tmp_path)

    # Save ep 0.
    w.create_episode()
    for _ in range(3):
        w.add_item(colors=colors, states=states, actions=actions)
    w.save_episode()

    # Start ep 1, abort it via missing cam key.
    w.create_episode()
    assert w.episode_id == 1
    w.add_item(colors={}, states=states, actions=actions)  # no color_0 → abort
    assert w.episode_corrupted
    assert w.episode_id == 0, "rolled back to 0 (ep 0 is the latest completed)"

    # Re-record as the same ID.
    w.create_episode()
    assert w.episode_id == 1
    for _ in range(3):
        w.add_item(colors=colors, states=states, actions=actions)
    assert w.save_episode() is True
    assert (Path(w.staging_dir) / "episode_000001").is_dir()

    w.close()


# ---------------------------------------------------------------------------
# input-validation abort tests
# ---------------------------------------------------------------------------


def test_missing_cam_key_aborts(tmp_path: Path, dummy_state, dummy_action):
    w = _make_writer(tmp_path)
    w.create_episode()
    w.add_item(colors={}, states=dummy_state, actions=dummy_action)
    assert w.episode_corrupted
    assert w.is_ready()
    w.close()


def test_wrong_frame_shape_aborts(tmp_path: Path, dummy_state, dummy_action):
    w = _make_writer(tmp_path)
    w.create_episode()
    bad = np.zeros((100, 100, 3), dtype=np.uint8)  # not 640x480
    w.add_item(colors={"color_0": bad}, states=dummy_state, actions=dummy_action)
    assert w.episode_corrupted
    w.close()


def test_wrong_state_dim_aborts(tmp_path: Path, dummy_frame, dummy_action):
    bad_state = {
        "left_arm":  {"qpos": [0.0] * 6},  # 6 instead of 7
        "right_arm": {"qpos": [0.0] * 7},
        "left_ee":   {"qpos": [0.0] * 6},
        "right_ee":  {"qpos": [0.0] * 6},
    }
    w = _make_writer(tmp_path)
    w.create_episode()
    w.add_item(colors={"color_0": dummy_frame}, states=bad_state, actions=dummy_action)
    assert w.episode_corrupted
    w.close()


def test_zero_frame_save_aborts(tmp_path: Path):
    """Calling save_episode with no frames must abort, not write empty parquet."""
    w = _make_writer(tmp_path)
    w.create_episode()
    assert w.save_episode() is False
    assert w.episode_corrupted
    assert w.is_ready()
    w.close()


# ---------------------------------------------------------------------------
# hot-path allocation invariant
# ---------------------------------------------------------------------------


def test_preallocated_buffers_not_reallocated_across_items(
    tmp_path: Path, dummy_inputs
):
    """State/action buffers must be allocated once in __init__ and reused."""
    colors, states, actions = dummy_inputs
    w = _make_writer(tmp_path)

    state_id_before = id(w._state_buf)
    action_id_before = id(w._action_buf)
    ts_id_before = id(w._ts_buf)
    state_base_before = w._state_buf.ctypes.data
    action_base_before = w._action_buf.ctypes.data

    w.create_episode()
    for _ in range(20):
        w.add_item(colors=colors, states=states, actions=actions)
    w.save_episode()

    assert id(w._state_buf) == state_id_before, "state buffer reallocated"
    assert id(w._action_buf) == action_id_before, "action buffer reallocated"
    assert id(w._ts_buf) == ts_id_before, "timestamp buffer reallocated"
    assert w._state_buf.ctypes.data == state_base_before
    assert w._action_buf.ctypes.data == action_base_before

    # Same buffers survive a second episode.
    w.create_episode()
    for _ in range(5):
        w.add_item(colors=colors, states=states, actions=actions)
    w.save_episode()
    assert w._state_buf.ctypes.data == state_base_before
    assert w._action_buf.ctypes.data == action_base_before

    w.close()


# ---------------------------------------------------------------------------
# state-ordering invariant
# ---------------------------------------------------------------------------


def test_state_flatten_order_is_arm_then_hand_left_then_right(
    tmp_path: Path, dummy_frame
):
    """Verify the 26D field order matches ROBOT_CONFIGS motors list exactly."""
    # Use distinctive values so we can assert per-part ordering.
    state = {
        "left_arm":  {"qpos": [1, 2, 3, 4, 5, 6, 7]},
        "right_arm": {"qpos": [8, 9, 10, 11, 12, 13, 14]},
        "left_ee":   {"qpos": [15, 16, 17, 18, 19, 20]},
        "right_ee":  {"qpos": [21, 22, 23, 24, 25, 26]},
    }
    w = _make_writer(tmp_path)
    w.create_episode()
    w.add_item(colors={"color_0": dummy_frame}, states=state, actions=state)
    w.save_episode()

    table = pq.read_table(Path(w.staging_dir) / "episode_000000" / "data.parquet")
    flat = table.column("observation.state")[0].as_py()
    assert flat == [float(x) for x in range(1, 27)], (
        f"26D flatten order incorrect: {flat}"
    )
    w.close()


# ---------------------------------------------------------------------------
# hot-path / worker-thread isolation
# ---------------------------------------------------------------------------


def test_rerun_logging_runs_on_worker_thread_not_hot_path(
    tmp_path: Path, dummy_inputs
):
    """Rerun IPC must not block ``add_item``. If it were inline, 5 frames
    would take 5*SLOW_MS >> the 60 Hz budget. Since it's on the worker
    thread, ``add_item`` should return near-instantly."""
    colors, states, actions = dummy_inputs

    SLOW_MS = 50  # would blow a 16.67 ms budget by 3x if inline

    class _SlowRerun:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def log_item_data(self, data: dict) -> None:
            self.calls.append(int(data["idx"]))
            time.sleep(SLOW_MS / 1000.0)

    # Construct with rerun_log=False so __init__ doesn't try to import
    # the real RerunLogger; then inject a fake and re-enable the flag.
    w = _make_writer(tmp_path, rerun_log=False)
    fake = _SlowRerun()
    w._rerun_logger = fake
    w.rerun_log = True

    w.create_episode()

    N = 5
    t0 = time.monotonic()
    for _ in range(N):
        w.add_item(colors=colors, states=states, actions=actions)
    hot_path_sec = time.monotonic() - t0

    # If Rerun were inline, this would be >= N * SLOW_MS = 250 ms.
    # Off-hot-path, it should be well under one SLOW_MS interval.
    budget_sec = (SLOW_MS / 1000.0) * 0.5
    assert hot_path_sec < budget_sec, (
        f"add_item appears to block on Rerun: {hot_path_sec*1000:.1f} ms "
        f"for {N} calls (budget {budget_sec*1000:.1f} ms)"
    )

    # save_episode waits for the worker to drain, so by the time it returns
    # all N rerun calls should have completed.
    assert w.save_episode() is True
    assert fake.calls == list(range(N)), (
        f"worker did not process all rerun items: {fake.calls}"
    )
    w.close()


# ---------------------------------------------------------------------------
# constants sanity
# ---------------------------------------------------------------------------


def test_module_constants_match_spec():
    assert STATE_DIM == 26
    assert ACTION_DIM == 26
    assert CAM_FEATURE_KEY == "observation.images.cam_head"
    assert DEFAULT_CAMERA_INCOMING_KEY == "color_0"
