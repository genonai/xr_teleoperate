"""Round-trip tests for the 30D/29D + fsm_mode writer schema (spec 2026-04-24).

These tests construct a writer, feed synthetic frames, finalize the episode,
then read back the Parquet file and confirm column types, shapes, and values.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from utils.lerobot_episode_writer import (  # type: ignore[import-not-found]
    ACTION_DIM,
    STATE_DIM,
    LeRobotEpisodeWriter,
)


@pytest.fixture
def tmp_task_dir(tmp_path: Path) -> Path:
    d = tmp_path / "task"
    d.mkdir()
    return d


def _make_states() -> dict:
    return {
        "left_arm":       {"qpos": [0.1] * 7, "qvel": [], "torque": []},
        "right_arm":      {"qpos": [0.2] * 7, "qvel": [], "torque": []},
        "left_ee":        {"qpos": [0.3] * 6, "qvel": [], "torque": []},
        "right_ee":       {"qpos": [0.4] * 6, "qvel": [], "torque": []},
        "base_achieved":  {"qpos": [0.5, 0.6, 0.7, 0.8], "qvel": [], "torque": []},
    }


def _make_actions() -> dict:
    return {
        "left_arm":  {"qpos": [1.1] * 7, "qvel": [], "torque": []},
        "right_arm": {"qpos": [1.2] * 7, "qvel": [], "torque": []},
        "left_ee":   {"qpos": [1.3] * 6, "qvel": [], "torque": []},
        "right_ee":  {"qpos": [1.4] * 6, "qvel": [], "torque": []},
        "base_cmd":  {"qpos": [0.21, -0.05, 0.0], "qvel": [], "torque": []},
    }


def _make_frame(h: int = 480, w: int = 640) -> np.ndarray:
    # bgr24 dummy frame.
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_dims_are_30_and_29():
    assert STATE_DIM == 30
    assert ACTION_DIM == 29


def test_single_episode_roundtrip_with_fsm(tmp_task_dir: Path):
    writer = LeRobotEpisodeWriter(
        task_dir=str(tmp_task_dir),
        frequency=30,
        rerun_log=False,
        max_episode_sec=1,
    )
    assert writer.create_episode()
    for i in range(5):
        writer.add_item(
            colors={"color_0": _make_frame()},
            states=_make_states(),
            actions=_make_actions(),
            fsm_mode=1 if i < 3 else 2,  # STAND for 3 frames, SQUAT for 2
        )
    assert writer.save_episode()

    # Locate the finalized episode dir.
    ep_dirs = sorted(p for p in (tmp_task_dir / "_staging").iterdir()
                     if p.is_dir() and not p.name.endswith(".tmp") and p.name.startswith("episode_"))
    assert len(ep_dirs) == 1
    table = pq.read_table(ep_dirs[0] / "data.parquet")

    cols = set(table.column_names)
    assert "observation.state" in cols
    assert "action" in cols
    assert "observation.fsm_mode" in cols

    # State/action fixed-size list dims.
    assert table.schema.field("observation.state").type.list_size == 30
    assert table.schema.field("action").type.list_size == 29

    # fsm_mode: int8, per-frame scalar.
    fsm_col = table["observation.fsm_mode"].to_pylist()
    assert fsm_col == [1, 1, 1, 2, 2]
    assert str(table.schema.field("observation.fsm_mode").type) == "int8"


def test_fsm_mode_no_stale_tail_across_episodes(tmp_task_dir: Path):
    """Buffer reuse invariant: parquet slice only contains current episode's data."""
    writer = LeRobotEpisodeWriter(
        task_dir=str(tmp_task_dir),
        frequency=30,
        rerun_log=False,
        max_episode_sec=1,
    )

    # Episode 1: 5 frames with fsm_mode = [1,1,1,2,2]
    assert writer.create_episode()
    for i in range(5):
        writer.add_item(
            colors={"color_0": _make_frame()},
            states=_make_states(),
            actions=_make_actions(),
            fsm_mode=1 if i < 3 else 2,
        )
    assert writer.save_episode()

    # Episode 2: 3 frames with fsm_mode = [0,0,0]
    assert writer.create_episode()
    for _ in range(3):
        writer.add_item(
            colors={"color_0": _make_frame()},
            states=_make_states(),
            actions=_make_actions(),
            fsm_mode=0,
        )
    assert writer.save_episode()

    ep_dirs = sorted(p for p in (tmp_task_dir / "_staging").iterdir()
                     if p.is_dir() and not p.name.endswith(".tmp") and p.name.startswith("episode_"))
    assert len(ep_dirs) == 2

    # Episode 1 must be exactly 5 rows of [1,1,1,2,2] — stale tail from
    # episode-2 buffer reuse would manifest here as wrong values.
    t1 = pq.read_table(ep_dirs[0] / "data.parquet")
    assert t1["observation.fsm_mode"].to_pylist() == [1, 1, 1, 2, 2]

    # Episode 2 must be exactly 3 rows of [0,0,0] — overwrite-as-you-go semantics.
    t2 = pq.read_table(ep_dirs[1] / "data.parquet")
    assert t2["observation.fsm_mode"].to_pylist() == [0, 0, 0]


def test_base_achieved_dim_mismatch_aborts(tmp_task_dir: Path):
    writer = LeRobotEpisodeWriter(
        task_dir=str(tmp_task_dir), frequency=30, rerun_log=False, max_episode_sec=1,
    )
    assert writer.create_episode()
    bad_states = _make_states()
    bad_states["base_achieved"]["qpos"] = [0.1, 0.2, 0.3]  # 3D, spec says 4D
    writer.add_item(
        colors={"color_0": _make_frame()},
        states=bad_states,
        actions=_make_actions(),
        fsm_mode=1,
    )
    assert writer.episode_corrupted


def test_default_robot_type_is_basevel_v1(tmp_task_dir: Path):
    writer = LeRobotEpisodeWriter(task_dir=str(tmp_task_dir), frequency=30, rerun_log=False)
    assert writer.robot_type == "Unitree_G1_Inspire_HeadOnly_Mono_BaseVel_v1"
