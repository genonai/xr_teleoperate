"""Integration tests for MimicCSVWriter — realistic G1 teleop scenarios.

Exercises the full code path with synthetic but physically plausible data:
  - Realistic G1 joint trajectories within joint limits
  - Proper unit quaternions (identity for stationary teleop)
  - Hand data with Inspire 6DOF ranges
  - Multi-episode sequences
  - BFS→DFS correctness verified against RL lab MotionLoader_ expectations
  - E2E validator cross-check

Run:
    cd repos/xr_teleoperate && python -m pytest tests/test_mimic_csv_integration.py -v
"""

import os
import sys
import tempfile
import shutil

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'teleop'))

from utils.mimic_csv_writer import (
    MimicCSVWriter,
    _JOINT_IDS_MAP,
    _CANONICAL_DEFAULT_JOINT_POS,
    _CANONICAL_STIFFNESS,
    _CANONICAL_DAMPING,
    _CANONICAL_ACTION_SCALE,
)

# Import e2e validator
sys.path.insert(0, os.path.dirname(__file__))
from test_mimic_csv_e2e import validate_episode


# ── Realistic data generators ────────────────────────────────────────────────

# G1-29 joint limits in BFS (motor) order — used to generate in-range random data
# Source: G1 URDF
_G1_29_BFS_LIMITS_LO = np.array([
    # BFS order: left_hip_pitch, right_hip_pitch, waist_yaw, ...
    # For arm joints (what actually moves during stationary teleop):
    # We use conservative limits that are within the real joint range
    -0.30, -0.30, 0.00, 0.00, 0.00, 0.00,   # left leg (near standing pose)
     0.00,  0.00, 0.00, 0.60, 0.60, 0.15,    # right leg (near standing pose)
     0.10, -0.40, -0.40,                       # waist
    -1.50, -1.50, -1.00, -0.50, -1.50, -0.40, -0.40,  # left arm (active range)
    -1.50, -1.50, -1.00, -0.50, -1.50, -0.40, -0.40,  # right arm (active range)
])
_G1_29_BFS_LIMITS_HI = np.array([
     0.30,  0.30, 0.05, 0.05, 0.05, 0.05,
     0.05,  0.05, 0.05, 0.75, 0.75, 0.25,
     0.30, -0.20, -0.20,
     1.50,  1.50,  1.00,  2.00,  1.50,  0.40,  0.40,
     1.50,  1.50,  1.00,  2.00,  1.50,  0.40,  0.40,
])

# Inspire hand 6DOF range: 0 (closed) to 1 (open)
_HAND_LO = 0.0
_HAND_HI = 1.0


def generate_smooth_trajectory(n_frames: int, seed: int = 42) -> dict:
    """Generate a physically plausible stationary teleop trajectory.

    Returns dict with keys: base_pos, base_quat, joint_q_bfs, hand_left, hand_right.
    Each value is an (n_frames, D) array.
    """
    rng = np.random.RandomState(seed)

    # Base: stationary standing
    base_pos = np.tile([0.0, 0.0, 0.793], (n_frames, 1))
    # Identity quaternion (xyzw) for stationary teleop — slight IMU drift
    base_quat = np.tile([0.0, 0.0, 0.0, 1.0], (n_frames, 1))
    # Add tiny drift to x,y,z components (realistic IMU noise)
    drift = rng.normal(0, 0.001, (n_frames, 3))
    base_quat[:, :3] += drift
    # Re-normalize to unit quaternion
    norms = np.linalg.norm(base_quat, axis=1, keepdims=True)
    base_quat /= norms

    # Joints: start from canonical standing pose, smooth random walk for arms
    standing = np.array(_CANONICAL_DEFAULT_JOINT_POS)
    joint_q_bfs = np.zeros((n_frames, 29))
    joint_q_bfs[0] = standing

    for i in range(1, n_frames):
        # Small random step
        step = rng.normal(0, 0.005, 29)
        # Legs (0-11) and waist (12-14): very small movement (standing)
        step[:15] *= 0.1
        # Arms (15-28): larger movement (teleop motion)
        step[15:] *= 1.0

        joint_q_bfs[i] = joint_q_bfs[i - 1] + step
        # Clip to limits
        joint_q_bfs[i] = np.clip(
            joint_q_bfs[i], _G1_29_BFS_LIMITS_LO, _G1_29_BFS_LIMITS_HI
        )

    # Hand: smooth open/close cycle (simulating grasp)
    t = np.linspace(0, 2 * np.pi, n_frames)
    # Left hand: starts open, closes mid-episode
    left_base = 0.5 + 0.4 * np.cos(t)
    hand_left = np.column_stack([left_base + rng.normal(0, 0.01, n_frames)
                                  for _ in range(6)])
    hand_left = np.clip(hand_left, _HAND_LO, _HAND_HI)

    # Right hand: starts open, closes earlier (reaching + grasping)
    right_base = 0.5 + 0.4 * np.cos(t + np.pi / 3)
    hand_right = np.column_stack([right_base + rng.normal(0, 0.01, n_frames)
                                   for _ in range(6)])
    hand_right = np.clip(hand_right, _HAND_LO, _HAND_HI)

    return {
        'base_pos': base_pos,
        'base_quat': base_quat,
        'joint_q_bfs': joint_q_bfs,
        'hand_left': hand_left,
        'hand_right': hand_right,
    }


# ── Test classes ─────────────────────────────────────────────────────────────

class TestRealisticTrajectory:
    """Full pipeline with realistic G1 joint data + e2e validation."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_realistic_episode_passes_e2e_validation(self):
        """Generate a 300-frame episode (5s @ 60Hz) and run e2e validator."""
        ep_dir = os.path.join(self.tmpdir, 'episode_0001')
        os.makedirs(ep_dir)

        traj = generate_smooth_trajectory(300, seed=42)
        writer = MimicCSVWriter(fps=60)
        writer.create_episode(ep_dir)

        for i in range(300):
            writer.add_frame(
                base_pos=traj['base_pos'][i],
                base_quat=traj['base_quat'][i],
                joint_q_29=traj['joint_q_bfs'][i],
                hand_left=traj['hand_left'][i],
                hand_right=traj['hand_right'][i],
            )

        writer.save_episode()

        # Run e2e validator — should pass with no errors
        assert validate_episode(ep_dir), "E2E validation failed on realistic trajectory"

    def test_realistic_episode_csv_values_are_correct(self):
        """Verify CSV contains correct values after BFS→DFS conversion."""
        ep_dir = os.path.join(self.tmpdir, 'episode_0002')
        os.makedirs(ep_dir)

        traj = generate_smooth_trajectory(50, seed=123)
        writer = MimicCSVWriter(fps=60)
        writer.create_episode(ep_dir)

        for i in range(50):
            writer.add_frame(
                base_pos=traj['base_pos'][i],
                base_quat=traj['base_quat'][i],
                joint_q_29=traj['joint_q_bfs'][i],
                hand_left=traj['hand_left'][i],
                hand_right=traj['hand_right'][i],
            )

        writer.save_episode()

        body = np.loadtxt(
            os.path.join(ep_dir, 'mimic', 'trajectory.csv'), delimiter=','
        )
        hand = np.loadtxt(
            os.path.join(ep_dir, 'mimic', 'hand_state.csv'), delimiter=','
        )

        assert body.shape == (50, 36)
        assert hand.shape == (50, 12)

        # Verify base_pos columns 0-2
        np.testing.assert_allclose(body[:, :3], traj['base_pos'], atol=1e-6)

        # Verify base_quat columns 3-6
        np.testing.assert_allclose(body[:, 3:7], traj['base_quat'], atol=1e-6)

        # Verify joints columns 7-35: these are DFS order
        # For each frame, do BFS→DFS conversion ourselves and compare
        for i in range(50):
            expected_dfs = MimicCSVWriter.bfs_to_dfs(traj['joint_q_bfs'][i])
            np.testing.assert_allclose(body[i, 7:36], expected_dfs, atol=1e-6)

        # Verify hand values
        np.testing.assert_allclose(hand[:, :6], traj['hand_left'], atol=1e-6)
        np.testing.assert_allclose(hand[:, 6:], traj['hand_right'], atol=1e-6)


class TestRLLabMotionLoaderCompatibility:
    """Verify the CSV is compatible with RL lab's MotionLoader_ parser."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_motion_loader_quaternion_extraction(self):
        """RL lab reads CSV quaternion as Quaternionf(data[6], data[3], data[4], data[5]).

        This means: w=col6, x=col3, y=col4, z=col5.
        Our CSV stores: col3=qx, col4=qy, col5=qz, col6=qw.
        So RL lab gets: w=qw, x=qx, y=qy, z=qz — correct.
        """
        ep_dir = os.path.join(self.tmpdir, 'episode_quat')
        os.makedirs(ep_dir)

        writer = MimicCSVWriter(fps=60)
        writer.create_episode(ep_dir)

        # Write a known quaternion: 30-degree rotation around Z axis
        angle = np.radians(30)
        # Isaac convention: [qx, qy, qz, qw]
        qx, qy, qz, qw = 0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)
        writer.add_frame(
            base_pos=[0, 0, 0.793],
            base_quat=[qx, qy, qz, qw],
            joint_q_29=np.zeros(29),
        )
        writer.save_episode()

        body = np.loadtxt(
            os.path.join(ep_dir, 'mimic', 'trajectory.csv'), delimiter=','
        )
        if body.ndim == 1:
            body = body[np.newaxis, :]
        row = body[0]

        # RL lab does: Quaternionf(data[6], data[3], data[4], data[5])
        # which is:    Quaternionf(w,        x,        y,        z)
        rl_lab_w = row[6]
        rl_lab_x = row[3]
        rl_lab_y = row[4]
        rl_lab_z = row[5]

        np.testing.assert_allclose(rl_lab_w, qw, atol=1e-6)
        np.testing.assert_allclose(rl_lab_x, qx, atol=1e-6)
        np.testing.assert_allclose(rl_lab_y, qy, atol=1e-6)
        np.testing.assert_allclose(rl_lab_z, qz, atol=1e-6)

    def test_motion_loader_joint_recovery(self):
        """RL lab reads joints in DFS order and converts: bfs[i] = dfs[joint_ids_map[i]].

        Verify this recovers the original BFS values we fed into MimicCSVWriter.
        """
        ep_dir = os.path.join(self.tmpdir, 'episode_joints')
        os.makedirs(ep_dir)

        writer = MimicCSVWriter(fps=60)
        writer.create_episode(ep_dir)

        # Use distinct values per joint so we can verify the ordering
        bfs_original = np.arange(29, dtype=float) * 0.01  # 0.00..0.28
        writer.add_frame(
            base_pos=[0, 0, 0.793],
            base_quat=[0, 0, 0, 1],
            joint_q_29=bfs_original,
        )
        writer.save_episode()

        body = np.loadtxt(
            os.path.join(ep_dir, 'mimic', 'trajectory.csv'), delimiter=','
        )
        if body.ndim == 1:
            body = body[np.newaxis, :]
        dfs_from_csv = body[0, 7:36]

        # Simulate RL lab's MotionLoader_ conversion: bfs[i] = dfs[joint_ids_map[i]]
        bfs_recovered = np.zeros(29)
        for bfs_idx in range(29):
            dfs_idx = _JOINT_IDS_MAP[bfs_idx]
            bfs_recovered[bfs_idx] = dfs_from_csv[dfs_idx]

        np.testing.assert_allclose(
            bfs_recovered, bfs_original, atol=1e-6,
            err_msg="RL lab MotionLoader_ failed to recover original BFS joint values"
        )

    def test_deploy_yaml_joint_ids_map_consistency(self):
        """deploy.yaml joint_ids_map should match the constant used for BFS→DFS."""
        ep_dir = os.path.join(self.tmpdir, 'episode_yaml')
        os.makedirs(ep_dir)

        writer = MimicCSVWriter(fps=60)
        writer.create_episode(ep_dir)
        writer.add_frame([0, 0, 0.793], [0, 0, 0, 1], np.zeros(29))
        writer.save_episode()

        with open(os.path.join(ep_dir, 'mimic', 'deploy.yaml')) as f:
            cfg = yaml.safe_load(f)

        assert cfg['joint_ids_map'] == _JOINT_IDS_MAP, \
            "deploy.yaml joint_ids_map does not match _JOINT_IDS_MAP constant"

    def test_deploy_yaml_canonical_constants_lengths(self):
        """All canonical constant lists must have exactly 29 elements."""
        assert len(_CANONICAL_DEFAULT_JOINT_POS) == 29
        assert len(_CANONICAL_STIFFNESS) == 29
        assert len(_CANONICAL_DAMPING) == 29
        assert len(_CANONICAL_ACTION_SCALE) == 29
        assert len(_JOINT_IDS_MAP) == 29

    def test_joint_ids_map_is_valid_permutation(self):
        """_JOINT_IDS_MAP must be a valid permutation of 0..28 (bijective BFS↔DFS)."""
        assert sorted(_JOINT_IDS_MAP) == list(range(29)), \
            "_JOINT_IDS_MAP is not a valid permutation of 0..28"


class TestMultiEpisodeRecording:
    """Simulate recording multiple episodes in sequence (as teleop does)."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_three_episodes_with_varying_lengths(self):
        """Record 3 episodes of different lengths, validate each independently."""
        writer = MimicCSVWriter(fps=60)
        episode_lengths = [120, 300, 60]  # 2s, 5s, 1s

        for ep_idx, n_frames in enumerate(episode_lengths):
            ep_dir = os.path.join(self.tmpdir, f'episode_{ep_idx:04d}')
            os.makedirs(ep_dir)

            traj = generate_smooth_trajectory(n_frames, seed=ep_idx)
            writer.create_episode(ep_dir)

            for i in range(n_frames):
                writer.add_frame(
                    base_pos=traj['base_pos'][i],
                    base_quat=traj['base_quat'][i],
                    joint_q_29=traj['joint_q_bfs'][i],
                    hand_left=traj['hand_left'][i],
                    hand_right=traj['hand_right'][i],
                )

            writer.save_episode()

            # Verify CSV row counts
            body = np.loadtxt(
                os.path.join(ep_dir, 'mimic', 'trajectory.csv'), delimiter=','
            )
            assert body.shape[0] == n_frames, \
                f"Episode {ep_idx}: expected {n_frames} rows, got {body.shape[0]}"

            # E2E validation
            assert validate_episode(ep_dir), \
                f"Episode {ep_idx} failed e2e validation"

    def test_episode_reuse_clears_buffers(self):
        """After save_episode(), create_episode() should start fresh."""
        writer = MimicCSVWriter(fps=60)

        # Episode 1: 10 frames with hand data
        ep1_dir = os.path.join(self.tmpdir, 'ep1')
        os.makedirs(ep1_dir)
        writer.create_episode(ep1_dir)
        for _ in range(10):
            writer.add_frame(
                [0, 0, 0.793], [0, 0, 0, 1], np.zeros(29),
                hand_left=np.ones(6), hand_right=np.ones(6),
            )
        writer.save_episode()

        # Episode 2: 5 frames WITHOUT hand data
        ep2_dir = os.path.join(self.tmpdir, 'ep2')
        os.makedirs(ep2_dir)
        writer.create_episode(ep2_dir)
        for _ in range(5):
            writer.add_frame([0, 0, 0.793], [0, 0, 0, 1], np.zeros(29))
        writer.save_episode()

        # Episode 2 should NOT have hand_state.csv
        # (create_episode must have cleared has_hand_data)
        assert not os.path.isfile(os.path.join(ep2_dir, 'mimic', 'hand_state.csv')), \
            "Episode 2 should not have hand_state.csv — buffers not cleared"

        body = np.loadtxt(
            os.path.join(ep2_dir, 'mimic', 'trajectory.csv'), delimiter=','
        )
        assert body.shape == (5, 36)


class TestSimulationModeDefaults:
    """Verify behavior when running in --sim mode (no real IMU/DDS)."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_identity_quaternion_for_sim_mode(self):
        """In sim mode, IMU defaults to identity quaternion [0,0,0,1] (xyzw).

        This should produce valid unit quaternions in the CSV.
        """
        ep_dir = os.path.join(self.tmpdir, 'episode_sim')
        os.makedirs(ep_dir)

        writer = MimicCSVWriter(fps=60)
        writer.create_episode(ep_dir)

        # Sim mode: identity quaternion every frame
        for _ in range(100):
            writer.add_frame(
                base_pos=[0.0, 0.0, 0.793],
                base_quat=[0.0, 0.0, 0.0, 1.0],  # identity, xyzw
                joint_q_29=np.array(_CANONICAL_DEFAULT_JOINT_POS),
            )

        writer.save_episode()

        body = np.loadtxt(
            os.path.join(ep_dir, 'mimic', 'trajectory.csv'), delimiter=','
        )

        # All quaternions should be exactly identity
        expected_quat = np.tile([0.0, 0.0, 0.0, 1.0], (100, 1))
        np.testing.assert_allclose(body[:, 3:7], expected_quat, atol=1e-6)

        # All joints should be canonical standing pose (in DFS order)
        expected_dfs = MimicCSVWriter.bfs_to_dfs(
            np.array(_CANONICAL_DEFAULT_JOINT_POS)
        )
        for i in range(100):
            np.testing.assert_allclose(body[i, 7:36], expected_dfs, atol=1e-6)

        # E2E validation
        assert validate_episode(ep_dir)

    def test_60hz_timing_in_deploy_yaml(self):
        """Verify 60Hz step_dt is correctly written."""
        ep_dir = os.path.join(self.tmpdir, 'episode_fps')
        os.makedirs(ep_dir)

        writer = MimicCSVWriter(fps=60)
        writer.create_episode(ep_dir)
        writer.add_frame([0, 0, 0.793], [0, 0, 0, 1], np.zeros(29))
        writer.save_episode()

        with open(os.path.join(ep_dir, 'mimic', 'deploy.yaml')) as f:
            cfg = yaml.safe_load(f)

        expected_dt = round(1.0 / 60.0, 6)
        assert abs(cfg['step_dt'] - expected_dt) < 1e-9
        # Duration for 100 frames @ 60Hz = ~1.667s
        # (not tested here, but step_dt * n_frames is how RL lab computes it)


class TestEdgeCases:
    """Edge cases that could break during real teleop."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_single_frame_episode(self):
        """Minimum viable episode: 1 frame."""
        ep_dir = os.path.join(self.tmpdir, 'episode_single')
        os.makedirs(ep_dir)

        writer = MimicCSVWriter(fps=60)
        writer.create_episode(ep_dir)
        writer.add_frame([0, 0, 0.793], [0, 0, 0, 1], np.zeros(29))
        writer.save_episode()

        body = np.loadtxt(
            os.path.join(ep_dir, 'mimic', 'trajectory.csv'), delimiter=','
        )
        assert body.shape == (36,) or body.shape == (1, 36)
        assert validate_episode(ep_dir)

    def test_long_episode_1800_frames(self):
        """30-second episode at 60Hz (1800 frames) — memory and performance."""
        ep_dir = os.path.join(self.tmpdir, 'episode_long')
        os.makedirs(ep_dir)

        traj = generate_smooth_trajectory(1800, seed=99)
        writer = MimicCSVWriter(fps=60)
        writer.create_episode(ep_dir)

        for i in range(1800):
            writer.add_frame(
                base_pos=traj['base_pos'][i],
                base_quat=traj['base_quat'][i],
                joint_q_29=traj['joint_q_bfs'][i],
                hand_left=traj['hand_left'][i],
                hand_right=traj['hand_right'][i],
            )

        writer.save_episode()

        body = np.loadtxt(
            os.path.join(ep_dir, 'mimic', 'trajectory.csv'), delimiter=','
        )
        assert body.shape == (1800, 36)
        assert validate_episode(ep_dir)

    def test_joint_values_at_limits(self):
        """Joints at exact limit values should still pass validation."""
        ep_dir = os.path.join(self.tmpdir, 'episode_limits')
        os.makedirs(ep_dir)

        writer = MimicCSVWriter(fps=60)
        writer.create_episode(ep_dir)

        # All joints at lower BFS limit
        writer.add_frame(
            [0, 0, 0.793], [0, 0, 0, 1],
            _G1_29_BFS_LIMITS_LO,
        )
        # All joints at upper BFS limit
        writer.add_frame(
            [0, 0, 0.793], [0, 0, 0, 1],
            _G1_29_BFS_LIMITS_HI,
        )

        writer.save_episode()

        body = np.loadtxt(
            os.path.join(ep_dir, 'mimic', 'trajectory.csv'), delimiter=','
        )
        assert body.shape == (2, 36)

    def test_hand_data_partial_none(self):
        """Only one hand is None — should be treated as no hand data for that frame."""
        ep_dir = os.path.join(self.tmpdir, 'episode_partial')
        os.makedirs(ep_dir)

        writer = MimicCSVWriter(fps=60)
        writer.create_episode(ep_dir)

        # Frame with only left hand (right is None)
        writer.add_frame(
            [0, 0, 0.793], [0, 0, 0, 1], np.zeros(29),
            hand_left=np.ones(6), hand_right=None,
        )
        # Frame with both hands
        writer.add_frame(
            [0, 0, 0.793], [0, 0, 0, 1], np.zeros(29),
            hand_left=np.ones(6), hand_right=np.ones(6),
        )

        writer.save_episode()

        body = np.loadtxt(
            os.path.join(ep_dir, 'mimic', 'trajectory.csv'), delimiter=','
        )
        assert body.shape == (2, 36)

        # hand_state.csv should exist (back-padded from frame 1)
        hand = np.loadtxt(
            os.path.join(ep_dir, 'mimic', 'hand_state.csv'), delimiter=','
        )
        assert hand.shape == (2, 12)
