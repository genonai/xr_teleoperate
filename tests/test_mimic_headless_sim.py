#!/usr/bin/env python3
"""Headless sim test for MimicCSVWriter — no VR/WebRTC required.

Generates a synthetic pick-and-place arm trajectory (sine-wave interpolation
between safe G1 joint configurations), records it through MimicCSVWriter,
and validates the output with the e2e validator.

Usage (inside Isaac Sim container):
    PYTHONPATH=/root/data/xr_teleoperate/teleop \
    /workspace/isaaclab/_isaac_sim/kit/python/bin/python3 \
        /root/data/xr_teleoperate/tests/test_mimic_headless_sim.py \
        --output-dir /root/data/mimic_sim_test \
        --num-frames 300 \
        --fps 60

Output:
    <output-dir>/episode_0000/mimic/trajectory.csv   (36 cols)
    <output-dir>/episode_0000/mimic/hand_state.csv   (12 cols)
    <output-dir>/episode_0000/mimic/deploy.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# ── Path setup ───────────────────────────────────────────────────────────────
# Allow importing from teleop/ when run standalone
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_TELEOP_DIR = os.path.join(_SCRIPT_DIR, '..', 'teleop')
if _TELEOP_DIR not in sys.path:
    sys.path.insert(0, _TELEOP_DIR)

from utils.mimic_csv_writer import MimicCSVWriter, _CANONICAL_DEFAULT_JOINT_POS

# Also allow importing the e2e validator
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_mimic_csv_e2e import validate_episode


# ── G1-29 waypoints for synthetic pick-and-place ─────────────────────────────
# BFS (motor) order: joints 0-14 are legs+waist, 15-21 left arm, 22-28 right arm
# We keep legs/waist at standing pose and move only the arms.

_STANDING = np.array(_CANONICAL_DEFAULT_JOINT_POS, dtype=np.float64)

# Right arm waypoints (BFS indices 22-28): reach → approach → grasp → lift → place
_RIGHT_ARM_WAYPOINTS = np.array([
    # Start: neutral standing pose (right arm BFS 22-28)
    [0.605, 0.596, 0.008, 0.003, 0.003, -0.003, -0.010],
    # Reach forward: shoulder pitch down, elbow bent
    [0.3, -0.3, 0.0, 1.2, 0.0, -0.2, 0.0],
    # Approach: lower elbow more
    [0.2, -0.4, 0.0, 1.5, 0.0, -0.3, 0.0],
    # Grasp position: steady
    [0.2, -0.4, 0.0, 1.5, 0.0, -0.3, 0.0],
    # Lift: shoulder pitch up
    [0.4, -0.3, 0.0, 1.2, 0.0, -0.2, 0.0],
    # Place: move to side
    [0.5, -0.5, 0.3, 1.0, 0.0, -0.1, 0.0],
    # Return to neutral
    [0.605, 0.596, 0.008, 0.003, 0.003, -0.003, -0.010],
], dtype=np.float64)

# Right hand (6 DOF Inspire): open → open → closing → closed → closed → opening → open
_RIGHT_HAND_WAYPOINTS = np.array([
    [0.9, 0.9, 0.9, 0.9, 0.9, 0.9],   # open
    [0.9, 0.9, 0.9, 0.9, 0.9, 0.9],   # still open (approaching)
    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],   # closing
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],   # closed (grasping)
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],   # still closed (lifting)
    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],   # opening (placing)
    [0.9, 0.9, 0.9, 0.9, 0.9, 0.9],   # open (done)
], dtype=np.float64)


def interpolate_waypoints(waypoints: np.ndarray, n_frames: int) -> np.ndarray:
    """Smoothly interpolate between waypoints over n_frames using linear interp."""
    n_waypoints, n_dof = waypoints.shape
    # Distribute frames evenly across waypoint segments
    t_waypoints = np.linspace(0, 1, n_waypoints)
    t_frames = np.linspace(0, 1, n_frames)

    result = np.zeros((n_frames, n_dof))
    for dof in range(n_dof):
        result[:, dof] = np.interp(t_frames, t_waypoints, waypoints[:, dof])

    return result


def generate_trajectory(n_frames: int) -> dict:
    """Generate a synthetic pick-and-place trajectory.

    Returns dict with keys: joint_q_bfs, hand_left, hand_right
    (all n_frames x D arrays in BFS order).
    """
    # Start from standing pose for all frames
    joint_q_bfs = np.tile(_STANDING, (n_frames, 1))

    # Interpolate right arm trajectory
    right_arm_traj = interpolate_waypoints(_RIGHT_ARM_WAYPOINTS, n_frames)
    joint_q_bfs[:, 22:29] = right_arm_traj

    # Add tiny noise to legs/waist (simulating standing micro-movements)
    rng = np.random.RandomState(42)
    joint_q_bfs[:, :15] += rng.normal(0, 0.001, (n_frames, 15))

    # Left arm: slight mirror movement (stays mostly still)
    left_arm_base = _STANDING[15:22]
    left_arm_noise = rng.normal(0, 0.002, (n_frames, 7))
    joint_q_bfs[:, 15:22] = left_arm_base + left_arm_noise

    # Hand trajectories
    right_hand = interpolate_waypoints(_RIGHT_HAND_WAYPOINTS, n_frames)
    # Left hand: stays open (not used in pick-and-place)
    left_hand = np.full((n_frames, 6), 0.9)
    left_hand += rng.normal(0, 0.01, (n_frames, 6))
    left_hand = np.clip(left_hand, 0.0, 1.0)

    return {
        'joint_q_bfs': joint_q_bfs,
        'hand_left': left_hand,
        'hand_right': right_hand,
    }


def run_headless_test(output_dir: str, n_frames: int, fps: float) -> bool:
    """Run the headless sim test. Returns True if all checks pass."""
    print(f"=== Headless MimicCSV Sim Test ===")
    print(f"  Output:     {output_dir}")
    print(f"  Frames:     {n_frames} ({n_frames / fps:.1f}s @ {fps:.0f}Hz)")
    print()

    # Generate trajectory
    print("[1/4] Generating synthetic pick-and-place trajectory...")
    traj = generate_trajectory(n_frames)

    # Base state (stationary teleop)
    base_pos = np.array([0.0, 0.0, 0.793])
    base_quat = np.array([0.0, 0.0, 0.0, 1.0])  # identity, xyzw

    # Record via MimicCSVWriter
    print("[2/4] Recording via MimicCSVWriter...")
    episode_dir = os.path.join(output_dir, 'episode_0000')
    os.makedirs(episode_dir, exist_ok=True)

    writer = MimicCSVWriter(fps=fps)
    writer.create_episode(episode_dir)

    t_start = time.time()
    for i in range(n_frames):
        writer.add_frame(
            base_pos=base_pos,
            base_quat=base_quat,
            joint_q_29=traj['joint_q_bfs'][i],
            hand_left=traj['hand_left'][i],
            hand_right=traj['hand_right'][i],
        )
    writer.save_episode()
    t_record = time.time() - t_start

    print(f"  Recording took {t_record:.3f}s ({n_frames / t_record:.0f} frames/s)")

    # Check output files exist
    print("[3/4] Checking output files...")
    mimic_dir = os.path.join(episode_dir, 'mimic')
    files = {
        'trajectory.csv': os.path.join(mimic_dir, 'trajectory.csv'),
        'hand_state.csv': os.path.join(mimic_dir, 'hand_state.csv'),
        'deploy.yaml': os.path.join(mimic_dir, 'deploy.yaml'),
    }
    for name, path in files.items():
        exists = os.path.isfile(path)
        size = os.path.getsize(path) if exists else 0
        status = f"{size:,} bytes" if exists else "MISSING"
        print(f"  {name}: {status}")
        if not exists:
            print(f"  ERROR: {name} not found!")
            return False

    # Quick sanity check on shapes
    body = np.loadtxt(files['trajectory.csv'], delimiter=',')
    hand = np.loadtxt(files['hand_state.csv'], delimiter=',')
    print(f"  trajectory.csv shape: {body.shape}")
    print(f"  hand_state.csv shape: {hand.shape}")

    # Run e2e validator
    print("[4/4] Running e2e validation...")
    passed = validate_episode(episode_dir)

    print()
    if passed:
        print("SUCCESS: All checks passed.")
        print(f"Output at: {episode_dir}/mimic/")
    else:
        print("FAILED: See errors above.")

    return passed


def main():
    parser = argparse.ArgumentParser(description="Headless sim test for MimicCSVWriter")
    parser.add_argument('--output-dir', type=str, default='/tmp/mimic_sim_test',
                        help='Directory to write test episode')
    parser.add_argument('--num-frames', type=int, default=300,
                        help='Number of frames to record (default: 300 = 5s @ 60Hz)')
    parser.add_argument('--fps', type=float, default=60.0,
                        help='Recording frequency (default: 60)')
    args = parser.parse_args()

    success = run_headless_test(args.output_dir, args.num_frames, args.fps)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
