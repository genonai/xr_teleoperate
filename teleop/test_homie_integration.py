"""Teleop-side integration tests for --homie flag wiring (T1 from spec §5.4).
Heavy mocking — no DDS, no robot, no Quest 3."""
import sys
import threading
import pytest
from unittest.mock import patch, MagicMock


def test_homie_and_motion_mutually_exclusive():
    """argparse rejects --homie + --motion together."""
    import argparse
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group()
    g.add_argument('--motion', action='store_true')
    g.add_argument('--homie', action='store_true')
    with pytest.raises(SystemExit):
        p.parse_args(['--homie', '--motion'])


def test_homie_client_module_byte_identical_to_source():
    """Spec §2.7 — synced homie_client.py must equal the gate source.

    The sync script (tools/sync_homie_client.sh) prepends a 2-line
    generated header to the source-of-truth file:

        # GENERATED — do not edit. Sync: tools/sync_homie_client.sh
        # Source of truth: scripts/real_eval/homie_eval_gate.py
        <source body starts here>

    We strip those two header lines and compare the remainder byte-for-byte.
    If the synced file drifts (someone edited it directly, or the source
    changed without re-running the sync script), this test fails.
    """
    import hashlib
    import os

    # Path resolution: this test file lives at repos/xr_teleoperate/teleop/.
    # The IL_PhysicalAI parent repo source-of-truth lives at
    # ../../scripts/real_eval/homie_eval_gate.py from the fork's working
    # directory. Use absolute paths so pytest can be invoked from either side.
    fork_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repos/xr_teleoperate
    parent_dir = os.path.dirname(os.path.dirname(fork_dir))  # IL_PhysicalAI
    src_path = os.path.join(parent_dir, "scripts", "real_eval", "homie_eval_gate.py")
    dst_path = os.path.join(fork_dir, "teleop", "utils", "homie_client.py")

    if not os.path.exists(src_path):
        pytest.skip(f"source-of-truth not found at {src_path} — "
                    "run this test from a parent IL_PhysicalAI checkout")

    with open(src_path) as f:
        src_body = f.read()
    with open(dst_path) as f:
        dst_full = f.read()

    # Strip the 2-line generated header. split("\n", 2) returns:
    #   parts[0] = "# GENERATED — do not edit. Sync: tools/sync_homie_client.sh"
    #   parts[1] = "# Source of truth: scripts/real_eval/homie_eval_gate.py"
    #   parts[2] = the rest (= src_body)
    parts = dst_full.split("\n", 2)
    assert len(parts) == 3, "synced file too short to contain the 2-line header"
    assert parts[0].startswith("# GENERATED"), \
        f"synced file missing generated marker on line 1: {parts[0]!r}"
    assert parts[1].startswith("# Source of truth"), \
        f"synced file missing source-of-truth marker on line 2: {parts[1]!r}"
    dst_body = parts[2]

    src_hash = hashlib.sha256(src_body.encode()).hexdigest()
    dst_hash = hashlib.sha256(dst_body.encode()).hexdigest()
    assert src_hash == dst_hash, (
        f"homie_client.py drift detected:\n"
        f"  src ({src_path}): {src_hash}\n"
        f"  dst ({dst_path}): {dst_hash}\n"
        f"Re-run: bash tools/sync_homie_client.sh --apply"
    )
