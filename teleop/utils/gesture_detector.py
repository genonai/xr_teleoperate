"""Gesture detector for hands-free record toggle during teleop data collection.

Detects the sequence: Open hand → Thumbs up → Open hand
Uses distance-based pose classification (frame-independent).

Usage:
    detector = GestureDetector()
    # In main loop:
    triggered = detector.update(hand_landmarks)  # (25, 3) array
    if triggered:
        RECORD_TOGGLE = True
"""

import time
import enum
import numpy as np
import logging_mp
logger = logging_mp.getLogger(__name__)


class Pose(enum.IntEnum):
    UNKNOWN = 0
    OPEN_HAND = 1
    THUMBS_UP = 2


class GestureState(enum.IntEnum):
    IDLE = 0        # waiting for open hand
    READY = 1       # open hand seen, waiting for thumbs up
    THUMBS_UP = 2   # thumbs up seen, waiting for open hand to confirm


# OpenXR 25-joint hand landmark indices
WRIST = 0
THUMB_TIP = 4
THUMB_MCP = 2
INDEX_TIP = 9
INDEX_MCP = 5
MIDDLE_TIP = 14
MIDDLE_MCP = 10
RING_TIP = 19
RING_MCP = 15
PINKY_TIP = 24
PINKY_MCP = 20


def _classify_pose(landmarks: np.ndarray) -> Pose:
    """Classify hand pose from (25, 3) landmarks.

    Uses distance ratios (tip-to-wrist vs mcp-to-wrist) so it works
    regardless of coordinate frame or hand orientation.
    """
    wrist = landmarks[WRIST]

    def tip_mcp_ratio(tip_idx, mcp_idx):
        d_tip = np.linalg.norm(landmarks[tip_idx] - wrist)
        d_mcp = np.linalg.norm(landmarks[mcp_idx] - wrist)
        return d_tip / max(d_mcp, 1e-6)

    thumb_ratio = tip_mcp_ratio(THUMB_TIP, THUMB_MCP)
    index_ratio = tip_mcp_ratio(INDEX_TIP, INDEX_MCP)
    middle_ratio = tip_mcp_ratio(MIDDLE_TIP, MIDDLE_MCP)
    ring_ratio = tip_mcp_ratio(RING_TIP, RING_MCP)
    pinky_ratio = tip_mcp_ratio(PINKY_TIP, PINKY_MCP)

    # Finger extended: tip farther from wrist than MCP (ratio > threshold)
    # Lower thresholds for more lenient detection
    thumb_ext = thumb_ratio > 1.05
    index_ext = index_ratio > 1.1
    middle_ext = middle_ratio > 1.1
    ring_ext = ring_ratio > 1.1
    pinky_ext = pinky_ratio > 1.1

    # Finger curled: tip close to or closer than MCP (ratio < threshold)
    index_curled = index_ratio < 0.95
    middle_curled = middle_ratio < 0.95
    ring_curled = ring_ratio < 0.95
    pinky_curled = pinky_ratio < 0.95

    four_extended = [index_ext, middle_ext, ring_ext, pinky_ext]
    four_curled = [index_curled, middle_curled, ring_curled, pinky_curled]

    if thumb_ext and all(four_extended):
        return Pose.OPEN_HAND
    if thumb_ext and all(four_curled):
        return Pose.THUMBS_UP
    return Pose.UNKNOWN


class GestureDetector:
    """Detects open_hand → thumbs_up → open_hand gesture sequence.

    Args:
        cooldown: Seconds to wait after trigger before allowing next trigger.
        persistence: Number of consecutive frames a pose must be held
                     before state transition (prevents flickering).
        debug: Log pose classification every N frames (0 = off).
    """

    def __init__(self, cooldown: float = 2.0, persistence: int = 5, debug: int = 30):
        self.cooldown = cooldown
        self.persistence = persistence
        self.debug = debug
        self.state = GestureState.IDLE
        self.last_trigger_time = 0.0
        self._pose_counter = 0
        self._last_pose = Pose.UNKNOWN
        self._frame_count = 0
        self._stable_pose = Pose.UNKNOWN  # pose that passed persistence check

    def update(self, landmarks: np.ndarray) -> bool:
        """Feed (25, 3) hand landmarks. Returns True if gesture completed."""
        if landmarks is None or landmarks.shape != (25, 3):
            return False

        self._frame_count += 1
        pose = _classify_pose(landmarks)

        # Debug logging
        if self.debug > 0 and self._frame_count % self.debug == 0:
            wrist = landmarks[WRIST]
            ratios = {
                "thumb": np.linalg.norm(landmarks[THUMB_TIP] - wrist) / max(np.linalg.norm(landmarks[THUMB_MCP] - wrist), 1e-6),
                "index": np.linalg.norm(landmarks[INDEX_TIP] - wrist) / max(np.linalg.norm(landmarks[INDEX_MCP] - wrist), 1e-6),
                "middle": np.linalg.norm(landmarks[MIDDLE_TIP] - wrist) / max(np.linalg.norm(landmarks[MIDDLE_MCP] - wrist), 1e-6),
                "ring": np.linalg.norm(landmarks[RING_TIP] - wrist) / max(np.linalg.norm(landmarks[RING_MCP] - wrist), 1e-6),
                "pinky": np.linalg.norm(landmarks[PINKY_TIP] - wrist) / max(np.linalg.norm(landmarks[PINKY_MCP] - wrist), 1e-6),
            }
            logger.debug(f"[gesture] pose={pose.name} state={self.state.name} ratios={{{', '.join(f'{k}:{v:.2f}' for k,v in ratios.items())}}}")

        # Count consecutive frames with same pose
        if pose == self._last_pose:
            self._pose_counter += 1
        else:
            self._pose_counter = 1
            self._last_pose = pose

        # Require pose to be stable for N frames before accepting
        if self._pose_counter < self.persistence:
            return False

        # Update stable pose when persistence threshold is first reached
        if pose != self._stable_pose:
            self._stable_pose = pose
        else:
            # Already processed this stable pose transition
            return False

        now = time.time()

        if self.state == GestureState.IDLE:
            if pose == Pose.OPEN_HAND:
                self.state = GestureState.READY
                logger.info(f"[gesture] IDLE → READY (open hand detected)")

        elif self.state == GestureState.READY:
            if pose == Pose.THUMBS_UP:
                self.state = GestureState.THUMBS_UP
                logger.info(f"[gesture] READY → THUMBS_UP (thumbs up detected)")
            elif pose == Pose.UNKNOWN:
                self.state = GestureState.IDLE

        elif self.state == GestureState.THUMBS_UP:
            if pose == Pose.OPEN_HAND and (now - self.last_trigger_time > self.cooldown):
                self.state = GestureState.IDLE
                self.last_trigger_time = now
                logger.info(f"[gesture] TRIGGERED! (open hand → thumbs up → open hand)")
                return True
            elif pose == Pose.UNKNOWN:
                self.state = GestureState.IDLE
                logger.info(f"[gesture] THUMBS_UP → IDLE (hand lost)")

        return False

    def reset(self):
        self.state = GestureState.IDLE
        self._pose_counter = 0
        self._last_pose = Pose.UNKNOWN
        self._stable_pose = Pose.UNKNOWN
