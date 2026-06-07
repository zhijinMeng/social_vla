"""Body orientation proxy for engagement score."""
from __future__ import annotations

import numpy as np

from social_vla.types import BBox


def bbox_orientation_score(bbox: BBox, frame_width: int) -> float:
    """
    Cheap proxy: person centered and facing camera ≈ higher score.
    Full MediaPipe Pose can replace this later.
    """
    cx_norm = bbox.cx / max(1, frame_width)
    center_score = 1.0 - min(1.0, abs(cx_norm - 0.5) * 2.0)
    aspect = bbox.w / max(1.0, bbox.h)
    # frontal upper-body tends to be wider
    aspect_score = float(np.clip((aspect - 0.25) / 0.35, 0.0, 1.0))
    return 0.6 * center_score + 0.4 * aspect_score


def approach_speed_score(velocity_px_s: float, max_speed: float = 120.0) -> float:
    """Normalize pixel velocity into [0, 1]."""
    return float(np.clip(velocity_px_s / max_speed, 0.0, 1.0))


def dwell_score(dwell_s: float, target_s: float = 3.0) -> float:
    return float(np.clip(dwell_s / target_s, 0.0, 1.0))
