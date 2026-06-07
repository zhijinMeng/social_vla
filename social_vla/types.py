"""Shared data types for Layer A perception and engagement."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SessionState(str, Enum):
    IDLE = "idle"
    ENGAGING = "engaging"
    IN_DIALOGUE = "in_dialogue"
    CLOSING = "closing"
    COOLDOWN = "cooldown"


@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    def expand(self, scale: float) -> BBox:
        cx, cy = self.cx, self.cy
        hw = self.w * (1.0 + scale) / 2.0
        hh = self.h * (1.0 + scale) / 2.0
        return BBox(cx - hw, cy - hh, cx + hw, cy + hh)

    def clip(self, width: int, height: int) -> BBox:
        return BBox(
            max(0.0, self.x1),
            max(0.0, self.y1),
            min(float(width), self.x2),
            min(float(height), self.y2),
        )

    def as_xyxy_int(self) -> tuple[int, int, int, int]:
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)


@dataclass
class PersonTrack:
    track_id: int
    bbox: BBox
    timestamp: float
    velocity: float = 0.0
    dwell_s: float = 0.0
    face_area_ratio: float = 0.0


@dataclass
class PerPersonScores:
    track_id: int
    lam_prob: float = 0.0
    talknet_prob: float = 0.0
    approach_speed: float = 0.0
    body_orient: float = 0.0
    speech_directed: float = 0.0
    dwell_time: float = 0.0
    engagement: float = 0.0
    weights: list[float] = field(default_factory=list)
    ready_lam: bool = False
    ready_talknet: bool = False


@dataclass
class FrameTick:
    timestamp: float
    frame_idx: int
    persons: list[PersonTrack]
    scores: list[PerPersonScores]
    vad_active: bool
    best_target_id: int | None = None
    trigger: str | None = None
    session_state: SessionState = SessionState.IDLE
    debug: dict[str, Any] = field(default_factory=dict)
