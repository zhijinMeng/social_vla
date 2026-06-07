"""Dynamic engagement score fusion (silent: LAM-heavy, audible: TalkNet-heavy)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from social_vla.perception.pose_features import approach_speed_score, bbox_orientation_score, dwell_score
from social_vla.types import PerPersonScores, PersonTrack


@dataclass
class EngagementConfig:
    # silent: lam, talknet, approach, body, dwell
    weights_silent: tuple[float, ...] = (0.35, 0.10, 0.20, 0.10, 0.25)
    weights_audible: tuple[float, ...] = (0.20, 0.35, 0.20, 0.15, 0.10)
    speech_directed_boost: float = 0.15


class EngagementScorer:
    def __init__(self, config: EngagementConfig | None = None):
        self.config = config or EngagementConfig()

    def fuse(
        self,
        person: PersonTrack,
        lam_prob: float,
        talknet_prob: float,
        vad_active: bool,
        frame_width: int,
        ready_lam: bool,
        ready_talknet: bool,
        speech_directed: float = 0.0,
    ) -> PerPersonScores:
        w = self.config.weights_audible if vad_active else self.config.weights_silent
        approach = approach_speed_score(person.velocity)
        body = bbox_orientation_score(person.bbox, frame_width)
        dwell = dwell_score(person.dwell_s)

        lam = lam_prob if ready_lam else 0.0
        talk = talknet_prob if (vad_active and ready_talknet) else 0.0

        features = [lam, talk, approach, body, dwell]
        score = float(np.dot(w, features))
        if vad_active and speech_directed > 0.5:
            score = min(1.0, score + self.config.speech_directed_boost)

        return PerPersonScores(
            track_id=person.track_id,
            lam_prob=lam,
            talknet_prob=talk,
            approach_speed=approach,
            body_orient=body,
            speech_directed=speech_directed,
            dwell_time=dwell,
            engagement=score,
            weights=list(w),
            ready_lam=ready_lam,
            ready_talknet=ready_talknet,
        )

    def pick_target(self, scores: list[PerPersonScores]) -> int | None:
        if not scores:
            return None
        best = max(scores, key=lambda s: (s.engagement, s.lam_prob, s.talknet_prob))
        return best.track_id
