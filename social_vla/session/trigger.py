"""Three-path trigger router: robot-initiated, user-initiated, turn."""
from __future__ import annotations

from dataclasses import dataclass

from social_vla.session.manager import SessionManager
from social_vla.types import PerPersonScores, SessionState


@dataclass
class TriggerConfig:
    score_threshold: float = 0.55
    score_high: float = 0.72
    consecutive_frames: int = 3
    ttm_threshold: float = 0.6
    utterance_min_s: float = 0.8
    vl_confidence: float = 0.75


@dataclass
class TriggerDecision:
    path: str | None
    target_id: int | None
    reason: str
    score: float = 0.0


class TriggerRouter:
    def __init__(self, config: TriggerConfig | None = None):
        self.config = config or TriggerConfig()
        self._streak: dict[int, int] = {}
        self._last_vad_active = False  # Track previous VAD state to detect edges

    def reset(self) -> None:
        self._streak.clear()
        self._last_vad_active = False

    def evaluate(
        self,
        scores: list[PerPersonScores],
        session: SessionManager,
        vad_active: bool,
        utterance_duration_s: float,
        doa_track_id: int | None,
        now: float,
    ) -> TriggerDecision:
        sm = session.session
        if sm.state == SessionState.IN_DIALOGUE:
            # Trigger turn only on VAD falling edge (1→0) after minimum duration
            vad_ended = self._last_vad_active and not vad_active
            if vad_ended and utterance_duration_s >= self.config.utterance_min_s:
                self._last_vad_active = vad_active
                return TriggerDecision("turn", sm.target_id, "user_utterance_end")
            self._last_vad_active = vad_active
            return TriggerDecision(None, sm.target_id, "in_dialogue_wait")

        if sm.state != SessionState.IDLE:
            return TriggerDecision(None, None, f"state_{sm.state.value}")

        # User-initiated on utterance end (same edge semantics as in-dialogue turn)
        vad_ended = self._last_vad_active and not vad_active
        if vad_ended and utterance_duration_s >= self.config.utterance_min_s:
            candidate = self._best_speech_directed(scores, doa_track_id)
            self._last_vad_active = vad_active
            if candidate is not None:
                return TriggerDecision("user_initiated", candidate, "vad_end+ttm/doa", score=0.0)
            return TriggerDecision(None, None, "vad_end_no_speaker")

        self._last_vad_active = vad_active

        # Robot-initiated via engagement score
        best = max(scores, key=lambda s: s.engagement) if scores else None
        if best is None:
            return TriggerDecision(None, None, "no_person")

        tid = best.track_id
        if not session.can_engage(tid, now):
            return TriggerDecision(None, None, "cooldown_or_busy")

        if best.engagement >= self.config.score_high:
            self._streak[tid] = self.config.consecutive_frames
        elif best.engagement >= self.config.score_threshold:
            self._streak[tid] = self._streak.get(tid, 0) + 1
        else:
            self._streak[tid] = 0

        silent_interest = (
            not vad_active
            and best.lam_prob >= 0.65
            and best.dwell_time >= 0.8
            and best.approach_speed < 0.35
        )

        if self._streak.get(tid, 0) >= self.config.consecutive_frames or silent_interest:
            return TriggerDecision(
                "robot_initiated",
                tid,
                "score_streak" if not silent_interest else "silent_interest",
                score=best.engagement,
            )

        return TriggerDecision(None, tid, "below_threshold", score=best.engagement)

    def _best_speech_directed(
        self, scores: list[PerPersonScores], doa_track_id: int | None
    ) -> int | None:
        if doa_track_id is not None:
            return doa_track_id
        eligible = [
            s
            for s in scores
            if s.talknet_prob >= self.config.ttm_threshold or s.speech_directed >= self.config.ttm_threshold
        ]
        if not eligible:
            return None
        best = max(eligible, key=lambda s: (s.talknet_prob, s.speech_directed, s.engagement))
        return best.track_id
