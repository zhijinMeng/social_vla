"""Layer A 10Hz perception + engagement + trigger pipeline."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def default_yolo_weights() -> str:
    for candidate in (
        _PROJECT_ROOT / "weights" / "yolov8n.pt",
        _PROJECT_ROOT / "yolov8n.pt",
    ):
        if candidate.is_file():
            return str(candidate)
    return "yolov8n.pt"

from social_vla.perception.face_crop import person_bbox_to_face_bbox
from social_vla.perception.lam_stream import StreamingLAMEngine
from social_vla.perception.pose_features import dwell_score
from social_vla.perception.talknet_stream import AudioMfccStream, StreamingTalkNetEngine
from social_vla.perception.yolo_tracker import SimpleByteTracker, YoloPersonDetector
from social_vla.scoring.engagement import EngagementScorer
from social_vla.session.manager import SessionManager
from social_vla.session.trigger import TriggerRouter
from social_vla.types import FrameTick, PerPersonScores, PersonTrack


@dataclass
class LayerAConfig:
    hz: float = 10.0
    mock_detector: bool = True
    mock_models: bool = True
    device: str = "cpu"
    yolo_model: str = ""
    lam_checkpoint: str | None = None
    talknet_checkpoint: str | None = None
    talknet_window_s: float = 1.0
    vad_mock: bool = False
    perception_only: bool = False


class LayerAPipeline:
    def __init__(self, config: LayerAConfig | None = None):
        self.config = config or LayerAConfig()
        yolo_model = self.config.yolo_model or default_yolo_weights()
        self.detector = YoloPersonDetector(
            model_name=yolo_model,
            device=self.config.device,
            mock=self.config.mock_detector,
        )
        self.tracker = SimpleByteTracker()
        self.lam = StreamingLAMEngine(
            checkpoint=self.config.lam_checkpoint,
            device=self.config.device,
            mock=self.config.mock_models,
        )
        self.talknet = StreamingTalkNetEngine(
            checkpoint=self.config.talknet_checkpoint,
            device=self.config.device,
            window_s=self.config.talknet_window_s,
            mock=self.config.mock_models,
        )
        self.audio = AudioMfccStream()
        self.scorer = EngagementScorer()
        self.session = SessionManager()
        self.trigger = TriggerRouter()
        self._frame_idx = 0
        self._last_tick = 0.0
        self._vad_active = False
        self._utterance_t0: float | None = None
        self._global_mfcc_track: int | None = None

    def push_audio_pcm(self, pcm: np.ndarray, active_track_id: int | None = None) -> None:
        rows = self.audio.push_pcm(pcm)
        tid = active_track_id or self._global_mfcc_track
        if tid is None:
            return
        self.talknet.push_audio_mfcc(tid, rows)

    def set_vad(self, active: bool, now: float | None = None) -> None:
        ts = now if now is not None else time.time()
        if active and not self._vad_active:
            self._utterance_t0 = ts
        if not active:
            self._utterance_t0 = None
        self._vad_active = active

    def process_frame(self, frame_bgr: np.ndarray, timestamp: float | None = None) -> FrameTick:
        ts = timestamp if timestamp is not None else time.time()
        self.session.tick(ts)
        h, w = frame_bgr.shape[:2]

        detections = self.detector.detect(frame_bgr)
        persons = self.tracker.update(detections, ts)
        active_ids = {p.track_id for p in persons}

        # prune stale tracks
        for tid in list(self.lam._buffers):
            if tid not in active_ids:
                self.lam.reset_track(tid)
        for tid in list(self.talknet._buffers):
            if tid not in active_ids:
                self.talknet.reset_track(tid)

        if persons:
            self._global_mfcc_track = persons[0].track_id

        scores: list[PerPersonScores] = []
        for person in persons:
            face_bbox = person_bbox_to_face_bbox(person.bbox)
            lam_prob, ready_lam = self.lam.update(person.track_id, frame_bgr, face_bbox, ts)
            talk_prob, ready_talk = self.talknet.update_visual(
                person.track_id, frame_bgr, person.bbox, ts
            )
            speech_directed = talk_prob if self._vad_active else 0.0
            fused = self.scorer.fuse(
                person=person,
                lam_prob=lam_prob,
                talknet_prob=talk_prob,
                vad_active=self._vad_active,
                frame_width=w,
                ready_lam=ready_lam,
                ready_talknet=ready_talk,
                speech_directed=speech_directed,
            )
            scores.append(fused)

        utter_dur = 0.0
        if self._vad_active and self._utterance_t0 is not None:
            utter_dur = ts - self._utterance_t0

        decision = self.trigger.evaluate(
            scores=scores,
            session=self.session,
            vad_active=self._vad_active,
            utterance_duration_s=utter_dur,
            doa_track_id=self._global_mfcc_track if self._vad_active else None,
            now=ts,
        )

        if not self.config.perception_only:
            if decision.path == "robot_initiated" and decision.target_id is not None:
                self.session.start_robot_initiated(decision.target_id, ts)
                self.session.enter_dialogue("[MOCK OPENING] 你好，想了解机器人吗？", ts)
            elif decision.path == "user_initiated" and decision.target_id is not None:
                self.session.start_user_initiated(
                    decision.target_id,
                    user_text="[MOCK USER]",
                    reply="[MOCK REPLY] 你好！我是展会引导机器人。",
                    now=ts,
                )

        self._frame_idx += 1
        tick = FrameTick(
            timestamp=ts,
            frame_idx=self._frame_idx,
            persons=persons,
            scores=scores,
            vad_active=self._vad_active,
            best_target_id=decision.target_id,
            trigger=decision.path,
            session_state=self.session.session.state,
            debug={
                "trigger_reason": decision.reason,
                "trigger_score": decision.score,
                "utterance_s": utter_dur,
                "perception_only": self.config.perception_only,
            },
        )
        self._last_tick = ts
        return tick

    def run_period(self, source, max_ticks: int | None = None) -> None:
        """Read from cv2.VideoCapture or callable returning BGR frames."""
        period = 1.0 / self.config.hz
        ticks = 0
        while max_ticks is None or ticks < max_ticks:
            t0 = time.time()
            if hasattr(source, "read"):
                ok, frame = source.read()
                if not ok:
                    break
            else:
                frame = source()
            if self.config.vad_mock:
                self.set_vad((ticks // 30) % 2 == 1, t0)
            tick = self.process_frame(frame, t0)
            self._print_tick(tick)
            ticks += 1
            elapsed = time.time() - t0
            time.sleep(max(0.0, period - elapsed))

    def _print_tick(self, tick: FrameTick) -> None:
        parts = [
            f"#{tick.frame_idx}",
            f"state={tick.session_state.value}",
            f"vad={int(tick.vad_active)}",
        ]
        if tick.scores:
            top = max(tick.scores, key=lambda s: s.engagement)
            lam_tag = "m" if top.ready_lam else "-"
            talk_tag = "m" if top.ready_talknet else "-"
            parts.append(
                f"id={top.track_id} eng={top.engagement:.2f} "
                f"lam={top.lam_prob:.2f}{lam_tag} talk={top.talknet_prob:.2f}{talk_tag}"
            )
        else:
            parts.append(f"persons={len(tick.persons)}")
        if tick.trigger:
            label = "WOULD_TRIGGER" if tick.debug.get("perception_only") else "TRIGGER"
            parts.append(f"{label}={tick.trigger}")
        if "audio_rms" in tick.debug:
            parts.append(f"rms={tick.debug['audio_rms']:.0f}")
        print(" | ".join(parts))

    def reset_session(self) -> None:
        from social_vla.session.manager import SessionManager

        self.session = SessionManager()
        self.trigger.reset()


def open_video_source(path: str | int):
    return cv2.VideoCapture(path)
