"""Streaming TalkNet active speaker detection."""
from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from social_vla.perception.face_crop import crop_face_rgb, face_rgb_to_talknet_gray, person_bbox_to_face_bbox
from social_vla.types import BBox

# TalkNet expects 25 video frames / sec and 100 MFCC frames / sec (4:1 ratio).
VIDEO_FPS = 25
MFCC_PER_SEC = 100
DEFAULT_WINDOW_S = 1.0


def _talknet_root() -> Path:
    return Path(__file__).resolve().parents[2] / "talknet-asd"


def _import_talknet_modules():
    root = str(_talknet_root())
    inserted = root not in sys.path
    if inserted:
        sys.path.insert(0, root)
    try:
        from model.talkNetModel import talkNetModel  # type: ignore
        from loss import lossAV  # type: ignore
        return talkNetModel, lossAV
    finally:
        if inserted and root in sys.path:
            sys.path.remove(root)


@dataclass
class _AvTrackBuffer:
    gray_frames: deque = field(default_factory=deque)
    mfcc_frames: deque = field(default_factory=deque)
    timestamps: deque = field(default_factory=deque)

    def push_visual(self, gray_112: np.ndarray, timestamp: float) -> None:
        self.gray_frames.append(gray_112)
        self.timestamps.append(timestamp)

    def push_mfcc(self, mfcc_row: np.ndarray) -> None:
        self.mfcc_frames.append(mfcc_row.astype(np.float32))

    def ready(self, window_s: float, min_visual: int) -> bool:
        return len(self.gray_frames) >= min_visual and len(self.mfcc_frames) >= int(window_s * MFCC_PER_SEC * 0.5)

    def trim(self, max_visual: int, max_mfcc: int) -> None:
        while len(self.gray_frames) > max_visual:
            self.gray_frames.popleft()
            self.timestamps.popleft()
        while len(self.mfcc_frames) > max_mfcc:
            self.mfcc_frames.popleft()

    def sample_window(self, window_s: float) -> tuple[np.ndarray, np.ndarray] | None:
        n_v = int(window_s * VIDEO_FPS)
        n_a = int(window_s * MFCC_PER_SEC)
        if len(self.gray_frames) < n_v or len(self.mfcc_frames) < n_a:
            # pad with edge frames if close
            if len(self.gray_frames) < 8 or len(self.mfcc_frames) < 32:
                return None
            n_v = min(n_v, len(self.gray_frames))
            n_a = min(n_a, len(self.mfcc_frames))
        visual = np.stack(list(self.gray_frames)[-n_v:], axis=0)
        audio = np.stack(list(self.mfcc_frames)[-n_a:], axis=0)
        # align lengths per TalkNet convention
        length = min((audio.shape[0] - audio.shape[0] % 4) / MFCC_PER_SEC, visual.shape[0] / VIDEO_FPS)
        n_a_use = int(round(length * MFCC_PER_SEC))
        n_v_use = int(round(length * VIDEO_FPS))
        if n_a_use < 4 or n_v_use < 4:
            return None
        return audio[:n_a_use], visual[:n_v_use]


class AudioMfccStream:
    """Incremental MFCC extractor from PCM chunks."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._pcm = np.array([], dtype=np.int16)

    def push_pcm(self, pcm: np.ndarray) -> list[np.ndarray]:
        self._pcm = np.concatenate([self._pcm, pcm.astype(np.int16)])
        rows: list[np.ndarray] = []
        win_samples = int(0.025 * self.sample_rate)
        hop_samples = int(0.010 * self.sample_rate)
        if len(self._pcm) < win_samples:
            return rows
        try:
            import python_speech_features
        except ImportError:
            return rows
        mfcc = python_speech_features.mfcc(
            self._pcm,
            self.sample_rate,
            numcep=13,
            winlen=0.025,
            winstep=0.010,
        )
        # only return newly available tail rows
        new_count = max(0, mfcc.shape[0] - getattr(self, "_last_mfcc_count", 0))
        self._last_mfcc_count = mfcc.shape[0]
        for i in range(mfcc.shape[0] - new_count, mfcc.shape[0]):
            rows.append(mfcc[i])
        # keep small tail for overlap
        keep = win_samples * 2
        if len(self._pcm) > keep:
            self._pcm = self._pcm[-keep:]
            self._last_mfcc_count = min(self._last_mfcc_count, 4)
        return rows


class StreamingTalkNetEngine:
    """Per-track AV sliding window TalkNet scorer."""

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        device: str | None = None,
        window_s: float = DEFAULT_WINDOW_S,
        mock: bool = False,
    ):
        self.mock = mock
        self.window_s = window_s
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.min_visual = max(8, int(window_s * VIDEO_FPS * 0.5))
        self.max_visual = int(window_s * VIDEO_FPS * 2)
        self.max_mfcc = int(window_s * MFCC_PER_SEC * 2)
        self._buffers: dict[int, _AvTrackBuffer] = {}
        self._model = None
        self._loss_av = None
        if not mock:
            self._load_model(checkpoint)

    def _load_model(self, checkpoint: str | Path | None) -> None:
        ckpt = Path(checkpoint) if checkpoint else Path(__file__).resolve().parents[2] / "weights" / "pretrain_TalkSet.model"
        if not ckpt.exists():
            self.mock = True
            return
        try:
            talkNetModel_cls, lossAV_cls = _import_talknet_modules()
            self._model = talkNetModel_cls().to(self.device)
            self._loss_av = lossAV_cls().to(self.device)
            state = torch.load(str(ckpt), map_location=self.device)
            model_state = {
                k.replace("model.", "", 1): v
                for k, v in state.items()
                if k.startswith("model.")
            }
            loss_state = {
                k.replace("lossAV.", "", 1): v
                for k, v in state.items()
                if k.startswith("lossAV.")
            }
            self._model.load_state_dict(model_state, strict=True)
            self._loss_av.load_state_dict(loss_state, strict=True)
            self._model.eval()
            self._loss_av.eval()
        except Exception:
            self.mock = True
            self._model = None
            self._loss_av = None

    def reset_track(self, track_id: int) -> None:
        self._buffers.pop(track_id, None)

    def push_audio_mfcc(self, track_id: int, mfcc_rows: list[np.ndarray]) -> None:
        if not mfcc_rows:
            return
        buf = self._buffers.setdefault(track_id, _AvTrackBuffer())
        for row in mfcc_rows:
            buf.push_mfcc(row)
        buf.trim(self.max_visual, self.max_mfcc)

    def update_visual(
        self,
        track_id: int,
        frame_bgr: np.ndarray,
        person_bbox: BBox,
        timestamp: float,
    ) -> tuple[float, bool]:
        buf = self._buffers.setdefault(track_id, _AvTrackBuffer())
        face_bbox = person_bbox_to_face_bbox(person_bbox, frame_bgr)
        face_rgb = crop_face_rgb(frame_bgr, face_bbox)
        gray = face_rgb_to_talknet_gray(face_rgb)
        buf.push_visual(gray, timestamp)
        buf.trim(self.max_visual, self.max_mfcc)
        if not buf.ready(self.window_s, self.min_visual):
            return 0.0, False
        if self.mock or self._model is None:
            return self._mock_score(timestamp), True
        sample = buf.sample_window(self.window_s)
        if sample is None:
            return 0.0, False
        audio_feat, visual_feat = sample
        dev = self.device
        with torch.no_grad():
            input_a = torch.FloatTensor(audio_feat).unsqueeze(0).to(dev)
            input_v = torch.FloatTensor(visual_feat).unsqueeze(0).to(dev)
            embed_a = self._model.forward_audio_frontend(input_a)
            embed_v = self._model.forward_visual_frontend(input_v)
            embed_a, embed_v = self._model.forward_cross_attention(embed_a, embed_v)
            out = self._model.forward_audio_visual_backend(embed_a, embed_v)
            raw = self._loss_av.forward(out, labels=None)
            score = float(np.mean(raw))
            prob = float(1.0 / (1.0 + np.exp(-score)))
        return prob, True

    def _mock_score(self, timestamp: float) -> float:
        base = 0.5 + 0.4 * np.sin(timestamp * 1.3 + 1.0)
        return float(np.clip(base, 0.0, 1.0))
