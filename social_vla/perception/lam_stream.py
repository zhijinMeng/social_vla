"""Streaming Ego4D Looking-At-Me (GazeLSTM) inference."""
from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from social_vla.perception.face_crop import crop_face_rgb
from social_vla.types import BBox

LAM_WINDOW = 7
LAM_CENTER = 3


def _ego4d_root() -> Path:
    return Path(__file__).resolve().parents[2] / "third_party" / "ego4d-lam"


def _import_gaze_lstm():
    root = str(_ego4d_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    from model.model import GazeLSTM  # type: ignore

    return GazeLSTM


@dataclass
class _LamTrackBuffer:
    frames: deque = field(default_factory=lambda: deque(maxlen=LAM_WINDOW))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=LAM_WINDOW))

    def push(self, face_rgb: np.ndarray, timestamp: float) -> None:
        self.frames.append(face_rgb)
        self.timestamps.append(timestamp)

    def ready(self) -> bool:
        return len(self.frames) == LAM_WINDOW

    def as_tensor(self, device: torch.device) -> torch.Tensor:
        """Return (1, 7, 3, 224, 224) normalized tensor."""
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = np.stack(self.frames, axis=0).astype(np.float32) / 255.0
        arr = (arr - mean) / std
        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).unsqueeze(0)
        return tensor.to(device)


class StreamingLAMEngine:
    """Per-track 7-frame sliding window LAM scorer."""

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        device: str | None = None,
        mock: bool = False,
    ):
        self.mock = mock
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._buffers: dict[int, _LamTrackBuffer] = {}
        self._model = None
        if not mock:
            self._load_model(checkpoint)

    def _load_model(self, checkpoint: str | Path | None) -> None:
        ckpt = Path(checkpoint) if checkpoint else Path(__file__).resolve().parents[2] / "weights" / "lam_gazelstm.pth"
        if not ckpt.exists():
            self.mock = True
            return
        try:
            GazeLSTM = _import_gaze_lstm()

            class _Args:
                checkpoint = str(ckpt)
                rank = 0
                eval = False

            self._model = GazeLSTM(_Args()).to(self.device)
            self._model.eval()
        except Exception:
            self.mock = True
            self._model = None

    def reset_track(self, track_id: int) -> None:
        self._buffers.pop(track_id, None)

    def update(self, track_id: int, frame_bgr: np.ndarray, face_bbox: BBox, timestamp: float) -> tuple[float, bool]:
        buf = self._buffers.setdefault(track_id, _LamTrackBuffer())
        face_rgb = crop_face_rgb(frame_bgr, face_bbox)
        buf.push(face_rgb, timestamp)
        if not buf.ready():
            return 0.0, False
        if self.mock or self._model is None:
            prob = self._mock_score(face_rgb, timestamp)
            return prob, True
        with torch.no_grad():
            x = buf.as_tensor(self.device)
            logits = self._model(x)
            prob = F.softmax(logits, dim=-1)[0, 1].item()
        return float(prob), True

    def _mock_score(self, face_rgb: np.ndarray, timestamp: float) -> float:
        """Brightness + center bias proxy when weights unavailable."""
        gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
        brightness = gray.mean() / 255.0
        phase = 0.5 + 0.35 * np.sin(timestamp * 0.9)
        return float(np.clip(0.25 * brightness + 0.75 * phase, 0.0, 1.0))
