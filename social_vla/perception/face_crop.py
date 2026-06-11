"""Face crop utilities shared by LAM and TalkNet."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from social_vla.types import BBox

_S3FD_ROOT = Path(__file__).resolve().parents[2] / "talknet-asd"
_s3fd_detector = None


def _get_s3fd(device: str = "cuda"):
    global _s3fd_detector
    if _s3fd_detector is not None:
        return _s3fd_detector
    inserted = str(_S3FD_ROOT) not in sys.path
    if inserted:
        sys.path.insert(0, str(_S3FD_ROOT))
    import os
    orig_cwd = os.getcwd()
    os.chdir(str(_S3FD_ROOT))
    try:
        from model.faceDetector.s3fd import S3FD  # type: ignore
        _s3fd_detector = S3FD(device=device)
    except Exception:
        _s3fd_detector = None
    finally:
        os.chdir(orig_cwd)
        if inserted and str(_S3FD_ROOT) in sys.path:
            sys.path.remove(str(_S3FD_ROOT))
    return _s3fd_detector


def person_bbox_to_face_bbox(person: BBox, frame: np.ndarray | None = None, device: str = "cuda") -> BBox:
    """Detect face within person bbox using S3FD; fall back to upper-40% heuristic."""
    fallback = BBox(person.x1, person.y1, person.x2, person.y1 + person.h * 0.40)
    if frame is None:
        return fallback
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = person.clip(fw, fh).as_xyxy_int()
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return fallback
    det = _get_s3fd(device)
    if det is None:
        return fallback
    try:
        faces = det.detect_faces(roi, conf_th=0.5, scales=[0.5])
    except Exception:
        return fallback
    if not len(faces):
        return fallback
    fx1, fy1, fx2, fy2, _ = max(faces, key=lambda f: (f[2] - f[0]) * (f[3] - f[1]))
    return BBox(x1 + fx1, y1 + fy1, x1 + fx2, y1 + fy2)



def crop_face_rgb(frame: np.ndarray, bbox: BBox, size: int = 224, scale: float = 0.1) -> np.ndarray:
    """Crop and resize face to RGB uint8 (H, W, 3)."""
    h, w = frame.shape[:2]
    box = bbox.expand(scale).clip(w, h)
    x1, y1, x2, y2 = box.as_xyxy_int()
    if x2 <= x1 or y2 <= y1:
        return np.zeros((size, size, 3), dtype=np.uint8)
    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    return cv2.resize(face, (size, size))


def face_rgb_to_talknet_gray(face_rgb_224: np.ndarray) -> np.ndarray:
    """TalkNet lip crop: center 112x112 grayscale from 224x224 face."""
    gray = cv2.cvtColor(face_rgb_224, cv2.COLOR_RGB2GRAY)
    center = 112
    half = 56
    return gray[center - half : center + half, center - half : center + half]
