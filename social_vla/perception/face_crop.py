"""Face crop utilities shared by LAM and TalkNet."""
from __future__ import annotations

import cv2
import numpy as np

from social_vla.types import BBox


def person_bbox_to_face_bbox(person: BBox, face_fraction: float = 0.45) -> BBox:
    """Heuristic face region from full-body person box (upper portion)."""
    h = person.h
    face_h = h * face_fraction
    return BBox(person.x1, person.y1, person.x2, person.y1 + face_h)


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
