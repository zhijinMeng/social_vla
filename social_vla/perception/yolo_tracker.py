"""YOLO person detection + lightweight IoU tracker (ByteTrack-style association)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from social_vla.perception.face_crop import person_bbox_to_face_bbox
from social_vla.types import BBox, PersonTrack


def _iou(a: BBox, b: BBox) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


@dataclass
class _TrackState:
    track_id: int
    bbox: BBox
    last_ts: float
    first_ts: float
    last_center: tuple[float, float]
    velocity_ema: float = 0.0
    missed: int = 0
    history: list[tuple[float, BBox]] = field(default_factory=list)


class SimpleByteTracker:
    """Greedy IoU tracker sufficient for Layer A prototyping."""

    def __init__(self, iou_thresh: float = 0.3, max_missed: int = 15):
        self.iou_thresh = iou_thresh
        self.max_missed = max_missed
        self._next_id = 1
        self._tracks: dict[int, _TrackState] = {}

    def update(self, detections: list[BBox], timestamp: float) -> list[PersonTrack]:
        assigned: set[int] = set()
        det_order = sorted(detections, key=lambda b: b.area, reverse=True)

        for det in det_order:
            best_id = None
            best_iou = self.iou_thresh
            for tid, st in self._tracks.items():
                if tid in assigned:
                    continue
                iou = _iou(st.bbox, det)
                if iou >= best_iou:
                    best_iou = iou
                    best_id = tid
            if best_id is None:
                tid = self._next_id
                self._next_id += 1
                cx, cy = det.cx, det.cy
                self._tracks[tid] = _TrackState(
                    track_id=tid,
                    bbox=det,
                    last_ts=timestamp,
                    first_ts=timestamp,
                    last_center=(cx, cy),
                    history=[(timestamp, det)],
                )
                assigned.add(tid)
            else:
                st = self._tracks[best_id]
                cx, cy = det.cx, det.cy
                dt = max(1e-3, timestamp - st.last_ts)
                dist = ((cx - st.last_center[0]) ** 2 + (cy - st.last_center[1]) ** 2) ** 0.5
                vel = dist / dt
                st.velocity_ema = 0.7 * st.velocity_ema + 0.3 * vel
                st.bbox = det
                st.last_ts = timestamp
                st.last_center = (cx, cy)
                st.missed = 0
                st.history.append((timestamp, det))
                assigned.add(best_id)

        for tid, st in list(self._tracks.items()):
            if tid not in assigned:
                st.missed += 1
                if st.missed > self.max_missed:
                    del self._tracks[tid]

        out: list[PersonTrack] = []
        for tid, st in self._tracks.items():
            if st.missed > 0:
                continue
            face = person_bbox_to_face_bbox(st.bbox)
            dwell = timestamp - st.first_ts
            frame_h = max(1.0, st.bbox.h)
            out.append(
                PersonTrack(
                    track_id=tid,
                    bbox=st.bbox,
                    timestamp=timestamp,
                    velocity=st.velocity_ema,
                    dwell_s=dwell,
                    face_area_ratio=min(1.0, face.area / max(1.0, st.bbox.area)),
                )
            )
        return sorted(out, key=lambda p: p.bbox.area, reverse=True)


class YoloPersonDetector:
    """Ultralytics YOLO wrapper; falls back to mock detections."""

    def __init__(self, model_name: str = "yolov8n.pt", device: str = "cpu", mock: bool = False):
        self.mock = mock
        self.device = device
        self._model = None
        if not mock:
            try:
                from ultralytics import YOLO

                self._model = YOLO(model_name)
            except Exception:
                self.mock = True

    def detect(self, frame: np.ndarray) -> list[BBox]:
        if self.mock or self._model is None:
            return self._mock_detect(frame)
        results = self._model(frame, verbose=False, classes=[0], device=self.device)
        boxes: list[BBox] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = box[:4]
                boxes.append(BBox(float(x1), float(y1), float(x2), float(y2)))
        return boxes

    def _mock_detect(self, frame: np.ndarray) -> list[BBox]:
        """Synthetic person in center for offline mock runs."""
        h, w = frame.shape[:2]
        t = time.time()
        cx = w * (0.5 + 0.08 * np.sin(t * 0.7))
        cy = h * (0.45 + 0.03 * np.cos(t * 0.5))
        bw, bh = w * 0.22, h * 0.55
        return [BBox(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)]
