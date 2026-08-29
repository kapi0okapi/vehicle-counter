"""Detection + multi-object tracking. ``Tracker`` is a protocol so the counting pipeline can be
tested with scripted tracks; ``UltralyticsTracker`` is the real implementation (YOLO + ByteTrack)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from vehicle_counter.config import CounterConfig


@dataclass
class Track:
    track_id: int
    class_name: str
    confidence: float
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels

    def anchor(self, kind: str = "bottom_center") -> tuple[float, float]:
        x0, y0, x1, y1 = self.box
        if kind == "bottom_center":
            return (x0 + x1) / 2, y1
        if kind == "center":
            return (x0 + x1) / 2, (y0 + y1) / 2
        raise ValueError(f"unknown anchor '{kind}'")


class Tracker(Protocol):
    def __call__(self, frame: np.ndarray) -> list[Track]: ...


class UltralyticsTracker:
    def __init__(self, cfg: CounterConfig):
        from ultralytics import YOLO

        self.cfg = cfg
        self.model = YOLO(os.path.expandvars(cfg.model))
        self.names = self.model.names

    def __call__(self, frame: np.ndarray) -> list[Track]:
        cfg = self.cfg
        result = self.model.track(
            frame,
            persist=True,  # keep tracker state between calls
            tracker=cfg.tracker,
            classes=cfg.classes,
            conf=cfg.conf,
            iou=cfg.iou,
            nms=cfg.nms,
            imgsz=cfg.imgsz,
            device=cfg.device or None,
            quantize=cfg.quantize,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None:  # no confirmed tracks in this frame
            return []
        return [
            Track(int(tid), self.names[int(cls)], float(conf), tuple(float(v) for v in xyxy))
            for tid, cls, conf, xyxy in zip(
                boxes.id.int().tolist(),
                boxes.cls.tolist(),
                boxes.conf.tolist(),
                boxes.xyxy.tolist(),
                strict=True,
            )
        ]
