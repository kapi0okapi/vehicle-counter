"""Counting configuration. YAML files in ``configs/`` override these defaults."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LineConfig:
    name: str = "line"
    start: tuple[float, float] = (0.0, 0.6)  # normalised image coordinates (x, y)
    end: tuple[float, float] = (1.0, 0.6)
    directions: tuple[str, str] = ("down", "up")


@dataclass
class CounterConfig:
    model: str = "yolo26s.pt"  # COCO weights (NMS-free YOLO26), or a fine-tuned checkpoint
    classes: list[int] | None = field(default_factory=lambda: [2, 3, 5, 7])  # COCO car/motorcycle/bus/truck
    conf: float = 0.3
    iou: float = 0.6  # NMS IoU, only used when nms is not False
    # False = YOLO26's end-to-end (one-to-one) head, no NMS; None = one-to-many head + NMS.
    # Ultralytics 8.4 defaults to None, so the NMS-free path must be requested explicitly.
    nms: bool | None = False
    imgsz: int = 960
    # any Ultralytics tracker: bytetrack | botsort | ocsort | deepocsort | tracktrack | fasttrack (.yaml)
    tracker: str = "bytetrack.yaml"
    device: str = ""  # "" lets Ultralytics pick cuda / mps / cpu
    quantize: int | None = None  # 16 = FP16 inference on CUDA
    vid_stride: int = 1  # process every n-th frame
    anchor: str = "bottom_center"  # point of the box that must cross the line: bottom_center | center
    min_hits: int = 3
    trail: int = 30  # frames of track history drawn in the output video
    lines: list[LineConfig] = field(default_factory=lambda: [LineConfig()])
