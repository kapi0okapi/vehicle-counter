"""UltralyticsTracker passes the YOLO26 inference options through and parses tracked boxes."""

import sys
import types

import numpy as np
import torch

from vehicle_counter.config import CounterConfig
from vehicle_counter.tracking import UltralyticsTracker


class FakeYOLO:
    calls: list[dict] = []

    def __init__(self, weights):
        self.names = {2: "car", 7: "truck"}

    def track(self, frame, **kwargs):
        FakeYOLO.calls.append(kwargs)
        boxes = types.SimpleNamespace(
            id=torch.tensor([4, 9]),
            cls=torch.tensor([2.0, 7.0]),
            conf=torch.tensor([0.9, 0.6]),
            xyxy=torch.tensor([[10.0, 20.0, 50.0, 60.0], [100.0, 100.0, 200.0, 180.0]]),
        )
        return [types.SimpleNamespace(boxes=boxes)]


def test_nms_free_head_and_quantize_are_forwarded(monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    cfg = CounterConfig(quantize=16)
    tracks = UltralyticsTracker(cfg)(np.zeros((240, 320, 3), np.uint8))

    kwargs = FakeYOLO.calls[-1]
    assert kwargs["nms"] is False  # Ultralytics 8.4 would otherwise run the one-to-many head + NMS
    assert kwargs["quantize"] == 16 and "half" not in kwargs
    assert kwargs["persist"] is True
    assert [(t.track_id, t.class_name) for t in tracks] == [(4, "car"), (9, "truck")]
    assert tracks[1].anchor() == (150.0, 180.0)
