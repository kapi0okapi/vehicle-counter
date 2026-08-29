"""Drawing tracks, count lines and running totals onto frames."""

from __future__ import annotations

from collections import defaultdict, deque

import cv2
import numpy as np

from vehicle_counter.counter import CountLine, LineCounter
from vehicle_counter.tracking import Track


def track_color(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id)
    return tuple(int(c) for c in rng.integers(60, 255, 3))


class Annotator:
    def __init__(self, trail: int = 30):
        self.trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=trail))

    def draw(
        self,
        frame: np.ndarray,
        tracks: list[Track],
        lines: list[CountLine],
        counter: LineCounter,
        anchor: str,
        fps: float | None = None,
    ) -> np.ndarray:
        for line in lines:
            a, b = tuple(map(int, line.start)), tuple(map(int, line.end))
            cv2.line(frame, a, b, (0, 220, 255), 2, cv2.LINE_AA)
            cv2.putText(
                frame, line.name, (a[0] + 5, a[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2
            )

        for track in tracks:
            color = track_color(track.track_id)
            x0, y0, x1, y1 = map(int, track.box)
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
            label = f"#{track.track_id} {track.class_name}"
            cv2.putText(frame, label, (x0, max(14, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            trail = self.trails[track.track_id]
            trail.append(tuple(map(int, track.anchor(anchor))))
            if len(trail) > 1:
                cv2.polylines(frame, [np.array(trail, np.int32)], False, color, 2, cv2.LINE_AA)

        rows = [
            f"{line}: " + "  ".join(f"{d} {n}" for d, n in dirs.items())
            for line, dirs in counter.totals().items()
        ] or ["no crossings yet"]
        if fps:
            rows.append(f"{fps:.1f} FPS")
        panel_h = 26 * len(rows) + 10
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (360, panel_h), (0, 0, 0), -1)
        frame[:] = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
        for i, text in enumerate(rows):
            cv2.putText(frame, text, (10, 26 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        return frame
