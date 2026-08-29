"""Line-crossing counter: counts each tracked object once per line, per direction and class.

A track is counted when the segment between its previous and current anchor point (by default
the bottom centre of the box, where the vehicle touches the road) intersects the count line.
Only the first crossing of a line by a track is counted, so a vehicle that jitters on the line
or is briefly lost is not counted twice.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CountLine:
    name: str
    start: tuple[float, float]
    end: tuple[float, float]
    # Names for crossings onto the positive / negative side of start->end. For a line drawn left to
    # right, the positive side is below it in the image, so the defaults read "down" / "up".
    directions: tuple[str, str] = ("down", "up")

    def scaled(self, width: int, height: int) -> CountLine:
        """Line given in normalised [0, 1] coordinates -> pixel coordinates."""
        return CountLine(
            self.name,
            (self.start[0] * width, self.start[1] * height),
            (self.end[0] * width, self.end[1] * height),
            self.directions,
        )


@dataclass
class CrossingEvent:
    frame: int
    track_id: int
    class_name: str
    line: str
    direction: str


def side(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Signed area test in image coordinates (y down): > 0 on the positive side of a->b, which is
    below a left-to-right line; < 0 on the other side; 0 on the line."""
    return float((b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0]))


def crossing_direction(p0: np.ndarray, p1: np.ndarray, a: np.ndarray, b: np.ndarray) -> int:
    """+1 / -1 if the movement p0->p1 crosses segment a-b onto its positive / negative side, else 0.
    Points exactly on the line count as the positive side."""
    pos0, pos1 = side(p0, a, b) >= 0, side(p1, a, b) >= 0
    if pos0 == pos1:
        return 0
    if side(a, p0, p1) * side(b, p0, p1) > 0:  # passes beside the segment, not through it
        return 0
    return 1 if pos1 else -1


@dataclass
class LineCounter:
    lines: list[CountLine]
    min_hits: int = 3  # ignore tracks seen in fewer frames (spurious detections)
    _last: dict[int, np.ndarray] = field(default_factory=dict)
    _hits: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    _counted: set[tuple[int, str]] = field(default_factory=set)
    counts: dict[str, dict[str, dict[str, int]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    )
    events: list[CrossingEvent] = field(default_factory=list)

    def update(
        self, frame: int, track_id: int, class_name: str, anchor: tuple[float, float]
    ) -> list[CrossingEvent]:
        point = np.asarray(anchor, dtype=float)
        previous = self._last.get(track_id)
        self._last[track_id] = point
        self._hits[track_id] += 1
        if previous is None or self._hits[track_id] < self.min_hits:
            return []
        new = []
        for line in self.lines:
            key = (track_id, line.name)
            if key in self._counted:
                continue
            sign = crossing_direction(previous, point, np.asarray(line.start), np.asarray(line.end))
            if sign:
                direction = line.directions[0] if sign > 0 else line.directions[1]
                self._counted.add(key)
                self.counts[line.name][direction][class_name] += 1
                new.append(CrossingEvent(frame, track_id, class_name, line.name, direction))
        self.events.extend(new)
        return new

    def totals(self) -> dict[str, dict[str, int]]:
        return {line: {d: sum(c.values()) for d, c in dirs.items()} for line, dirs in self.counts.items()}

    def as_dict(self) -> dict:
        return {line: {d: dict(c) for d, c in dirs.items()} for line, dirs in self.counts.items()}
