import numpy as np
import pytest

from vehicle_counter.counter import CountLine, LineCounter, crossing_direction

A, B = np.array([0.0, 10.0]), np.array([100.0, 10.0])  # horizontal line, drawn left to right


@pytest.mark.parametrize(
    ("p0", "p1", "expected"),
    [
        ((50, 5), (50, 15), 1),  # moving down crosses onto the positive side
        ((50, 15), (50, 5), -1),
        ((50, 5), (50, 8), 0),  # approaches but does not cross
        ((150, 5), (150, 15), 0),  # crosses the infinite line beside the segment
        ((50, 5), (50, 10), 1),  # ending exactly on the line counts
    ],
)
def test_crossing_direction(p0, p1, expected):
    assert crossing_direction(np.array(p0, float), np.array(p1, float), A, B) == expected


def test_each_track_is_counted_once():
    counter = LineCounter([CountLine("l", (0, 10), (100, 10))], min_hits=1)
    # jitter back and forth across the line
    for frame, y in enumerate([0, 5, 12, 8, 14, 20]):
        counter.update(frame, track_id=1, class_name="car", anchor=(50, y))
    assert counter.totals() == {"l": {"down": 1}}
    assert len(counter.events) == 1


def test_directions_and_classes():
    counter = LineCounter([CountLine("l", (0, 10), (100, 10), ("south", "north"))], min_hits=1)
    for frame in range(3):
        counter.update(frame, 1, "car", (20, 5 + 5 * frame))  # 5 -> 15: down
        counter.update(frame, 2, "bus", (60, 15 - 5 * frame))  # 15 -> 5: up
        counter.update(frame, 3, "car", (80, 16 - 5 * frame))
    assert counter.as_dict() == {"l": {"south": {"car": 1}, "north": {"bus": 1, "car": 1}}}


def test_short_tracks_are_ignored():
    counter = LineCounter([CountLine("l", (0, 10), (100, 10))], min_hits=3)
    counter.update(0, 7, "car", (50, 5))
    counter.update(1, 7, "car", (50, 15))  # crosses on its second observation
    assert counter.totals() == {}


def test_scaled_line():
    line = CountLine("l", (0.0, 0.5), (1.0, 0.5)).scaled(640, 480)
    assert line.start == (0.0, 240.0) and line.end == (640.0, 240.0)
