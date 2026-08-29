"""Counting pipeline on a rendered video with scripted tracks (no detector needed)."""

import json

import cv2
import numpy as np

from vehicle_counter.config import CounterConfig, LineConfig
from vehicle_counter.counter import CountLine
from vehicle_counter.evaluate_counts import ground_truth_totals
from vehicle_counter.pipeline import count_video
from vehicle_counter.tracking import Track

W, H, FRAMES = 320, 240, 40


def trajectories(frame: int) -> list[Track]:
    """Car 1 drives down through the line, car 2 up, truck 3 stays above the line."""
    y1 = 20 + 5 * frame
    y2 = 220 - 5 * frame
    return [
        Track(1, "car", 0.9, (40, y1 - 20, 80, y1)),
        Track(2, "car", 0.9, (150, y2 - 20, 190, y2)),
        Track(3, "truck", 0.9, (250, 30, 300, 60)),
    ]


class ScriptedTracker:
    def __init__(self):
        self.frame = 0

    def __call__(self, frame: np.ndarray) -> list[Track]:
        tracks = trajectories(self.frame)
        self.frame += 1
        return tracks


def make_video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (W, H))
    for f in range(FRAMES):
        frame = np.full((H, W, 3), 40, np.uint8)
        for t in trajectories(f):
            x0, y0, x1, y1 = map(int, t.box)
            cv2.rectangle(frame, (x0, y0), (x1, y1), (200, 200, 200), -1)
        writer.write(frame)
    writer.release()


def test_count_video_with_scripted_tracks(tmp_path):
    video = tmp_path / "clip.mp4"
    make_video(video)
    cfg = CounterConfig(lines=[LineConfig("middle", (0.0, 0.5), (1.0, 0.5))])
    result = count_video(video, cfg, tracker=ScriptedTracker(), out_dir=tmp_path / "out")
    assert result.frames == FRAMES
    assert result.totals == {"middle": {"down": 1, "up": 1}}
    assert (tmp_path / "out" / "annotated.mp4").stat().st_size > 0
    events = (tmp_path / "out" / "events.csv").read_text().strip().splitlines()
    assert len(events) == 3  # header + two crossings
    assert json.loads((tmp_path / "out" / "summary.json").read_text())["counts"]["middle"]["down"] == {
        "car": 1
    }


def test_ground_truth_totals_from_mot(tmp_path):
    rows = []
    for f in range(FRAMES):
        for t in trajectories(f):
            x0, y0, x1, y1 = t.box
            rows.append(f"{f},{t.track_id},{x0},{y0},{x1 - x0},{y1 - y0},1,0,1")
    (tmp_path / "gt.txt").write_text("\n".join(rows))
    lines = [CountLine("middle", (0, H / 2), (W, H / 2))]
    assert ground_truth_totals(tmp_path / "gt.txt", lines, "bottom_center") == {
        "middle": {"down": 1, "up": 1}
    }
