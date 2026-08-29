"""Video -> tracks -> line crossings -> counts, annotated video and an event log."""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2

from vehicle_counter.annotate import Annotator
from vehicle_counter.config import CounterConfig
from vehicle_counter.counter import CountLine, CrossingEvent, LineCounter
from vehicle_counter.tracking import Tracker, UltralyticsTracker

log = logging.getLogger(__name__)


@dataclass
class CountResult:
    counts: dict
    totals: dict
    frames: int
    source_fps: float
    processing_fps: float
    events: list[CrossingEvent] = field(default_factory=list)


def open_source(source: str | int) -> cv2.VideoCapture:
    """File path, RTSP/HTTP URL or webcam index."""
    capture = cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
    if not capture.isOpened():
        raise OSError(f"cannot open video source {source}")
    return capture


def count_video(
    source: str | int,
    cfg: CounterConfig,
    tracker: Tracker | None = None,
    out_dir: Path | None = None,
    show: bool = False,
    max_frames: int | None = None,
) -> CountResult:
    capture = open_source(source)
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    lines = [CountLine(c.name, c.start, c.end, c.directions).scaled(width, height) for c in cfg.lines]
    counter = LineCounter(lines, min_hits=cfg.min_hits)
    tracker = tracker or UltralyticsTracker(cfg)
    annotator = Annotator(cfg.trail) if (out_dir or show) else None
    writer = None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(out_dir / "annotated.mp4"), fourcc, source_fps / cfg.vid_stride, (width, height)
        )

    frame_idx, processed, busy, fps = 0, 0, 0.0, None
    try:
        while max_frames is None or processed < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_idx % cfg.vid_stride:
                frame_idx += 1
                continue
            started = time.perf_counter()
            tracks = tracker(frame)
            for track in tracks:
                counter.update(frame_idx, track.track_id, track.class_name, track.anchor(cfg.anchor))
            elapsed = time.perf_counter() - started
            busy += elapsed
            fps = 1 / elapsed if fps is None else 0.9 * fps + 0.1 / max(elapsed, 1e-6)
            if annotator:
                annotator.draw(frame, tracks, lines, counter, cfg.anchor, fps)
            if writer:
                writer.write(frame)
            if show:
                cv2.imshow("vehicle-counter", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            processed += 1
            frame_idx += 1
    finally:
        capture.release()
        if writer:
            writer.release()

    result = CountResult(
        counter.as_dict(),
        counter.totals(),
        processed,
        source_fps,
        processed / busy if busy else 0.0,
        counter.events,
    )
    if out_dir:
        write_outputs(result, out_dir)
    log.info("processed %d frames at %.1f FPS, totals %s", processed, result.processing_fps, result.totals)
    return result


def write_outputs(result: CountResult, out_dir: Path) -> None:
    with (out_dir / "events.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time_s", "track_id", "class", "line", "direction"])
        for e in result.events:
            writer.writerow(
                [
                    e.frame,
                    round(e.frame / result.source_fps, 2),
                    e.track_id,
                    e.class_name,
                    e.line,
                    e.direction,
                ]
            )
    summary = {k: v for k, v in asdict(result).items() if k != "events"}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
