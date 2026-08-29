"""Counting accuracy against annotated tracks.

Ground-truth counts come from running the same line counter over the annotated trajectories
(MOT format), so the metric isolates detection + tracking errors from the counting rule.

    vehicle-count-eval --sequences data/detrac_yolo/sequences --config configs/detrac.yaml

Each sequence folder needs ``video.mp4`` and ``gt.txt`` (frame,id,x,y,w,h,...), as written by
``vehicle_counter.datasets.ua_detrac``.

Every camera gets its own count line, as an operator would draw it: across the road,
perpendicular to the main direction of travel of the annotated vehicles, through the middle of the
traffic and as long as the band of lanes they use (``--line auto``, the default). ``--line config``
uses the lines of the config file for every sequence instead.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from vehicle_counter.config import CounterConfig, LineConfig
from vehicle_counter.counter import CountLine, LineCounter
from vehicle_counter.pipeline import count_video
from vehicle_counter.tracking import Track
from vehicle_counter.utils.config import load_config


def ground_truth_totals(gt_file: Path, lines: list[CountLine], anchor: str) -> dict[str, dict[str, int]]:
    rows = np.loadtxt(gt_file, delimiter=",", ndmin=2)
    by_frame: dict[int, list] = defaultdict(list)
    for row in rows:
        by_frame[int(row[0])].append(row)
    counter = LineCounter(lines, min_hits=1)
    for frame in sorted(by_frame):
        for row in by_frame[frame]:
            track = Track(int(row[1]), "vehicle", 1.0, (row[2], row[3], row[2] + row[4], row[3] + row[5]))
            counter.update(frame, track.track_id, track.class_name, track.anchor(anchor))
    return counter.totals()


def _anchors(rows: np.ndarray, anchor: str) -> np.ndarray:
    x, y, w, h = rows[:, 2], rows[:, 3], rows[:, 4], rows[:, 5]
    return np.stack([x + w / 2, y + h if anchor == "bottom_center" else y + h / 2], axis=1)


def flow_line(gt_file: Path, width: int, height: int, anchor: str, margin: float = 0.1) -> LineConfig | None:
    """Count line across the traffic of one camera, from its annotated tracks (normalised coordinates).

    The direction of travel is the principal axis of the tracks' unit displacement vectors, which
    is the same for both carriageways of a road; the line is perpendicular to it, passes through the
    median anchor position and spans the 2nd-98th percentile of the anchors across the road, plus a
    margin. ``None`` when no track moves far enough to define a direction.
    """
    rows = np.loadtxt(gt_file, delimiter=",", ndmin=2)
    points = _anchors(rows, anchor)
    moves = []
    for track_id in np.unique(rows[:, 1]):
        mask = rows[:, 1] == track_id
        order = np.argsort(rows[mask, 0])
        track = points[mask][order]
        step = track[-1] - track[0]
        if np.linalg.norm(step) > 0.05 * min(width, height):
            moves.append(step / np.linalg.norm(step))
    if not moves:
        return None
    moves = np.array(moves)
    eigval, eigvec = np.linalg.eigh(moves.T @ moves)
    travel = eigvec[:, np.argmax(eigval)]
    across = np.array([-travel[1], travel[0]])
    centre = np.median(points, axis=0)
    offset = (points - centre) @ across
    lo, hi = np.percentile(offset, [2, 98])
    lo, hi = lo - margin * (hi - lo), hi + margin * (hi - lo)
    start, end = centre + lo * across, centre + hi * across
    size = np.array([width, height], float)
    start, end = np.clip(start / size, 0, 1), np.clip(end / size, 0, 1)
    return LineConfig(
        "line", (float(start[0]), float(start[1])), (float(end[0]), float(end[1])), ("in", "out")
    )


def compare(gt: dict, pred: dict, lines: list[CountLine]) -> list[dict]:
    rows = []
    for line in lines:
        for direction in line.directions:
            g = gt.get(line.name, {}).get(direction, 0)
            p = pred.get(line.name, {}).get(direction, 0)
            rows.append(
                {"line": line.name, "direction": direction, "gt": g, "pred": p, "abs_err": abs(p - g)}
            )
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/detrac.yaml"))
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--out", type=Path, default=Path("runs/count_eval.json"))
    parser.add_argument(
        "--line", choices=("auto", "config"), default="auto", help="count line per camera, see above"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    cfg = load_config(CounterConfig, args.config, args.set)

    report, total_err, total_gt = {}, 0, 0
    for seq_dir in sorted(p for p in args.sequences.iterdir() if (p / "gt.txt").exists()):
        capture = cv2.VideoCapture(str(seq_dir / "video.mp4"))
        width, height = (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        capture.release()
        seq_cfg = cfg
        if args.line == "auto":
            line = flow_line(seq_dir / "gt.txt", width, height, cfg.anchor)
            if line is not None:
                seq_cfg = dataclasses.replace(cfg, lines=[line])
        lines = [CountLine(c.name, c.start, c.end, c.directions).scaled(width, height) for c in seq_cfg.lines]
        gt = ground_truth_totals(seq_dir / "gt.txt", lines, cfg.anchor)
        pred = count_video(seq_dir / "video.mp4", seq_cfg).totals
        rows = compare(gt, pred, lines)
        per_row = [line for line in seq_cfg.lines for _ in line.directions]
        for row, line in zip(rows, per_row, strict=True):
            row["line_xy"] = [list(line.start), list(line.end)]
        report[seq_dir.name] = rows
        total_err += sum(r["abs_err"] for r in rows)
        total_gt += sum(r["gt"] for r in rows)
        print(seq_dir.name, "  ".join(f"{r['direction']}: {r['pred']}/{r['gt']}" for r in rows))

    summary = {
        "sequences": len(report),
        "vehicles_gt": total_gt,
        "abs_count_error": total_err,
        "relative_count_error": total_err / total_gt if total_gt else None,
        "model": cfg.model,
        "tracker": cfg.tracker,
        "line": args.line,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "per_sequence": report}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
