"""Count vehicles crossing one or more lines in a video file, stream or webcam.

    vehicle-count --source traffic.mp4 --out runs/count/traffic
    vehicle-count --source rtsp://camera/stream --config configs/two_lines.yaml --show
    vehicle-count --source 0 --set model=yolo26n.pt --set imgsz=640

Outputs: annotated.mp4, events.csv (one row per crossing) and summary.json.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from vehicle_counter.config import CounterConfig
from vehicle_counter.pipeline import count_video
from vehicle_counter.utils.config import load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", required=True, help="video file, stream URL or webcam index")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--out", type=Path, default=Path("runs/count"))
    parser.add_argument("--show", action="store_true", help="live preview window (needs GUI OpenCV)")
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(CounterConfig, args.config if args.config.exists() else None, args.set)
    result = count_video(args.source, cfg, out_dir=args.out, show=args.show, max_frames=args.max_frames)
    for line, directions in result.counts.items():
        for direction, classes in directions.items():
            detail = ", ".join(f"{name} {n}" for name, n in sorted(classes.items()))
            print(f"{line} {direction}: {sum(classes.values())} ({detail})")
    print(f"{result.frames} frames, {result.processing_fps:.1f} FPS -> {args.out}")


if __name__ == "__main__":
    main()
