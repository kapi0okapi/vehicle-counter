"""Fine-tune a YOLO detector (Ultralytics) on a traffic dataset, resuming automatically.

    vehicle-train --data data/detrac_yolo/data.yaml --model yolo26s.pt --name detrac_yolo26s

If ``<project>/<name>/weights/last.pt`` exists, training resumes from it, so a job stopped by
the Slurm time limit continues when it is resubmitted.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="yolo26s.pt")
    parser.add_argument("--name", default="detrac_yolo26s")
    parser.add_argument("--project", default=os.environ.get("RUNS", "runs"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="")
    parser.add_argument("--optimizer", default="auto", help="SGD | AdamW | auto (Ultralytics' choice)")
    parser.add_argument("--lr0", type=float, help="initial learning rate (default: the optimizer's)")
    parser.add_argument("--freeze", type=int, help="freeze the first N layers (the backbone is 0-10)")
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args(argv)

    from ultralytics import YOLO

    project = Path(os.path.expandvars(args.project)).resolve()
    last = project / args.name / "weights" / "last.pt"
    if last.exists():
        print(f"resuming from {last}")
        YOLO(str(last)).train(resume=True)
        return
    extra = {k: v for k, v in (("lr0", args.lr0), ("freeze", args.freeze)) if v is not None}
    YOLO(args.model).train(
        data=os.path.expandvars(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device or None,
        project=str(project),
        name=args.name,
        exist_ok=True,
        cos_lr=True,
        optimizer=args.optimizer,
        patience=args.patience,
        close_mosaic=10,
        plots=True,
        nms=False,  # validate (and pick best.pt) on YOLO26's NMS-free head, which tracking uses
        **extra,
    )


if __name__ == "__main__":
    main()
