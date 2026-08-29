"""UA-DETRAC -> YOLO detection dataset + per-sequence ground truth for count evaluation.

Expected layout of the official release:

    Insight-MVT_Annotation_Train/MVI_20011/img00001.jpg ...
    DETRAC-Train-Annotations-XML/MVI_20011.xml ...

Output:

    <out>/images/{train,val}/<seq>_<frame>.jpg     ignored regions greyed out
    <out>/labels/{train,val}/<seq>_<frame>.txt     YOLO format
    <out>/data.yaml
    <out>/sequences/<val seq>/video.mp4 + gt.txt   MOT format, for vehicle-count-eval

Train and validation are split by sequence (camera site), so validation frames never come
from a video the detector has seen.

    python -m vehicle_counter.datasets.ua_detrac --images .../Insight-MVT_Annotation_Train \\
        --annotations .../DETRAC-Train-Annotations-XML --out data/detrac_yolo
"""

from __future__ import annotations

import argparse
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

CLASSES = ("car", "van", "bus", "others")
GREY = (114, 114, 114)  # same as the YOLO letterbox padding


@dataclass
class Target:
    track_id: int
    box: tuple[float, float, float, float]  # left, top, width, height
    vehicle_type: str


@dataclass
class Sequence:
    name: str
    ignored: list[tuple[float, float, float, float]] = field(default_factory=list)
    frames: dict[int, list[Target]] = field(default_factory=dict)


def _box(element) -> tuple[float, float, float, float]:
    return tuple(float(element.get(k)) for k in ("left", "top", "width", "height"))


def parse_xml(path: Path) -> Sequence:
    root = ET.parse(path).getroot()
    seq = Sequence(root.get("name") or path.stem)
    for region in root.iter("ignored_region"):
        seq.ignored.extend(_box(b) for b in region.findall("box"))
    for frame in root.iter("frame"):
        targets = []
        for target in frame.iter("target"):
            attribute = target.find("attribute")
            targets.append(
                Target(int(target.get("id")), _box(target.find("box")), attribute.get("vehicle_type"))
            )
        seq.frames[int(frame.get("num"))] = targets
    return seq


def yolo_lines(targets: list[Target], width: int, height: int) -> list[str]:
    lines = []
    for t in targets:
        left, top, w, h = t.box
        x0, y0 = max(0.0, left), max(0.0, top)
        x1, y1 = min(float(width), left + w), min(float(height), top + h)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        cls = CLASSES.index(t.vehicle_type) if t.vehicle_type in CLASSES else CLASSES.index("others")
        cx, cy = (x0 + x1) / 2 / width, (y0 + y1) / 2 / height
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {(x1 - x0) / width:.6f} {(y1 - y0) / height:.6f}")
    return lines


def grey_out(image: np.ndarray, regions: list[tuple[float, float, float, float]]) -> np.ndarray:
    for left, top, w, h in regions:
        cv2.rectangle(image, (int(left), int(top)), (int(left + w), int(top + h)), GREY, -1)
    return image


def write_sequence_gt(seq: Sequence, image_dir: Path, out_dir: Path, fps: float = 25.0) -> None:
    """Full-frame-rate video plus MOT ground truth (frame,id,x,y,w,h,conf,class,visibility)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(seq.frames)
    first = cv2.imread(str(image_dir / f"img{frames[0]:05d}.jpg"))
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(out_dir / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    with (out_dir / "gt.txt").open("w") as gt:
        for num in frames:
            # unannotated regions greyed out, as in the training images, so that vehicles the
            # ground truth leaves out are not counted either
            writer.write(grey_out(cv2.imread(str(image_dir / f"img{num:05d}.jpg")), seq.ignored))
            for t in seq.frames[num]:
                cls = CLASSES.index(t.vehicle_type) if t.vehicle_type in CLASSES else 3
                gt.write(
                    f"{num - frames[0]},{t.track_id},{t.box[0]:.2f},{t.box[1]:.2f},{t.box[2]:.2f},"
                    f"{t.box[3]:.2f},1,{cls},1\n"
                )
    writer.release()


def convert(
    images_root: Path,
    xml_root: Path,
    out: Path,
    val_fraction: float = 0.15,
    frame_step: int = 5,
    seed: int = 0,
    write_videos: bool = True,
) -> dict[str, int]:
    xmls = sorted(xml_root.glob("*.xml"))
    if not xmls:
        raise FileNotFoundError(f"no XML annotations in {xml_root}")
    names = [x.stem for x in xmls]
    order = np.random.default_rng(seed).permutation(len(names))
    val = {names[i] for i in order[: max(1, round(len(names) * val_fraction))]}
    counts = {"train": 0, "val": 0}

    for xml in xmls:
        seq = parse_xml(xml)
        split = "val" if xml.stem in val else "train"
        image_dir = images_root / xml.stem
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for num, targets in sorted(seq.frames.items()):
            if (num - 1) % frame_step:  # consecutive frames are near-duplicates
                continue
            image = cv2.imread(str(image_dir / f"img{num:05d}.jpg"))
            if image is None:
                log.warning("missing frame %s/%d", xml.stem, num)
                continue
            stem = f"{xml.stem}_{num:05d}"
            cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), grey_out(image, seq.ignored))
            lines = yolo_lines(targets, image.shape[1], image.shape[0])
            (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            counts[split] += 1
        if split == "val" and write_videos:
            write_sequence_gt(seq, image_dir, out / "sequences" / xml.stem)
        log.info("%s -> %s", xml.stem, split)

    names_yaml = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names_yaml}\n"
    )
    return counts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--frame-step", type=int, default=5)
    parser.add_argument("--no-videos", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    counts = convert(
        args.images,
        args.annotations,
        args.out,
        args.val_fraction,
        args.frame_step,
        write_videos=not args.no_videos,
    )
    print(f"images: {counts}, data.yaml in {args.out}")


if __name__ == "__main__":
    main()
