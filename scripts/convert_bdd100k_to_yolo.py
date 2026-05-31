#!/usr/bin/env python3
"""Convert BDD100K image labels JSON to Ultralytics YOLO detection labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterator

from PIL import Image


BDD_DET_CLASSES = [
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motor",
    "bike",
    "traffic light",
    "traffic sign",
]


def iter_json_array(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[dict]:
    """Stream a top-level JSON array without loading the full BDD file."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as f:
        buf = ""
        idx = 0
        eof = False
        started = False

        while True:
            if idx >= len(buf) and not eof:
                chunk = f.read(chunk_size)
                eof = chunk == ""
                buf = buf[idx:] + chunk
                idx = 0

            while idx < len(buf) and buf[idx].isspace():
                idx += 1

            if not started:
                if idx >= len(buf):
                    if eof:
                        raise ValueError(f"{path} does not contain a JSON array")
                    continue
                if buf[idx] != "[":
                    raise ValueError(f"{path} must start with a JSON array")
                idx += 1
                started = True
                continue

            while idx < len(buf) and buf[idx].isspace():
                idx += 1

            if idx < len(buf) and buf[idx] == ",":
                idx += 1
                continue

            if idx < len(buf) and buf[idx] == "]":
                break

            try:
                obj, end = decoder.raw_decode(buf, idx)
            except json.JSONDecodeError:
                if eof:
                    raise
                chunk = f.read(chunk_size)
                eof = chunk == ""
                buf = buf[idx:] + chunk
                idx = 0
                continue

            yield obj
            idx = end

            if idx > chunk_size:
                buf = buf[idx:]
                idx = 0


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size


def convert_box_to_yolo(
    box: dict, img_w: int, img_h: int
) -> tuple[float, float, float, float] | None:
    x1 = max(0.0, min(float(box["x1"]), img_w))
    y1 = max(0.0, min(float(box["y1"]), img_h))
    x2 = max(0.0, min(float(box["x2"]), img_w))
    y2 = max(0.0, min(float(box["y2"]), img_h))

    bw = x2 - x1
    bh = y2 - y1
    if bw <= 1.0 or bh <= 1.0:
        return None

    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0
    return xc / img_w, yc / img_h, bw / img_w, bh / img_h


def convert_split(
    split: str,
    images_dir: Path,
    json_path: Path,
    labels_dir: Path,
    class_to_id: dict[str, int],
) -> Counter:
    labels_dir.mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    seen_images: set[str] = set()

    for item in iter_json_array(json_path):
        image_name = item["name"]
        seen_images.add(image_name)
        image_path = images_dir / image_name
        label_path = labels_dir / f"{Path(image_name).stem}.txt"

        if not image_path.exists():
            stats["missing_images"] += 1
            continue

        img_w, img_h = image_size(image_path)
        lines: list[str] = []

        for label in item.get("labels", []):
            category = label.get("category")
            box = label.get("box2d")
            if box is None:
                stats["ignored_non_box_labels"] += 1
                continue
            if category not in class_to_id:
                stats[f"ignored_category:{category}"] += 1
                continue

            yolo_box = convert_box_to_yolo(box, img_w, img_h)
            if yolo_box is None:
                stats["ignored_invalid_boxes"] += 1
                continue

            class_id = class_to_id[category]
            xc, yc, bw, bh = yolo_box
            lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            stats[f"class:{category}"] += 1

        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        stats["images"] += 1
        stats["boxes"] += len(lines)

    for image_path in images_dir.glob("*.jpg"):
        if image_path.name not in seen_images:
            (labels_dir / f"{image_path.stem}.txt").write_text("", encoding="utf-8")
            stats["images_without_json_entry"] += 1

    print(f"[{split}] wrote labels to {labels_dir}")
    for key, value in stats.most_common():
        print(f"[{split}] {key}: {value}")
    return stats


def write_yaml(output_path: Path, archive_root: Path, classes: list[str]) -> None:
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(classes))
    text = f"""# BDD100K detection data for Ultralytics YOLO
path: {archive_root.as_posix()}
train: train/images
val: val/images

names:
{names}
"""
    output_path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, default=Path(r"E:\projectbdd\archive"))
    parser.add_argument("--classes", nargs="+", default=BDD_DET_CLASSES)
    parser.add_argument("--yaml-out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive_root = args.archive_root
    class_to_id = {name: idx for idx, name in enumerate(args.classes)}

    split_cfg = {
        "train": (
            archive_root / "train" / "images",
            archive_root / "train" / "annotations" / "bdd100k_labels_images_train.json",
            archive_root / "train" / "labels",
        ),
        "val": (
            archive_root / "val" / "images",
            archive_root / "val" / "annotations" / "bdd100k_labels_images_val.json",
            archive_root / "val" / "labels",
        ),
    }

    for split, (images_dir, json_path, labels_dir) in split_cfg.items():
        if not images_dir.exists():
            raise FileNotFoundError(images_dir)
        if not json_path.exists():
            raise FileNotFoundError(json_path)
        convert_split(split, images_dir, json_path, labels_dir, class_to_id)

    yaml_out = args.yaml_out or archive_root / "bdd100k_yolo.yaml"
    write_yaml(yaml_out, archive_root, args.classes)
    print(f"[yaml] wrote {yaml_out}")


if __name__ == "__main__":
    main()
