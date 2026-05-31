#!/usr/bin/env python3
"""Fuse BDD100K road segmentation overlay with YOLO detection boxes."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from ultralytics import YOLO

from train_bdd_bisenetv2 import BDD_CLASSES, LiteBiSeNetV2


DET_COLORS = {
    "person": (0, 165, 255),
    "rider": (255, 0, 255),
    "car": (255, 255, 0),
    "truck": (255, 128, 0),
    "bus": (255, 128, 0),
    "train": (180, 180, 180),
    "motor": (255, 0, 255),
    "bike": (255, 0, 255),
    "traffic light": (0, 0, 255),
    "traffic sign": (255, 0, 0),
}


def load_seg_model(ckpt_path: Path, device: torch.device) -> LiteBiSeNetV2:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = LiteBiSeNetV2(num_classes=len(BDD_CLASSES))
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()
    return model


def preprocess_seg(image_rgb: Image.Image, width: int, height: int) -> torch.Tensor:
    image = TF.resize(image_rgb, (height, width), interpolation=TF.InterpolationMode.BILINEAR)
    tensor = TF.to_tensor(image)
    tensor = TF.normalize(
        tensor,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return tensor.unsqueeze(0)


@torch.no_grad()
def predict_seg_mask(
    model: LiteBiSeNetV2,
    image_rgb: Image.Image,
    device: torch.device,
    width: int,
    height: int,
) -> np.ndarray:
    orig_w, orig_h = image_rgb.size
    tensor = preprocess_seg(image_rgb, width, height).to(device)
    logits = model(tensor)
    pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    return cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)


def blend_drivable_overlay(image_bgr: np.ndarray, seg_mask: np.ndarray, alpha: float) -> np.ndarray:
    # BDD/Cityscapes train_id 0 is road. This is a road/drivable approximation.
    road_mask = seg_mask == 0
    overlay = image_bgr.copy()
    overlay[road_mask] = (144, 238, 144)  # BGR light green
    blended = cv2.addWeighted(overlay, alpha, image_bgr, 1.0 - alpha, 0)
    blended[~road_mask] = image_bgr[~road_mask]
    return blended


def draw_yolo_boxes(image_bgr: np.ndarray, result, conf_thres: float) -> np.ndarray:
    names = result.names
    boxes = result.boxes
    if boxes is None:
        return image_bgr

    out = image_bgr.copy()
    for box in boxes:
        conf = float(box.conf[0])
        if conf < conf_thres:
            continue
        cls_id = int(box.cls[0])
        name = names[cls_id]
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        color = DET_COLORS.get(name, (255, 255, 255))
        label = f"{name} {conf:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(y1, th + baseline + 4)
        cv2.rectangle(out, (x1, y_text - th - baseline - 4), (x1 + tw + 4, y_text), color, -1)
        cv2.putText(
            out,
            label,
            (x1 + 2, y_text - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return out


def collect_images(source: Path, limit: int, seed: int) -> list[Path]:
    if source.is_file():
        return [source]
    images = sorted(
        p for p in source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if limit > 0 and len(images) > limit:
        random.Random(seed).shuffle(images)
        images = sorted(images[:limit])
    return images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--seg-ckpt", type=Path, required=True)
    parser.add_argument("--yolo-ckpt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seg_model = load_seg_model(args.seg_ckpt, device)
    yolo_model = YOLO(str(args.yolo_ckpt))
    images = collect_images(args.source, args.limit, args.seed)
    if not images:
        raise RuntimeError(f"No images found in {args.source}")

    for idx, image_path in enumerate(images, start=1):
        image_rgb = Image.open(image_path).convert("RGB")
        image_bgr = cv2.cvtColor(np.array(image_rgb), cv2.COLOR_RGB2BGR)

        seg_mask = predict_seg_mask(seg_model, image_rgb, device, args.width, args.height)
        seg_overlay = blend_drivable_overlay(image_bgr, seg_mask, args.alpha)

        yolo_result = yolo_model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            verbose=False,
            device=0 if device.type == "cuda" else "cpu",
        )[0]
        fused = draw_yolo_boxes(seg_overlay, yolo_result, args.conf)

        out_path = args.out_dir / image_path.name
        cv2.imwrite(str(out_path), fused)
        print(f"[{idx}/{len(images)}] saved {out_path}")


if __name__ == "__main__":
    main()
