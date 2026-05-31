#!/usr/bin/env python3
"""Train a lightweight BiSeNetV2-style segmentation baseline on BDD100K masks.

This script intentionally depends only on PyTorch, torchvision, Pillow and NumPy.
It avoids MMCV custom CUDA ops so it can run on newer GPUs once PyTorch supports
the device architecture.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF


BDD_CLASSES = (
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
)


@dataclass
class TrainConfig:
    data_root: str = "/mnt/e/projectbdd/archivemask"
    output_dir: str = "/mnt/e/projectbdd/bdd_bisenetv2_runs/baseline_512x288"
    width: int = 512
    height: int = 288
    num_classes: int = 19
    ignore_index: int = 255
    batch_size: int = 4
    workers: int = 4
    epochs: int = 20
    lr: float = 0.01
    weight_decay: float = 5e-4
    val_every: int = 1
    train_limit: int = 0
    val_limit: int = 0
    seed: int = 42
    amp: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BDDSegDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split: str,
        size: tuple[int, int],
        limit: int = 0,
        augment: bool = False,
    ) -> None:
        self.data_root = data_root
        self.split = split
        self.size = size
        self.augment = augment
        self.image_dir = data_root / "images" / split
        self.label_dir = data_root / "labels" / split

        if not self.image_dir.exists():
            raise FileNotFoundError(self.image_dir)
        if not self.label_dir.exists():
            raise FileNotFoundError(self.label_dir)

        samples: list[tuple[Path, Path]] = []
        for image_path in sorted(self.image_dir.glob("*.jpg")):
            label_path = self.label_dir / f"{image_path.stem}_train_id.png"
            if label_path.exists():
                samples.append((image_path, label_path))

        if limit > 0:
            samples = samples[:limit]
        if not samples:
            raise RuntimeError(f"No paired samples found for split={split}")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, label_path = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path)

        if self.augment and random.random() < 0.5:
            image = TF.hflip(image)
            label = TF.hflip(label)

        image = TF.resize(image, self.size, interpolation=TF.InterpolationMode.BILINEAR)
        label = TF.resize(label, self.size, interpolation=TF.InterpolationMode.NEAREST)

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(
            image_tensor,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        label_tensor = torch.from_numpy(np.array(label, dtype=np.int64))
        return image_tensor, label_tensor


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        padding = kernel // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class DepthwiseSeparableConv(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__(
            ConvBNAct(in_ch, in_ch, kernel=3, stride=stride, groups=in_ch),
            ConvBNAct(in_ch, out_ch, kernel=1, stride=1),
        )


class ContextBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBNAct(channels, channels, kernel=1),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.conv(x)


class LiteBiSeNetV2(nn.Module):
    """A compact two-branch real-time segmentation model inspired by BiSeNetV2."""

    def __init__(self, num_classes: int = 19) -> None:
        super().__init__()
        self.detail = nn.Sequential(
            ConvBNAct(3, 32, stride=2),
            ConvBNAct(32, 32),
            ConvBNAct(32, 64, stride=2),
            ConvBNAct(64, 64),
            ConvBNAct(64, 128, stride=2),
            ConvBNAct(128, 128),
        )
        self.semantic = nn.Sequential(
            ConvBNAct(3, 16, stride=2),
            DepthwiseSeparableConv(16, 32, stride=2),
            DepthwiseSeparableConv(32, 64, stride=2),
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 128, stride=2),
            ContextBlock(128),
        )
        self.semantic_proj = ConvBNAct(128, 128, kernel=1)
        self.fuse = nn.Sequential(
            ConvBNAct(256, 128),
            DepthwiseSeparableConv(128, 128),
            nn.Conv2d(128, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        detail = self.detail(x)
        semantic = self.semantic(x)
        semantic = self.semantic_proj(semantic)
        semantic = F.interpolate(
            semantic, size=detail.shape[-2:], mode="bilinear", align_corners=False
        )
        out = self.fuse(torch.cat([detail, semantic], dim=1))
        return F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)


def fast_hist(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore: int) -> torch.Tensor:
    mask = target != ignore
    pred = pred[mask]
    target = target[mask]
    valid = (target >= 0) & (target < num_classes)
    pred = pred[valid]
    target = target[valid]
    hist = torch.bincount(
        num_classes * target + pred,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)
    return hist


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
) -> dict[str, float]:
    model.eval()
    hist = torch.zeros((num_classes, num_classes), device=device, dtype=torch.float64)
    total_loss = 0.0
    seen = 0
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        pred = logits.argmax(dim=1)
        hist += fast_hist(pred, labels, num_classes, ignore_index)
        total_loss += loss.item() * images.size(0)
        seen += images.size(0)

    intersection = torch.diag(hist)
    union = hist.sum(dim=1) + hist.sum(dim=0) - intersection
    iou = intersection / union.clamp_min(1)
    miou = iou.mean().item()
    pixel_acc = intersection.sum() / hist.sum().clamp_min(1)
    return {
        "loss": total_loss / max(seen, 1),
        "mIoU": miou,
        "pixel_acc": pixel_acc.item(),
    }


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = BDDSegDataset(
        Path(cfg.data_root),
        "train",
        (cfg.height, cfg.width),
        limit=cfg.train_limit,
        augment=True,
    )
    val_ds = BDDSegDataset(
        Path(cfg.data_root),
        "val",
        (cfg.height, cfg.width),
        limit=cfg.val_limit,
        augment=False,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=True,
    )

    model = LiteBiSeNetV2(num_classes=cfg.num_classes).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=0.9,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer,
        total_iters=max(cfg.epochs * len(train_loader), 1),
        power=0.9,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_index)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")

    best_miou = -1.0
    global_step = 0
    print(f"device={device} train={len(train_ds)} val={len(val_ds)} classes={cfg.num_classes}")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        start = time.time()
        running = 0.0
        for step, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=cfg.amp and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running += loss.item()
            global_step += 1
            if step % 50 == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"epoch={epoch:03d} step={step:05d}/{len(train_loader)} "
                    f"loss={running / step:.4f} lr={lr:.6f}"
                )

        print(f"epoch={epoch:03d} train_loss={running / max(len(train_loader), 1):.4f} time={time.time()-start:.1f}s")

        if epoch % cfg.val_every == 0:
            metrics = evaluate(model, val_loader, device, cfg.num_classes, cfg.ignore_index)
            print(
                f"epoch={epoch:03d} val_loss={metrics['loss']:.4f} "
                f"mIoU={metrics['mIoU']:.4f} pixel_acc={metrics['pixel_acc']:.4f}"
            )
            last_path = output_dir / "last.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "classes": BDD_CLASSES,
                    "metrics": metrics,
                    "config": asdict(cfg),
                },
                last_path,
            )
            if metrics["mIoU"] > best_miou:
                best_miou = metrics["mIoU"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "classes": BDD_CLASSES,
                        "metrics": metrics,
                        "config": asdict(cfg),
                    },
                    output_dir / "best.pt",
                )
                print(f"saved best checkpoint: mIoU={best_miou:.4f}")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=TrainConfig.data_root)
    parser.add_argument("--output-dir", default=TrainConfig.output_dir)
    parser.add_argument("--width", type=int, default=TrainConfig.width)
    parser.add_argument("--height", type=int, default=TrainConfig.height)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--workers", type=int, default=TrainConfig.workers)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--train-limit", type=int, default=TrainConfig.train_limit)
    parser.add_argument("--val-limit", type=int, default=TrainConfig.val_limit)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    cfg = TrainConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        width=args.width,
        height=args.height,
        batch_size=args.batch_size,
        workers=args.workers,
        epochs=args.epochs,
        lr=args.lr,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        amp=not args.no_amp,
    )
    return cfg


if __name__ == "__main__":
    train(parse_args())
