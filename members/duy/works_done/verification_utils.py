"""Helpers for Duy's diagnostic checks, NOT the team's official trainer/metrics.

Huy and Tri own the production loss/metrics. These small reference formulas
exist so the baseline can be verified before those modules are delivered.
"""

from __future__ import annotations

import csv
import hashlib
import os
import platform
import random
import subprocess
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = ROOT / "members/khanh/works_done/dataset_splits.csv"


def seed_everything(seed: int, threads: int = 4) -> None:
    if seed < 0 or threads < 1:
        raise ValueError("seed must be >=0 and threads must be >=1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def device_from_name(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch environment has no available CUDA device")
    return torch.device(name)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance() -> dict:
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    return {"created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_base_commit": git("rev-parse", "HEAD"),
            "source_branch": git("branch", "--show-current"),
            "source_dirty": bool(git("status", "--porcelain")),
            "python": platform.python_version(), "torch": str(torch.__version__),
            "platform": platform.platform(),
            "khanh_dataset_sha256": file_sha256(ROOT / "members/khanh/works_done/dataset.py"),
            "source_sha256": {p.name: file_sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}


def diagnostic_loss(logits: Tensor, masks: Tensor, smooth: float = 1e-6) -> Tensor:
    """0.5 BCE + 0.5 per-image soft Dice; never threshold during training."""
    if logits.shape != masks.shape:
        raise ValueError(f"Logits and masks must have equal shapes, got {logits.shape} / {masks.shape}")
    probs = logits.sigmoid()
    axes = (1, 2, 3)
    overlap = (probs * masks).sum(dim=axes)
    soft_dice = (2 * overlap + smooth) / (probs.sum(dim=axes) + masks.sum(dim=axes) + smooth)
    return 0.5 * nn.functional.binary_cross_entropy_with_logits(logits, masks) + 0.5 * (1 - soft_dice).mean()


def diagnostic_scores(logits: Tensor, masks: Tensor) -> dict:
    """Per-image binary Dice/IoU, including the both-empty convention."""
    pred = logits.sigmoid() >= 0.5
    truth = masks >= 0.5
    axes = (1, 2, 3)
    intersection = (pred & truth).sum(dim=axes).float()
    pred_area, true_area = pred.sum(dim=axes).float(), truth.sum(dim=axes).float()
    denominator = pred_area + true_area
    union = denominator - intersection
    dice = torch.where(denominator == 0, torch.ones_like(denominator), 2 * intersection / denominator.clamp_min(1))
    iou = torch.where(union == 0, torch.ones_like(union), intersection / union.clamp_min(1))
    return {"dice": dice.cpu().tolist(), "iou": iou.cpu().tolist(),
            "mean_dice": dice.mean().item(), "mean_iou": iou.mean().item()}


def fixed_train_batch(count: int, image_size: int, seed: int, manifest: Path = DEFAULT_MANIFEST):
    """Reuse Khanh's Dataset; read only selected train images, never val/test.

    Disable augmentation on this LOCAL dataset for a memorization diagnostic.
    Capture and report upstream constructor warnings; do not edit Khanh's code.
    """
    from members.khanh.works_done.dataset import KvasirDataset, get_transforms

    manifest = manifest.resolve()
    with manifest.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    counts = Counter(row["split"] for row in rows)
    if counts != Counter({"train": 700, "val": 150, "test": 150}):
        raise ValueError(f"Expected the agreed 700/150/150 manifest; got {dict(counts)}")
    ids = [row["image_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Manifest contains duplicate image IDs; fix before using the baseline")
    if count < 1 or count > counts["train"]:
        raise ValueError("count must be between 1 and 700")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dataset = KvasirDataset(csv_path=str(manifest), split="train", project_root=str(ROOT),
                               img_size=(image_size, image_size))
        dataset.transform = get_transforms("val", (image_size, image_size))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()
    samples = [dataset[i] for i in indices]
    images = torch.stack([x for x, _ in samples])
    masks = torch.stack([y for _, y in samples])
    if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError("Dataset images must be finite normalized RGB [0,1]")
    if not torch.isfinite(masks).all() or not ((masks == 0) | (masks == 1)).all():
        raise ValueError("Dataset masks must be finite and binary")
    chosen = dataset.df.iloc[indices]
    metadata = {"split": "train", "split_counts": dict(counts), "dataset_indices": indices,
                "image_ids": chosen["image_id"].tolist(),
                "manifest_relative_path": os.path.relpath(manifest, ROOT).replace("\\", "/"),
                "manifest_sha256": file_sha256(manifest),
                "preprocessing": "Khanh get_transforms('val') applied ONLY to selected TRAIN rows; no augmentation",
                "validation_or_test_images_read": 0,
                "upstream_warnings": sorted(set(str(w.message) for w in caught))}
    return images, masks, metadata
