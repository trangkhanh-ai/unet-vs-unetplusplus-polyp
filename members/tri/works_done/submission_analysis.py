"""Thống kê ghép cặp và phân tích head; dùng lại evaluator/model của repo."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from members.khanh.works_done.dataset import get_dataloader
from members.huy.works_done.run_utils import PROJECT_ROOT
from .checkpoint_io import (load_checkpoint, checkpoint_manifest, checkpoint_threshold,
                            resolve_device, resolve_project_path, sha256_file, verify_protocol_lock)
from .metrics import binary_mask_metrics


def bootstrap_ci(values, samples=10000, seed=0):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Bootstrap cần ít nhất hai giá trị hữu hạn")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(1)
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


def paired_statistics(first, second):
    # Ghép bằng ID, không dựa vào vị trí dòng trong hai CSV.
    if first.image_id.duplicated().any() or second.image_id.duplicated().any():
        raise ValueError("CSV có ID trùng")
    if set(first.image_id) != set(second.image_id):
        raise ValueError("Hai CSV không cùng tập ảnh")
    paired = first[["image_id", "dice"]].merge(second[["image_id", "dice"]], on="image_id", validate="one_to_one")
    diff = (paired.dice_y - paired.dice_x).to_numpy()
    from scipy.stats import wilcoxon
    low, high = bootstrap_ci(diff)
    std = diff.std(ddof=1)
    return dict(delta=float(diff.mean()), ci_low=low, ci_high=high,
                p_wilcoxon=float(wilcoxon(diff).pvalue) if np.any(diff) else 1.,
                cohen_dz=float(diff.mean() / std) if std > 0 else None,
                images_up=int((diff > 0).sum()), images_down=int((diff < 0).sum()),
                images_equal=int((diff == 0).sum()))


def export_statistics(evaluation_paths, labels, output):
    output = resolve_project_path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Chọn output thống kê mới")
    if len(labels) != len(evaluation_paths) or len(set(labels)) != len(labels):
        raise ValueError("Labels và các folder đánh giá phải tương ứng, không trùng")
    tables = {label: pd.read_csv(Path(path) / "per_image.csv") for label, path in zip(labels, evaluation_paths)}
    comparisons, intervals = {}, {}
    for i, a in enumerate(labels):
        intervals[a] = bootstrap_ci(tables[a].dice)
        for b in labels[i + 1:]:
            comparisons[f"{b}_vs_{a}"] = paired_statistics(tables[a], tables[b])
    result = dict(bootstrap_samples=10000, bootstrap_seed=0, paired_dice=comparisons,
                  dice_confidence_intervals=intervals,
                  scope="Conditional image-level analysis of these fixed checkpoints; not multi-seed uncertainty. Patient/video independence is unverified.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "paired_stats.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return result


@torch.inference_mode()
def evaluate_heads(checkpoint_path, split, output, device="cpu", batch_size=1, protocol_lock=None):
    """Xuất cả 4 head và mean trong MỘT lượt forward; phân tích bổ sung riêng."""
    output = resolve_project_path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Chọn output phân tích head mới")
    device = resolve_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    path = resolve_project_path(checkpoint_path)
    model, checkpoint = load_checkpoint(path, device)
    if not checkpoint["model_kwargs"].get("deep_supervision"):
        raise ValueError("Phân tích head chỉ dành cho U-Net++ deep supervision")
    if split not in {"val", "test"} or batch_size < 1:
        raise ValueError("Cần split val/test và batch size dương")
    if split == "test":
        if not protocol_lock:
            raise ValueError("Phân tích test cần khóa của checkpoint đã chọn trước đó")
        verify_protocol_lock(checkpoint, protocol_lock, path)
    manifest = checkpoint_manifest(checkpoint)
    loader = get_dataloader(split, csv_path=manifest, project_root=PROJECT_ROOT,
                            batch_size=batch_size, img_size=tuple(checkpoint["preprocessing"]["image_size"]),
                            return_id=True, augment=False, shuffle=False)
    threshold = checkpoint_threshold(checkpoint)
    rows = []
    for images, targets, ids in loader:
        probabilities = [logits.sigmoid() for logits in model(images.to(device))]
        policies = {str(i + 1): p for i, p in enumerate(probabilities)}
        policies["mean"] = torch.stack(probabilities).mean(0)
        for policy, p in policies.items():
            metrics = binary_mask_metrics(p >= threshold, targets.to(device).bool())
            for k, image_id in enumerate(ids):
                rows.append(dict(image_id=image_id, head=policy, **{m: float(v[k]) for m, v in metrics.items()}))
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "per_image_heads.csv", index=False)
    means = frame.groupby("head")[["dice", "iou", "precision", "recall"]].mean()
    means.to_csv(output / "heads_summary.csv")
    result = dict(split=split, checkpoint_sha256=sha256_file(path), threshold=threshold,
                  image_count=len(loader.dataset), selected_policy="last (fixed before test)",
                  role="Exploratory reproduction; does not replace the official test results",
                  metrics=means.to_dict(orient="index"))
    (output / "analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
