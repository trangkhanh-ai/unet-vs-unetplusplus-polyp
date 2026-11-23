"""Evaluate and time actual truncated U-Net++ paths with the locked C checkpoint."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from members.khanh.works_done.dataset import get_dataloader
from members.tri.works_done.checkpoint_io import (
    checkpoint_manifest, checkpoint_threshold, load_checkpoint, resolve_device, verify_protocol_lock,
)
from members.tri.works_done.metrics import binary_metrics


class PrunedHead(torch.nn.Module):
    """Keep exactly the dependency graph for the selected output head."""

    def __init__(self, full_model, level):
        super().__init__()
        if level not in (1, 2, 3, 4) or not full_model.deep_supervision:
            raise ValueError("Expected a deep-supervised model and a level from 1 to 4.")
        self.level = level
        self.pool = full_model.pool
        self.encoder = torch.nn.ModuleList(getattr(full_model, f"conv{i}_0") for i in range(level + 1))
        self.columns = torch.nn.ModuleList(
            torch.nn.ModuleList(getattr(full_model, f"conv{i}_{j}") for i in range(level - j + 1))
            for j in range(1, level + 1)
        )
        self.head = full_model.heads[level - 1]

    def forward(self, image):
        grid = [[self.encoder[0](image)]]
        for block in self.encoder[1:]:
            grid.append([block(self.pool(grid[-1][0]))])
        for column, blocks in enumerate(self.columns, start=1):
            for row, block in enumerate(blocks):
                grid[row].append(block(grid[row + 1][column - 1], *grid[row]))
        return self.head(grid[0][-1])


@torch.inference_mode()
def benchmark(output, requested_device, repeats):
    if repeats < 3:
        raise ValueError("Use at least three repeats for a median.")
    output.mkdir(parents=True, exist_ok=False)
    lock_path = ROOT / "configs/protocol_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    entry = next(item for item in lock["experiments"] if item["label"] == "C")
    device = resolve_device(requested_device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint_path = ROOT / entry["checkpoint_path"].replace(chr(92), "/")
    full, checkpoint = load_checkpoint(checkpoint_path, device)
    verified = verify_protocol_lock(checkpoint, lock_path, checkpoint_path)
    if verified["label"] != "C":
        raise ValueError("The benchmark requires locked checkpoint C.")
    manifest = checkpoint_manifest(checkpoint)
    threshold = checkpoint_threshold(checkpoint)
    variants = [PrunedHead(full, level).eval() for level in range(1, 5)]
    measures = []
    maximum_error = np.zeros(4)
    reference_errors = np.zeros(4, dtype=np.int64)
    batch_size = 8
    for split in ("val", "test"):
        loader = get_dataloader(split, csv_path=manifest, project_root=ROOT, batch_size=batch_size,
                                img_size=tuple(checkpoint["preprocessing"]["image_size"]),
                                return_id=True, augment=False, shuffle=False)
        ids_seen = set()
        for images, targets, ids in loader:
            if ids_seen.intersection(ids):
                raise ValueError("Duplicate image IDs in evaluation.")
            ids_seen.update(ids)
            images, targets = images.to(device), targets.to(device)
            references = full(images)
            for index, variant in enumerate(variants):
                logits = variant(images)
                maximum_error[index] = max(maximum_error[index], float((logits - references[index]).abs().max()))
                reference_errors[index] += int(torch.count_nonzero((logits >= 0) != (references[index] >= 0)))
                for image_id, score in zip(ids, binary_metrics(logits, targets, threshold)["dice"].cpu().tolist()):
                    measures.append({"split": split, "image_id": image_id, "level": index + 1, "dice": score})
        if len(ids_seen) != 150:
            raise ValueError(f"Expected 150 images in {split}.")
    if maximum_error.max() > 1e-5 or reference_errors.any():
        raise AssertionError((maximum_error.tolist(), reference_errors.tolist()))
    del checkpoint, targets, references, logits
    metrics = pd.DataFrame(measures)
    metrics.to_csv(output / "pruning_per_image.csv", index=False)

    # Fixed fp32 batch, warm each path and rotate the measurement order.
    images = images[:1].repeat(batch_size, 1, 1, 1)
    for variant in variants:
        for _ in range(10):
            variant(images)
    times = [[] for _ in variants]
    timing_steps = 12
    for repeat in range(repeats):
        for offset in range(len(variants)):
            index = (repeat + offset) % len(variants)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = perf_counter()
            for _ in range(timing_steps):
                variants[index](images)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times[index].append(1000 * (perf_counter() - start) / timing_steps / batch_size)

    means = metrics.pivot_table(index="level", columns="split", values="dice")
    rows = []
    for index, variant in enumerate(variants):
        level = index + 1
        rows.append({
            "level": level, "nodes": (level + 1) * (level + 2) // 2,
            "parameters": sum(p.numel() for p in variant.parameters()),
            "dice_val": float(means.loc[level, "val"]), "dice_test": float(means.loc[level, "test"]),
            "median_ms_per_image": float(np.median(times[index])),
            "min_ms_per_image": min(times[index]), "max_ms_per_image": max(times[index]),
            "max_abs_logit_delta": float(maximum_error[index]), "mask_pixel_mismatches": int(reference_errors[index]),
        })
    pd.DataFrame(rows).to_csv(output / "pruning_summary.csv", index=False)
    report = {
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch_version": str(torch.__version__), "batch_size": batch_size, "threshold": threshold,
        "checkpoint_sha256": entry["checkpoint_sha256"], "manifest_sha256": lock["manifest_sha256"],
        "lock_sha256": verified["lock_sha256"], "repeats": repeats, "forwards_per_repeat": timing_steps,
        "timings_ms_per_image": times, "rows": rows,
        "scope": "FP32 forward only on a fixed resident batch; each truncated path runs one head. Validation and test Dice use 150 distinct images each.",
    }
    details = json.dumps(report, indent=2)
    (output / "pruning_summary.json").write_text(details, encoding="utf-8")
    print(details)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    benchmark(args.output, args.device, args.repeats)
