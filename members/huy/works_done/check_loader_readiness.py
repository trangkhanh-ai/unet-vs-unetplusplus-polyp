"""Measure and verify the team's actual train/val loader without training a model."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time
import warnings

sys.dont_write_bytecode = True
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def group_seed():
    source = ROOT / "members/khanh/works_done/prepare_data_splits.py"
    tree = ast.parse(source.read_text(encoding="utf-8-sig"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "create_stratified_splits")
    seed = ast.literal_eval(function.args.defaults[-1])
    if not isinstance(seed, int):
        raise ValueError("Group's split seed is not an integer literal.")
    return seed


def digest_tensor(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def run(output):
    import cv2
    import numpy as np
    import pandas as pd
    import torch
    import albumentations as A
    from members.khanh.works_done.dataset import get_dataloaders

    seed = group_seed()
    torch.set_num_threads(2)
    dependencies = {name: importlib.metadata.version(name) for name in (
        "torch", "numpy", "pandas", "opencv-python-headless", "albumentations", "matplotlib")}
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable, "python": platform.python_version(), "platform": platform.platform(),
        "processor": platform.processor(), "modules": {"torch": str(torch.__version__), "cv2": cv2.__version__,
        "numpy": np.__version__, "pandas": pd.__version__, "albumentations": A.__version__},
        "direct_audit_packages": dependencies, "cuda_available": torch.cuda.is_available(),
        "audit_settings": {"batch_size": 4, "image_size": [256, 256], "num_workers": 0,
            "torch_cpu_threads": torch.get_num_threads(), "opencv_threads": cv2.getNumThreads(),
            "seed": seed, "seed_source": "create_stratified_splits default in group's script",
            "test_loader_iterated": False, "model_used": False,
            "batch_size_source": "Khanh's existing handover smoke check"},
    }

    def reset_global_seed():
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def build(seed_augmentation):
        reset_global_seed()
        train, val, _ = get_dataloaders(batch_size=4, num_workers=0)
        if seed_augmentation:
            train.dataset.transform.set_random_seed(seed)
        return train, val

    def sample_hashes(seed_augmentation, batches):
        loader, _ = build(seed_augmentation)
        iterator = iter(loader)
        result = []
        for _ in range(batches):
            images, masks = next(iterator)
            result.append({"images": digest_tensor(images), "masks": digest_tensor(masks)})
        return result

    # Read-only sampler check: uses the actual sampler and records its real dataset IDs.
    orders = []
    for _ in range(2):
        train, _ = build(False)
        indices = list(iter(train.sampler))
        orders.append([str(train.dataset.df.iloc[i].image_id) for i in indices])
    report["sampler"] = {"samples": len(orders[0]), "same_order_after_global_seed_reset": orders[0] == orders[1],
        "order_sha256": [hashlib.sha256("\n".join(order).encode()).hexdigest() for order in orders],
        "first_ids": orders[0][:8],
        "scope": "Direct sampler iteration; recorded separately from batch iteration's base-seed consumption."}

    global_a, global_b = sample_hashes(False, 2), sample_hashes(False, 2)
    report["global_seed_only"] = {"batches_per_run": 2, "samples_per_run": 8,
        "all_batches_equal": global_a == global_b, "run_a": global_a, "run_b": global_b}
    print(f"Global seeds only, identical first 2 batches: {global_a == global_b}", flush=True)

    # Full train pass checks all 700 actual images in order, including existing augmentations.
    def full_digest():
        loader, _ = build(True)
        image_hash, mask_hash = hashlib.sha256(), hashlib.sha256()
        count = 0
        for images, masks in loader:
            image_hash.update(images.contiguous().numpy().tobytes())
            mask_hash.update(masks.contiguous().numpy().tobytes())
            count += len(images)
        return {"samples": count, "images_sha256": image_hash.hexdigest(), "masks_sha256": mask_hash.hexdigest()}

    full_a, full_b = full_digest(), full_digest()
    report["global_and_compose_seed"] = {"run_a": full_a, "run_b": full_b,
        "all_samples_equal": full_a == full_b,
        "scope": "Fresh loaders, same process/environment, workers=0. Not model or checkpoint reproducibility."}
    print(f"Global + Compose seed, identical full train pass: {full_a == full_b}", flush=True)
    if full_a != full_b or orders[0] != orders[1]:
        raise RuntimeError("Seeded loader replay did not match; do not claim reproducibility.")

    # Timing is separate from the above hashing so digest cost is not included.
    rows = []
    for split_index, split in enumerate(("train", "val")):
        warm_loader = build(True)[split_index]
        next(iter(warm_loader))
        for repetition in range(1, 4):
            loader = build(True)[split_index]
            samples, timings = 0, []
            start = time.perf_counter()
            iterator = iter(loader)
            while True:
                before = time.perf_counter()
                try:
                    images, masks = next(iterator)
                except StopIteration:
                    break
                timings.append(time.perf_counter() - before)
                samples += len(images)
            duration = time.perf_counter() - start
            rows.append({"split": split, "repetition": repetition, "samples": samples,
                "batches": len(timings), "total_seconds": duration, "images_per_second": samples/duration,
                "median_batch_ms": 1000*statistics.median(timings),
                "p95_batch_ms": 1000*float(np.quantile(timings, .95))})
        print(f"{split}: 3 loader-only timing passes complete", flush=True)
    report["loader_timing"] = {"measurements": rows,
        "scope": "CPU disk decode + existing transforms + collation. No model, GPU transfer, backward, loss or test. Warm OS cache likely; not cold-start or training throughput."}
    report["effective_transforms"] = {"train": train.dataset.transform.to_dict()}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = run(output)
        report["warnings"] = sorted({f"{w.category.__name__}: {w.message}" for w in caught})
    (output / "loader_readiness.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Evidence written to {output}", flush=True)


if __name__ == "__main__":
    main()
