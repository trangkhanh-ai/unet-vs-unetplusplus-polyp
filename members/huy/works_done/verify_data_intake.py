"""Audit the existing Kvasir manifest and Khanh's actual train/val loaders.

No synthetic data, model, split generation, training or test-set scoring.
Run from the repository root with --output pointing to a NEW evidence folder.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import warnings

sys.dont_write_bytecode = True
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
MANIFEST = ROOT / "members/khanh/works_done/dataset_splits.csv"
LOADER_SOURCE = ROOT / "members/khanh/works_done/dataset.py"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def resolve_data_path(value):
    relative = Path(value.replace("\\", "/"))
    path = (ROOT / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(ROOT):
        raise ValueError(f"Path outside repository: {value}")
    return path


def audit_manifest(pd, cv2, np):
    frame = pd.read_csv(MANIFEST)
    required = {"image_id", "image_path", "mask_path", "split", "polyp_area_ratio",
                "polyp_category", "kfold_5", "width", "height"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("Missing manifest columns or empty manifest.")
    if frame[list(required)].isna().any().any():
        raise ValueError("Missing values in required manifest fields.")
    issues = []
    duplicates = frame[frame.image_id.duplicated(keep=False)].image_id.tolist()
    if duplicates:
        issues.append({"duplicate_ids": duplicates})
    if set(frame.split) != {"train", "val", "test"}:
        issues.append({"unexpected_split_labels": sorted(set(frame.split))})
    split_ids = {name: set(frame.loc[frame.split == name, "image_id"]) for name in ("train", "val", "test")}
    overlaps = {f"{a}/{b}": sorted(split_ids[a] & split_ids[b])
                for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    if any(overlaps.values()):
        issues.append({"split_overlap": overlaps})
    hashes, pixel_hashes = defaultdict(list), defaultdict(list)
    counts = Counter()
    for row in frame.itertuples(index=False):
        try:
            image_path, mask_path = resolve_data_path(row.image_path), resolve_data_path(row.mask_path)
            if image_path.stem != row.image_id or mask_path.stem != row.image_id:
                raise ValueError("Image/mask filenames do not match ID.")
            image = cv2.imread(str(image_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None:
                raise ValueError("Unreadable image/mask.")
            if image.shape[:2] != mask.shape or image.shape[:2] != (row.height, row.width):
                raise ValueError("Image/mask/manifest dimensions disagree.")
            binary = mask > 127  # Identical threshold to Khanh's current code.
            ratio = float(binary.mean())
            category = "empty" if not binary.any() else "small" if ratio < .05 else "medium" if ratio < .20 else "large"
            if abs(ratio - row.polyp_area_ratio) > 0.000000501:
                raise ValueError("Area ratio differs from manifest beyond six-decimal rounding.")
            if category != row.polyp_category:
                raise ValueError("Area category differs from current group thresholds.")
            counts["valid_pairs"] += 1
            counts["empty_masks"] += int(not binary.any())
            hashes[sha256(image_path)].append({"id": row.image_id, "split": row.split})
            pixel_key = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
            pixel_hashes[pixel_key].append({"id": row.image_id, "split": row.split})
        except Exception as error:
            issues.append({"image_id": str(row.image_id), "error": str(error)})
    expected = {
        kind: {resolve_data_path(value) for value in frame[f"{kind}_path"]}
        for kind in ("image", "mask")
    }
    unlisted = {}
    for kind, directory in (("image", "images"), ("mask", "masks")):
        found = {p.resolve() for p in (ROOT / "Kvasir-SEG" / directory).iterdir()
                 if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
        unlisted[kind] = sorted(str(p.relative_to(ROOT)) for p in found - expected[kind])
    byte_duplicates = [items for items in hashes.values() if len(items) > 1]
    decoded_duplicates = [items for items in pixel_hashes.values() if len(items) > 1]
    cross_split_duplicates = [items for items in decoded_duplicates if len({i["split"] for i in items}) > 1]
    if cross_split_duplicates:
        issues.append({"exact_duplicate_images_cross_splits": cross_split_duplicates})
    return frame, {
        "rows": len(frame), "split_counts": {k: int(v) for k, v in frame.split.value_counts().items()},
        "category_counts": pd.crosstab(frame.split, frame.polyp_category).to_dict(orient="index"),
        "fold_counts": {str(k): int(v) for k, v in frame.kfold_5.value_counts().sort_index().items()},
        "duplicate_ids": duplicates, "split_overlaps": overlaps, "unlisted_files": unlisted,
        "raw_file_checks": dict(counts), "byte_duplicate_image_groups": byte_duplicates,
        "decoded_duplicate_image_groups": decoded_duplicates,
        "issues": issues, "limitation": "Exact duplicates only; no near-duplicate or patient/video independence verification.",
    }


def check_batches(loader, expected_ids, torch):
    if set(loader.dataset.df.image_id) != expected_ids:
        raise ValueError("Loader membership differs from manifest.")
    seen, batches = 0, 0
    minimum, maximum = 1.0, 0.0
    observed_mask_values, empty_masks = set(), 0
    first_batch = None
    for images, masks in loader:
        batch = len(images)
        if tuple(images.shape) != (batch, 3, 256, 256) or tuple(masks.shape) != (batch, 1, 256, 256):
            raise ValueError("Unexpected batch dimensions.")
        if images.dtype != torch.float32 or masks.dtype != torch.float32:
            raise ValueError("Expected float32 images and masks.")
        if not torch.isfinite(images).all() or not torch.isfinite(masks).all():
            raise ValueError("Non-finite batch.")
        if not ((masks == 0) | (masks == 1)).all():
            raise ValueError("Non-binary transformed mask.")
        low, high = images.min().item(), images.max().item()
        if low < -1e-6 or high > 1 + 1e-6:
            raise ValueError("Images outside expected [0,1] range.")
        minimum, maximum = min(minimum, low), max(maximum, high)
        observed_mask_values.update(masks.unique().tolist())
        empty_masks += int((masks.flatten(1).sum(1) == 0).sum())
        if first_batch is None:
            first_batch = {"image_shape": list(images.shape), "mask_shape": list(masks.shape)}
        seen += batch
        batches += 1
    if seen != len(expected_ids):
        raise ValueError("Loader did not visit expected number of samples.")
    return {"samples": seen, "batches": batches, "first_batch": first_batch,
            "image_dtype": "float32", "mask_dtype": "float32",
            "image_range": [minimum, maximum], "mask_values": sorted(observed_mask_values),
            "empty_masks_after_transform": empty_masks,
            "sampler": type(loader.sampler).__name__, "drop_last": loader.drop_last}


def inspect_val_reference(dataset, cv2, np, torch):
    # Real first validation sample only; independently verify channel order/resize/mask processing.
    row = dataset.df.iloc[0]
    a, ma = dataset[0]
    b, mb = dataset[0]
    original = cv2.cvtColor(cv2.imread(str(resolve_data_path(row.image_path))), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(resolve_data_path(row.mask_path)), cv2.IMREAD_GRAYSCALE)
    image_ref = cv2.resize(original, (256, 256), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
    mask_ref = cv2.resize((mask > 127).astype(np.float32), (256, 256), interpolation=cv2.INTER_NEAREST)
    image_error = float(np.abs(a.permute(1, 2, 0).numpy() - image_ref).max())
    repeat_equal = torch.equal(a, b) and torch.equal(ma, mb)
    mask_equal = np.array_equal(ma[0].numpy(), mask_ref)
    if not repeat_equal or not mask_equal or image_error > 1e-6:
        raise ValueError("Validation preprocessing differs from the expected actual source pipeline.")
    return {"image_id": str(row.image_id), "repeat_identical": repeat_equal,
            "rgb_resize_normalization_max_abs_error": image_error, "nearest_mask_matches": mask_equal}


def export_real_train_preview(dataset, output, cv2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    count = min(4, len(dataset))
    figure, axes = plt.subplots(count, 4, figsize=(12, 3 * count), squeeze=False)
    ids = []
    for i in range(count):
        row = dataset.df.iloc[i]
        ids.append(str(row.image_id))
        original = cv2.cvtColor(cv2.imread(str(resolve_data_path(row.image_path))), cv2.COLOR_BGR2RGB)
        original_mask = cv2.imread(str(resolve_data_path(row.mask_path)), cv2.IMREAD_GRAYSCALE) > 127
        transformed, transformed_mask = dataset[i]
        axes[i, 0].imshow(original)
        axes[i, 1].imshow(original_mask, cmap="gray", vmin=0, vmax=1)
        axes[i, 2].imshow(transformed.permute(1, 2, 0).numpy())
        axes[i, 3].imshow(transformed_mask[0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_title(str(row.image_id), fontsize=7)
        for j in range(4):
            axes[i, j].axis("off")
    for ax, title in zip(axes[0], ("Original image", "Original mask", "Khanh train transform", "Transformed mask")):
        ax.text(.5, 1.16, title, transform=ax.transAxes, ha="center", fontsize=10)
    figure.suptitle("Actual Kvasir-SEG train samples - existing group pipeline", y=.995)
    figure.tight_layout(rect=(0, 0, 1, .97))
    figure.savefig(output / "real_train_preview.png", dpi=140)
    plt.close(figure)
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "repository": str(ROOT), "commit": git("rev-parse", "HEAD"),
              "branch": git("branch", "--show-current"), "git_status_before": git("status", "--short"),
              "python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
              "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (MANIFEST, LOADER_SOURCE)},
              "scope": "Existing data and train/val loader audit. Test metadata/file integrity only; no model or test scoring."}
    started = time.perf_counter()
    exit_code = 0
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            import cv2
            import numpy as np
            import pandas as pd
            import torch
            import albumentations as A
            from members.khanh.works_done.dataset import get_dataloaders
            report["modules"] = {"torch": str(torch.__version__), "numpy": np.__version__,
                                 "pandas": pd.__version__, "cv2": cv2.__version__, "albumentations": A.__version__}
            report["cuda_available"] = torch.cuda.is_available()
            report["cuda_build"] = torch.version.cuda
            torch.set_num_threads(2)  # Audit process resource limit, not a training setting.
            frame, report["manifest"] = audit_manifest(pd, cv2, np)
            print("Manifest audit complete", flush=True)
            if report["manifest"]["issues"]:
                raise ValueError("Manifest integrity issues; inspect report before accepting data.")
            train, val, test = get_dataloaders(batch_size=4, num_workers=0)
            report["audit_settings"] = {"batch_size": 4, "source": "Khanh's __main__ handover check",
                                         "training_batch_size_decided": False,
                                         "test_loader_iterated": False,
                                         "augmentation": "Unmodified existing pipeline; no new dataset generated."}
            # Group's manifest preparation uses seed 42. This seeds the audit order/transforms only.
            torch.manual_seed(42)
            train.dataset.transform.set_random_seed(42)
            report["audit_settings"]["audit_seed_from_group_manifest_script"] = 42
            report["effective_transforms"] = {"train": train.dataset.transform.to_dict(), "val": val.dataset.transform.to_dict()}
            report["loader"] = {}
            for split, loader in (("train", train), ("val", val)):
                ids = set(frame.loc[frame.split == split, "image_id"])
                report["loader"][split] = check_batches(loader, ids, torch)
                print(f"{split}: {report['loader'][split]['samples']} actual samples passed", flush=True)
            report["val_reference"] = inspect_val_reference(val.dataset, cv2, np, torch)
            report["preview_train_ids"] = export_real_train_preview(train.dataset, output, cv2)
            report["status"] = "technical_checks_passed_review_warnings"
        except Exception as error:
            report["status"] = "failed"
            report["error"] = f"{type(error).__name__}: {error}"
            exit_code = 1
        report["warnings"] = sorted({f"{w.category.__name__}: {w.message}" for w in captured})
    report["elapsed_seconds"] = time.perf_counter() - started
    report["source_unchanged"] = all(sha256(ROOT / name) == digest for name, digest in report["source_sha256"].items())
    if not report["source_unchanged"]:
        report["status"] = "failed_source_changed_during_audit"
        exit_code = 1
    save_json(output / "audit.json", report)
    packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())
    (output / "environment_packages.txt").write_text("\n".join(packages) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "error": report.get("error"),
                      "warnings": report["warnings"], "output": str(output)}, ensure_ascii=True, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
