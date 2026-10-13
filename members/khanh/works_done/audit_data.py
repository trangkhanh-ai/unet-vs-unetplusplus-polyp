"""Kiểm tra chất lượng/EDA từ manifest cố định; không tạo split mới."""
import argparse
import hashlib
import json
from pathlib import Path
from collections import defaultdict

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .dataset import PROJECT_ROOT, DEFAULT_MANIFEST, read_image_unicode, validate_manifest, get_transforms


def audit_dataset(output_dir, manifest=DEFAULT_MANIFEST):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = Path(manifest)
    frame = pd.read_csv(manifest)
    validate_manifest(frame)
    issues, measurements = [], []
    decoded_hashes = defaultdict(list)
    for row in frame.itertuples(index=False):
        image_path = PROJECT_ROOT / row.image_path.replace("\\", "/")
        mask_path = PROJECT_ROOT / row.mask_path.replace("\\", "/")
        image = read_image_unicode(image_path, cv2.IMREAD_COLOR)
        mask = read_image_unicode(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            issues.append({"image_id": row.image_id, "error": "unreadable"})
            continue
        if image.shape[:2] != mask.shape:
            issues.append({"image_id": row.image_id, "error": "shape_mismatch"})
            continue
        ratio = float((mask > 127).mean())
        actual_group = "empty" if ratio == 0 else "small" if ratio < .05 else "medium" if ratio < .20 else "large"
        if (image.shape[1], image.shape[0]) != (row.width, row.height):
            issues.append({"image_id": row.image_id, "error": "metadata_dimensions"})
        if abs(ratio - row.polyp_area_ratio) > 1e-6 or row.polyp_category != actual_group:
            issues.append({"image_id": row.image_id, "error": "metadata_area"})
        # Shape cũng tham gia hash, tránh cùng byte nhưng khác hình.
        digest = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
        decoded_hashes[digest].append({"image_id": row.image_id, "split": row.split})
        measurements.append({"image_id": row.image_id, "split": row.split,
                             "width": image.shape[1], "height": image.shape[0],
                             "polyp_area_ratio": ratio, "polyp_category": actual_group,
                             "decoded_image_sha256": digest})
    groups = [values for values in decoded_hashes.values() if len(values) > 1]
    cross_split = [values for values in groups if len({x["split"] for x in values}) > 1]
    data = pd.DataFrame(measurements)
    counts = frame["split"].value_counts().to_dict()
    if counts != {"train": 700, "val": 150, "test": 150}:
        issues.append({"error": "unexpected_holdout_counts", "counts": counts})
    if cross_split:
        issues.append({"error": "exact_duplicates_across_split", "groups": cross_split})
    image_ids = {p.stem for p in (PROJECT_ROOT / "Kvasir-SEG/images").glob("*.jpg")}
    mask_ids = {p.stem for p in (PROJECT_ROOT / "Kvasir-SEG/masks").glob("*.jpg")}
    if image_ids != mask_ids or image_ids != set(frame["image_id"]):
        issues.append({"error": "folder_manifest_ID_mismatch"})
    report = {
        "status": "passed" if not issues else "failed",
        "scope": "data quality audit; no model evaluation or tuning",
        "manifest": str(manifest.relative_to(PROJECT_ROOT)),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "pairs": len(measurements), "split_counts": counts,
        "area_group_counts": pd.crosstab(data["split"], data["polyp_category"]).to_dict(),
        "width_range": [int(data["width"].min()), int(data["width"].max())],
        "height_range": [int(data["height"].min()), int(data["height"].max())],
        "empty_masks": int((data["polyp_area_ratio"] == 0).sum()),
        "exact_duplicate_groups": groups, "issues": issues,
        "limitations": [
            "Near-duplicate similarity and patient/video independence are not verified",
            "Mask area groups are image-area ratios, not clinical polyp size",
            "Masks are JPEG; binary label is grayscale >127",
        ],
    }
    (output_dir / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    data.to_csv(output_dir / "image_metadata.csv", index=False)
    counts_table = pd.crosstab(data["split"], data["polyp_category"]).reindex(["train", "val", "test"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    counts_table.plot.bar(ax=axes[0], title="Fixed holdout counts", rot=0)
    axes[1].hist(data["polyp_area_ratio"], bins=30, color="#207e8b")
    axes[1].set(title="Foreground area fraction", xlabel="GT mask area / image area", ylabel="Images")
    axes[2].scatter(data["width"], data["height"], s=6, alpha=.4)
    axes[2].set(title="Original resolutions", xlabel="Width", ylabel="Height")
    fig.savefig(output_dir / "data_overview.png", dpi=140)
    plt.close(fig)
    # Preview chọn từ TRAIN, gồm các nhóm diện tích, trước/sau transform.
    selected = pd.concat([frame.loc[(frame.split == "train") & (frame.polyp_category == category)].head(2)
                          for category in ("small", "medium", "large")])
    transform = get_transforms("train", seed=42)
    fig, axes = plt.subplots(len(selected), 4, figsize=(10, 2.1 * len(selected)), squeeze=False)
    for index, row in enumerate(selected.itertuples(index=False)):
        rgb = cv2.cvtColor(read_image_unicode(PROJECT_ROOT / row.image_path.replace("\\", "/"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask = (read_image_unicode(PROJECT_ROOT / row.mask_path.replace("\\", "/"), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        sample = transform(image=rgb, mask=mask)
        aug_image = sample["image"].permute(1, 2, 0).numpy()
        views = [rgb, mask, aug_image, sample["mask"].numpy()]
        for col, view in enumerate(views):
            axes[index, col].imshow(view, cmap="gray" if col in (1, 3) else None)
            axes[index, col].axis("off")
        axes[index, 0].set_title(f"{row.image_id}\n{row.polyp_category}", fontsize=7)
    for col, title in enumerate(("TRAIN RGB", "GT mask", "Augmented RGB", "Augmented GT")):
        axes[0, col].set_xlabel(title)
        axes[0, col].xaxis.set_label_position("top")
        axes[0, col].xaxis.set_visible(True)
    fig.suptitle("Data pipeline preview — fixed TRAIN IDs only", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "train_augmentation_preview.png", dpi=130)
    plt.close(fig)
    print(json.dumps({"status": report["status"], "pairs": report["pairs"], "split_counts": counts, "issues": issues}, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/data_audit")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    result = audit_dataset(args.output, args.manifest)
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
