"""Screen fixed Kvasir-SEG splits for visually similar image pairs."""
import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from members.khanh.works_done.dataset import DEFAULT_MANIFEST, read_image_unicode, validate_manifest
from members.tri.works_done.checkpoint_io import sha256_file


def perceptual_hash(gray):
    """Pack 63 low-frequency DCT comparisons into an integer, excluding DC."""
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    coefficients = cv2.dct(small.astype(np.float32))[:8, :8].flatten()[1:]
    bits = np.packbits(coefficients > np.median(coefficients))
    return int.from_bytes(bits.tobytes(), "big")


def audit(output):
    manifest = pd.read_csv(DEFAULT_MANIFEST).sort_values("image_id")
    validate_manifest(manifest)
    images = {}
    hashes = {}
    for row in manifest.itertuples():
        path = ROOT / row.image_path.replace(chr(92), "/")
        gray = read_image_unicode(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(path)
        images[row.image_id] = path
        hashes[row.image_id] = (row.split, perceptual_hash(gray))

    cutoff = 6
    pairs = []
    checked = Counter()
    for (left_id, (left_split, left_hash)), (right_id, (right_split, right_hash)) in combinations(hashes.items(), 2):
        if left_split == right_split:
            continue
        split_pair = "/".join(sorted((left_split, right_split)))
        checked[split_pair] += 1
        distance = (left_hash ^ right_hash).bit_count()
        if distance <= cutoff:
            pairs.append((left_id, right_id, split_pair, distance))
    pairs.sort(key=lambda row: (row[-1], row[0], row[1]))
    output.mkdir(parents=True, exist_ok=False)
    table = pd.DataFrame(pairs, columns=["left_id", "right_id", "split_pair", "hamming_distance"])
    table.to_csv(output / "similarity_candidates.csv", index=False)
    test_ids = set(manifest.loc[manifest.split.eq("test"), "image_id"])
    flagged = test_ids & (set(table.left_id) | set(table.right_id))
    sensitivity = []
    for model in "ABC":
        scores = pd.read_csv(ROOT / f"results/{model}_test/per_image.csv")
        if set(scores.image_id) != test_ids or not scores.image_id.is_unique:
            raise ValueError(f"Reference test scores for {model} do not match the manifest.")
        retained = scores.loc[~scores.image_id.isin(flagged)]
        sensitivity.append({
            "model": model, "flagged_test_images": len(flagged), "retained_test_images": len(retained),
            "reference_dice": float(scores.dice.mean()), "retained_dice": float(retained.dice.mean()),
        })
    pd.DataFrame(sensitivity).to_csv(output / "similarity_sensitivity.csv", index=False)
    for start in range(0, len(pairs), 9):
        page = Image.new("RGB", (990, 3 * 173), "white")
        labels = ImageDraw.Draw(page)
        for index, (left, right, splits, distance) in enumerate(pairs[start:start + 9], start=start):
            x, y = (index - start) % 3 * 330, (index - start) // 3 * 173
            for side, image_id in enumerate((left, right)):
                with Image.open(images[image_id]) as source:
                    thumb = source.convert("RGB")
                    thumb.thumbnail((155, 139))
                page.paste(thumb, (x + side * 160, y + 27))
            labels.text((x + 4, y + 7), f"#{index + 1}  {splits}  d={distance}", fill="black")
        page.save(output / f"candidates_{start // 9 + 1:02d}.png")
    report = {
        "images": len(manifest), "candidate_threshold": cutoff, "hash_bits": 63,
        "manifest_sha256": sha256_file(DEFAULT_MANIFEST),
        "pairs_screened": sum(checked.values()), "screened_by_split": dict(checked),
        "candidate_count": len(pairs), "candidates_by_split": dict(Counter(p[2] for p in pairs)),
        "flagged_test_images": sorted(flagged), "sensitivity": sensitivity,
        "scope": "Image-level candidate screening. Patient/video groups require source identifiers.",
    }
    details = json.dumps(report, indent=2)
    (output / "similarity_summary.json").write_text(details, encoding="utf-8")
    print(details)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args().output)
