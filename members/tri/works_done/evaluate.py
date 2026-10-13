"""Đánh giá checkpoint đã chọn bằng validation; không huấn luyện/tune trên test.

Chạy từ repo: python -m members.tri.works_done.evaluate --help
Output phải là thư mục mới hoặc rỗng; dùng thư mục mới để giữ kết quả cũ.
"""

import argparse
from datetime import datetime, timezone
import json
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch

from members.khanh.works_done.dataset import get_dataloader, read_image_unicode
from .checkpoint_io import (PROJECT_ROOT, load_checkpoint, checkpoint_manifest,
    checkpoint_threshold, resolve_device, resolve_project_path, sha256_file, verify_protocol_lock)
from .metrics import binary_metrics, last_logits


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def environment_details(device):
    return {"device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "platform": platform.platform(), "torch": str(torch.__version__), "python": platform.python_version(),
            "torch_threads": torch.get_num_threads(), "precision": "float32", "amp": False,
            "cuda_build": torch.version.cuda, "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32}


def draw_examples(rows, metadata, masks_dir, output, image_size):
    """Định trước bằng ID sắp xếp; không chọn ảnh đẹp để che thất bại."""
    import cv2
    selected = sorted(rows, key=lambda row: row["image_id"])[:6]
    figure, axes = plt.subplots(len(selected), 4, figsize=(12, 3 * len(selected)), squeeze=False)
    for index, row in enumerate(selected):
        source = metadata.loc[row["image_id"]]
        image = read_image_unicode(resolve_project_path(source["image_path"]), cv2.IMREAD_COLOR)
        truth = read_image_unicode(resolve_project_path(source["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if image is None or truth is None:
            raise FileNotFoundError(f"Cannot draw image/mask {row['image_id']}")
        image = cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2RGB), (image_size[1], image_size[0]))
        truth = cv2.resize((truth > 127).astype(np.uint8), (image_size[1],image_size[0]), interpolation=cv2.INTER_NEAREST)
        prediction = np.asarray(Image.open(masks_dir / f"{row['image_id']}.png")) > 127
        overlay = image.astype(float).copy()
        overlay[prediction] = .6 * overlay[prediction] + .4 * np.array([0,255,0])
        for ax, array, title in zip(axes[index], (image,truth,prediction,overlay.astype(np.uint8)),
                                     (row["image_id"],"Ground truth",f"Prediction | Dice {row['dice']:.3f}","Overlay")):
            ax.imshow(array,cmap="gray" if array.ndim == 2 else None)
            ax.set_title(title,fontsize=9)
            ax.axis("off")
    figure.suptitle("First six image IDs in sorted order; metrics at model input resolution",fontsize=12)
    figure.tight_layout()
    figure.savefig(output,dpi=130)
    plt.close(figure)


def evaluate_checkpoint(checkpoint_path, split, output, device="auto", batch_size=8,
                        protocol_lock=None, num_workers=0, warmup_batches=2):
    if split not in {"val","test"}:
        raise ValueError("Evaluation split must be val or test")
    if batch_size < 1 or num_workers < 0 or warmup_batches < 0:
        raise ValueError("batch_size >= 1, num_workers >= 0, warmup_batches >= 0 required")
    device = resolve_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint_path = resolve_project_path(checkpoint_path)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    # Khóa được kiểm tra TRƯỚC khi tạo/lặp DataLoader test.
    authorization = None
    if split == "test":
        if protocol_lock is None:
            raise ValueError("Test evaluation requires --protocol-lock created after validation selection")
        authorization = verify_protocol_lock(checkpoint,protocol_lock,checkpoint_path)
    manifest = checkpoint_manifest(checkpoint)
    all_metadata = pd.read_csv(manifest)
    actual_counts = {str(k): int(v) for k,v in all_metadata["split"].value_counts().items()}
    if actual_counts != checkpoint.get("split_counts"):
        raise ValueError("Current manifest split counts differ from checkpoint")
    ids = all_metadata["image_id"].astype(str)
    if ids.duplicated().any():
        raise ValueError("Manifest contains duplicate image IDs")
    image_size = tuple(checkpoint["preprocessing"]["image_size"])
    threshold = checkpoint_threshold(checkpoint)
    loader = get_dataloader(csv_path=str(manifest), project_root=str(PROJECT_ROOT), split=split,
        img_size=image_size, batch_size=batch_size, num_workers=num_workers, seed=checkpoint["config"]["seed"],
        return_id=True, augment=False, shuffle=False)
    output = resolve_project_path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output directory is not empty; choose a fresh directory to preserve prior results")
    output.mkdir(parents=True,exist_ok=True)
    checkpoint_sha = sha256_file(checkpoint_path)
    masks_dir = output / "masks"
    masks_dir.mkdir(exist_ok=True)
    metadata = all_metadata.set_index("image_id")
    rows, inference_seconds, loading_seconds, transfer_seconds, batch_times = [], 0.0, 0.0, 0.0, []
    iterator = iter(loader)
    for batch_index in range(len(loader)):
        start = time.perf_counter()
        images, targets, batch_ids = next(iterator)
        loading_seconds += time.perf_counter() - start
        synchronize(device)
        start = time.perf_counter()
        images, targets = images.to(device), targets.to(device)
        synchronize(device)
        transfer_seconds += time.perf_counter() - start
        with torch.inference_mode():
            if batch_index == 0:
                for _ in range(warmup_batches):
                    last_logits(model(images))
                synchronize(device)
            start = time.perf_counter()
            logits = last_logits(model(images))
            synchronize(device)
            seconds = time.perf_counter() - start
            inference_seconds += seconds
            batch_times.append({"images": int(images.shape[0]), "seconds": seconds})
            metrics = binary_metrics(logits,targets,threshold)
            predictions = (logits.sigmoid() >= threshold).cpu().numpy()[:,0]
        for local_index,image_id in enumerate(batch_ids):
            image_id = str(image_id)
            row = {"image_id": image_id, "split": split,
                   **{key: float(value[local_index].item()) for key,value in metrics.items()}}
            original = metadata.loc[image_id]
            row.update({"polyp_category": original.get("polyp_category", "unknown"),
                        "polyp_area_ratio_original": float(original.get("polyp_area_ratio",np.nan)),
                        "height_original": int(original["height"]), "width_original": int(original["width"]),
                        "target_area_ratio_at_input": float(targets[local_index].mean().item())})
            rows.append(row)
            Image.fromarray(predictions[local_index].astype(np.uint8)*255).save(masks_dir / f"{image_id}.png")
    table = pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)
    expected_ids = set(loader.dataset.df["image_id"].astype(str))
    if len(table) != len(expected_ids) or set(table["image_id"]) != expected_ids:
        raise ValueError("Evaluation omitted or duplicated images")
    table.to_csv(output / "per_image.csv",index=False,encoding="utf-8")
    groups = {}
    for name, group in table.groupby("polyp_category"):
        groups[str(name)] = {"count": len(group), **{key:float(group[key].mean()) for key in ("dice","iou","precision","recall")}}
    summary = {"format_version":1, "created_at":datetime.now(timezone.utc).isoformat(),
        "status":"evaluated", "split":split, "evaluation_kind":"locked_test" if split == "test" else "validation",
        "model_name":checkpoint["model_name"], "model_kwargs":checkpoint["model_kwargs"], "config":checkpoint["config"],
        "checkpoint":str(checkpoint_path), "checkpoint_sha256":checkpoint_sha, "selected_epoch":checkpoint["epoch"],
        "best_validation":checkpoint.get("best"), "manifest_sha256":checkpoint["manifest_sha256"], "split_counts":actual_counts,
        "preprocessing":checkpoint["preprocessing"], "threshold":threshold, "head_policy":"last", "aggregation":"per_image_then_mean",
        "image_count":len(table), "metrics":{key:float(table[key].mean()) for key in ("dice","iou","precision","recall")},
        "by_polyp_category":groups, "parameters":sum(p.numel() for p in model.parameters()), "protocol_authorization":authorization,
        "timing":{"scope":"model forward only; excludes loading, transfer, sigmoid, metrics, export; float32 without AMP",
            "warmup_batches":warmup_batches,"batch_size":batch_size,"num_workers":num_workers,
            "forward_seconds":inference_seconds,"images_per_second":len(table)/inference_seconds,
            "milliseconds_per_image":1000*inference_seconds/len(table),"data_loading_seconds":loading_seconds,
            "transfer_seconds":transfer_seconds,"batches":batch_times},
        "environment":environment_details(device), "qualitative_selection":"first six sorted image IDs",
        "evaluation_source_hashes":{name:sha256_file(Path(__file__).resolve().parent / name)
                                    for name in ("evaluate.py","checkpoint_io.py","metrics.py")},
        "resolution_note":"All metrics and exported masks use model input resolution, not original image resolution",
        "files":{"per_image":"per_image.csv","masks":"masks/","qualitative":"qualitative.png"}}
    draw_examples(rows,metadata,masks_dir,output / "qualitative.png",image_size)
    (output / "summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--split",choices=["val","test"],required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--device",default="auto")
    parser.add_argument("--batch-size",type=int,default=8)
    parser.add_argument("--num-workers",type=int,default=0)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--warmup-batches",type=int,default=2)
    parser.add_argument("--protocol-lock")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be >= 1")
    torch.set_num_threads(args.threads)
    result = evaluate_checkpoint(args.checkpoint,args.split,args.output,args.device,args.batch_size,
                                 args.protocol_lock,args.num_workers,args.warmup_batches)
    print(json.dumps({"split":result["split"],"count":result["image_count"],"metrics":result["metrics"]},indent=2))


if __name__ == "__main__":
    main()
