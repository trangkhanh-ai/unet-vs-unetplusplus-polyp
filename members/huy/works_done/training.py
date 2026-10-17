"""Trainer chung cho U-Net, U-Net++ và U-Net++ có deep supervision.

Luồng: train -> validation -> chọn best -> checkpoint. Không tạo test loader.
"""

import csv
import math
import os
import shutil
import time
from pathlib import Path

import torch

from members.duy.works_done.model_factory import build_model
from members.khanh.works_done.dataset import get_dataloader
from members.tri.works_done.metrics import binary_metrics, last_logits

from .losses import BCESoftDiceLoss, deep_supervision_loss
from .run_utils import (
    PROJECT_ROOT, atomic_checkpoint, atomic_json, capture_rng_state,
    choose_device, collect_provenance, configure_reproducibility,
    manifest_path, read_manifest_metadata, resolve_config, restore_rng_state,
    training_autocast_dtype,
)


HISTORY_COLUMNS = [
    "epoch", "train_loss", "val_loss", "val_dice", "val_iou", "val_precision",
    "val_recall", "learning_rate", "epoch_seconds", "peak_gpu_memory_bytes",
    "amp_skipped_steps", "is_best", "patience_count",
]


def _write_history(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _prepare_epoch(loader, seed: int, epoch: int) -> None:
    # Cùng seed+epoch => cùng thứ tự ảnh và chuỗi augmentation giữa A/B/C.
    loader.dataset.set_epoch(epoch)
    if loader.generator is None:
        raise RuntimeError("Data loader must have its own seeded torch.Generator")
    loader.generator.manual_seed((seed + epoch) % 2**32)


def _train_epoch(model, loader, optimizer, scaler, criterion, device, amp: bool,
                 amp_dtype: torch.dtype = torch.bfloat16) -> dict:
    model.train()
    total_loss, images_seen, skipped_steps = 0.0, 0, 0
    for batch in loader:
        images, masks = batch[:2]
        images = images.to(device, non_blocking=device.type == "cuda")
        masks = masks.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
            output = model(images)
            loss = deep_supervision_loss(output, masks, criterion)
        if not math.isfinite(loss.item()):
            raise FloatingPointError("Training loss is non-finite; stop before saving an invalid run")
        scale_before = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        skipped_steps += int(scaler.get_scale() < scale_before)
        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        images_seen += batch_size
    if images_seen != len(loader.dataset):
        raise RuntimeError("Training did not visit every train image exactly once")
    return {"loss": total_loss / images_seen, "amp_skipped_steps": skipped_steps}


@torch.inference_mode()
def _validate_epoch(model, loader, criterion, device, amp: bool, threshold: float) -> dict:
    model.eval()
    totals = {"loss": 0.0, "dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
    images_seen = 0
    for batch in loader:
        images, masks = batch[:2]
        images = images.to(device, non_blocking=device.type == "cuda")
        masks = masks.to(device, non_blocking=device.type == "cuda")
        # Validation dùng FP32 như evaluator cuối: cùng precision khi chọn
        # best checkpoint và báo điểm. AMP chỉ tăng tốc forward/backward train.
        logits = last_logits(model(images))
        # Validation luôn cùng objective một head cuối cho cả ba mô hình.
        totals["loss"] += criterion.per_image(logits, masks).double().sum().item()
        metrics = binary_metrics(logits, masks, threshold=threshold)
        for name in ("dice", "iou", "precision", "recall"):
            totals[name] += metrics[name].sum().item()
        images_seen += images.shape[0]
    if images_seen != len(loader.dataset):
        raise RuntimeError("Validation did not visit every validation image exactly once")
    result = {name: value / images_seen for name, value in totals.items()}
    if any(not math.isfinite(value) for value in result.values()):
        raise FloatingPointError("Validation produced non-finite loss or metrics")
    return result


def _check_resume(checkpoint: dict, config: dict, metadata: dict, provenance: dict) -> None:
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    old_config = checkpoint["config"]
    changed = [key for key in config if key not in {"max_epochs", "device"}
               and config[key] != old_config.get(key)]
    if changed:
        raise ValueError(f"Resume changes the experimental protocol: {changed}")
    if checkpoint["manifest_sha256"] != metadata["manifest_sha256"]:
        raise ValueError("Resume manifest SHA256 differs from the original split")
    if checkpoint["provenance"]["source_hashes"] != provenance["source_hashes"]:
        raise ValueError("Resume execution source changed; use a new output directory for a new run")
    if config["max_epochs"] < checkpoint["epoch"]:
        raise ValueError("max_epochs cannot be smaller than the completed checkpoint epoch")
    if checkpoint["patience_count"] >= config["early_stopping_patience"]:
        raise ValueError("This run already reached early stopping; start a new run instead of resuming it")


def _check_resume_destination(checkpoint: dict, resume: Path, output_dir: Path) -> Path | None:
    """Không lùi epoch rồi ghi đè lịch sử; fork phải giữ đúng best của nguồn."""
    if output_dir == resume.parent:
        last_path = output_dir / "last.pt"
        if not last_path.is_file():
            raise FileNotFoundError("An existing run needs last.pt before in-place resume")
        current_last = torch.load(last_path, map_location="cpu", weights_only=True)
        if checkpoint["epoch"] != current_last["epoch"]:
            raise ValueError("Cannot rewind an existing run; resume its last.pt or choose a new output directory")
        return None
    if output_dir.exists():
        raise FileExistsError("A separate resumed output directory must be new")
    best_path = resume.parent / "best.pt"
    if not best_path.is_file():
        raise FileNotFoundError("Resume needs the original best.pt next to its checkpoint")
    previous_best = torch.load(best_path, map_location="cpu", weights_only=True)
    if previous_best["epoch"] != checkpoint["best"]["epoch"] or previous_best["best"] != checkpoint["best"]:
        raise ValueError("Source best.pt differs from the best recorded in the resume checkpoint")
    for key in ("model_name", "model_kwargs", "manifest_sha256", "split_counts"):
        if previous_best[key] != checkpoint[key]:
            raise ValueError(f"Source best.pt differs from resume checkpoint in {key}")
    for key, value in checkpoint["config"].items():
        if key not in {"max_epochs", "device"} and previous_best["config"].get(key) != value:
            raise ValueError(f"Source best.pt uses a different experimental config: {key}")
    if previous_best["provenance"]["source_hashes"] != checkpoint["provenance"]["source_hashes"]:
        raise ValueError("Source best.pt uses different execution code")
    return best_path


def train_experiment(config: dict, output_dir: Path | str,
                     resume: Path | str | None = None) -> dict:
    """Chạy hoặc tiếp tục một thí nghiệm; trả nội dung summary.json.

    Resume cho phép đổi max_epochs và device. Cùng phần cứng, môi trường và
    protocol mới có thể kiểm tra tái lập từng trọng số; đổi CPU/GPU không đảm bảo
    kết quả bit-for-bit. Không cho đổi split, model, LR, loss hay augmentation.
    """
    config = resolve_config(config)
    configure_reproducibility(config["seed"], config["torch_threads"])
    device = choose_device(config["device"])
    amp = config["amp"] and device.type == "cuda"
    amp_dtype = training_autocast_dtype(config, device)
    training_precision = f"amp_{config['amp_dtype']}" if amp else "float32"
    # BF16 không cần loss scaling; giữ GradScaler cho protocol FP16 cũ.
    use_grad_scaler = amp and amp_dtype == torch.float16
    metadata = read_manifest_metadata(manifest_path(config))
    provenance = collect_provenance(device)
    provenance["environment"]["training_precision"] = training_precision
    provenance["environment"]["grad_scaler_enabled"] = use_grad_scaler
    output_dir = Path(output_dir).resolve()
    checkpoint, previous_best_path = None, None
    if resume is None:
        if output_dir.exists():
            raise FileExistsError(f"Output already exists: {output_dir}; choose a new run name or --resume")
    else:
        resume = Path(resume).resolve()
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        _check_resume(checkpoint, config, metadata, provenance)
        previous_best_path = _check_resume_destination(checkpoint, resume, output_dir)
        if output_dir != resume.parent and config["max_epochs"] == checkpoint["epoch"]:
            raise ValueError("Resume into a new directory requires at least one additional epoch")
    output_dir.mkdir(parents=True, exist_ok=True)

    loader_kwargs = {
        "csv_path": str(manifest_path(config)), "project_root": str(PROJECT_ROOT),
        "batch_size": config["batch_size"], "num_workers": config["num_workers"],
        "img_size": tuple(config["image_size"]), "seed": config["seed"],
        "return_id": False, "pin_memory": device.type == "cuda",
    }
    train_loader = get_dataloader("train", augment=config["augmentation"],
                                  shuffle=True, **loader_kwargs)
    val_loader = get_dataloader("val", augment=False, shuffle=False, **loader_kwargs)
    model = build_model(config["model"], **config["model_kwargs"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = BCESoftDiceLoss(**config["loss"])
    scaler = torch.amp.GradScaler(device.type, enabled=use_grad_scaler)
    history = []
    best = {"epoch": 0, "val_dice": -1.0, "val_iou": 0.0, "val_loss": None}
    patience_count, start_epoch = 0, 1
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if use_grad_scaler and checkpoint["scaler_state"]:
            scaler.load_state_dict(checkpoint["scaler_state"])
        history = checkpoint["history"]
        best, patience_count = checkpoint["best"], checkpoint["patience_count"]
        start_epoch = checkpoint["epoch"] + 1
        restore_rng_state(checkpoint["rng_state"])
        if previous_best_path is not None:
            shutil.copy2(previous_best_path, output_dir / "best.pt")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    atomic_json(output_dir / "resolved_config.json", config)
    atomic_json(output_dir / "environment.json", provenance)
    print(f"Model={config['model']} | device={device} | train={training_precision} | "
          f"parameters={parameter_count:,}", flush=True)
    print(f"Train={len(train_loader.dataset)} | val={len(val_loader.dataset)} | test images read=0", flush=True)

    status = "completed"
    for epoch in range(start_epoch, config["max_epochs"] + 1):
        _prepare_epoch(train_loader, config["seed"], epoch)
        _prepare_epoch(val_loader, config["seed"], epoch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        train = _train_epoch(model, train_loader, optimizer, scaler, criterion,
                             device, amp, amp_dtype)
        validation = _validate_epoch(model, val_loader, criterion, device, amp,
                                     config["evaluation"]["threshold"])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        is_best = validation["dice"] > best["val_dice"] or (
            validation["dice"] == best["val_dice"] and
            (best["val_loss"] is None or validation["loss"] < best["val_loss"])
        )
        if is_best:
            best = {"epoch": epoch, "val_dice": validation["dice"],
                    "val_iou": validation["iou"], "val_loss": validation["loss"]}
            patience_count = 0
        else:
            patience_count += 1
        stop = patience_count >= config["early_stopping_patience"]
        final_epoch = stop or epoch == config["max_epochs"]
        epoch_status = "early_stopped" if stop else ("completed" if final_epoch else "training")
        row = {
            "epoch": epoch, "train_loss": train["loss"], "val_loss": validation["loss"],
            "val_dice": validation["dice"], "val_iou": validation["iou"],
            "val_precision": validation["precision"], "val_recall": validation["recall"],
            "learning_rate": optimizer.param_groups[0]["lr"], "epoch_seconds": seconds,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "amp_skipped_steps": train["amp_skipped_steps"], "is_best": is_best,
            "patience_count": patience_count,
        }
        history.append(row)
        saved = {
            "format_version": 1, "model_name": config["model"],
            "model_kwargs": config["model_kwargs"], "model_state": model.state_dict(),
            "config": config, "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(), "epoch": epoch, "best": best,
            "patience_count": patience_count, "history": history,
            "rng_state": capture_rng_state(), **metadata,
            "preprocessing": {"image_size": config["image_size"], "normalization": "rgb_float_0_1",
                              "mask": "binary_0_1"},
            "provenance": provenance, "status": epoch_status,
            "validation_precision": "float32",
            "training_precision": training_precision,
            "grad_scaler_enabled": use_grad_scaler,
        }
        atomic_checkpoint(output_dir / "last.pt", saved)
        if is_best:
            atomic_checkpoint(output_dir / "best.pt", saved)
        _write_history(output_dir / "history.csv", history)
        summary = {
            "status": epoch_status, "model_name": config["model"], "config": config,
            "device": str(device), "amp_effective": amp, "parameter_count": parameter_count,
            "training_precision": training_precision,
            "grad_scaler_enabled": use_grad_scaler,
            "validation_precision": "float32",
            "epochs_completed": epoch, "best": best, **metadata,
            "train_images": len(train_loader.dataset), "validation_images": len(val_loader.dataset),
            "test_images_read": 0, "history_csv": "history.csv", "best_checkpoint": "best.pt",
            "last_checkpoint": "last.pt", "training_and_validation_seconds": sum(item["epoch_seconds"] for item in history),
            "provenance": provenance,
        }
        atomic_json(output_dir / "summary.json", summary)
        print(f"Epoch {epoch:03d}/{config['max_epochs']} | train loss={train['loss']:.5f} | "
              f"val loss={validation['loss']:.5f} | Dice={validation['dice']:.5f} | "
              f"IoU={validation['iou']:.5f} | {seconds:.1f}s | best={is_best} | "
              f"patience={patience_count}/{config['early_stopping_patience']}", flush=True)
        if stop:
            status = "early_stopped"
            break
    # max_epochs bằng epoch đã hoàn thành: đọc lại summary/checkpoint hiện có.
    if not history:
        raise RuntimeError("No training epochs completed")
    if start_epoch > config["max_epochs"]:
        import json
        summary_path = output_dir / "summary.json"
        if summary_path.exists():
            return json.loads(summary_path.read_text(encoding="utf-8"))
        raise ValueError("Resume into a new directory requires at least one additional epoch")
    return summary
