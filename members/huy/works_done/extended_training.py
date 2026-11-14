"""Thí nghiệm mở rộng riêng; dùng model, Dataset, loss và epoch loop của repo.

Không thay tám file đã gắn SHA256 với các checkpoint chính thức.
"""
import copy
import json
import math
from pathlib import Path
import re
import time

import pandas as pd
import torch
from torch import nn

from members.duy.works_done.model_factory import build_model
from members.khanh.works_done.dataset import get_dataloader
from .losses import BCESoftDiceLoss
from .training import _prepare_epoch, _train_epoch, _validate_epoch
from .run_utils import (
    PROJECT_ROOT, atomic_checkpoint, atomic_json, capture_rng_state, choose_device,
    collect_provenance, configure_reproducibility, manifest_path,
    read_manifest_metadata, resolve_config, restore_rng_state, sha256,
    training_autocast_dtype,
)


def resolve_extended_config(config):
    raw = copy.deepcopy(config)
    scheduler = raw.pop("scheduler", "none")
    warmup = raw.pop("warmup_epochs", 0)
    init = raw.pop("initialization", "default")
    resolved = resolve_config({**raw, "scheduler": "none"})
    if scheduler not in {"none", "cosine"} or init not in {"default", "he"}:
        raise ValueError("Scheduler phải là none/cosine; initialization phải là default/he")
    if type(warmup) is not int or not 0 <= warmup < resolved["max_epochs"]:
        raise ValueError("warmup_epochs phải nằm trong [0, max_epochs)")
    resolved.update(scheduler=scheduler, warmup_epochs=warmup, initialization=init)
    return resolved


def learning_rate_at(epoch, config):
    warm, total, lr = config["warmup_epochs"], config["max_epochs"], config["learning_rate"]
    if warm and epoch <= warm:
        return lr * epoch / warm
    if config["scheduler"] == "cosine":
        progress = (epoch - warm) / (total - warm)
        return 1e-6 + (lr - 1e-6) * (1 + math.cos(math.pi * progress)) / 2
    return lr


def _initialize_he(model):
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


def _check_resume(state, config, metadata, provenance, run):
    required = {"config", "rng_state", "optimizer_state", "scaler_state", "history",
                "best", "patience_count", "epoch", "provenance", "manifest_sha256"}
    if not required.issubset(state) or state.get("trainer_kind") != "extension_20261111":
        raise ValueError("Resume cần last.pt của trainer mở rộng; không dùng trọng số rút gọn")
    for key, value in config.items():
        if key == "device":
            continue
        if key == "max_epochs" and config["scheduler"] == "none":
            if value < state["config"][key]:
                raise ValueError("Không được giảm ngân sách epoch khi resume")
            continue
        if state["config"].get(key) != value:
            raise ValueError(f"Cấu hình resume khác: {key}; tạo run mới")
    if state["manifest_sha256"] != metadata["manifest_sha256"]:
        raise ValueError("Manifest đã thay đổi")
    if state["provenance"]["source_hashes"] != provenance["source_hashes"]:
        raise ValueError("Source đã thay đổi; tạo run mới")
    if state["patience_count"] >= config["early_stopping_patience"]:
        raise ValueError("Run đã early-stop; không tiếp tục cùng run")
    if len(state["history"]) != state["epoch"]:
        raise ValueError("Lịch sử checkpoint không khớp epoch")
    best = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    if best["epoch"] != state["best"]["epoch"] or best["best"] != state["best"]:
        raise ValueError("best.pt không khớp last.pt")
    history = pd.read_csv(run / "history.csv")
    if history["epoch"].tolist() != list(range(1, state["epoch"] + 1)):
        raise ValueError("Lịch sử trên đĩa khác checkpoint; không ghi đè")


def train_extended(config, name, resume=False):
    """Lưu runs/<name>/{best,last}.pt; không có --force để tránh ghi đè kết quả."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name):
        raise ValueError("Tên run chỉ dùng chữ, số, gạch dưới và gạch nối")
    if name in {"A_unet_seed42", "B_unetplusplus_seed42", "C_unetplusplus_ds_seed42"}:
        raise ValueError("Không ghi vào ba run chính thức")
    config = resolve_extended_config(config)
    run = PROJECT_ROOT / "runs" / name
    last = run / "last.pt"
    # Kiểm tra trước khi tạo folder, nạp ảnh hoặc sửa config.
    if resume and not last.is_file():
        raise FileNotFoundError(f"Không có checkpoint để resume: {last}")
    if not resume and run.exists() and any(run.iterdir()):
        raise FileExistsError("Run đã có dữ liệu; chọn tên mới hoặc --resume")
    configure_reproducibility(config["seed"], config["torch_threads"])
    device = choose_device(config["device"])
    dtype = training_autocast_dtype(config, device)
    amp = config["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp and dtype == torch.float16)
    metadata = read_manifest_metadata(manifest_path(config))
    provenance = collect_provenance(device)
    for relative in ("members/huy/works_done/extended_training.py", "CacFileSource.py"):
        provenance["source_hashes"][relative] = sha256(PROJECT_ROOT / relative)
    state = None
    if resume:
        state = torch.load(last, map_location="cpu", weights_only=True)
        _check_resume(state, config, metadata, provenance, run)
    model = build_model(config["model"], **config["model_kwargs"]).to(device)
    if config["initialization"] == "he":
        _initialize_he(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    history, best, patience, start = [], {"epoch": 0, "val_dice": -1., "val_loss": None}, 0, 1
    if state is not None:
        model.load_state_dict(state["model_state"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        scaler.load_state_dict(state["scaler_state"])
        history, best = state["history"], state["best"]
        patience, start = state["patience_count"], state["epoch"] + 1
        restore_rng_state(state["rng_state"])
    if start > config["max_epochs"]:
        return json.loads((run / "summary.json").read_text(encoding="utf-8"))
    loader_args = dict(csv_path=manifest_path(config), project_root=PROJECT_ROOT,
                       batch_size=config["batch_size"], img_size=tuple(config["image_size"]),
                       seed=config["seed"], num_workers=0)
    train_loader = get_dataloader("train", augment=config["augmentation"], **loader_args)
    val_loader = get_dataloader("val", augment=False, **loader_args)
    criterion = BCESoftDiceLoss(**config["loss"])
    run.mkdir(parents=True, exist_ok=True)
    atomic_json(run / "resolved_config.json", config)
    print(f"Run mở rộng {name}: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test đọc=0", flush=True)
    for epoch in range(start, config["max_epochs"] + 1):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate_at(epoch, config)
        _prepare_epoch(train_loader, config["seed"], epoch)
        _prepare_epoch(val_loader, config["seed"], epoch)
        begin = time.perf_counter()
        train = _train_epoch(model, train_loader, optimizer, scaler, criterion, device, amp, dtype)
        val = _validate_epoch(model, val_loader, criterion, device, False, config["evaluation"]["threshold"])
        improved = val["dice"] > best["val_dice"] or (
            val["dice"] == best["val_dice"] and (best["val_loss"] is None or val["loss"] < best["val_loss"]))
        if improved:
            best = {"epoch": epoch, "val_dice": val["dice"], "val_iou": val["iou"], "val_loss": val["loss"]}
            patience = 0
        else:
            patience += 1
        stopped = patience >= config["early_stopping_patience"]
        status = "early_stopped" if stopped else "completed" if epoch == config["max_epochs"] else "training"
        history.append(dict(epoch=epoch, train_loss=train["loss"], val_loss=val["loss"],
                            val_dice=val["dice"], val_iou=val["iou"],
                            learning_rate=optimizer.param_groups[0]["lr"],
                            epoch_seconds=time.perf_counter() - begin, is_best=improved,
                            patience_count=patience))
        payload = dict(format_version=1, trainer_kind="extension_20261111", status=status,
                       model_name=config["model"], model_kwargs=config["model_kwargs"],
                       model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                       scaler_state=scaler.state_dict(), config=config, epoch=epoch,
                       best=best, history=history, patience_count=patience,
                       rng_state=capture_rng_state(), provenance=provenance, **metadata,
                       preprocessing={"image_size": config["image_size"],
                                      "normalization": "rgb_float_0_1", "mask": "binary_0_1"})
        atomic_checkpoint(last, payload)
        if improved:
            atomic_checkpoint(run / "best.pt", payload)
        pd.DataFrame(history).to_csv(run / "history.csv", index=False)
        summary = dict(status=status, model_name=config["model"], config=config, best=best,
                       epochs_completed=epoch, test_images_read=0,
                       parameter_count=sum(p.numel() for p in model.parameters()),
                       trainer_kind="extension_20261111", **metadata)
        atomic_json(run / "summary.json", summary)
        print(f"Epoch {epoch}/{config['max_epochs']} | val Dice={val['dice']:.4f} | best={best['epoch']}", flush=True)
        if stopped:
            break
    return summary
