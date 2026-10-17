"""Cấu hình, tái lập và lưu tệp cho trainer; không đọc ảnh test."""

import copy
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_CONFIG = {
    "model": "unet",
    "model_kwargs": {"in_channels": 3, "out_channels": 1, "base_channels": 32,
                     "batch_norm": False, "deep_supervision": False},
    "manifest": "members/khanh/works_done/dataset_splits.csv",
    "image_size": [256, 256], "seed": 42, "batch_size": 8, "num_workers": 0,
    "learning_rate": 0.001, "max_epochs": 50, "early_stopping_patience": 10,
    "device": "auto", "amp": True, "amp_dtype": "bfloat16",
    "torch_threads": 4, "augmentation": True,
    "scheduler": "none",
    "loss": {"bce_weight": 0.5, "soft_dice_weight": 0.5, "smooth": 1e-6},
    "evaluation": {"threshold": 0.5, "head_policy": "last",
                   "aggregation": "per_image_then_mean"},
}


def resolve_config(config: dict) -> dict:
    """Bổ sung mặc định và báo lỗi rõ khi cấu hình không thuộc protocol."""
    if not isinstance(config, dict):
        raise TypeError("Config must be a dictionary")
    unknown = set(config) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    result = copy.deepcopy(DEFAULT_CONFIG)
    for key, value in config.items():
        if key in {"model_kwargs", "loss", "evaluation"}:
            if not isinstance(value, dict):
                raise TypeError(f"{key} must be an object")
            extra = set(value) - set(DEFAULT_CONFIG[key])
            if extra:
                raise ValueError(f"Unknown {key} keys: {sorted(extra)}")
            result[key].update(copy.deepcopy(value))
        else:
            result[key] = copy.deepcopy(value)
    if result["model"] not in {"unet", "unetplusplus"}:
        raise ValueError("model must be unet or unetplusplus")
    kwargs = result["model_kwargs"]
    if kwargs["in_channels"] != 3 or kwargs["out_channels"] != 1:
        raise ValueError("Kvasir protocol expects RGB input and one polyp output channel")
    if type(kwargs["base_channels"]) is not int or kwargs["base_channels"] < 1:
        raise ValueError("base_channels must be a positive integer")
    for key in ("batch_norm", "deep_supervision"):
        if type(kwargs[key]) is not bool:
            raise TypeError(f"model_kwargs.{key} must be a boolean")
    if result["model"] == "unet" and kwargs["deep_supervision"]:
        raise ValueError("Plain U-Net cannot use deep supervision")
    size = result["image_size"]
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError("image_size must contain [height, width]")
    if any(type(n) is not int or n < 16 or n % 16 for n in size):
        raise ValueError("Height and width must be >=16 and divisible by 16")
    result["image_size"] = list(size)
    for key in ("batch_size", "max_epochs", "early_stopping_patience", "torch_threads"):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(result["seed"]) is not int or not 0 <= result["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if result["num_workers"] != 0:
        raise ValueError("This reproducible course protocol requires num_workers=0")
    for key in ("amp", "augmentation"):
        if type(result[key]) is not bool:
            raise TypeError(f"{key} must be a boolean")
    if result["amp_dtype"] not in {"bfloat16", "float16"}:
        raise ValueError("amp_dtype must be bfloat16 or float16")
    if result["device"] not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu or cuda")
    if result["scheduler"] != "none":
        raise ValueError("The agreed protocol uses scheduler='none'")
    if not math.isfinite(result["learning_rate"]) or result["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive and finite")
    weights = result["loss"]
    if any(not math.isfinite(value) for value in weights.values()):
        raise ValueError("Loss values must be finite")
    if min(weights["bce_weight"], weights["soft_dice_weight"]) < 0:
        raise ValueError("Loss weights must be nonnegative")
    if weights["bce_weight"] + weights["soft_dice_weight"] <= 0 or weights["smooth"] <= 0:
        raise ValueError("Loss weight sum and Dice smooth must be positive")
    evaluation = result["evaluation"]
    if not math.isfinite(evaluation["threshold"]) or not 0 < evaluation["threshold"] < 1:
        raise ValueError("Prediction threshold must be in (0, 1)")
    if evaluation["head_policy"] != "last" or evaluation["aggregation"] != "per_image_then_mean":
        raise ValueError("Evaluation uses last head and per_image_then_mean")
    if not isinstance(result["manifest"], str) or not result["manifest"]:
        raise ValueError("manifest must be a nonempty path string")
    return result


def manifest_path(config: dict) -> Path:
    path = Path(config["manifest"])
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_manifest_metadata(path: Path) -> dict:
    """Chỉ đọc CSV: kiểm tra ID/split; không mở bất kỳ ảnh test nào."""
    frame = pd.read_csv(path)
    required = {"image_id", "image_path", "mask_path", "split"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Manifest lacks columns: {sorted(required - set(frame.columns))}")
    if frame[list(required)].isna().any().any():
        raise ValueError("Manifest has missing IDs, paths or split labels")
    if frame["image_id"].duplicated().any():
        raise ValueError("Manifest image IDs must be unique across all splits")
    if frame["image_path"].duplicated().any() or frame["mask_path"].duplicated().any():
        raise ValueError("Manifest image/mask paths must be unique across all splits")
    counts = frame["split"].value_counts().to_dict()
    if set(counts) != {"train", "val", "test"}:
        raise ValueError("Manifest must contain nonempty train, val and test splits")
    return {"manifest_sha256": sha256(path),
            "split_counts": {key: int(value) for key, value in counts.items()}}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_value(*args) -> str | None:
    try:
        process = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True,
                                 text=True, encoding="utf-8", check=True)
        return process.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def collect_provenance(device: torch.device) -> dict:
    names = [
        "members/khanh/works_done/dataset.py",
        "members/duy/works_done/unet.py", "members/duy/works_done/model_factory.py",
        "members/phuc/works_done/unetplusplus.py",
        "members/huy/works_done/losses.py", "members/huy/works_done/training.py",
        "members/huy/works_done/run_utils.py", "members/tri/works_done/metrics.py",
    ]
    source_hashes = {name: sha256(PROJECT_ROOT / name)
                     for name in names if (PROJECT_ROOT / name).is_file()}
    environment = {"python": sys.version, "platform": platform.platform(),
                   "torch": str(torch.__version__), "numpy": np.__version__,
                   "device": str(device), "cuda_build": torch.version.cuda,
                   "cuda_available": torch.cuda.is_available(),
                   "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                   "torch_threads": torch.get_num_threads()}
    if device.type == "cuda":
        environment["gpu"] = torch.cuda.get_device_name(device)
        environment["gpu_total_memory_bytes"] = torch.cuda.get_device_properties(device).total_memory
    return {"created_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": git_value("rev-parse", "HEAD"),
            "branch": git_value("branch", "--show-current"),
            "dirty": bool(git_value("status", "--porcelain")),
            "source_hashes": source_hashes, "environment": environment}


def configure_reproducibility(seed: int, threads: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; install GPU PyTorch or use --device cpu")
    return torch.device("cuda" if requested != "cpu" and torch.cuda.is_available() else "cpu")


def training_autocast_dtype(config: dict, device: torch.device) -> torch.dtype:
    """BF16 có miền giá trị rộng hơn FP16; báo lỗi nếu GPU không hỗ trợ.

    CPU luôn train FP32 vì trainer chỉ bật AMP trên CUDA. FP16 được giữ
    cho thí nghiệm cũ; ba cấu hình chính thức v2 đều khai báo bfloat16.
    """
    dtype = torch.bfloat16 if config["amp_dtype"] == "bfloat16" else torch.float16
    if config["amp"] and device.type == "cuda" and dtype == torch.bfloat16:
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 AMP requested but this CUDA GPU does not support BF16")
    return dtype


def capture_rng_state() -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {"algorithm": numpy_state[0], "keys": numpy_state[1].tolist(),
                  "position": numpy_state[2], "has_gauss": numpy_state[3],
                  "cached_gaussian": numpy_state[4]},
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state["algorithm"], np.asarray(numpy_state["keys"], dtype=np.uint32),
                         numpy_state["position"], numpy_state["has_gauss"],
                         numpy_state["cached_gaussian"]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def atomic_checkpoint(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)
