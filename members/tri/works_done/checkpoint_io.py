"""Nạp checkpoint an toàn, kiểm tra dữ liệu và khóa giao thức trước tập test."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from members.duy.works_done.model_factory import build_model

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_project_path(path) -> Path:
    result = Path(str(path).replace("\\", "/"))
    return result.resolve() if result.is_absolute() else (PROJECT_ROOT / result).resolve()


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; choose --device cpu")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only cpu/cuda/auto devices are supported")
    return device


def load_checkpoint(path, device="cpu") -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """weights_only=True không chạy pickle chứa class/function tùy ý."""
    path = resolve_project_path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    required = {"format_version", "model_name", "model_kwargs", "model_state", "config", "preprocessing", "manifest_sha256", "epoch"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError("Unsupported checkpoint: use best.pt from the shared Huy trainer")
    if checkpoint["format_version"] != 1 or checkpoint["epoch"] < 1:
        raise ValueError("Checkpoint format must be version 1 with at least one completed epoch")
    # Model/preprocessing code đổi sau training có thể khiến benchmark sai.
    hashes = checkpoint.get("provenance", {}).get("source_hashes", {})
    inference_sources = {"members/khanh/works_done/dataset.py", "members/duy/works_done/unet.py",
                         "members/duy/works_done/model_factory.py", "members/phuc/works_done/unetplusplus.py",
                         "members/tri/works_done/metrics.py"}
    for name, recorded_hash in hashes.items():
        normalized = str(name).replace("\\", "/")
        if normalized in inference_sources and sha256_file(PROJECT_ROOT / normalized) != recorded_hash:
            raise ValueError(f"Inference source changed since training: {normalized}")
    preprocessing = checkpoint["preprocessing"]
    if preprocessing.get("normalization") != "rgb_float_0_1" or preprocessing.get("mask") != "binary_0_1":
        raise ValueError("Unsupported preprocessing; expected RGB [0,1] and mask 0/1")
    size = preprocessing.get("image_size")
    if not isinstance(size, (list, tuple)) or len(size) != 2 or any(not isinstance(x,int) or x < 16 or x % 16 for x in size):
        raise ValueError("Checkpoint image_size must be [H,W], both positive multiples of 16")
    if checkpoint["config"].get("image_size") != list(size):
        raise ValueError("Checkpoint preprocessing size differs from its training config")
    if checkpoint["config"].get("model") != checkpoint["model_name"] or checkpoint["config"].get("model_kwargs") != checkpoint["model_kwargs"]:
        raise ValueError("Checkpoint model metadata differs from its training config")
    model = build_model(checkpoint["model_name"], **checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint


def checkpoint_threshold(checkpoint) -> float:
    config = checkpoint["config"]
    threshold = config.get("evaluation", {}).get("threshold", config.get("threshold", .5))
    if not isinstance(threshold, (float,int)) or not 0 < threshold < 1:
        raise ValueError("Checkpoint threshold must be between 0 and 1")
    return float(threshold)


def checkpoint_manifest(checkpoint) -> Path:
    config = checkpoint["config"]
    path = resolve_project_path(config.get("manifest", "members/khanh/works_done/dataset_splits.csv"))
    if sha256_file(path) != checkpoint["manifest_sha256"]:
        raise ValueError("Dataset manifest changed since training; do not mix split versions")
    return path


def verify_protocol_lock(checkpoint, lock_path, checkpoint_path) -> Dict[str, Any]:
    """Khóa có checkpoint/config chốt từ validation, trước khi đọc ảnh test.

    Đây là dấu vết kiểm toán; timestamp không chứng minh lịch sử thật nếu người
    dùng tự sửa file. Mỗi thí nghiệm phải trùng SHA và cấu hình đã khóa.
    """
    lock = json.loads(resolve_project_path(lock_path).read_text(encoding="utf-8"))
    if lock.get("schema_version") != 1 or lock.get("status") != "locked_before_test":
        raise ValueError("Test requires a schema_version=1, locked_before_test protocol lock")
    if not lock.get("created_at"):
        raise ValueError("Protocol lock must record created_at")
    if lock.get("manifest_sha256") != checkpoint["manifest_sha256"]:
        raise ValueError("Protocol lock manifest differs from checkpoint")
    if lock.get("split_counts") != checkpoint.get("split_counts"):
        raise ValueError("Protocol lock split counts differ from checkpoint")
    actual_path = resolve_project_path(checkpoint_path)
    actual_sha = sha256_file(actual_path)
    candidates = [entry for entry in lock.get("experiments", [])
                  if entry.get("checkpoint_sha256") == actual_sha
                  and resolve_project_path(entry.get("checkpoint_path", "")) == actual_path]
    if len(candidates) != 1:
        raise ValueError("Exactly one protocol experiment must authorize this checkpoint path and SHA256")
    entry = candidates[0]
    if entry.get("config") != checkpoint["config"] or entry.get("model") != checkpoint["model_name"]:
        raise ValueError("Checkpoint model/config differs from locked experiment")
    if entry.get("seed") != checkpoint["config"].get("seed"):
        raise ValueError("Checkpoint seed differs from locked experiment")
    protocol = lock.get("protocol", {})
    if protocol.get("image_size") != checkpoint["preprocessing"]["image_size"]:
        raise ValueError("Locked image size differs from preprocessing")
    if protocol.get("threshold") != checkpoint_threshold(checkpoint) or protocol.get("head_policy") != "last":
        raise ValueError("Locked threshold/head policy is incompatible with evaluation")
    for key in ("base_channels", "batch_norm"):
        if protocol.get(key) != checkpoint["model_kwargs"].get(key):
            raise ValueError(f"Locked {key} differs from model configuration")
    for key in ("batch_size","learning_rate","max_epochs","early_stopping_patience","seed","augmentation","loss"):
        if key in protocol and protocol[key] != checkpoint["config"].get(key):
            raise ValueError(f"Locked {key} differs from training configuration")
    return {"label": entry.get("label"), "lock_sha256": sha256_file(resolve_project_path(lock_path)), "created_at": lock["created_at"]}
