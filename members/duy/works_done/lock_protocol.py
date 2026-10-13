"""Ghi khóa checkpoint đã chọn bằng validation trước lần đánh giá test."""
import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path

import torch

from members.huy.works_done.run_utils import PROJECT_ROOT, sha256
from members.tri.works_done.compare import common_training_config


def create_protocol_lock(run_paths, labels, output):
    if len(run_paths) != len(labels) or len(run_paths) < 2 or len(set(labels)) != len(labels):
        raise ValueError("Cần >=2 run với label duy nhất tương ứng")
    experiments, reference = [], None
    for run_name, label in zip(run_paths, labels):
        run = Path(run_name)
        run = run if run.is_absolute() else PROJECT_ROOT / run
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        if summary["status"] not in {"completed", "early_stopped"}:
            raise ValueError(f"Run chưa kết thúc: {run}")
        checkpoint_path = run / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["best"] != summary["best"] or checkpoint["epoch"] != summary["best"]["epoch"]:
            raise ValueError("best.pt không khớp checkpoint chọn từ validation")
        if checkpoint["split_counts"] != {"train": 700, "val": 150, "test": 150}:
            raise ValueError("Khóa chính thức yêu cầu holdout 700/150/150")
        if sha256(PROJECT_ROOT / checkpoint["config"]["manifest"]) != checkpoint["manifest_sha256"]:
            raise ValueError("Manifest đã đổi sau huấn luyện")
        if reference is not None:
            if common_training_config(reference["config"]) != common_training_config(checkpoint["config"]):
                raise ValueError("Các run khác protocol chung")
            if reference["manifest_sha256"] != checkpoint["manifest_sha256"]:
                raise ValueError("Các run dùng manifest khác")
            if reference["provenance"]["source_hashes"] != checkpoint["provenance"]["source_hashes"]:
                raise ValueError("Các run dùng mã thực thi khác phiên bản")
        else:
            reference = checkpoint
        experiments.append({
            "label": label, "model": checkpoint["model_name"], "seed": checkpoint["config"]["seed"],
            "checkpoint_path": str(checkpoint_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "checkpoint_sha256": sha256(checkpoint_path), "config": checkpoint["config"],
        })
    cfg = reference["config"]
    protocol = {name: cfg[name] for name in (
        "image_size", "batch_size", "learning_rate", "max_epochs", "early_stopping_patience",
        "seed", "augmentation", "loss", "num_workers", "amp", "amp_dtype", "scheduler")}
    protocol.update({name: cfg["model_kwargs"][name] for name in ("base_channels", "batch_norm")})
    protocol.update({"threshold": cfg["evaluation"]["threshold"], "head_policy": "last"})
    result = {"schema_version": 1, "status": "locked_before_test",
              "created_at": datetime.now(timezone(timedelta(hours=7))).isoformat(),
              "manifest_sha256": reference["manifest_sha256"],
              "split_counts": reference["split_counts"], "protocol": protocol,
              "experiments": experiments}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("Khóa đã tồn tại; giữ lịch sử, tạo tên mới cho thí nghiệm mới")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", default="configs/protocol_lock.json")
    args = parser.parse_args()
    lock = create_protocol_lock(args.runs, args.labels, args.output)
    print(json.dumps({"status": lock["status"], "experiments": [x["label"] for x in lock["experiments"]]}, indent=2))


if __name__ == "__main__":
    main()
