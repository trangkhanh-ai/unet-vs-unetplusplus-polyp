"""CLI huấn luyện; chạy từ thư mục gốc repo."""

import argparse
import json
from pathlib import Path

from .training import train_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Train U-Net/U-Net++ using the shared course protocol")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--max-epochs", type=int, help="Pilot run or extend the same protocol on resume")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.device is not None:
        config["device"] = args.device
    if args.max_epochs is not None:
        config["max_epochs"] = args.max_epochs
    summary = train_experiment(config, args.output, args.resume)
    print(json.dumps({"status": summary["status"], "best": summary["best"],
                      "epochs_completed": summary["epochs_completed"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
