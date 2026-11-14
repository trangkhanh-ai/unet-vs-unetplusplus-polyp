"""Điểm chạy bộ nộp. Source thật nằm trong members/<tên>/works_done của repo.

Không sao chép lại model, Dataset hay loss. Chạy --help để xem ví dụ.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
RUNS = {"A": "A_unet_seed42", "B": "B_unetplusplus_seed42", "C": "C_unetplusplus_ds_seed42"}
CONFIGS = {"A": "unet", "B": "unetplusplus", "C": "unetplusplus_ds"}


def output_path(value, label):
    return value or f"results/{label}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"


def run_project(*arguments):
    subprocess.run([sys.executable, "-X", "utf8", "-B", str(ROOT / "project.py"), *map(str, arguments)], cwd=ROOT, check=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("check", "audit", "splits", "compare"):
        s = sub.add_parser(name)
        s.add_argument("--output")
        if name == "compare":
            s.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
            s.add_argument("--batch-size", type=int, default=1)
    s = sub.add_parser("prepare"); s.add_argument("--include-last", action="store_true")
    s = sub.add_parser("layers"); s.add_argument("--model", choices=RUNS, default="B")
    s = sub.add_parser("evaluate")
    s.add_argument("--weights", "--checkpoint", dest="checkpoint", required=True)
    s.add_argument("--split", choices=("val", "test"), default="test")
    s.add_argument("--head", choices=("last", "mean", "1", "2", "3", "4", "all"), default="last")
    s.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    s.add_argument("--batch-size", type=int, default=1)
    s.add_argument("--protocol-lock", default="configs/protocol_lock.json")
    s.add_argument("--output")
    s = sub.add_parser("predict")
    s.add_argument("--weights", "--checkpoint", dest="checkpoint", required=True)
    s.add_argument("--image", required=True); s.add_argument("--out")
    s.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    s = sub.add_parser("demo"); s.add_argument("--port", type=int, default=8501)
    s = sub.add_parser("train")
    s.add_argument("--model", choices=RUNS, required=True)
    s.add_argument("--name", required=True, help="Tên run mới trong runs/; không dùng tên run chính thức")
    s.add_argument("--protocol", choices=("reference", "extension"), default="reference")
    s.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    for flag, field in (("epochs", "max_epochs"), ("patience", "early_stopping_patience"), ("batch_size", "batch_size"), ("seed", "seed")):
        s.add_argument("--" + flag, type=int, dest=field)
    s.add_argument("--lr", type=float, dest="learning_rate")
    s.add_argument("--scheduler", choices=("none", "cosine"), default="none")
    s.add_argument("--warmup", type=int, default=0)
    s.add_argument("--init", choices=("default", "he"), default="default")
    s.add_argument("--batch_norm", action="store_true")
    s.add_argument("--resume", action="store_true")
    return p


def main():
    args = parser().parse_args()
    if args.command == "prepare":
        from scripts.prepare_project import prepare
        prepare(args.include_last)
    elif args.command == "check":
        run_project("test", "--output", output_path(args.output, "checks"))
        subprocess.run([sys.executable, "-X", "utf8", "-B", "-m", "unittest", "members.duy.works_done.test_submission"], cwd=ROOT, check=True)
    elif args.command in {"audit", "splits"}:
        # Không tự chia lại dữ liệu đi kèm checkpoint đã công bố.
        run_project("audit", "--output", output_path(args.output, "data_audit_recheck"))
    elif args.command == "compare":
        output = output_path(args.output, "reproduced")
        subprocess.run([sys.executable, "-X", "utf8", "-B", str(ROOT / "scripts/reproduce_results.py"),
                        "--device", args.device, "--batch-size", str(args.batch_size), "--output", output], cwd=ROOT, check=True)
        from members.tri.works_done.submission_analysis import export_statistics
        paths = [ROOT / output / f"{label}_test" for label in RUNS]
        print(json.dumps(export_statistics(paths, list(RUNS), ROOT / output / "statistics"), indent=2))
    elif args.command == "evaluate":
        output = output_path(args.output, f"eval_{args.split}_{args.head}")
        if args.head != "last":
            from members.tri.works_done.submission_analysis import evaluate_heads
            result = evaluate_heads(args.checkpoint, args.split, output, args.device, args.batch_size, args.protocol_lock)
            print(json.dumps(result if args.head == "all" else result["metrics"]["4" if args.head == "last" else args.head], indent=2))
        else:
            run_project("evaluate", "--checkpoint", args.checkpoint, "--split", args.split,
                        "--output", output, "--device", args.device, "--batch-size", args.batch_size,
                        "--protocol-lock", args.protocol_lock)
    elif args.command == "predict":
        run_project("predict", "--checkpoint", args.checkpoint, "--input", args.image,
                    "--output", output_path(args.out, "prediction"), "--device", args.device)
    elif args.command == "demo":
        run_project("demo", "--server.port", args.port)
    elif args.command == "layers":
        from members.duy.works_done.model_factory import build_model
        # Hook shape trực tiếp; không yêu cầu torchview để đọc model.
        cfg = json.loads((ROOT / "configs" / (CONFIGS[args.model] + ".json")).read_text())
        model = build_model(cfg["model"], **cfg["model_kwargs"])
        import torch
        torch.set_num_threads(4)
        rows, hooks = [], []
        def hook(name):
            def record(layer, inputs, output):
                rows.append((name, type(layer).__name__, list(output.shape), sum(p.numel() for p in layer.parameters(recurse=False))))
            return record
        for name, layer in model.named_modules():
            if not list(layer.children()):
                hooks.append(layer.register_forward_hook(hook(name)))
        with torch.inference_mode():
            model(torch.zeros(1, 3, 256, 256))
        for handle in hooks:
            handle.remove()
        for row in rows:
            print(*row, sep=" | ")
    elif args.command == "train":
        import re
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.name) or args.name in RUNS.values():
            raise ValueError("Chọn tên run mới chỉ gồm chữ/số/gạch dưới/gạch nối")
        cfg = json.loads((ROOT / "configs" / (CONFIGS[args.model] + ".json")).read_text())
        cfg["device"] = args.device
        for key in ("max_epochs", "early_stopping_patience", "batch_size", "seed", "learning_rate"):
            if getattr(args, key) is not None:
                cfg[key] = getattr(args, key)
        extension = args.scheduler != "none" or args.warmup or args.init != "default" or args.batch_norm
        if extension and args.protocol != "extension":
            raise ValueError("Các thay đổi optimizer/init/BN phải dùng --protocol extension")
        if args.protocol == "extension":
            from members.huy.works_done.extended_training import train_extended
            cfg.update(scheduler=args.scheduler, warmup_epochs=args.warmup, initialization=args.init)
            cfg["model_kwargs"]["batch_norm"] = args.batch_norm
            result = train_extended(cfg, args.name, args.resume)
        else:
            from members.huy.works_done.training import train_experiment
            run = ROOT / "runs" / args.name
            if args.resume and not (run / "last.pt").is_file():
                raise FileNotFoundError("Không có last.pt để resume")
            result = train_experiment(cfg, run, run / "last.pt" if args.resume else None)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
