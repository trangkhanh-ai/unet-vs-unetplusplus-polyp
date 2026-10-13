"""Đánh giá lại A/B/C và tạo bảng/hình trong một output mới.

python scripts/reproduce_results.py --device cuda
Không train lại và không thay kết quả tham chiếu đã công bố.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.prepare_project import prepare

RUNS = [("A", "A_unet_seed42"), ("B", "B_unetplusplus_seed42"), ("C", "C_unetplusplus_ds_seed42")]


def run(arguments):
    print("Chạy: project.py " + " ".join(arguments), flush=True)
    subprocess.run([sys.executable, "-X", "utf8", "-B", str(ROOT / "project.py"), *arguments],
                   cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Inference batch; 1 uses less VRAM, reference timing used 8")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size phải dương")
    output = ROOT / (args.output or "results/reproduced_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    output = output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Output phải ở trong repo")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Chọn output mới để giữ minh chứng cũ")
    prepare()
    output.mkdir(parents=True, exist_ok=True)
    evaluations = []
    for label, folder in RUNS:
        evaluation = output / (label + "_test")
        run(["evaluate", "--checkpoint", f"runs/{folder}/best.pt", "--split", "test",
             "--device", args.device, "--protocol-lock", "configs/protocol_lock.json",
             "--batch-size", str(args.batch_size),
             "--output", str(evaluation)])
        evaluations.append(str(evaluation))
    run(["compare", "--evaluations", *evaluations, "--output", str(output / "comparison")])
    current = json.loads((output / "comparison/comparison.json").read_text(encoding="utf-8"))
    reference = json.loads((ROOT / "results/comparison/comparison.json").read_text(encoding="utf-8"))
    deltas = [{"model": now["experiment"], "dice_delta": now["dice"] - old["dice"],
               "iou_delta": now["iou"] - old["iou"]}
              for now, old in zip(current["comparison"], reference["comparison"])]
    (output / "reference_deltas.json").write_text(json.dumps(deltas, indent=2), encoding="utf-8")
    print("Hoàn tất. Mở " + str(output / "comparison"), flush=True)
    print("Sai khác so với bản tham chiếu (không bảo đảm bitwise khi đổi môi trường): " + json.dumps(deltas))


if __name__ == "__main__":
    main()
