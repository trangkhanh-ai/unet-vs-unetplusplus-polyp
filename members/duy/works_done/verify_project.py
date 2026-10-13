"""Một lệnh chạy tất cả kiểm tra hợp đồng và ghi minh chứng."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import unittest

import torch

TEST_MODULES = [
    "members.khanh.works_done.test_dataset",
    "members.duy.works_done.test_unet",
    "members.phuc.works_done.test_unetplusplus",
    "members.huy.works_done.test_training",
    "members.tri.works_done.test_metrics",
    "members.tri.works_done.test_evaluation",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/verification"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    suite = unittest.defaultTestLoader.loadTestsFromNames(TEST_MODULES)
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        "status": "passed" if result.wasSuccessful() else "failed",
        "tests_run": result.testsRun,
        "failure_count": len(result.failures), "error_count": len(result.errors),
        "skipped_count": len(result.skipped),
        "seconds": time.perf_counter() - started,
        "modules": TEST_MODULES,
        "python": platform.python_version(), "torch": str(torch.__version__),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "tested_scope": "contracts/synthetic integration; no official Kvasir test evaluation",
    }
    (args.output / "test_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())
    (args.output / "environment_packages.txt").write_text("\n".join(packages) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
