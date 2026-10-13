"""Cài môi trường riêng của repo từ Python3.12, CPU hoặc CUDA12.8.

python scripts/setup_environment.py --device cuda
Thêm --visualizations hoặc --reports để cài công cụ xuất hình/Word/slide.
"""
import argparse
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--visualizations", action="store_true")
    parser.add_argument("--reports", action="store_true")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Dùng Python3.12 để theo đúng môi trường đồ án")
    directory = ROOT / ".venv"
    python = directory / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(directory)
    wheel = "cu128" if args.device == "cuda" else "cpu"
    commands = [
        ["-m", "pip", "install", f"torch==2.11.0+{wheel}", "--index-url", f"https://download.pytorch.org/whl/{wheel}"],
        ["-m", "pip", "install", "-r", "requirements.txt"],
    ]
    if args.visualizations:
        commands.append(["-m", "pip", "install", "-r", "requirements-viz.txt"])
    if args.reports:
        commands.append(["-m", "pip", "install", "-r", "requirements-report.txt"])
    commands.append(["-m", "pip", "check"])
    for command in commands:
        subprocess.run([str(python), *command], cwd=ROOT, check=True)
    if args.visualizations and sys.platform == "win32":
        subprocess.run([str(python), "scripts/prepare_visualization_tools.py"], cwd=ROOT, check=True)
    print("Đã cài. Python: " + str(python), flush=True)


if __name__ == "__main__":
    main()
