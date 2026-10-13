"""Điểm chạy chung; phần nghiệp vụ vẫn nằm trong folder từng thành viên."""
import importlib
from pathlib import Path
import sys

COMMANDS = {
    "audit": "members.khanh.works_done.audit_data",
    "test": "members.duy.works_done.verify_project",
    "train": "members.huy.works_done.train",
    "lock": "members.duy.works_done.lock_protocol",
    "evaluate": "members.tri.works_done.evaluate",
    "compare": "members.tri.works_done.compare",
    "predict": "members.tri.works_done.predict",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("--help", "-h"):
        print("Polyp project: python project.py <command> [arguments]")
        print("Commands: audit, test, train, lock, evaluate, compare, predict, demo")
        print("Ví dụ: python project.py train --config configs/unet.json --output runs/A_unet_seed42")
        return
    command, arguments = sys.argv[1], sys.argv[2:]
    if command == "demo":
        from streamlit.web import cli
        script = Path(__file__).resolve().parent / "members/tri/works_done/demo.py"
        sys.argv = ["streamlit", "run", str(script), "--server.address", "127.0.0.1",
                    "--server.headless", "true", *arguments]
        raise SystemExit(cli.main())
    if command not in COMMANDS:
        raise SystemExit(f"Command không hợp lệ: {command}; chạy --help")
    module = importlib.import_module(COMMANDS[command])
    sys.argv = [COMMANDS[command], *arguments]
    module.main()


if __name__ == "__main__":
    main()
