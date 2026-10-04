"""Create/check SHA-256 evidence for actual inputs. Does not generate or change data."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = Path("members/khanh/works_done/dataset_splits.csv")


def resolve(relative):
    path = (ROOT / Path(relative.replace("\\", "/"))).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Snapshot paths must remain inside the repository.")
    return path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_snapshot(destination):
    paths = {MANIFEST.as_posix(), "members/khanh/works_done/dataset.py",
        "members/khanh/works_done/prepare_data_splits.py", "members/README.md",
        "members/huy/works_done/check_loader_readiness.py", "members/huy/works_done/input_snapshot.py"}
    with (ROOT / MANIFEST).open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            paths.update(row[key].replace("\\", "/") for key in ("image_path", "mask_path"))
    result = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "files": {name: sha256(resolve(name)) for name in sorted(paths)}}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Saved {len(paths)} actual-file checksums to {destination}")


def check_snapshot(source):
    data = json.loads(source.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("Unknown snapshot version.")
    changed = []
    for name, expected in data["files"].items():
        path = resolve(name)
        if not path.is_file() or sha256(path) != expected:
            changed.append(name)
    print(json.dumps({"checked_files": len(data["files"]), "unchanged": not changed, "changed_or_missing": changed}, indent=2))
    return 1 if changed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--create", type=Path)
    group.add_argument("--check", type=Path)
    args = parser.parse_args()
    if args.create:
        create_snapshot(args.create)
    else:
        raise SystemExit(check_snapshot(args.check))
