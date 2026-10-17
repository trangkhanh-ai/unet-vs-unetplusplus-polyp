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


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True,
                                   encoding="utf-8").strip()


def create_snapshot(destination):
    # Snapshot the current sources, not the removed historical members/README.md.
    # Include untracked Python additions so HEAD is never presented as their revision.
    paths = {MANIFEST.as_posix(), ".gitattributes", "README.md", "project.py"}
    paths.update(p.relative_to(ROOT).as_posix() for p in (ROOT / "members").rglob("*.py"))
    paths.update(p.relative_to(ROOT).as_posix() for p in (ROOT / "scripts").glob("*.py"))
    paths.update(p.relative_to(ROOT).as_posix() for p in (ROOT / "configs").glob("*.json"))
    paths.update(p.name for p in ROOT.glob("requirements*.txt"))
    with (ROOT / MANIFEST).open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            paths.update(row[key].replace("\\", "/") for key in ("image_path", "mask_path"))
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    result = {"schema_version": 2, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hash_policy": "sha256_raw_bytes_all_files; no text or image normalization",
        "scope": "listed source/config/dependency files and every original manifest image/mask",
        "git_head_commit": git("rev-parse", "HEAD"),
        "git_head_tree": git("rev-parse", "HEAD^{tree}"),
        "branch": git("branch", "--show-current"),
        "working_tree_dirty": bool(status), "working_tree_status": status.splitlines(),
        "revision_note": "HEAD is the base revision; files hashes describe executed working-tree bytes, including uncommitted files.",
        "attributes_sha256": sha256(ROOT / ".gitattributes"),
        "files": {name: sha256(resolve(name)) for name in sorted(paths)}}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Saved {len(paths)} actual-file checksums to {destination}")


def check_snapshot(source):
    data = json.loads(source.read_text(encoding="utf-8"))
    if data.get("schema_version") not in (1, 2):
        raise ValueError("Unknown snapshot version.")
    changed = []
    for name, expected in data["files"].items():
        path = resolve(name)
        if not path.is_file() or sha256(path) != expected:
            changed.append(name)
    print(json.dumps({"schema_version": data["schema_version"],
        "checked_files": len(data["files"]), "unchanged": not changed,
        "changed_or_missing": changed,
        "note": "Only listed files are checked. An old snapshot is historical evidence, not the current source revision."}, indent=2))
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
