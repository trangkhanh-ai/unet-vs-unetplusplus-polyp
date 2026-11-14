"""Đóng gói repo thành 1.ChuongTrinh.zip, giữ nguyên byte source và checkpoint.

Không đưa venv, Git, cache, run pilot hoặc file Word vào ZIP chương trình.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RUNS = ("A_unet_seed42", "B_unetplusplus_seed42", "C_unetplusplus_ds_seed42")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def selected_files(include_last=False):
    paths = set()
    for name in ("CacFileSource.py", "project.py", "README.md", ".gitattributes",
                 "requirements.txt", "requirements-duy.txt", "requirements-viz.txt", "requirements-report.txt"):
        p = ROOT / name
        if p.is_file():
            paths.add(p)
    for folder in ("scripts", "configs", "Kvasir-SEG"):
        for p in (ROOT / folder).rglob("*"):
            if p.is_file() and p.suffix.lower() in {".py", ".ps1", ".json", ".csv", ".png", ".jpg", ".jpeg"}:
                paths.add(p)
    for owner in ("duy", "khanh", "phuc", "huy", "tri"):
        for p in (ROOT / "members" / owner / "works_done").rglob("*"):
            if p.is_file() and p.suffix.lower() in {".py", ".json", ".csv", ".txt", ".png", ".svg"}:
                # Artifacts có thể gồm dữ liệu chẩn đoán cục bộ; chỉ lấy hình kiến trúc.
                relative = p.relative_to(ROOT).as_posix()
                if "/artifacts/" in relative and "/architecture/" not in relative:
                    continue
                paths.add(p)
    for name in RUNS:
        run = ROOT / "runs" / name
        for p in run.glob("*"):
            if p.is_file() and (p.suffix in {".json", ".csv", ".txt"} or p.name == "best.pt" or include_last and p.name == "last.pt"):
                paths.add(p)
    result_folders = ("A_test", "B_test", "C_test", "comparison", "data_audit", "report_figures",
                      "model_graphs", "verification", "verification_bf16_v2", "submission_checks_20261111",
                      "submission_reproduction_20261111", "submission_heads_20261111_val", "submission_heads_20261111_test")
    for folder in result_folders:
        for p in (ROOT / "results" / folder).rglob("*"):
            if p.is_file() and p.suffix.lower() in {".json", ".csv", ".png", ".svg", ".gv", ".txt"}:
                paths.add(p)
    return sorted(p for p in paths if "__pycache__" not in p.parts)


def build(output, include_last=False):
    release = json.loads((ROOT / "configs/pretrained_release.json").read_text(encoding="utf-8"))
    for relative, expected in release["source_hashes"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Source hoặc manifest đã đổi: {relative}")
    for asset in release["assets"]:
        if asset["kind"] == "best" or include_last:
            if digest(ROOT / asset["path"]) != asset["sha256"]:
                raise ValueError(f"Checkpoint khác: {asset['path']}")
    paths = selected_files(include_last)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Giữ bộ nộp cũ; chọn tên ZIP mới")
    output.parent.mkdir(parents=True, exist_ok=True)
    inventory = [dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, sha256=digest(p)) for p in paths]
    info = dict(schema_version=1, folder="1.ChuongTrinh", file_count=len(paths),
                checkpoint_policy="Original best.pt retained; no slimming or reserialization",
                files=inventory)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for p, item in zip(paths, inventory):
            archive.write(p, "1.ChuongTrinh/" + item["path"])
        archive.writestr("1.ChuongTrinh/submission_manifest.json", json.dumps(info, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"ZIP CRC failed: {bad}")
    receipt = dict(path=str(output), bytes=output.stat().st_size, sha256=digest(output), files=len(paths))
    output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="reports/submission_20261111/1.ChuongTrinh.zip")
    p.add_argument("--include-last", action="store_true")
    args = p.parse_args()
    build(args.output, args.include_last)
