"""Lấy checkpoint đã huấn luyện từ GitHub Release, kiểm tra SHA256.

Chỉ dùng thư viện chuẩn. Không cần token GitHub cho release công khai.
Mặc định lấy ba best.pt; thêm --include-last để tiếp tục huấn luyện.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_sources(manifest):
    for relative, expected in manifest["source_hashes"].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or sha256_file(path) != expected:
            raise ValueError(f"Source/manifest không khớp bản huấn luyện: {relative}")


def check_data():
    csv_path = ROOT / "members/khanh/works_done/dataset_splits.csv"
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    missing = [row[key] for row in rows for key in ("image_path", "mask_path")
               if not (ROOT / row[key].replace("\\", "/")).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Thiếu {len(missing)} ảnh/mask. Đặt Kvasir-SEG/images và masks ở repo root; "
            "giữ đúng ID trong CSV. Xem mục dữ liệu trong README.")
    return len(rows)


def download_asset(asset):
    target = (ROOT / asset["path"]).resolve()
    if not target.is_relative_to(ROOT):
        raise ValueError("Checkpoint path nằm ngoài repo")
    if target.exists():
        if sha256_file(target) != asset["sha256"]:
            raise ValueError(f"Không ghi đè checkpoint khác: {asset['path']}")
        print(f"Đã có và đúng SHA256: {asset['path']}", flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".pt.part")
    request = urllib.request.Request(asset["url"], headers={"User-Agent": "polyp-course-project"})
    received = 0
    next_notice = 16 * 1024 * 1024
    print(f"Tải {asset['name']} ({asset['bytes'] / 1024**2:.1f} MiB)", flush=True)
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as stream:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            stream.write(chunk)
            received += len(chunk)
            if received >= next_notice:
                print(f"  {received / asset['bytes']:.0%}", flush=True)
                next_notice += 16 * 1024 * 1024
    if received != asset["bytes"] or sha256_file(partial) != asset["sha256"]:
        raise ValueError(f"Download chưa đúng kích thước/SHA256: {asset['name']}")
    partial.replace(target)
    print(f"Đã xác minh: {asset['path']}", flush=True)


def prepare(include_last=False):
    manifest = json.loads((ROOT / "configs/pretrained_release.json").read_text(encoding="utf-8"))
    check_sources(manifest)
    count = check_data()
    for asset in manifest["assets"]:
        if asset["kind"] == "best" or include_last:
            download_asset(asset)
    print(f"Sẵn sàng: {count} cặp dữ liệu, source đúng SHA, checkpoint hợp lệ.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-last", action="store_true")
    args = parser.parse_args()
    prepare(args.include_last)


if __name__ == "__main__":
    main()
