"""Đóng gói code, tài liệu và kết quả; ghi rõ checkpoint lớn nằm ngoài ZIP.

Chạy sau khi báo cáo và slide đã kiểm tra. Không đọc ảnh dữ liệu gốc.
"""
from pathlib import Path
import hashlib
import json
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT = PROJECT_ROOT / "reports" / "ban_giao"
ORIGINAL_PLAN = PROJECT_ROOT.parent / "KE_HOACH_5_THANH_VIEN_UNETPP_KVASIR_SEG.docx"
FOLDERS = ("members", "configs", "docs", "scripts", "results", "runs", "reports")
EXTENSIONS = {".py", ".ps1", ".md", ".json", ".csv", ".txt", ".log", ".png", ".svg", ".docx", ".pptx"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_name(path):
    if path == ORIGINAL_PLAN:
        return "plans/ke_hoach_ban_dau.docx"
    return path.relative_to(PROJECT_ROOT).as_posix()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    files = [PROJECT_ROOT / name for name in ("project.py", "README.md", ".gitignore", "requirements.txt", "requirements-lock-gpu.txt")]
    for folder in FOLDERS:
        for path in (PROJECT_ROOT / folder).rglob("*"):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            relative = path.relative_to(PROJECT_ROOT)
            if "__pycache__" in relative.parts or path.is_relative_to(OUTPUT):
                continue
            if (relative.parts[:2] == ("reports", "qa")
                    and path.name not in {"word_quality_check.json", "slides_quality_check.json"}):
                continue
            files.append(path)
    files = sorted(set(files))
    if ORIGINAL_PLAN.is_file():
        files.append(ORIGINAL_PLAN)
    weights = []
    checkpoint_paths = []
    for folder in ("runs", "members"):
        checkpoint_paths.extend(p for p in (PROJECT_ROOT / folder).rglob("*")
                                if p.is_file() and p.suffix.lower() in {".pt", ".pth"})
    for path in sorted(set(checkpoint_paths)):
        weights.append({"path": path.relative_to(PROJECT_ROOT).as_posix(),
                       "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = {
        "purpose": "Source, documentation and measured outputs for the course project",
        "excluded": ["Kvasir-SEG original images/masks", "Python environments", "Git internals",
                     "large .pt checkpoints (stored locally and listed separately)", "internal render QA", "probability .npy arrays"],
        "notes": ["plans/ke_hoach_ban_dau.docx giữ nguyên kế hoạch ban đầu; giao diện code hiện tại được giải thích trong docs/INTERFACES.md"],
        "files": [{"path": archive_name(p), "bytes": p.stat().st_size,
                   "sha256": sha256(p)} for p in files],
        "local_checkpoints": weights,
    }
    manifest_path = OUTPUT / "DANH_MUC_BAN_GIAO.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    archive_path = OUTPUT / "CODE_TAI_LIEU_KET_QUA_DO_AN.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, archive_name(path))
        archive.write(manifest_path, "DANH_MUC_BAN_GIAO.json")
    print(json.dumps({"archive": str(archive_path), "files": len(files),
                      "bytes": archive_path.stat().st_size, "sha256": sha256(archive_path),
                      "local_checkpoints": len(weights)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
