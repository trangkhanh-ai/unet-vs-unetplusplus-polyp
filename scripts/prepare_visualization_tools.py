"""Download a portable, SHA-checked Graphviz on Windows for Torchview.

Does not install system-wide software or change the registry/PATH permanently.
Linux/macOS: install Graphviz with your package manager and use `dot` on PATH.
"""
from pathlib import Path
import hashlib
import os
import re
import subprocess
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
URL = "https://gitlab.com/api/v4/projects/4207231/packages/generic/graphviz-releases/16.1.0/windows_10_cmake_Release_Graphviz-16.1.0-win64.zip"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if os.name != "nt":
        raise SystemExit("Install Graphviz using your package manager, then check: dot -V")
    # Graphviz's Windows plugin loader needs a cache path without the repo's
    # Vietnamese folder names. LOCALAPPDATA belongs to the current user.
    directory = Path(os.environ["LOCALAPPDATA"]) / "polyp-course-tools" / "Graphviz-16.1.0"
    existing = list(directory.glob("**/bin/dot.exe")) if directory.exists() else []
    if existing:
        subprocess.run([str(existing[0]), "-c"], cwd=existing[0].parent, check=True)
        subprocess.run([str(existing[0]), "-V"], check=True)
        print(existing[0])
        return
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "graphviz.zip"
    with urllib.request.urlopen(URL + ".sha256", timeout=60) as response:
        digest_text = response.read().decode("utf-8")
    expected = re.search(r"\b[0-9a-fA-F]{64}\b", digest_text)
    if expected is None:
        raise RuntimeError("Graphviz's official checksum could not be read")
    print("Downloading official Graphviz 16.1.0 portable archive", flush=True)
    urllib.request.urlretrieve(URL, archive)
    if sha256(archive) != expected.group(0).lower():
        raise RuntimeError("Graphviz archive checksum mismatch")
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            target = (directory / member.filename).resolve()
            if not target.is_relative_to(directory.resolve()):
                raise RuntimeError("Unsafe archive member")
        source.extractall(directory)
    dot = next(directory.glob("**/bin/dot.exe"), None)
    if dot is None:
        raise RuntimeError("Graphviz archive did not contain bin/dot.exe")
    # Portable folder belongs to this repo; registration writes only here.
    subprocess.run([str(dot), "-c"], cwd=dot.parent, check=True)
    subprocess.run([str(dot), "-V"], check=True)
    print(dot)


if __name__ == "__main__":
    main()
