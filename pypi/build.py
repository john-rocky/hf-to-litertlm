#!/usr/bin/env python3
"""Build the hf-to-litertlm wheel and sdist into pypi/dist/.

    python3 pypi/build.py

The package is a launcher (hf_to_litertlm/cli.py) plus hf_to_litertlm/payload.zip: the files in
PAYLOAD, read from this checkout's HEAD, and MANIFEST.json (each file's sha256 and the commit).
The launcher unpacks them and runs scripts/convert.py with `uv run --frozen`, so PAYLOAD must hold
every file the one-command path opens. A package or payload file with uncommitted changes stops the
build: the wheel is always a commit of this repo.
"""
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PAYLOAD = [
    ".python-version", "pyproject.toml", "uv.lock", "LICENSE",
    "scripts/convert.py", "scripts/verify_lock.py", "scripts/check_bundle.py",
    "scripts/verify_quality.py", "scripts/litert_lm_cli_verifier.py",
    "scripts/chat_template_dual.py", "scripts/failure_line.py", "scripts/failure_hints.json",
    "scripts/section_identity.py", "tools/chat_template_prefix/desk_render.py",
    "recipes/qwen3_int4_block32_octav.json", "recipes/curated_index.json",
    "lfm_work/add_executor_metadata.py", "granite_work/drop_start_token.py",
]
PACKAGE_FILES = ["pyproject.toml", "README.md", "hf_to_litertlm/__init__.py",
                 "hf_to_litertlm/__main__.py", "hf_to_litertlm/cli.py"]


def git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                          check=True).stdout


def payload_zip():
    """payload.zip from HEAD's blobs and git modes, entries sorted, every timestamp 1980-01-01."""
    dirty = git("status", "--porcelain", "--", *PAYLOAD, *(f"pypi/{p}" for p in PACKAGE_FILES)).strip()
    if dirty:
        sys.exit("FAIL build: package or payload files differ from HEAD -> next: commit them\n" + dirty)
    commit = git("rev-parse", "HEAD").strip()
    modes = {}
    for line in git("ls-tree", "HEAD", "--", *PAYLOAD).splitlines():
        meta, rel = line.split("\t", 1)
        modes[rel] = int(meta.split()[0], 8) & 0o777
    files = {}
    for rel in PAYLOAD:
        files[rel] = subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{rel}"],
                                    capture_output=True, check=True).stdout
        if rel.endswith(".py"):
            compile(files[rel], rel, "exec")
    manifest = {"repo": "https://github.com/john-rocky/hf-to-litertlm", "commit": commit,
                "files": {rel: hashlib.sha256(b).hexdigest() for rel, b in sorted(files.items())}}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for rel, data in [("MANIFEST.json", (json.dumps(manifest, indent=2) + "\n").encode())] + \
                sorted(files.items()):
            info = zipfile.ZipInfo(rel, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = modes.get(rel, 0o644) << 16
            z.writestr(info, data)
    return buf.getvalue(), manifest


def main():
    data, manifest = payload_zip()
    dist = HERE / "dist"
    with tempfile.TemporaryDirectory(prefix="hf-to-litertlm-build-") as td:
        stage = Path(td)
        for rel in PACKAGE_FILES:
            (stage / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(HERE / rel, stage / rel)
        shutil.copyfile(REPO / "LICENSE", stage / "LICENSE")
        (stage / "hf_to_litertlm" / "payload.zip").write_bytes(data)
        subprocess.run(["uv", "build", "--out-dir", str(dist), str(stage)], check=True)
    print(f"payload: {len(manifest['files'])} files at {manifest['commit'][:12]}, "
          f"payload.zip {len(data):,} B sha256 {hashlib.sha256(data).hexdigest()}")
    for f in sorted(dist.iterdir()):
        print(f"{f.name}  {f.stat().st_size:,} B  sha256 {hashlib.sha256(f.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
