"""Check the six CC0 photographs of the fixtures (photos.json) against Wikimedia Commons and against each other.

    python fixtures/fetch_photos.py            # downloads each thumbnail URL once (about 1.2 MB) and compares
    python fixtures/fetch_photos.py --local    # no network: the local files only

The photographs the checks read are in images/ (PNG) and the pinned thumbnails they were decoded from in photos/.
For each photograph this prints whether:
  - images/<file> has the recorded SHA-256 (`sha256`) and photos/<name>.jpg the pinned one (`thumbnail.sha256`);
  - Pillow decodes photos/<name>.jpg to the RGB pixels of images/<file> (`rgb_sha256`) here;
  - Commons still serves the pinned thumbnail at `thumbnail.url` (skipped with --local).
It writes nothing. Exit 0 when every check holds, 1 otherwise (a thumbnail that Commons now encodes differently is
reported, and the files here stay what the checks used). Needs Pillow (host/requirements-host.txt)."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
USER_AGENT = "clef-flash-litert-fixtures/1.0 (python-urllib)"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> int:
    from PIL import Image

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--local", action="store_true", help="skip the download")
    a = ap.parse_args()
    bad = 0
    for p in json.loads((HERE / "photos.json").read_text())["photos"]:
        png = (HERE / p["file"]).read_bytes()
        jpg = (HERE / p["thumbnail"]["file"]).read_bytes()
        rgb = sha(Image.open(io.BytesIO(jpg)).convert("RGB").tobytes())
        notes = [f"png {'ok' if sha(png) == p['sha256'] else 'DIFFERS'}",
                 f"thumbnail {'ok' if sha(jpg) == p['thumbnail']['sha256'] else 'DIFFERS'}",
                 f"decode {'= the png' if rgb == p['rgb_sha256'] else 'DIFFERS from the png'}"]
        ok = sha(png) == p["sha256"] and sha(jpg) == p["thumbnail"]["sha256"] and rgb == p["rgb_sha256"]
        if not a.local:
            req = urllib.request.Request(p["thumbnail"]["url"], headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as resp:
                remote = resp.read()
            if sha(remote) == p["thumbnail"]["sha256"]:
                notes.append("Commons serves the pinned thumbnail")
            else:
                same = sha(Image.open(io.BytesIO(remote)).convert("RGB").tobytes()) == p["rgb_sha256"]
                notes.append(f"Commons now serves other bytes ({len(remote):,} B, SHA-256 {sha(remote)[:16]}...; "
                             f"pixels {'the same' if same else 'differ'})")
                ok = False
        bad += not ok
        print(f"{p['name']} ({', '.join(p['records'])}; {p['author']}, {p['license']}): " + "; ".join(notes))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
