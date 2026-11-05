#!/usr/bin/env python3
"""Download what the checks read and this repository does not hold, into the work directory (the parent of this folder). Standard library only.

    python3 conversion/fetch_inputs.py author                # author/: the author's score.py and evaluate.py, the three official examples (a pinned commit of OpenBMB/MiniCPM-OCR)
    python3 conversion/fetch_inputs.py fixtures              # bench/: the rows, and the pictures of the records in sets/fixtures.json
    python3 conversion/fetch_inputs.py family [--limit N]    # bench/: the rows, and the pictures of the benchmark's olmOCR family (446 records, 892 files)

Every file of the commit and the benchmark's rows are checked against their SHA-256. A file that exists is kept, so a stopped run resumes. The pictures and texts stay in these
folders, on your disk: they are not part of this repository. Only the olmOCR family is fetched; openbmb/JevOCRBench's DATA_LICENSE.md and NOTICE give its terms (ODC-BY-1.0,
https://huggingface.co/datasets/allenai/olmOCR-bench, cite Poznanski et al., arXiv:2502.18443), and the official examples are crops of OmniDocBench pages (research use only)."""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMMIT = "8177040f0d6c824ccf87bc18b938b6a0f91b6d0e"
GITHUB = f"https://raw.githubusercontent.com/OpenBMB/MiniCPM-OCR/{COMMIT}/"
REVISION = "a26c4b32dcd3a95490842ca637b80edd2b71d975"
BENCH = f"https://huggingface.co/datasets/openbmb/JevOCRBench/resolve/{REVISION}/"
ROWS = "data/jevocrbench.jsonl"
ROWS_SHA256 = "7a4546d121055b7a03ef68967138d3b15aa30d270c253594b1d9e61b09e57f1d"
AUTHOR = {
    "scripts/minicpm-ocr-jev/eval/score.py": "de26bfc2b92b2c8df64caa826f8606784948bc24a59438c50605af4f34198fa2",
    "scripts/minicpm-ocr-jev/eval/evaluate.py": "f56309e00c7756804e798046c83ea6f947fd684739025a28a9dd00fe0dfb0d49",
    "scripts/minicpm-ocr-jev/examples/01_text_putnam/case.json": "a606b4df23ff288fd608a03101eef9ecc8a8f2e13d97ee14dc69a220a8692754",
    "scripts/minicpm-ocr-jev/examples/01_text_putnam/block.png": "a05e3bf48c674d25b9dfe684fec813098816943895551161e0da3a8ce13eb069",
    "scripts/minicpm-ocr-jev/examples/01_text_putnam/reference_render.png": "3f703d528157b3f25092db29a3cff7ab68e067669eb9093f80582f3a28d37a9d",
    "scripts/minicpm-ocr-jev/examples/02_formula_putnam/case.json": "c45f7dee39a26a6b395f097e09c73aa70511366ad8e3b0795f38045183849283",
    "scripts/minicpm-ocr-jev/examples/02_formula_putnam/block.png": "a63f5675a76abfc6e7809051fa33b4c96ec358a5220cabecdf6a7f6401c374f9",
    "scripts/minicpm-ocr-jev/examples/02_formula_putnam/reference_render.png": "62267b9c26275fae150bd7d3298f0f9d37fadaa7744abef46bd6e163ce1e543c",
    "scripts/minicpm-ocr-jev/examples/03_table_gb_standard/case.json": "f8b572c41b4cb2a0f53712b4ed4cf2ad2c7db6574821b9d47d2cb79111daf101",
    "scripts/minicpm-ocr-jev/examples/03_table_gb_standard/block.png": "3a4835f45bebb1d23c945294ca6f187e90a0f06e604a6c3993a2b39da36c481d",
    "scripts/minicpm-ocr-jev/examples/03_table_gb_standard/reference_render.png": "6675d7cdf5b09c09afe79573658fd8d055fd72988e6145bf3d48cbbfe16d363b",
}


def fetch(url: str, path: str, sha256: str = "", pace: float = 0.0) -> None:
    if os.path.exists(path):
        if sha256 and hashlib.sha256(open(path, "rb").read()).hexdigest() != sha256:
            sys.exit(f"{path} exists and its SHA-256 is not {sha256}")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for attempt in range(6):
        time.sleep(pace)
        try:
            data = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "fetch_inputs/1.0"}), timeout=120).read()
            break
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == 5:
                sys.exit(f"{url}: {e}")
            time.sleep(30 * (attempt + 1) if getattr(e, "code", 0) == 429 else 5 * (attempt + 1))
    if sha256 and hashlib.sha256(data).hexdigest() != sha256:
        sys.exit(f"{url}: SHA-256 {hashlib.sha256(data).hexdigest()}, want {sha256}")
    with open(path + ".tmp", "wb") as f:
        f.write(data)
    os.rename(path + ".tmp", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=("author", "fixtures", "family"))
    ap.add_argument("--limit", type=int, default=0, help="family: only the first N records")
    ap.add_argument("--pace", type=float, default=0.7, help="seconds before each download of a picture (the Hub limits calls per IP)")
    a = ap.parse_args()
    if a.what == "author":
        for rel, want in AUTHOR.items():
            fetch(GITHUB + rel, os.path.join(WORK, "author", rel), want)
        print(f"author/: {len(AUTHOR)} files, SHA-256 checked")
        return
    fetch(BENCH + ROWS, os.path.join(WORK, "bench", ROWS), ROWS_SHA256)
    rows = {r["item_id"]: r for r in map(json.loads, open(os.path.join(WORK, "bench", ROWS), encoding="utf-8"))}
    if a.what == "fixtures":
        fx = json.load(open(os.path.join(WORK, "conversion", "sets", "fixtures.json")))
        files = {rows[d["item_id"]][k] for d in fx["decide"] for k in ("image_path", "render_image_path")}
        files |= {rows[t["item_id"]]["image_path" if t["which"] == 0 else "render_image_path"] for t in fx["tower"]}
    else:
        family = sorted((r for r in rows.values() if str(r["source"]).startswith("olmocr_")), key=lambda r: r["item_id"])
        if a.limit:
            family = family[:a.limit]
        files = {r[k] for r in family for k in ("image_path", "render_image_path")}
    for i, rel in enumerate(sorted(files)):
        fetch(BENCH + rel, os.path.join(WORK, "bench", rel), pace=a.pace)
        if i % 100 == 0:
            print(f"{i + 1} / {len(files)}", flush=True)
    print(f"bench/: rows checked, {len(files)} pictures")


if __name__ == "__main__":
    main()
