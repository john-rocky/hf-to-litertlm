"""The source's three official examples on this repository's LiteRT files, through the PyTorch-free host.

    python examples/run_example.py                         # the GPU at float32 (Metal on a Mac), the two multi-signature files
    python examples/run_example.py --check                 # also compare with the author's published scores, and test the refusals
    python examples/run_example.py --accel cpu --weight-cache /path/to/folder --check     # CPU (XNNPACK), weight cache in that folder
    python examples/run_example.py --accel cpu --form files --check                       # one file per bucket

The examples are the source's own: a crop of a document page, a render of an OCR text, and the text. Their pictures come from
OmniDocBench, so they are not in this repository: the script downloads them into memory from a pinned commit of
github.com/OpenBMB/MiniCPM-OCR (scripts/minicpm-ocr-jev/examples/) and checks each file's SHA-256 first.

--check passes when each example's prob_1 is within 0.02 of the author's `observed_prob_yes` (the author's tutorial: "A score within
0.02 of the reference is normal"), and every request the host must refuse comes back with the expected status: a literal pad token in
the text, 16-bit pictures, a broken file, a request above the top bucket. It prints CHECK PASS or CHECK FAIL and exits 1 on a failure.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import struct
import sys
import tempfile
import time
import urllib.request
import zlib

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "host"))

COMMIT = "8177040f0d6c824ccf87bc18b938b6a0f91b6d0e"
BASE = f"https://raw.githubusercontent.com/OpenBMB/MiniCPM-OCR/{COMMIT}/scripts/minicpm-ocr-jev/examples"
CASES = ("01_text_putnam", "02_formula_putnam", "03_table_gb_standard")
TOLERANCE = 0.02
SHA256 = {
    "01_text_putnam/case.json": "a606b4df23ff288fd608a03101eef9ecc8a8f2e13d97ee14dc69a220a8692754",
    "01_text_putnam/block.png": "a05e3bf48c674d25b9dfe684fec813098816943895551161e0da3a8ce13eb069",
    "01_text_putnam/reference_render.png": "3f703d528157b3f25092db29a3cff7ab68e067669eb9093f80582f3a28d37a9d",
    "02_formula_putnam/case.json": "c45f7dee39a26a6b395f097e09c73aa70511366ad8e3b0795f38045183849283",
    "02_formula_putnam/block.png": "a63f5675a76abfc6e7809051fa33b4c96ec358a5220cabecdf6a7f6401c374f9",
    "02_formula_putnam/reference_render.png": "62267b9c26275fae150bd7d3298f0f9d37fadaa7744abef46bd6e163ce1e543c",
    "03_table_gb_standard/case.json": "f8b572c41b4cb2a0f53712b4ed4cf2ad2c7db6574821b9d47d2cb79111daf101",
    "03_table_gb_standard/block.png": "3a4835f45bebb1d23c945294ca6f187e90a0f06e604a6c3993a2b39da36c481d",
    "03_table_gb_standard/reference_render.png": "6675d7cdf5b09c09afe79573658fd8d055fd72988e6145bf3d48cbbfe16d363b",
}


def fetch(name: str, sha256: dict = SHA256, base: str = BASE) -> bytes:
    """One file of the pinned commit, into memory, its SHA-256 checked first."""
    data = urllib.request.urlopen(f"{base}/{name}", timeout=60).read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != sha256[name]:
        raise SystemExit(f"{name}: SHA-256 {digest}, want {sha256[name]}")
    return data


def png(width: int, height: int, depth: int, color_type: int, rows: np.ndarray) -> bytes:
    """A PNG written by hand: Pillow does not write 16-bit RGB. rows = one row per line, big-endian samples."""
    def chunk(tag, body):
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body))

    raw = b"".join(b"\x00" + rows[i].tobytes() for i in range(height))
    head = struct.pack(">IIBBBBB", width, height, depth, color_type, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", head) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def refusals(host, block: bytes, render: bytes, text: str) -> dict:
    """Each request the host must refuse: {name: (expected status, the host's answer)}."""
    h, w = 64, 96
    ramp = (np.arange(h * w * 4, dtype=np.uint32) * 11 % 65536).astype(">u2").reshape(h, w, 4)
    out = {}
    img = lambda data: Image.open(io.BytesIO(data))  # noqa: E731
    with tempfile.TemporaryDirectory() as d:
        def score_files(name, data, status, suffix=".png"):
            p = os.path.join(d, name + suffix)
            with open(p, "wb") as f:
                f.write(data)
            out[name] = (status, host.score(p, img(render), text))

        out["image_pad_token_in_text"] = ("literal_pad_token", host.score(img(block), img(render), f"{text} <|image_pad|>"))
        out["video_pad_token_in_text"] = ("literal_pad_token", host.score(img(block), img(render), f"{text} <|video_pad|>"))
        score_files("rgb_16_bit_png", png(w, h, 16, 2, ramp[:, :, :3]), "not_8_bit")
        score_files("rgba_16_bit_png", png(w, h, 16, 6, ramp), "not_8_bit")
        score_files("gray_16_bit_png", png(w, h, 16, 0, ramp[:, :, 0]), "not_8_bit")
        score_files("truncated_png", block[:len(block) // 2], "unreadable_picture")
        score_files("not_a_picture", b"this is not a picture\n", "unreadable_picture")
        out["missing_file"] = ("unreadable_picture", host.score(os.path.join(d, "no_such_file.png"), img(render), text))
        out["a_side_of_0"] = ("unreadable_picture", host.score(Image.new("RGB", (0, 16)), img(render), text))
    max_len, max_n = host.decide.max_len, host.tower.max_n
    out["text_above_the_top_length"] = ("too_large", host.score(img(block), img(render), "word " * (max_len + 300)))
    if max_n < 16384:     # 2,048 x 2,048 is the picture budget's largest grid, 16,384 patches
        big = Image.new("RGB", (2048, 2048), "white")
        out["picture_above_the_top_bucket"] = ("too_large", host.score(big, img(render), text))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--accel", choices=("gpu", "cpu"), default="gpu")
    ap.add_argument("--form", choices=("multi", "files"), default="multi")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--keep", type=int, default=1, help="--form files: bucket files compiled at a time per kind")
    ap.add_argument("--weight-cache", default="", help="--form multi on the CPU: the folder of the XNNPACK weight cache files")
    ap.add_argument("--repo", default=ROOT, help="the folder that holds the graphs, tokenizer.json and tables/ (default: this repository)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default="", help="write the numbers as JSON (no picture, no text)")
    a = ap.parse_args()
    if a.accel == "cpu" and a.form == "multi" and not a.weight_cache:
        ap.error("--form multi on the CPU needs --weight-cache FOLDER (the first load writes the cache files there), or use --form files")
    from decide_host import load_host

    files = {name: fetch(name) for name in SHA256}
    t0 = time.time()
    results, refused, fails = {}, {}, []
    with load_host(a.repo, accelerator=a.accel, form=a.form, threads=a.threads, keep=a.keep, weight_cache_dir=a.weight_cache or None) as host:
        print(f"loaded in {time.time() - t0:.1f} s, accel {a.accel}, form {a.form}, top length {host.decide.max_len}, top patches {host.tower.max_n}", flush=True)
        for name in CASES:
            case = json.loads(files[f"{name}/case.json"])
            role = case["reference_render_of"]
            text = next(c["text"] for c in case["candidates"] if c["role"] == role)
            observed = case["observed_prob_yes"][role]
            block, render = files[f"{name}/{case['block_image']}"], files[f"{name}/{case['reference_render']}"]
            out = host.score(Image.open(io.BytesIO(block)), Image.open(io.BytesIO(render)), text)
            if out["status"] != "ok":
                fails.append(f"{name}: {out}")
                continue
            diff = out["prob_1"] - observed
            results[name] = {"case": case["case_id"], "category": case["category"], "prob_1": out["prob_1"], "observed": observed, "diff": diff,
                             "logits": out["logits"], "T": out["T"], "n": out["n"], "grids": out["grids"], "pooled_id": out["pooled_id"], "ids_sha": out["ids_sha"],
                             "hidden_sha": hashlib.sha256(out["hidden_pooled"].tobytes()).hexdigest()[:16],
                             "decide_bucket": host.decide.last_bucket, "tower_bucket": host.tower.last_bucket}
            print(f"{name}: prob_1 {out['prob_1']:.5f}, author {observed:.5f}, difference {diff:+.5f}, {out['T']} tokens, patches {out['n']}", flush=True)
            if abs(diff) > TOLERANCE:
                fails.append(f"{name}: prob_1 {out['prob_1']} vs {observed}, more than {TOLERANCE} apart")
        if a.check:
            last = json.loads(files[f"{CASES[0]}/case.json"])
            text = next(c["text"] for c in last["candidates"] if c["role"] == last["reference_render_of"])
            for name, (want, got) in refusals(host, files[f"{CASES[0]}/block.png"], files[f"{CASES[0]}/reference_render.png"], text).items():
                doc = {k: v for k, v in got.items() if k in ("status", "why", "mode", "raw", "n", "T", "limit")}
                refused[name] = {"expected": want, **doc}
                print(f"refused {name}: {json.dumps(doc)}", flush=True)
                if got["status"] != want:
                    fails.append(f"{name}: status {got['status']}, want {want}")
    print(f"wall {time.time() - t0:.1f} s (compiles included)")
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"accel": a.accel, "form": a.form, "threads": a.threads, "results": results, "refusals": refused, "fails": fails}, f, indent=1)
    if a.check:
        print("\n".join(fails) if fails else "", end="\n" if fails else "")
        print("CHECK PASS" if not fails else f"CHECK FAIL ({len(fails)} differences)")
        sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
