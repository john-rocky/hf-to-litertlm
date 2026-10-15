"""Shared helpers for the EmbeddingGemma 2 Transcribe conversion: fixtures, fixed windows, per-token cosine, sha256.

The fixtures are the reader fixtures of github.com/john-rocky/eg2-transcribe (`reader/fixtures/`): MANIFEST.json is in
git, the arrays are rebuilt by its `make_fixtures.py` and checked here against the MANIFEST's sha256.
numpy and the standard library only.
"""
import hashlib
import json
import os
import re

import numpy as np

FPS = 100                           # log-mel frames per second (hop 160 at 16 kHz); 4 frames = 1 soft token
WINDOW_TOKENS = {30: 750, 10: 250}  # tokens per window (the reader's signature length)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_fixtures(fixture_dir, names=("features.npy", "soft_tokens.npy"), long=False):
    """sha256 of the fixture arrays against MANIFEST.json. -> number of files checked; raises on a mismatch."""
    m = json.load(open(os.path.join(fixture_dir, "MANIFEST.json")))
    todo = [(os.path.join(fixture_dir, e["path"], n), e["sha256"][n]) for e in m["entries"] for n in names]
    if long:
        todo.append((os.path.join(fixture_dir, m["long"]["path"], "features.npy"), m["long"]["sha256"]["features.npy"]))
    for path, want in todo:
        got = sha256_file(path)
        if got != want:
            raise SystemExit(f"{path}: sha256 {got} is not the MANIFEST's {want} (rebuild with make_fixtures.py)")
    return len(todo)


def fixture_ids(fixture_dir):
    base = os.path.join(fixture_dir, "test-clean-20")
    return sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))


def load_fixture(fixture_dir, uid, what):
    """what = "features" (log-mel [n_frames, 128]) or "soft_tokens" (the reference soft tokens [n_tokens, 512])."""
    return np.load(os.path.join(fixture_dir, "test-clean-20", uid, what + ".npy")).astype(np.float32)


def windows_of(mel, W):
    """Fixed windows of W frames, the last one right-padded with zeros -> [(mel_w [W, 128], mask_w [W] float, n_real)]."""
    out = []
    for s in range(0, mel.shape[0], W):
        seg = mel[s:s + W]
        n = seg.shape[0]
        mw = np.zeros((W, mel.shape[1]), np.float32)
        mw[:n] = seg
        mk = np.zeros((W,), np.float32)
        mk[:n] = 1.0
        out.append((mw, mk, n))
    return out


def long_windows(fixture_dir, W, n=10):
    """n full windows of W frames spread over the long clip (long-speckled-band/features.npy) -> [(name, mel_w, mask_w)]."""
    feats = np.load(os.path.join(fixture_dir, "long-speckled-band", "features.npy"), mmap_mode="r")
    nfull = feats.shape[0] // W
    return [(f"long#w{j}", np.ascontiguousarray(feats[j * W:(j + 1) * W]).astype(np.float32), np.ones(W, np.float32))
            for j in np.linspace(0, nfull - 1, n).round().astype(int)]


def rowwise_cos(a, b, eps=1e-12):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return (a * b).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), eps)


def release_table(path):
    """{utterance id: the reader's transcript} from the fixture table of the reader's RELEASE.md."""
    rows = {}
    for line in open(path, encoding="utf-8"):
        m = re.match(r"^\| `([0-9]+-[0-9]+-[0-9]+)` \| (\d+) / (\d+) \| (.*) \|\s*$", line)
        if m:
            rows[m.group(1)] = m.group(4).strip()
    return rows
