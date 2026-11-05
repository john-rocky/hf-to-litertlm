"""The olmOCR family of the author's benchmark on this repository's LiteRT files, through the PyTorch-free host: 446 records of
openbmb/JevOCRBench at a pinned revision, each an original block, a render of a candidate text, the text and a label.

    python examples/score_olmocr_set.py --accel gpu --cache-dir /path/to/folder                 # all 446 records
    python examples/score_olmocr_set.py --accel gpu --cache-dir /path/to/folder --limit 20      # the first 20

Prints the AUROC (rank method with averaged ties, as the author's eval/evaluate.py), overall and per category, on the records the host
scored; a record above the top bucket comes back `too_large` and is counted apart. Nothing of the benchmark is in this repository: the
script downloads the rows and the two pictures of each record into --cache-dir, one file at a time, and writes one row of numbers per
record to <cache-dir>/scores_<accel>_<form>.jsonl (no picture, no text). A stopped run resumes. Only this family is read: the
benchmark's other families carry other terms (its DATA_LICENSE.md).

Attribution, as the benchmark's NOTICE and the license ask: this family derives from olmOCR-bench
(https://huggingface.co/datasets/allenai/olmOCR-bench), Open Data Commons Attribution License v1.0
(https://opendatacommons.org/licenses/by/1-0/). Poznanski et al., "olmOCR: Unlocking Trillions of Tokens in PDFs with Vision Language
Models," arXiv:2502.18443 (2025), https://arxiv.org/abs/2502.18443.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "host"))

REVISION = "a26c4b32dcd3a95490842ca637b80edd2b71d975"
BASE = f"https://huggingface.co/datasets/openbmb/JevOCRBench/resolve/{REVISION}"
ROWS = "data/jevocrbench.jsonl"
ROWS_SHA256 = "7a4546d121055b7a03ef68967138d3b15aa30d270c253594b1d9e61b09e57f1d"
FAMILY_PREFIX, FAMILY_RECORDS = "olmocr_", 446
ATTRIBUTION = __doc__[__doc__.index("Attribution, as"):].strip()


def fetch(cache: str, rel: str, pace: float) -> str:
    """One file of the pinned revision into the cache (kept), with pauses and retries: the Hub limits calls per IP."""
    path = os.path.join(cache, rel)
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for attempt in range(6):
        time.sleep(pace)
        try:
            req = urllib.request.Request(f"{BASE}/{rel}", headers={"User-Agent": "score_olmocr_set/1.0"})
            data = urllib.request.urlopen(req, timeout=120).read()
            break
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == 5:
                raise SystemExit(f"{rel}: {e}") from e
            time.sleep(30 * (attempt + 1) if getattr(e, "code", 0) == 429 else 5 * (attempt + 1))
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.rename(tmp, path)
    return path


def auroc(pairs: list) -> float | None:
    """pairs = [(score, label), ...]: Mann-Whitney U with averaged ranks on ties."""
    pos = sum(1 for _, y in pairs if y == 1)
    neg = len(pairs) - pos
    if not pos or not neg:
        return None
    ordered = sorted(pairs, key=lambda p: p[0])
    rank_sum, i = 0.0, 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1][0] == ordered[i][0]:
            j += 1
        rank_sum += sum((i + j) / 2.0 + 1.0 for k in range(i, j + 1) if ordered[k][1] == 1)
        i = j + 1
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--accel", choices=("gpu", "cpu"), default="gpu")
    ap.add_argument("--form", choices=("multi", "files"), default="multi")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--keep", type=int, default=1, help="--form files: bucket files compiled at a time per kind (on Metal a 256-token decision file holds 8.2 GB once compiled and peaks at 14.3 GB while it compiles)")
    ap.add_argument("--weight-cache", default="", help="--form multi on the CPU: the folder of the XNNPACK weight cache files")
    ap.add_argument("--repo", default=ROOT, help="the folder that holds the graphs, tokenizer.json and tables/ (default: this repository)")
    ap.add_argument("--cache-dir", required=True, help="where the benchmark's rows, pictures and the score rows go")
    ap.add_argument("--limit", type=int, default=0, help="only the first N records of the family")
    ap.add_argument("--pace", type=float, default=0.7, help="seconds before each download (the Hub limits calls per IP)")
    a = ap.parse_args()
    if a.accel == "cpu" and a.form == "multi" and not a.weight_cache:
        ap.error("--form multi on the CPU needs --weight-cache FOLDER (the first load writes the cache files there), or use --form files")
    from decide_host import load_host

    os.makedirs(a.cache_dir, exist_ok=True)
    rows_path = fetch(a.cache_dir, ROWS, a.pace)
    digest = hashlib.sha256(open(rows_path, "rb").read()).hexdigest()
    if digest != ROWS_SHA256:
        raise SystemExit(f"{ROWS}: SHA-256 {digest}, want {ROWS_SHA256}")
    rows = [json.loads(line) for line in open(rows_path, encoding="utf-8")]
    family = [r for r in rows if str(r["source"]).startswith(FAMILY_PREFIX)]
    if len(family) != FAMILY_RECORDS:
        raise SystemExit(f"{len(family)} records in the family, want {FAMILY_RECORDS}")
    family.sort(key=lambda r: r["item_id"])
    if a.limit:
        family = family[:a.limit]
    out_path = os.path.join(a.cache_dir, f"scores_{a.accel}_{a.form}.jsonl")
    done = {}
    if os.path.exists(out_path):
        done = {r["item_id"]: r for r in map(json.loads, open(out_path, encoding="utf-8"))}
    with load_host(a.repo, accelerator=a.accel, form=a.form, threads=a.threads, keep=a.keep, weight_cache_dir=a.weight_cache or None) as host, \
            open(out_path, "a", encoding="utf-8") as out:
        print(f"{len(family)} records, {len(done)} already scored, accel {a.accel}, form {a.form}, top length {host.decide.max_len}, top patches {host.tower.max_n}", flush=True)
        for i, r in enumerate(family):
            if r["item_id"] in done:
                continue
            image, render = fetch(a.cache_dir, r["image_path"], a.pace), fetch(a.cache_dir, r["render_image_path"], a.pace)
            got = host.score(image, render, str(r["ocr_text"] or ""))
            row = {"item_id": r["item_id"], "category": r["category"], "label": 1 if r["label"] == "pos" else 0, "source": r["source"], "status": got["status"]}
            for key in ("prob_1", "logits", "T", "n", "ids_sha", "limit"):
                if key in got:
                    row[key] = got[key]
            if "hidden_pooled" in got:
                row["hidden_sha"] = hashlib.sha256(got["hidden_pooled"].tobytes()).hexdigest()[:16]
            out.write(json.dumps(row) + "\n")
            out.flush()
            done[r["item_id"]] = row
            if i % 20 == 0:
                print(f"{i + 1} / {len(family)}: record {r['item_id']}, status {got['status']}, prob_1 {got.get('prob_1')}", flush=True)
    ids = {r["item_id"] for r in family}
    scored = [r for i, r in done.items() if i in ids and r["status"] == "ok"]
    other = [r for i, r in done.items() if i in ids and r["status"] != "ok"]
    print(f"AUROC {100 * (auroc([(r['prob_1'], r['label']) for r in scored]) or 0):.3f} on {len(scored)} records; {len(other)} not scored: "
          + json.dumps({s: sum(r['status'] == s for r in other) for s in sorted({r['status'] for r in other})}))
    for cat in sorted({r["category"] for r in scored}):
        sub = [r for r in scored if r["category"] == cat]
        value = auroc([(r["prob_1"], r["label"]) for r in sub])
        print(f"  {cat}: AUROC {100 * value:.3f} on {len(sub)} records" if value is not None else f"  {cat}: one class only, {len(sub)} records")
    print(f"rows: {out_path}")
    print(ATTRIBUTION)


if __name__ == "__main__":
    main()
