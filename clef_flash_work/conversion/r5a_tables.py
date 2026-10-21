#!/usr/bin/env python3
"""Round 5a: the host's two vocabulary tables, each the checkpoint's BF16 tensor copied byte for byte into a one-key
safetensors file without __metadata__ (memory decision-graph-embeds-input-form: metadata reorders between writes and
moves the sha256), and the round's ledger of large untracked files.

    venv-host/bin/python scripts/r5a_tables.py          (from K) -> out/r5a/embed_tokens.safetensors,
                                                                    out/r5a/lm_head.safetensors, results/files_r5a.json

Source = model-00001-of-00004.safetensors of the pinned snapshot (f0b7d3cd; its sha256 = the LFS oid 8b45a8e9...,
results/checkpoint_check.json): `model.language_model.embed_tokens.weight` and `lm_head.weight`, BF16 [248320, 4096]
each (tie_word_embeddings false). Checks per file: the header has one key and no __metadata__, dtype / shape / offsets;
the data region equals the source byte range (block compare, sha256 of both); the safetensors library opens the file
and reports the key, dtype, shape and no metadata; the file sha256. Refuses to overwrite a file."""
import datetime
import hashlib
import json
import sys
import time
from pathlib import Path

K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "host"))
from clef_litert.tables import copy_bf16_tensor, read_header, sha256_file  # noqa: E402

SNAP = K / "hf/hub/models--Cloudflare--clef-flash/snapshots/f0b7d3cd65296abfe459e6c228b618f5d8b26d2f"
SHARD = "model-00001-of-00004.safetensors"
TABLES = (("model.language_model.embed_tokens.weight", "embed_tokens.safetensors", "embed_tokens.weight",
           "the row graph's embeds: the rows of the ids, widened to float32 by the host"),
          ("lm_head.weight", "lm_head.safetensors", "lm_head.weight",
           "the joint head's lexical vectors: the mean of the rows of each option span's ids"))
SHAPE = [248320, 4096]


def now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def compare_region(path: Path, offset: int, src: Path, src_offset: int, n: int, block: int = 1 << 24) -> dict:
    """Block compare of path[offset:offset+n] with src[src_offset:src_offset+n] -> equality and both sha256."""
    ha, hb = hashlib.sha256(), hashlib.sha256()
    equal, done = True, 0
    with open(path, "rb") as fa, open(src, "rb") as fb:
        fa.seek(offset)
        fb.seek(src_offset)
        while done < n:
            k = min(block, n - done)
            a, b = fa.read(k), fb.read(k)
            if len(a) != k or len(b) != k:
                raise IOError(f"short read at {done}")
            equal &= a == b
            ha.update(a)
            hb.update(b)
            done += k
    return {"bytes_equal": bool(equal), "sha256_file_region": ha.hexdigest(), "sha256_source_region": hb.hexdigest()}


def main() -> int:
    from safetensors import safe_open

    out_dir = K / "out/r5a"
    out_dir.mkdir(parents=True, exist_ok=True)
    src = SNAP / SHARD
    rows, ok = [], True
    for src_key, name, key, role in TABLES:
        dst = out_dir / name
        t0 = time.perf_counter()
        rec = copy_bf16_tensor(src, src_key, dst, key)
        t_copy = time.perf_counter() - t0
        header, data_start = read_header(dst)
        e = header.get(key, {})
        checks = {"one_key_no_metadata": list(header) == [key],
                  "dtype_shape_offsets": e == {"dtype": "BF16", "shape": SHAPE, "data_offsets": [0, SHAPE[0] * SHAPE[1] * 2]},
                  "data_offset": data_start == rec["data_offset"],
                  "file_bytes": dst.stat().st_size == data_start + SHAPE[0] * SHAPE[1] * 2}
        t1 = time.perf_counter()
        region = compare_region(dst, data_start, src, rec["source"]["byte_range"][0], SHAPE[0] * SHAPE[1] * 2)
        checks["data_equals_checkpoint_bytes"] = region["bytes_equal"]
        checks["data_sha256_equals_copy_record"] = region["sha256_source_region"] == rec["source"]["data_sha256"]
        with safe_open(str(dst), framework="numpy") as f:
            sl = f.get_slice(key)
            lib = {"keys": list(f.keys()), "metadata": f.metadata(), "dtype": str(sl.get_dtype()),
                   "shape": list(sl.get_shape())}
        checks["safetensors_library_reads"] = (lib["keys"] == [key] and lib["metadata"] is None
                                               and lib["dtype"].lower() in ("bf16", "bfloat16") and lib["shape"] == SHAPE)
        sha = sha256_file(dst)
        t_check = time.perf_counter() - t1
        ok &= all(checks.values())
        rows.append({"path": f"out/r5a/{name}", "bytes": dst.stat().st_size, "sha256": sha,
                     "sha256_from": "scripts/r5a_tables.py (read back after the write)", "created_at": now(), "round": "r5a",
                     "role": role, "key": key, "source": {"snapshot_file": SHARD, "revision": SNAP.name, **rec["source"]},
                     "header_bytes": rec["header_bytes"], "data_offset": rec["data_offset"], "region": region,
                     "safetensors_library": lib, "checks": checks, "seconds": {"copy": round(t_copy, 1),
                                                                                "check": round(t_check, 1)},
                     "deleted_at": None, "moved_to": None})
        print(json.dumps({"file": name, "bytes": rows[-1]["bytes"], "sha256": sha, "checks": checks,
                          "seconds": rows[-1]["seconds"]}), flush=True)
    doc = {"what": "round 5a: large files kept outside git (the host's tables)", "written_at": now(),
           "files": rows, "bytes_total": sum(r["bytes"] for r in rows), "all_checks_pass": bool(ok)}
    (K / "results/files_r5a.json").write_text(json.dumps(doc, indent=1) + "\n")
    print("all checks pass" if ok else "CHECK FAILED", doc["bytes_total"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
