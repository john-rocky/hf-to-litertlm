"""Step 1: the 11 small files of LiquidAI/d1-omni-600M at the pinned revision into the default HF cache.

    HF_HUB_DISABLE_XET=1 venv-ref/bin/python scripts/fetch_small.py

Never asks for model.safetensors. Each file's sha256 must equal the copy in hf_small/.
Also reads the Hub metadata (main sha, per-file size, the LFS sha256 of model.safetensors) without downloading it,
and watches blobs/*.incomplete (a weight download would show up there).
-> results/small_files.json
"""
import hashlib
import json
import os
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, try_to_load_from_cache

REPO = "LiquidAI/d1-omni-600M"
REV = "414f8d6438174f5b2133a9c21a478fc42625e308"
SMALL = [".gitattributes", "LICENSE", "README.md", "audio.py", "config.json", "encoder.py", "modeling_d1.py",
         "prompt.py", "tokenizer.json", "tokenizer_config.json", "vision.py"]
WEIGHTS = "model.safetensors"
WEIGHTS_SHA256 = "0713bb05270c2685ad106522f4092bceeeb3a93cf79b401f399a712296c911e1"
WEIGHTS_BYTES = 2_348_774_500
K = Path(__file__).resolve().parents[1]
CACHE_REPO = Path.home() / ".cache/huggingface/hub/models--LiquidAI--d1-omni-600M"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def incomplete():
    blobs = CACHE_REPO / "blobs"
    return [{"name": p.name, "bytes": p.stat().st_size} for p in blobs.glob("*.incomplete")] if blobs.exists() else []


def main():
    assert WEIGHTS not in SMALL
    assert os.environ.get("HF_HUB_DISABLE_XET") == "1", "run with HF_HUB_DISABLE_XET=1"
    out = {"repo": REPO, "revision": REV, "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "env_HF_HUB_DISABLE_XET": os.environ.get("HF_HUB_DISABLE_XET")}

    # Hub metadata (no file content): the main sha now, and the pinned revision's file list with sizes / LFS oids.
    api = HfApi()
    main_info = api.model_info(REPO)
    out["main_sha_now"] = main_info.sha
    out["main_sha_equals_pin"] = main_info.sha == REV
    info = api.model_info(REPO, revision=REV, files_metadata=True)
    hub_files = {}
    for s in info.siblings:
        rec = {"size": s.size, "blob_id": s.blob_id}
        if s.lfs is not None:
            lfs = s.lfs
            rec["lfs_sha256"] = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
            rec["lfs_size"] = lfs.get("size") if isinstance(lfs, dict) else getattr(lfs, "size", None)
        hub_files[s.rfilename] = rec
    out["hub_files_at_revision"] = hub_files
    out["hub_file_list_equals_launch"] = sorted(hub_files) == sorted(SMALL + [WEIGHTS])
    w = hub_files.get(WEIGHTS, {})
    out["weights_lfs_check"] = {"lfs_sha256": w.get("lfs_sha256"), "size": w.get("size"),
                                "sha256_equals_launch": w.get("lfs_sha256") == WEIGHTS_SHA256,
                                "size_equals_launch": w.get("size") == WEIGHTS_BYTES}

    pre_cached = {f: bool(try_to_load_from_cache(REPO, f, revision=REV) not in (None,)) for f in SMALL}
    out["incomplete_before"] = incomplete()
    rows = []
    t0 = time.time()
    for f in SMALL:
        p = hf_hub_download(REPO, f, revision=REV)
        bad = [x for x in incomplete() if x["bytes"] > 100 * 1024 * 1024]
        assert not bad, f"a large .incomplete appeared (weight download?): {bad}"
        copy = K / "hf_small" / f
        digest = sha256(p)
        rows.append({"name": f, "bytes": os.path.getsize(p), "sha256": digest, "snapshot_path": p,
                     "resolved_blob": os.path.realpath(p), "pre_cached": pre_cached[f],
                     "hub_size": hub_files.get(f, {}).get("size"),
                     "size_equals_hub": os.path.getsize(p) == hub_files.get(f, {}).get("size"),
                     "hf_small_sha256": sha256(copy), "sha256_equals_hf_small": digest == sha256(copy)})
    out["seconds"] = round(time.time() - t0, 2)
    out["files"] = rows
    out["all_equal_hf_small"] = all(r["sha256_equals_hf_small"] for r in rows)
    out["all_size_equal_hub"] = all(r["size_equals_hub"] for r in rows)
    out["snapshot_dir"] = str(CACHE_REPO / "snapshots" / REV)
    out["snapshot_in_default_cache"] = all(r["snapshot_path"].startswith(out["snapshot_dir"]) for r in rows)
    out["weights_in_snapshot"] = (CACHE_REPO / "snapshots" / REV / WEIGHTS).exists()
    out["incomplete_after"] = incomplete()
    out["cache_repo_bytes"] = sum(p.stat().st_size for p in (CACHE_REPO / "blobs").iterdir() if p.is_file())
    assert out["all_equal_hf_small"], "sha256 differs from hf_small/"
    assert not out["weights_in_snapshot"]
    (K / "results/small_files.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("main_sha_now", "main_sha_equals_pin", "hub_file_list_equals_launch",
                                          "weights_lfs_check", "all_equal_hf_small", "all_size_equal_hub",
                                          "snapshot_in_default_cache", "weights_in_snapshot", "incomplete_after",
                                          "cache_repo_bytes", "seconds")}, indent=1))
    print("pre_cached:", sum(pre_cached.values()), "/", len(SMALL))


if __name__ == "__main__":
    main()
