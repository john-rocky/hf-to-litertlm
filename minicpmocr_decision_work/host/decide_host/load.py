"""Load this repository's LiteRT files as one PyTorch-free host.

    import sys
    sys.path.insert(0, "host")                                   # from the repository root
    from decide_host import load_host

    with load_host(".", accelerator="gpu") as host:              # the GPU at float32 (Metal on a Mac)
        out = host.score("block.png", "render.png", "the OCR text")     # {"status": "ok", "prob_1": ..., "T": ..., "n": [...], ...}
        best = host.best_of("block.png", [("render_a.png", "text a"), ("render_b.png", "text b")])

Two forms of the graphs (`form=`):

- "multi" (the default, for a desktop): multi_decide_L4096_fp16fc.tflite and multi_tower_N4096_fp16fc.tflite, every bucket as a signature
  of one file over one copy of the weights, up to 4,096 tokens and 4,096 patches a picture. On the GPU with
  GpuOptions.constant_tensor_sharing; on the CPU with an XNNPACK weight cache in `weight_cache_dir` (required there, written on the first
  load).
- "files" (the form for a phone): one file per bucket (decide_L<L>[_qb1024]_fp16fc.tflite, tower_N<N>[_qb1024]_fp16fc.tflite), whichever of
  them are in the repository folder, compiled when a request first needs one (`keep`: at most that many compiled at a time per kind, the
  least recently used closed first).

A request above the top bucket comes back as {"status": "too_large", "n": [...], "T": ..., "limit": {...}}. The GPU always runs at
float32 (GpuOptions(enforce_f32=True)): the delegate's default precision breaks this model's final norm.
"""
from __future__ import annotations

import os

from . import litert_backends as B
from .host import Host

MULTI_DECIDE, MULTI_TOWER = "multi_decide_L4096_fp16fc.tflite", "multi_tower_N4096_fp16fc.tflite"


def load_host(repo: str = ".", accelerator: str = "cpu", form: str = "multi", threads: int = 4, keep: int = 1,
              weight_cache_dir: str | None = None) -> Host:
    """The host over the files of `repo` (this repository's root). accelerator: "cpu" (XNNPACK, `threads`) or "gpu" (the GPU at
    float32). form: "multi" or "files" (module docstring). keep: bucket files compiled at a time per kind ("files").
    weight_cache_dir: where the "multi" form on the CPU keeps its XNNPACK weight cache files (required there)."""
    if accelerator not in ("cpu", "gpu"):
        raise ValueError("accelerator is 'cpu' or 'gpu'")
    repo = os.path.abspath(repo)
    acc = "cpu" if accelerator == "cpu" else "gpu_f32"
    tables = os.path.join(repo, "tables", "tables.safetensors")
    if form == "multi":
        for name in (MULTI_DECIDE, MULTI_TOWER):
            if not os.path.exists(os.path.join(repo, name)):
                raise FileNotFoundError(f"{name} is not in {repo}: download it, or use form=\"files\"")
        if accelerator == "cpu" and not weight_cache_dir:
            raise ValueError("the multi form on the CPU needs weight_cache_dir= (a folder; the first load writes the cache files there)")
        sharing = accelerator == "gpu"
        wc = {}
        if accelerator == "cpu":
            os.makedirs(weight_cache_dir, exist_ok=True)
            wc = {n: os.path.join(weight_cache_dir, os.path.splitext(n)[0] + ".xnnpack_cache") for n in (MULTI_DECIDE, MULTI_TOWER)}
        decide = B.LiteRTMultiDecide(os.path.join(repo, MULTI_DECIDE), acc, threads, sharing, wc.get(MULTI_DECIDE, ""))
        tower = B.LiteRTMultiTower(os.path.join(repo, MULTI_TOWER), acc, threads, sharing, wc.get(MULTI_TOWER, ""))
    elif form == "files":
        if keep < 1:
            raise ValueError("keep is at least 1: the file a request needs stays compiled while it runs")
        dec, tow = B.discover(repo, "decide", "fp16fc"), B.discover(repo, "tower", "fp16fc")
        if not dec or not tow:
            raise FileNotFoundError(f"no decide_L<L>_fp16fc.tflite or tower_N<N>_fp16fc.tflite file in {repo}")
        decide = B.LiteRTDecide(dec, acc, threads, keep=keep)
        tower = B.LiteRTTower(tow, acc, threads, keep=keep)
    else:
        raise ValueError("form is 'multi' or 'files'")
    return Host(tower, decide, repo, tables=tables)
