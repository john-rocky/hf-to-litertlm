"""Load this repository's LiteRT files as one PyTorch-free host.

    import sys
    sys.path.insert(0, "host")                                   # from the repository root
    from decide_host import load_host

    with load_host(".", accelerator="gpu") as host:              # the GPU (Metal on a Mac)
        out = host.score("block.png", "render.png", "the OCR text")     # {"status": "ok", "prob_1": ..., "T": ..., "n": [...], ...}
        best = host.best_of("block.png", [("render_a.png", "text a"), ("render_b.png", "text b")])

Two forms of the graphs (`form=`):

- "multi" (the default, for a desktop): multi_decide_L4096_f16sfc.tflite and multi_tower_N4096_fp16fc.tflite, every bucket as a signature
  of one file over one copy of the weights, up to 4,096 tokens and 4,096 patches a picture. On the GPU with
  GpuOptions.constant_tensor_sharing; on the CPU with an XNNPACK weight cache in `weight_cache_dir` (required there, written on the first
  load).
- "files" (the form for a phone): one file per bucket (decide_L<L>[_qb1024]_f16sfc.tflite, tower_N<N>[_qb1024]_fp16fc.tflite), whichever of
  them are in the repository folder, compiled when a request first needs one (`keep`: at most that many compiled at a time per kind, the
  least recently used closed first).

A request above the top bucket comes back as {"status": "too_large", "n": [...], "T": ..., "limit": {...}}. On the GPU the decision graph
runs at the delegate's default precision and the tower at float32 (GpuOptions(enforce_f32=True)): the decision files are the f16s form,
whose final norm is scaled to stay inside float16's range, and the tower has no such form. On the CPU both run at float32.
"""
from __future__ import annotations

import os

from . import litert_backends as B
from .host import Host

DECIDE_FORM, TOWER_FORM = "f16sfc", "fp16fc"
MULTI_DECIDE, MULTI_TOWER = f"multi_decide_L4096_{DECIDE_FORM}.tflite", f"multi_tower_N4096_{TOWER_FORM}.tflite"


def load_host(repo: str = ".", accelerator: str = "cpu", form: str = "multi", threads: int = 4, keep: int = 1,
              weight_cache_dir: str | None = None) -> Host:
    """The host over the files of `repo` (this repository's root). accelerator: "cpu" (XNNPACK, `threads`) or "gpu" (the decision graph
    at the GPU's default precision, the tower at float32). form: "multi" or "files" (module docstring). keep: bucket files compiled at a
    time per kind ("files"). weight_cache_dir: where the "multi" form on the CPU keeps its XNNPACK weight cache files (required there)."""
    if accelerator not in ("cpu", "gpu"):
        raise ValueError("accelerator is 'cpu' or 'gpu'")
    repo = os.path.abspath(repo)
    acc_decide, acc_tower = ("cpu", "cpu") if accelerator == "cpu" else ("gpu_default", "gpu_f32")
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
        decide = B.LiteRTMultiDecide(os.path.join(repo, MULTI_DECIDE), acc_decide, threads, sharing, wc.get(MULTI_DECIDE, ""))
        tower = B.LiteRTMultiTower(os.path.join(repo, MULTI_TOWER), acc_tower, threads, sharing, wc.get(MULTI_TOWER, ""))
    elif form == "files":
        if keep < 1:
            raise ValueError("keep is at least 1: the file a request needs stays compiled while it runs")
        dec, tow = B.discover(repo, "decide", DECIDE_FORM), B.discover(repo, "tower", TOWER_FORM)
        if not dec or not tow:
            raise FileNotFoundError(f"no decide_L<L>_{DECIDE_FORM}.tflite or tower_N<N>_{TOWER_FORM}.tflite file in {repo}")
        decide = B.LiteRTDecide(dec, acc_decide, threads, keep=keep)
        tower = B.LiteRTTower(tow, acc_tower, threads, keep=keep)
    else:
        raise ValueError("form is 'multi' or 'files'")
    return Host(tower, decide, repo, tables=tables)
