"""Load this repository's d1-Omni-2B LiteRT files as one PyTorch-free host.

    import sys
    sys.path.insert(0, "host")                       # from the repository root
    from d1_litert import load_host

    with load_host(".", accelerator="gpu") as host:  # the GPU at float32 (Metal on a Mac); "cpu" = XNNPACK
        out = host.system_one(state, questions, media=[picture, clip])  # {"answers": {...}, "usage": {...}}
        outs = host.system_one_batch([(state, questions, media), ...])  # one response per request

`system_one`, `system_one_batch` and `run` are the provider's System One API (its `api.py`, `prompt.py` and the other
torch-free modules in provider/, imported unchanged); d1_host.D1Host ports the rest to numpy. Two forms of the
decision graphs (`form=`):

- "files" (the default): one file per length bucket, d1-omni-2b_decide_L<L>[_qb1024]_fp16fc.tflite for L = 128, 256,
  512, 1024, 2048, 4096 and 6144, compiled when a sequence first needs it (`keep`: at most that many compiled at a
  time, the least recently used closed first), and the B = 8 file d1-omni-2b_decide_b8_L128_fp16fc.tflite for a provider
  batch of at least 8 sequences that all fit 128 tokens (8 sequences a call).
- "multi8": d1-omni-2b_decide_multi8_fp16fc.tflite, the seven buckets and the B = 8 graph as eight signatures of one file
  over one copy of the weights, on one CompiledModel: on the GPU with GpuOptions.constant_tensor_sharing, on the CPU
  with an XNNPACK weight cache file (`weight_cache=`, written on the first load: 5,868,852,600 bytes).

Both forms route a provider batch the same way: decide_b8_L128 when it has at least 8 sequences of at most 128 tokens,
else each sequence on the smallest bucket that holds it. Pictures run on the tower files
d1-omni-2b_tower_G<GH>x<GW>[_qb1152]_fp16fc.tflite (the smallest canvas that holds the picture's patch grid), audio
clips on the audio encoder files d1-omni-2b_audio_T<T>_fp16fc.tflite (T = 504, 1000, 2000, 3000 mel frames; the
smallest that holds the clip, at most 30 s). The GPU runs at float32 (GpuOptions(enforce_f32=True)): its default
precision moves the answers outside the checked tolerance.
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from b8_single_omni import B8SingleDecide  # noqa: E402
from d1_batch_omni import D1HostBatchedOmni  # noqa: E402
from lrt_backend import LiteRTAudio, LiteRTDecide, LiteRTTower, audio_paths, graph_paths  # noqa: E402
from multi8_omni import Multi8Decide  # noqa: E402

PREFIX = "d1-omni-2b"


class FilesDecide:
    """The single decide files behind the host's decide interface: one sequence runs on the bucket files
    (lrt_backend.LiteRTDecide); a provider batch of at least `batch_min` sequences that all fit the B = 8 file's 128
    tokens runs on that file (b8_single_omni.B8SingleDecide, compiled on first use), else one sequence a call."""

    def __init__(self, buckets: LiteRTDecide, batch_path: str | None, accelerator: str, threads: int,
                 batch_min: int = 8):
        self.buckets, self.batch_path = buckets, batch_path
        self.accelerator, self.threads, self.batch_min = accelerator, threads, max(1, int(batch_min))
        self.batch = None
        hit = re.search(r"_decide_b(\d+)_L(\d+)_", os.path.basename(batch_path or ""))
        self.B, self.BL = (int(hit.group(1)), int(hit.group(2))) if hit else (0, 0)
        self.max_len = buckets.max_len
        self.last = {}

    def many(self, seqs):
        seqs = [(np.asarray(e, np.float32), int(s), list(q)) for e, s, q in seqs]
        if self.BL and len(seqs) >= self.batch_min and all(e.shape[0] <= self.BL for e, _, _ in seqs):
            if self.batch is None:
                self.batch = B8SingleDecide(self.batch_path, self.accelerator, "fp32", self.threads)
            out = self.batch.many(seqs)
            self.last = {"sequences": len(seqs), "calls": self.batch.last["calls"],
                         "route": os.path.basename(self.batch_path)}
        else:
            out = [self.buckets(e, s, q) for e, s, q in seqs]
            self.last = {"sequences": len(seqs), "calls": len(seqs), "route": "buckets"}
        return out

    def __call__(self, embeds, state_len, question):
        return self.buckets(embeds, state_len, question)

    def close(self):
        self.buckets.close()
        if self.batch is not None:
            self.batch.close()
            self.batch = None


class Host(D1HostBatchedOmni):
    """D1HostBatchedOmni (d1_batch_omni.py) with `close()` and a context manager for its backends."""

    def close(self):
        for backend in (self.decide, self.tower, self.audio):
            if backend is not None:
                backend.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def load_host(repo: str = ".", accelerator: str = "cpu", form: str = "files", threads: int = 8, keep: int = 1,
              weight_cache: str | None = None) -> Host:
    """The host over the files of `repo` (this repository's root). accelerator: "cpu" (XNNPACK, `threads`) or "gpu"
    (the GPU at float32). form: "files" or "multi8" (module docstring). keep: bucket files compiled at a time ("files").
    weight_cache: the XNNPACK weight cache file of the "multi8" form on the CPU (required there)."""
    if accelerator not in ("cpu", "gpu"):
        raise ValueError("accelerator is 'cpu' or 'gpu'")
    repo = os.path.abspath(repo)
    dec_paths, tow_paths = graph_paths(repo, "fp16fc")
    if form == "files":
        if not dec_paths:
            raise FileNotFoundError(f"no {PREFIX}_decide_L<L>_fp16fc.tflite file in {repo}")
        batch = os.path.join(repo, f"{PREFIX}_decide_b8_L128_fp16fc.tflite")
        decide = FilesDecide(LiteRTDecide(dec_paths, accelerator, "fp32", threads, keep=keep),
                             batch if os.path.exists(batch) else None, accelerator, threads)
    elif form == "multi8":
        path = os.path.join(repo, f"{PREFIX}_decide_multi8_fp16fc.tflite")
        if accelerator == "cpu" and not weight_cache:
            raise ValueError("the multi8 form on the CPU needs weight_cache= (a file path; the first load writes it)")
        decide = Multi8Decide(path, accelerator, "fp32", threads, sharing=accelerator == "gpu",
                              weight_cache=weight_cache if accelerator == "cpu" else "", batch_min=8)
    else:
        raise ValueError("form is 'files' or 'multi8'")
    tower = LiteRTTower(tow_paths, accelerator, "fp32", threads) if tow_paths else None
    aud = audio_paths(repo, "fp16fc")
    audio = LiteRTAudio(aud, accelerator, "fp32", threads) if aud else None
    return Host(os.path.join(repo, "provider"), decide, tower,
                embed_table=os.path.join(repo, "tables", "embed_table.npy"), audio=audio)
