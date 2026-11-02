"""The decide backend over the eight-signature file (d1-2b_decide_multi8_fp16fc.tflite: signatures decide_L128 ..
decide_L6144_qb1024 = the seven length buckets, and decide_b8_L128 = the batch graph of 8 sequences of the L128 bucket)
on ONE CompiledModel: on the GPU at float32 with GpuOptions.constant_tensor_sharing, or on the CPU (XNNPACK) with the
XNNPACK weight cache file (lrt_backend.compiled_options and lrt_backend.MultiGraph: buffers made per signature on first
use and kept).

  dec = Multi8Decide(path, "gpu", sharing=True)                      # or Multi8Decide(path, "cpu", weight_cache=...)
  hidden = dec(embeds [T, 2048], state_len, question)               # one sequence: the smallest bucket that holds it
  hiddens = dec.many([(embeds, state_len, question), ...])          # a provider batch (d1_batch.D1HostBatched)

Routing of `many`: when every sequence of the batch fits 128 tokens and there are at least `batch_min` of them
(default 8 = full calls), they go through decide_b8_L128 eight a call (the last call padded with padding rows: embeds 0,
sees_q [0, 1, 0..], sees_k 0, never read); otherwise one call a sequence on its smallest bucket. batch_min=1 sends every
L128 sequence through the batch signature. numpy + ai-edge-litert only (no torch).
"""
from __future__ import annotations

import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from d1_layout import SEES, make_sees  # noqa: E402  (numpy only)
from lrt_backend import MultiGraph  # noqa: E402


def _sees_batch(rows, B, L):
    """d1_layout_b.make_sees_batch, copied (numpy): real rows from make_sees, the rest padding rows."""
    q = np.zeros((B, L, SEES), np.float32)
    k = np.zeros((B, L, SEES), np.float32)
    q[:, :, 1] = 1.0
    for b, (state_len, tags) in enumerate(rows):
        qb, kb = make_sees(state_len, tags, L)
        q[b], k[b] = qb[0], kb[0]
    return q, k


class Multi8Decide:
    def __init__(self, path: str, accelerator: str = "gpu", precision: str = "fp32", threads: int = 8,
                 sharing: bool = False, weight_cache: str = "", batch_min: int = 8):
        self.path, self.accelerator, self.precision = path, accelerator, precision
        self.m = MultiGraph(path, accelerator, precision, threads, sharing, weight_cache)
        self.sigs, self.bsig = {}, None
        for name in self.m.signatures:
            hit = re.fullmatch(r"decide_L(\d+)(_qb\d+)?", name)
            if hit:
                self.sigs[int(hit.group(1))] = name
            hit = re.fullmatch(r"decide_b(\d+)_L(\d+)", name)
            if hit:
                self.bsig, self.B, self.BL = name, int(hit.group(1)), int(hit.group(2))
        if not self.sigs:
            self.m.close()
            raise ValueError(f"{path}: no decide_L<n> signature ({sorted(self.m.signatures)})")
        self.paths = {L: path for L in self.sigs}
        self.max_len = max(self.sigs)
        self.batch_min = max(1, int(batch_min))
        self.sharing, self.weight_cache = sharing, weight_cache
        self.load_s = self.m.load_s
        self.fully_accelerated = self.m.fully_accelerated
        self.loads = [{"file": os.path.basename(path), "load_s": round(self.m.load_s, 2),
                       "fully_accelerated": self.m.fully_accelerated, "signatures": sorted(self.m.signatures),
                       "sharing": sharing, "weight_cache": weight_cache or None}]
        self.calls = []
        self.last = {}
        self.graphs = {}          # the timing script reads load seconds from here; the multi file has one load

    def _one(self, embeds, state_len, question):
        T = embeds.shape[0]
        fits = [L for L in sorted(self.sigs) if L >= T]
        if not fits:
            raise ValueError(f"a sequence of {T} tokens is longer than the largest graph ({self.max_len})")
        L = fits[0]
        e = np.zeros((1, L, embeds.shape[1]), np.float32)
        e[0, :T] = embeds
        q, k = make_sees(state_len, question, L)
        t0 = time.perf_counter()
        h = self.m(self.sigs[L], embeds=e, sees_q=q, sees_k=k)["hidden"]
        dt = time.perf_counter() - t0
        self.calls.append({"sig": self.sigs[L], "rows": 1, "run_s": dt})
        self.last = {"L": L, "T": T, "run_s": dt, "calls": 1}
        return h[0, :T]

    def _batch(self, chunk):
        e = np.zeros((self.B, self.BL, chunk[0][0].shape[1]), np.float32)
        for b, (embeds, _, _) in enumerate(chunk):
            e[b, : embeds.shape[0]] = embeds
        q, k = _sees_batch([(s, qq) for _, s, qq in chunk], self.B, self.BL)
        t0 = time.perf_counter()
        h = self.m(self.bsig, embeds=e, sees_q=q, sees_k=k)["hidden"]
        dt = time.perf_counter() - t0
        self.calls.append({"sig": self.bsig, "rows": len(chunk), "run_s": dt})
        return [h[b, : chunk[b][0].shape[0]] for b in range(len(chunk))]

    def many(self, seqs):
        """[(embeds [T, d], state_len, question tags)] -> [hidden [T, d]] (the routing in the module doc)."""
        seqs = [(np.asarray(e, np.float32), int(s), list(q)) for e, s, q in seqs]
        for e, _, q in seqs:
            if len(q) != e.shape[0]:
                raise ValueError("question tags and embeds differ in length")
        n0 = len(self.calls)
        if self.bsig and len(seqs) >= self.batch_min and all(e.shape[0] <= self.BL for e, _, _ in seqs):
            out = []
            for i in range(0, len(seqs), self.B):
                out.extend(self._batch(seqs[i:i + self.B]))
            route = self.bsig
        else:
            out = [self._one(e, s, q) for e, s, q in seqs]
            route = "buckets"
        self.last = {"sequences": len(seqs), "calls": len(self.calls) - n0, "route": route}
        return out

    def __call__(self, embeds, state_len, question):
        return self._one(np.asarray(embeds, np.float32), int(state_len), list(question))

    def close(self):
        self.m.close()
