"""The decide backend of the B = 8 single file (d1-omni-2b_decide_b8_L128_fp16fc.tflite, signature decide_b8_L128)
for the batch host (d1_batch_omni.D1HostBatchedOmni): eight sequences a call, right-padded to 128, the last call
padded with padding rows, over lrt_backend.MultiGraph (one CompiledModel, buffers made once; the GPU at float32 without
constant_tensor_sharing and the CPU without a weight cache, as the B = 8 single file was checked). numpy +
ai-edge-litert only (no torch)."""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import lrt_backend as _lb  # noqa: E402
from d1_layout import SEES, make_sees  # noqa: E402
from lrt_backend import MultiGraph  # noqa: E402

assert os.path.samefile(_lb.__file__, os.path.join(HERE, "lrt_backend.py")), _lb.__file__


class B8SingleDecide:
    def __init__(self, path: str, accelerator: str = "gpu", precision: str = "fp32", threads: int = 8,
                 sharing: bool = False, weight_cache: str = ""):
        self.path, self.accelerator, self.precision = path, accelerator, precision
        self.m = MultiGraph(path, accelerator, precision, threads, sharing, weight_cache)
        assert len(self.m.signatures) == 1, sorted(self.m.signatures)
        self.bsig = next(iter(self.m.signatures))
        shape = list(self.m.model.get_input_tensor_details(self.bsig)["embeds"]["shape"])
        self.B, self.BL, self.hidden = shape
        self.max_len = self.BL
        self.load_s = self.m.load_s
        self.fully_accelerated = self.m.fully_accelerated
        self.loads = [{"file": os.path.basename(path), "load_s": round(self.m.load_s, 2),
                       "fully_accelerated": self.m.fully_accelerated, "signature": self.bsig, "B": self.B,
                       "L": self.BL, "sharing": sharing, "weight_cache": weight_cache or None}]
        self.calls = []
        self.last = {}
        self.batch_min = 1

    def _batch(self, chunk):
        e = np.zeros((self.B, self.BL, self.hidden), np.float32)
        q = np.zeros((self.B, self.BL, SEES), np.float32)
        k = np.zeros((self.B, self.BL, SEES), np.float32)
        q[:, :, 1] = 1.0                            # padding rows (d1-2B's make_sees_batch)
        for b, (embeds, state_len, tags) in enumerate(chunk):
            e[b, : embeds.shape[0]] = embeds
            qb, kb = make_sees(state_len, tags, self.BL)
            q[b], k[b] = qb[0], kb[0]
        t0 = time.perf_counter()
        h = self.m(self.bsig, embeds=e, sees_q=q, sees_k=k)["hidden"]
        dt = time.perf_counter() - t0
        self.calls.append({"sig": self.bsig, "rows": len(chunk), "run_s": dt})
        return [h[b, : chunk[b][0].shape[0]] for b in range(len(chunk))]

    def many(self, seqs):
        seqs = [(np.asarray(e, np.float32), int(s), list(q)) for e, s, q in seqs]
        for e, _, q in seqs:
            if e.shape[0] > self.BL or len(q) != e.shape[0]:
                raise ValueError(f"a sequence of {e.shape[0]} tokens does not fit the batch graph's L {self.BL}")
        n0, out = len(self.calls), []
        for i in range(0, len(seqs), self.B):
            out.extend(self._batch(seqs[i:i + self.B]))
        self.last = {"sequences": len(seqs), "calls": len(self.calls) - n0, "route": self.bsig}
        return out

    def __call__(self, embeds, state_len, question):
        return self.many([(embeds, state_len, question)])[0]

    def close(self):
        self.m.close()
