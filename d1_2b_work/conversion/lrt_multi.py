"""Gate backends for the seven-signature decide file and for GpuOptions.constant_tensor_sharing (kept apart from
lrt_backend.py):

  LiteRTMultiDecide(path)  the host's decide backend over out/graphs/d1-2b_decide_multi7_fp16fc.tflite: ONE
                           CompiledModel holds every bucket (signature decide_L<n>[_qb<m>]), a sequence runs on the
                           smallest bucket that holds it (the padding of lrt_backend.LiteRTDecide)
  SharedTower(paths)       lrt_backend.LiteRTTower with the sharing option on each canvas file

    dec = LiteRTMultiDecide(path, "gpu", "fp32", sharing=True)
    hidden = dec(embeds [T, 2048], state_len, question)
"""
from __future__ import annotations

import os
import re

from lrt_backend import LiteRTDecide, LiteRTTower, MultiGraph


class _Sig:
    """One signature of a MultiGraph, called like lrt_backend.Graph (`own` = close the model with it)."""

    def __init__(self, m: MultiGraph, sig: str, own: bool = False):
        self.m, self.sig, self.own = m, sig, own

    @property
    def load_s(self) -> float:
        return self.m.load_s

    @property
    def fully_accelerated(self):
        return self.m.fully_accelerated

    def __call__(self, **arrays) -> dict:
        return self.m(self.sig, **arrays)

    def close(self):
        if self.own:
            self.m.close()


class LiteRTMultiDecide(LiteRTDecide):
    def __init__(self, path: str, accelerator: str = "gpu", precision: str = "fp32", threads: int = 8,
                 sharing: bool = True, weight_cache: str = ""):
        m = MultiGraph(path, accelerator, precision, threads, sharing, weight_cache)
        sigs = {}
        for name in m.signatures:
            hit = re.fullmatch(r"decide_L(\d+)(_qb\d+)?", name)
            if hit:
                sigs[int(hit.group(1))] = name
        if not sigs:
            m.close()
            raise ValueError(f"{path}: no decide_L<n> signature ({sorted(m.signatures)})")
        super().__init__({L: path for L in sigs}, accelerator, precision, threads)
        self.m, self.sigs, self.sharing = m, sigs, sharing
        self.loads.append({"file": os.path.basename(path), "load_s": round(m.load_s, 2),
                           "fully_accelerated": m.fully_accelerated, "signatures": len(sigs), "sharing": sharing})

    def graph(self, L: int) -> _Sig:
        return _Sig(self.m, self.sigs[L])

    def close(self):
        self.m.close()


class SharedTower(LiteRTTower):
    def __init__(self, paths: dict, accelerator: str = "gpu", precision: str = "fp32", threads: int = 8,
                 keep: bool = True, sharing: bool = True):
        super().__init__(paths, accelerator, precision, threads, keep)
        self.sharing = sharing

    def graph(self, canvas) -> _Sig:
        if canvas not in self.graphs:
            if not self.keep:
                for g in self.graphs.values():
                    g.close()
                self.graphs.clear()
            m = MultiGraph(self.paths[canvas], self.accelerator, self.precision, self.threads, self.sharing)
            self.graphs[canvas] = _Sig(m, "tower", own=True)
        return self.graphs[canvas]
