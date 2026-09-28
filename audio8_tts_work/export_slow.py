"""Export the slow AR (24L) as one .tflite with signatures prefill_<P>... + decode, fp32 weights.

Inputs (named): codes int32 [1,11,T], input_pos int32 [T], mask fp32 [1,1,T,CACHE] (additive, 0 = attend),
k_<i>/v_<i> fp32 [1,2,CACHE,64] x24.  Outputs: logits [1,4097], hidden [1,1,896], k_<i>/v_<i> updated.

  CACHE=2048 PREFILL=64,256 ~/venvs/lt094dev/bin/python3 audio8_tts_work/export_slow.py
"""
import os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import arktts_port as P
import litert_torch

CACHE = int(os.environ.get("CACHE", str(C.MAX_SEQ)))
PREFILL = [int(x) for x in os.environ.get("PREFILL", "64,256").split(",")]
OUT = os.path.join(C.OUT, "slow")
os.makedirs(OUT, exist_ok=True)
path = os.path.join(OUT, f"slow_fp32_c{CACHE}{os.environ.get('SUFFIX', '')}.tflite")

w = P.load_main_weights()
slow = P.SlowAR(w, CACHE).eval()
KV = [f"{kv}_{i}" for i in range(C.N_LAYER) for kv in ("k", "v")]


def make_wrapper(m):
    src = (f"def forward(self, codes, input_pos, mask, {', '.join(KV)}):\n"
           f"    out = self.m(codes, input_pos, mask, {', '.join(KV)})\n"
           f"    return {{'logits': out[0], 'hidden': out[1], " + ", ".join(f"'{n}': out[{2+i}]" for i, n in enumerate(KV)) + "}\n")
    ns = {}
    exec(src, ns)
    cls = type("SlowWrapper", (torch.nn.Module,), {"forward": ns["forward"]})
    obj = cls()
    obj.m = m
    return obj.eval()


def sample(T):
    kw = {"codes": torch.zeros(1, C.NUM_CB + 1, T, dtype=torch.int32),
          "input_pos": torch.arange(T, dtype=torch.int32),
          "mask": torch.zeros(1, 1, T, CACHE)}
    for n in KV:
        kw[n] = torch.zeros(1, C.KV_HEADS, CACHE, C.HEAD_DIM)
    return kw


wrap = make_wrapper(slow)
t0 = time.time()
conv = None
for Pn in PREFILL:
    conv = (litert_torch.signature if conv is None else conv.signature)(f"prefill_{Pn}", wrap, sample_kwargs=sample(Pn))
conv = conv.signature("decode", wrap, sample_kwargs=sample(1))
model = conv.convert()
model.export(path)
print(f"exported {path} {os.path.getsize(path)/1e6:.0f} MB in {time.time()-t0:.0f}s")
print("SLOW_EXPORT_DONE")
