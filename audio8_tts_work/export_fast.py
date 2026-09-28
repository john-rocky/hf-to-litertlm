"""Export the fast AR (4L, codebook predictor) as fast_fp32.tflite, one signature `step`.

Inputs: hidden [1,1,896] f32, token [1] i32, use_hidden [1] f32 (1.0 at position 0), pos [1] i32,
mask [1,1,1,10] f32 additive, k_all/v_all [4,1,2,10,64] f32.  Outputs: logits [1,4096], k_all, v_all.

  ~/venvs/lt094dev/bin/python3 audio8_tts_work/export_fast.py
"""
import os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import arktts_port as P
import litert_torch

OUT = os.path.join(C.OUT, "fast")
os.makedirs(OUT, exist_ok=True)
path = os.path.join(OUT, f"fast_fp32{os.environ.get('SUFFIX', '')}.tflite")
w = P.load_main_weights()
fast = P.FastAR(w).eval()


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, hidden, token, use_hidden, pos, mask, k_all, v_all):
        lg, k, v = self.m(hidden, token, use_hidden, pos, mask, k_all, v_all)
        return {"logits": lg, "k_all": k, "v_all": v}


kw = {"hidden": torch.zeros(1, 1, C.DIM), "token": torch.zeros(1, dtype=torch.int32),
      "use_hidden": torch.ones(1), "pos": torch.zeros(1, dtype=torch.int32),
      "mask": torch.zeros(1, 1, 1, C.NUM_CB),
      "k_all": torch.zeros(C.N_FAST, 1, C.KV_HEADS, C.NUM_CB, C.HEAD_DIM),
      "v_all": torch.zeros(C.N_FAST, 1, C.KV_HEADS, C.NUM_CB, C.HEAD_DIM)}
t0 = time.time()
model = litert_torch.signature("step", Wrap(fast), sample_kwargs=kw).convert()
model.export(path)
print(f"exported {path} {os.path.getsize(path)/1e6:.0f} MB in {time.time()-t0:.0f}s")
print("FAST_EXPORT_DONE")
