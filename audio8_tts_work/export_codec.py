"""Export the codec decoder (codes [1,10,T] int32 -> wav [1,1,T*2048]) at static T, fp32, then check
the tflite against the oracle waveform (right-padding: causal graph => valid frames are exact).

  T=64 ~/venvs/lt094dev/bin/python3 audio8_tts_work/export_codec.py
"""
import os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import arktts_port as P
import litert_torch

T = int(os.environ.get("T", "64"))
OUT = os.path.join(C.OUT, "codec"); os.makedirs(OUT, exist_ok=True)
path = os.path.join(OUT, f"codec_decoder_fp32_T{T}{os.environ.get('SUFFIX', '')}.tflite")
codec, mod = P.load_codec()
dec = P.CodecDecoder(codec).eval()


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, codes):
        return {"wav": self.m(codes)}


with torch.no_grad():  # eager pass populates the rope-table cache for this static T
    dec(torch.zeros(1, C.NUM_CB, T, dtype=torch.int32))
t0 = time.time()
model = litert_torch.signature("decode", Wrap(dec), sample_kwargs={"codes": torch.zeros(1, C.NUM_CB, T, dtype=torch.int32)}).convert()
model.export(path)
print(f"exported {path} {os.path.getsize(path)/1e6:.0f} MB in {time.time()-t0:.0f}s", flush=True)

# ---- gate vs oracle wav (first T frames of a case, right-padded window) ----
from ai_edge_litert.interpreter import Interpreter
it = Interpreter(model_path=path, num_threads=8)
run = it.get_signature_runner("decode")
d = np.load(f"{C.OUT}/oracle/en_ref_0.npz")
codes, wav = d["codes"], d["wav"]
n = min(T, codes.shape[1])
buf = np.zeros((1, C.NUM_CB, T), np.int32); buf[0, :, :n] = codes[:, :n]
t1 = time.perf_counter(); out = run(codes=buf)["wav"][0, 0]; dt = time.perf_counter() - t1
t1 = time.perf_counter(); out = run(codes=buf)["wav"][0, 0]; dt2 = time.perf_counter() - t1
ref = wav[: n * C.FRAME]; got = out[: n * C.FRAME]
print(f"T{T}: {n} valid frames: max|d| {np.abs(got-ref).max():.3e} corr {np.corrcoef(got, ref)[0,1]:.6f} "
      f"| invoke {dt*1e3:.0f} ms (2nd {dt2*1e3:.0f} ms) for {T*C.FRAME/C.SR:.2f}s audio, 8 threads")
print("CODEC_EXPORT_DONE")
