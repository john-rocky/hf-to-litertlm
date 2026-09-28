"""Export the codec encoder (voice registration): audio [1,1,N] fp32 -> codes [1,10,N/2048] int32, static N.
Buckets by seconds (SEC env, default 10): N = ceil(SEC*44100/2048)*2048.  Verifies vs the oracle reference codes.
  SEC=10 ~/venvs/lt094dev/bin/python3 audio8_tts_work/export_codec_encoder.py
"""
import os, sys, time, math
import numpy as np, torch, soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import arktts_port as P
import litert_torch

SEC = float(os.environ.get("SEC", "10"))
N = int(math.ceil(SEC * C.SR / C.FRAME)) * C.FRAME
OUT = os.path.join(C.OUT, "codec"); os.makedirs(OUT, exist_ok=True)
path = os.path.join(OUT, f"codec_encoder_fp32_{int(SEC)}s{os.environ.get('SUFFIX', '')}.tflite")
codec, mod = P.load_codec()
enc = P.CodecEncoder(codec).eval()


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, audio):
        return {"codes": self.m(audio)}


w = Wrap(enc)
with torch.no_grad():
    w(torch.zeros(1, 1, N))  # rope cache for this static length
t0 = time.time()
litert_torch.signature("encode", w, sample_kwargs={"audio": torch.zeros(1, 1, N)}).convert().export(path)
print(f"exported {path} {os.path.getsize(path)/1e6:.0f} MB (N={N}, {N/C.SR:.2f}s) in {time.time()-t0:.0f}s", flush=True)

from ai_edge_litert.interpreter import Interpreter
run = Interpreter(model_path=path, num_threads=8).get_signature_runner("encode")
for key in C.REFS:
    a, sr = sf.read(f"{C.FIX}/ref_{key}_44k.wav", dtype="float32")
    assert len(a) <= N
    buf = np.zeros((1, 1, N), np.float32); buf[0, 0, : len(a)] = a
    t1 = time.perf_counter(); codes = run(audio=buf)["codes"][0]; dt = time.perf_counter() - t1
    oc = np.load(f"{C.FIX}/ref_{key}_codes.npy"); n = oc.shape[1]
    # the oracle encodes the clip padded to a frame multiple; frames beyond the clip are padding-derived
    print(f"ref {key}: codes match {(codes[:, :n] == oc).mean():.4f} ({(codes[:, :n] != oc).sum()} of {oc.size} differ) | invoke {dt*1e3:.0f} ms")
print("ENCODER_EXPORT_DONE")
