"""slow AR tflite vs oracle (teacher forcing): chunked prefill of prompt[:-1] + decode of every column.
  ~/venvs/lt094dev/bin/python3 audio8_tts_work/verify_tflite_slow.py [path] [n_cases]
"""
import os, sys, glob, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from ai_edge_litert.interpreter import Interpreter

path = sys.argv[1] if len(sys.argv) > 1 else f"{C.OUT}/slow/slow_fp32_c2048.tflite"
n_cases = int(sys.argv[2]) if len(sys.argv) > 2 else 3
NTHREADS = int(os.environ.get("NTHREADS", "4"))
NEG = -1e9
t0 = time.time()
it = Interpreter(model_path=path, num_threads=NTHREADS)
sigs = it.get_signature_list()
prefill_sigs = sorted(int(k.split("_")[1]) for k in sigs if k.startswith("prefill_"))
pre = {P: it.get_signature_runner(f"prefill_{P}") for P in prefill_sigs}
dec = it.get_signature_runner("decode")
CACHE = dec.get_input_details()["mask"]["shape"][-1]
KV = [n for n in dec.get_input_details() if n[:2] in ("k_", "v_")]
print(f"{os.path.basename(path)}: sigs {sorted(sigs)} cache {CACHE} kv {len(KV)} load {time.time()-t0:.1f}s "
      f"outputs {sorted(dec.get_output_details())[:3]}...")

for f in sorted(glob.glob(f"{C.OUT}/oracle/*.npz"))[:n_cases]:
    d = np.load(f); prompt, codes, sem = d["prompt"], d["codes"], d["semantic"]
    P, T = prompt.shape[1], codes.shape[1]
    full = np.concatenate([prompt, np.concatenate([sem[None], codes], 0)], 1)
    kv = {n: np.zeros((1, C.KV_HEADS, CACHE, C.HEAD_DIM), np.float32) for n in KV}
    t_pre = time.perf_counter(); s = 0; npre = 0
    while s < P - 1:
        rem = P - 1 - s
        Tn = next((p for p in prefill_sigs if p >= rem), prefill_sigs[-1]); n = min(rem, Tn)
        buf = np.zeros((1, C.NUM_CB + 1, Tn), np.int32); buf[0, :, :n] = prompt[:, s:s + n]
        mask = np.full((1, 1, Tn, CACHE), NEG, np.float32)
        for i in range(Tn):
            mask[0, 0, i, : s + i + 1] = 0.0
        out = pre[Tn](codes=buf, input_pos=np.arange(s, s + Tn, dtype=np.int32), mask=mask, **kv)
        kv = {k: out[k] for k in KV}; s += n; npre += 1
    t_pre = time.perf_counter() - t_pre
    pl, ph = [], []
    t_dec = time.perf_counter()
    for t in range(T + 1):
        pos = P - 1 + t
        mask = np.full((1, 1, 1, CACHE), NEG, np.float32); mask[..., : pos + 1] = 0.0
        out = dec(codes=full[:, pos].reshape(1, C.NUM_CB + 1, 1).astype(np.int32), input_pos=np.array([pos], np.int32), mask=mask, **kv)
        kv = {k: out[k] for k in KV}
        pl.append(out["logits"][0]); ph.append(out["hidden"][0, 0])
    t_dec = (time.perf_counter() - t_dec) / (T + 1)
    pl, ph = np.stack(pl), np.stack(ph)
    ol, oh = d["slow_logits"][: T + 1], d["fast_hidden"][: T + 1]
    dl, dh = np.abs(pl - ol), np.abs(ph - oh)
    print(f"{os.path.basename(f)}: P {P} T {T} | logits max|d| {dl.max():.3e} mean {dl.mean():.2e} argmax {(pl.argmax(1)==ol.argmax(1)).mean():.3f} "
          f"| hidden max|d| {dh.max():.3e} | prefill {t_pre*1e3:.0f}ms ({npre} chunks) decode {t_dec*1e3:.1f}ms/step ({NTHREADS} thr)")
print("VERIFY_SLOW_DONE")
