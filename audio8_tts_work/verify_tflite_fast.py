"""fast AR tflite vs oracle fast logits (first 8 frames x 9 steps of each case).
  ~/venvs/lt094dev/bin/python3 audio8_tts_work/verify_tflite_fast.py [path] [n_cases]
"""
import os, sys, glob, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from ai_edge_litert.interpreter import Interpreter

path = sys.argv[1] if len(sys.argv) > 1 else f"{C.OUT}/fast/fast_fp32.tflite"
n_cases = int(sys.argv[2]) if len(sys.argv) > 2 else 4
NEG = -1e9
it = Interpreter(model_path=path, num_threads=4)
step = it.get_signature_runner("step")
print("inputs", {k: (v["shape"].tolist(), str(v["dtype"].__name__)) for k, v in step.get_input_details().items()})
print("outputs", {k: v["shape"].tolist() for k, v in step.get_output_details().items()})
masks = [np.where(np.arange(C.NUM_CB) <= p, 0.0, NEG).astype(np.float32).reshape(1, 1, 1, -1) for p in range(C.NUM_CB)]
tot_d, am, tot, calls, t_call = 0.0, 0, 0, 0, 0.0
for f in sorted(glob.glob(f"{C.OUT}/oracle/*.npz"))[:n_cases]:
    d = np.load(f); fl, oh, codes = d["fast_logits_first8"], d["fast_hidden"], d["codes"]
    for fi in range(len(fl)):
        k = np.zeros((C.N_FAST, 1, C.KV_HEADS, C.NUM_CB, C.HEAD_DIM), np.float32); v = np.zeros_like(k)
        h = oh[fi].reshape(1, 1, -1).astype(np.float32)
        def run(tok, use, p, k, v):
            global calls, t_call
            t0 = time.perf_counter()
            o = step(hidden=h, token=np.array([tok], np.int32), use_hidden=np.array([use], np.float32),
                     pos=np.array([p], np.int32), mask=masks[p], k_all=k, v_all=v)
            t_call += time.perf_counter() - t0; calls += 1
            return o["logits"][0], o["k_all"], o["v_all"]
        _, k, v = run(0, 1.0, 0, k, v)
        cur = int(codes[0, fi])
        for p in range(1, C.NUM_CB):
            lg, k, v = run(cur, 0.0, p, k, v)
            ref = fl[fi, p - 1]
            tot_d = max(tot_d, float(np.abs(lg - ref).max())); am += int(lg.argmax() == ref.argmax()); tot += 1
            cur = int(codes[p, fi])
print(f"{os.path.basename(path)}: fast logits max|d| {tot_d:.3e} argmax match {am}/{tot}  ({calls} calls, {1e3*t_call/calls:.2f} ms/call, 4 threads)")
print("VERIFY_FAST_DONE")
