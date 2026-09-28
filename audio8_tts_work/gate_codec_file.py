"""Gate one codec decoder .tflite against the oracle waveforms (all cases that fit in T frames): corr, max|d|,
per-case time.   ~/venvs/lt094dev/bin/python3 audio8_tts_work/gate_codec_file.py <file> [nthreads]"""
import os, sys, glob, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from ai_edge_litert.interpreter import Interpreter
path = sys.argv[1]; nt = int(sys.argv[2]) if len(sys.argv) > 2 else 8
it = Interpreter(model_path=path, num_threads=nt); run = it.get_signature_runner(it.get_signature_list() and list(it.get_signature_list())[0])
IN = list(run.get_input_details())[0]; OUTN = list(run.get_output_details())[0]
T = run.get_input_details()[IN]["shape"][-1]
corrs, ds, dts = [], [], []
for f in sorted(glob.glob(f"{C.OUT}/oracle/*.npz")):
    d = np.load(f); codes, wav = d["codes"], d["wav"]
    if codes.shape[1] > T: continue
    n = codes.shape[1]; buf = np.zeros((1, C.NUM_CB, T), np.int32); buf[0, :, :n] = codes
    t1 = time.perf_counter(); out = run(**{IN: buf})[OUTN][0, 0]; dts.append(time.perf_counter() - t1)
    ref, got = wav[: n * C.FRAME], out[: n * C.FRAME]
    corrs.append(np.corrcoef(got, ref)[0, 1]); ds.append(np.abs(got - ref).max())
print(f"{os.path.basename(path)} ({os.path.getsize(path)/1e6:.0f} MB, T{T}): {len(corrs)} cases | corr min {min(corrs):.6f} mean {np.mean(corrs):.6f} "
      f"| max|d| {max(ds):.3e} | invoke median {np.median(dts)*1e3:.0f} ms ({nt} thr)")
