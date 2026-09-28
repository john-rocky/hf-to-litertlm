"""Codec-only gate: decode the ORACLE codes of every case with one codec .tflite (single call, cases that fit in T),
write wavs to out/codec_gate/<tag>/, so asr/spk gates isolate the codec's own quantization effect.
  ~/venvs/lt094dev/bin/python3 audio8_tts_work/codec_only_gate.py <codec.tflite> <tag>"""
import os, sys, glob, time
import numpy as np, soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from ai_edge_litert.interpreter import Interpreter
path, tag = sys.argv[1], sys.argv[2]
outd = f"{C.OUT}/codec_gate/{tag}"; os.makedirs(outd, exist_ok=True)
it = Interpreter(model_path=path, num_threads=8); run = it.get_signature_runner(list(it.get_signature_list())[0])
IN = list(run.get_input_details())[0]; ON = list(run.get_output_details())[0]
T = run.get_input_details()[IN]["shape"][-1]
corrs, dts = [], []
for f in sorted(glob.glob(f"{C.OUT}/oracle/*.npz")):
    d = np.load(f); codes, wav = d["codes"], d["wav"]; n = codes.shape[1]
    assert n <= T
    buf = np.zeros((1, C.NUM_CB, T), np.int32); buf[0, :, :n] = codes
    t1 = time.perf_counter(); out = run(**{IN: buf})[ON][0, 0][: n * C.FRAME]; dts.append(time.perf_counter() - t1)
    corrs.append(np.corrcoef(out, wav[: n * C.FRAME])[0, 1])
    sf.write(f"{outd}/{os.path.basename(f)[:-4]}.wav", out, C.SR, subtype="PCM_16")
print(f"{tag}: {len(corrs)} cases | wav corr vs oracle min {min(corrs):.6f} mean {np.mean(corrs):.6f} | invoke median {np.median(dts)*1e3:.0f} ms (8 thr) -> {outd}")
