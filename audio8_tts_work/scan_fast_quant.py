"""Per-group int8 sensitivity scan of the fast AR: quantize one weight group at a time (aeq regex on the
op scope) and measure logits error / argmax agreement vs the oracle on the first frames.
  ~/venvs/lt094dev/bin/python3 audio8_tts_work/scan_fast_quant.py [src]
"""
import copy, os, sys, glob, subprocess
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import ai_edge_quantizer.recipe as aqr
from ai_edge_quantizer import quantizer
from ai_edge_litert.interpreter import Interpreter

src = sys.argv[1] if len(sys.argv) > 1 else f"{C.OUT}/fast/fast_fp32_v2.tflite"
NEG = -1e9
cases = sorted(glob.glob(f"{C.OUT}/oracle/*.npz"))[:3]
masks = [np.where(np.arange(C.NUM_CB) <= p, 0.0, NEG).astype(np.float32).reshape(1, 1, 1, -1) for p in range(C.NUM_CB)]


def evaluate(path):
    it = Interpreter(model_path=path, num_threads=4); step = it.get_signature_runner("step")
    maxd, sumd, am, tot = 0.0, 0.0, 0, 0
    for f in cases:
        d = np.load(f); fl, oh, codes = d["fast_logits_first8"], d["fast_hidden"], d["codes"]
        for fi in range(len(fl)):
            k = np.zeros((C.N_FAST, 1, C.KV_HEADS, C.NUM_CB, C.HEAD_DIM), np.float32); v = np.zeros_like(k)
            h = oh[fi].reshape(1, 1, -1).astype(np.float32)
            o = step(hidden=h, token=np.zeros(1, np.int32), use_hidden=np.ones(1, np.float32), pos=np.zeros(1, np.int32), mask=masks[0], k_all=k, v_all=v)
            k, v = o["k_all"], o["v_all"]; cur = int(codes[0, fi])
            for p in range(1, C.NUM_CB):
                o = step(hidden=h, token=np.array([cur], np.int32), use_hidden=np.zeros(1, np.float32), pos=np.array([p], np.int32), mask=masks[p], k_all=k, v_all=v)
                k, v = o["k_all"], o["v_all"]; lg = o["logits"][0]; ref = fl[fi, p - 1]
                dd = np.abs(lg - ref); maxd = max(maxd, float(dd.max())); sumd += float(dd.mean()); am += int(lg.argmax() == ref.argmax()); tot += 1
                cur = int(codes[p, fi])
    return maxd, sumd / tot, am, tot


def quant(recipe, dst):
    qt = quantizer.Quantizer(src); qt.load_quantization_recipe(recipe); qt.quantize().export_model(dst)
    return dst


fc8 = copy.deepcopy(aqr.dynamic_wi8_afp32()[0])
emb8 = copy.deepcopy(fc8); emb8["operation"] = "EMBEDDING_LOOKUP"
groups = {
    "emb_only": [emb8],
    "head_only": [dict(fc8, regex=".*step_logits_output.*")],
    **{f"block{i}_only": [dict(fc8, regex=f".*ArBlock_{i};.*")] for i in range(C.N_FAST)},
    "fc_all_no_emb": [fc8],
    "fc_all_no_head": [dict(fc8, regex=".*ArBlock_.*"), emb8],
}
print(f"{'variant':<16} {'MB':>5} {'max|d|':>8} {'mean|d|':>8} {'argmax':>9}")
fp = evaluate(src); print(f"{'fp32':<16} {os.path.getsize(src)/1e6:>5.0f} {fp[0]:>8.3f} {fp[1]:>8.4f} {fp[2]:>4}/{fp[3]}")
for name, recipe in groups.items():
    dst = f"{C.OUT}/fast/scan_{name}.tflite"
    try:
        quant(recipe, dst); r = evaluate(dst)
        print(f"{name:<16} {os.path.getsize(dst)/1e6:>5.0f} {r[0]:>8.3f} {r[1]:>8.4f} {r[2]:>4}/{r[3]}", flush=True)
    except Exception as e:
        print(f"{name:<16} ERROR {str(e)[:120]}")
