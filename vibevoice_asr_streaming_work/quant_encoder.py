#!/usr/bin/env python3
"""int8-dynamic (FULLY_CONNECTED only, `wi8fc`, the BitNet lane's shipped encoder recipe) from the
fp32 26-frame encoder tflite, with parity vs the fp32 tflite on a real window (clip02, first 26 frames).
  python quant_encoder.py [--fp32 out/audio_encoder_26f_fp32.tflite]"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "vibevoice_asr_work"))
import common as C  # noqa: E402


def run(path, frames):
    from ai_edge_litert.interpreter import Interpreter
    sig = Interpreter(model_path=path, num_threads=8).get_signature_runner()
    sig(audio=frames)
    t0 = time.time()
    out = sig(audio=frames)["features"]
    return out, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", default=os.path.join(HERE, "out", "audio_encoder_26f_fp32.tflite"))
    args = ap.parse_args()
    from ai_edge_quantizer import quantizer, recipe_manager, qtyping
    OP = qtyping.TFLOperationName
    wi8 = args.fp32.replace("_fp32.tflite", "_wi8fc.tflite")
    if os.path.exists(wi8):
        os.remove(wi8)
    rm = recipe_manager.RecipeManager()
    rm.add_dynamic_config(regex=".*", operation_name=OP.FULLY_CONNECTED, num_bits=8)
    qt = quantizer.Quantizer(args.fp32, rm.get_quantization_recipe())
    assert not qt.need_calibration
    t0 = time.time()
    qt.quantize().export_model(wi8)
    print(f"wi8fc {os.path.getsize(wi8)/1e6:.1f} MB (fp32 {os.path.getsize(args.fp32)/1e6:.1f} MB) in {time.time()-t0:.0f}s")
    wav = C.load_wav(os.path.join(C.WORK, "fixtures", "clip02.wav"))
    T = 26
    frames = np.zeros((1, T, C.HOP), np.float32)
    frames[0] = wav[:T * C.HOP].reshape(T, C.HOP)
    a, ta = run(args.fp32, frames)
    b, tb = run(wi8, frames)
    err = float(np.abs(a - b).max())
    cos = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))
    r = {"wi8fc": wi8, "size_mb": round(os.path.getsize(wi8) / 1e6, 1), "max_abs_err_vs_fp32": err,
         "rel_err": err / float(np.abs(a).max()), "cos": cos, "invoke_s_fp32": round(ta, 3), "invoke_s_wi8fc": round(tb, 3)}
    print(json.dumps(r))
    json.dump(r, open(os.path.join(HERE, "out", "audio_encoder_26f_wi8fc_report.json"), "w"), indent=1)
    print("QUANT_DONE")


if __name__ == "__main__":
    main()
