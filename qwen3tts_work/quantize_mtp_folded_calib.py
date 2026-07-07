"""Calibrated static int8 (A16W8) quantization of the folded MTP graph.

Data-free failed (int8 blockwise no-ops; int4 blockwise data-free -> garbage,
wav corr 0.002). This uses ai-edge-quantizer STATIC calibration: feed
representative (past_hidden, cb0_embed) frames so activation ranges are
measured, then int8 weights + int16 activations.

Calibration frames come from the fp32 pipeline's reference trajectory
(e2e_ref: 52 real frames of hiddens + cb0 codes). Extend CALIB_NPZS with more
sentences for a wider range.

  .venv/bin/python qwen3tts_work/quantize_mtp_folded_calib.py [ai16|ai8]
"""
import os
import sys

import numpy as np
from ai_edge_quantizer import quantizer, recipe

MODE = sys.argv[1] if len(sys.argv) > 1 else "ai16"
SRC = "out/qwen3tts-mtp/mtp_folded.tflite"
DST = f"out/qwen3tts-mtp/mtp_folded_static_{MODE}.tflite"

codec_emb = np.load("out/qwen3tts-host/codec_embedding.npy")  # [3072,1024]
zero_noise = np.zeros((15, 2048), np.float32)

# Build calibration feeds from the reference trajectory (52 real frames).
ref = np.load("qwen3tts_work/ref/e2e_ref.npz")
feeds = []
for i in range(ref["codes"].shape[0]):
    cb0 = int(ref["codes"][i, 0])
    feeds.append({
        "args_0": ref["hiddens"][i].reshape(1, 1, 1024).astype(np.float32),
        "args_1": codec_emb[cb0].reshape(1, 1, 1024).astype(np.float32),
        "args_2": zero_noise,
    })
print(f"calibration frames: {len(feeds)}")

qt = quantizer.Quantizer(SRC)
rec = recipe.static_wi8_ai16() if MODE == "ai16" else recipe.static_wi8_ai8()
qt.load_quantization_recipe(rec)
print("need_calibration:", qt.need_calibration)
calib = qt.calibrate({"serving_default": feeds})
res = qt.quantize(calib)
res.export_model(DST)
print(f"{MODE} -> {DST} {os.path.getsize(DST)//1024//1024} MB "
      f"(fp32 {os.path.getsize(SRC)//1024//1024} MB)")
print("QUANT_DONE")
