"""Folded-MTP compression lane v2: fp16 float-casting and GPTQ weight-only.

Prior verdicts on the folded graph (hostloop gate):
  - dynamic OCTAV int8 BLOCKWISE_32 -> silent no-op (532 MB unchanged)
  - dynamic OCTAV int4 BLOCKWISE_32 -> garbage (cb0 7.7%, wav corr 0.002)
  - static  wi8_ai16 calibrated     -> runaway 512 frames AND slow (int16 kernels)

This tries the two remaining in-toolchain paths:
  fp16   : FLOAT_CASTING weights fp32->fp16 (~half size, no activation change)
  gptq8  : GPTQ int8 weight-only, calibrated on the 52 reference frames
           (DEAD END: ai-edge-quantizer GPTQ is an unimplemented stub)
  gptq4  : GPTQ int4 weight-only, same calibration (same stub)
  recover4 : DEQUANTIZED_WEIGHT_RECOVERY int4 BLOCKWISE_32 on the torch-GPTQ
           fake-quant export (mtp_folded_gptqfq.tflite, see gptq_mtp_folded.py)
           -> true int4 flatbuffer that stays int4 in RAM on device.

  .venv/bin/python qwen3tts_work/quantize_mtp_folded_v2.py fp16|gptq8|gptq4|recover4
"""
import os
import sys

import numpy as np
from ai_edge_quantizer import quantizer
from ai_edge_quantizer import recipe as aqr

MODE = sys.argv[1]
SRC = "out/qwen3tts-mtp/mtp_folded.tflite"
DST = f"out/qwen3tts-mtp/mtp_folded_{MODE}.tflite"

if MODE == "recover4":
    SRC = "out/qwen3tts-mtp/mtp_folded_gptqfq.tflite"
    DST = "out/qwen3tts-mtp/mtp_folded_gptq_int4.tflite"
    rules = [{
        "regex": ".*",
        "operation": "*",
        "algorithm_key": aqr.AlgorithmName.DEQUANTIZED_WEIGHT_RECOVERY,
        "op_config": {
            "weight_tensor_config": {
                "num_bits": 4,
                "symmetric": True,
                "granularity": "BLOCKWISE_32",
                "dtype": "INT",
            },
            "compute_precision": "FLOAT",
            "explicit_dequantize": True,
            "skip_checks": False,
            "min_weight_elements": 0,
        },
    }]
    needs_calib = False
elif MODE == "fp16":
    rules = [{
        "regex": ".*",
        "operation": "*",
        "algorithm_key": aqr.AlgorithmName.FLOAT_CASTING,
        "op_config": {
            "weight_tensor_config": {
                "num_bits": 16,
                "symmetric": True,
                "granularity": "TENSORWISE",
                "dtype": "FLOAT",
            },
            "compute_precision": "FLOAT",
            "explicit_dequantize": True,
            "skip_checks": False,
            "min_weight_elements": 0,
        },
    }]
    needs_calib = False
else:
    bits = 8 if MODE == "gptq8" else 4
    rules = [{
        "regex": ".*",
        "operation": "*",
        "algorithm_key": aqr.AlgorithmName.GPTQ,
        "op_config": {
            "weight_tensor_config": {
                "num_bits": bits,
                "symmetric": True,
                "granularity": "CHANNELWISE",
                "dtype": "INT",
            },
            "compute_precision": "FLOAT",
            "explicit_dequantize": True,
            "skip_checks": False,
            "min_weight_elements": 0,
        },
    }]
    needs_calib = True

qt = quantizer.Quantizer(SRC)
qt.load_quantization_recipe(rules)
print("need_calibration:", qt.need_calibration)

calib_result = None
if needs_calib or qt.need_calibration:
    codec_emb = np.load("out/qwen3tts-host/codec_embedding.npy")
    ref = np.load("qwen3tts_work/ref/e2e_ref.npz")
    zero_noise = np.zeros((15, 2048), np.float32)
    feeds = []
    for i in range(ref["codes"].shape[0]):
        cb0 = int(ref["codes"][i, 0])
        feeds.append({
            "args_0": ref["hiddens"][i].reshape(1, 1, 1024).astype(np.float32),
            "args_1": codec_emb[cb0].reshape(1, 1, 1024).astype(np.float32),
            "args_2": zero_noise,
        })
    print(f"calibrating on {len(feeds)} frames…")
    if qt.need_calibration:
        calib_result = qt.calibrate({"serving_default": feeds})
    else:
        # GPTQ weight-only: recipe_manager says "no calibration needed" but
        # quantize() demands QSVs — call the Calibrator directly (its default
        # PRESERVE_ALL_TENSORS mode also caches the activations GPTQ needs).
        from ai_edge_quantizer import calibrator as aq_calibrator
        cal = aq_calibrator.Calibrator(SRC, num_threads=16)
        cal.calibrate({"serving_default": feeds}, qt._recipe_manager)
        calib_result = cal.get_model_qsvs()

res = qt.quantize(calib_result)
res.export_model(DST)
print(f"{MODE} -> {DST} {os.path.getsize(DST)//1024//1024} MB "
      f"(fp32 {os.path.getsize(SRC)//1024//1024} MB)")
print("QUANT_V2_DONE")
