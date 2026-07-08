"""Quantize the Qwen3-TTS codec decoder (the now-dominant device stage).

The codec is FLOP-bound (conv/transconv/SnakeBeta stack, ~318 GFLOP at T=64),
unlike the bandwidth-bound MTP — so int8 conv/GEMM kernels are the lever. Tries
data-free dynamic int8 (DRQ, weights int8 in RAM + int8 kernels) and fp16, and
gates each standalone against the fp32 reference waveform (corr + max|d|).

  .venv/bin/python qwen3tts_work/quantize_codec.py drq8|fp16
"""
import os
import sys

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from ai_edge_quantizer import quantizer
from ai_edge_quantizer import recipe as aqr

MODE = sys.argv[1] if len(sys.argv) > 1 else "drq8"
SRC = "out/qwen3tts-codec/codec_decode_T64.tflite"
DST = f"out/qwen3tts-codec/codec_decode_T64_{MODE}.tflite"
T = 64

qt = quantizer.Quantizer(SRC)
if MODE == "drq8":
    qt.load_quantization_recipe(aqr.dynamic_wi8_afp32())
elif MODE == "fp16":
    qt.load_quantization_recipe([{
        "regex": ".*", "operation": "*",
        "algorithm_key": aqr.AlgorithmName.FLOAT_CASTING,
        "op_config": {
            "weight_tensor_config": {
                "num_bits": 16, "symmetric": True,
                "granularity": "TENSORWISE", "dtype": "FLOAT",
            },
            "compute_precision": "FLOAT", "explicit_dequantize": True,
            "skip_checks": False, "min_weight_elements": 0,
        },
    }])
else:
    raise SystemExit(f"unknown mode {MODE}")

res = qt.quantize()
res.export_model(DST)
print(f"{MODE} -> {DST} {os.path.getsize(DST)//1024//1024} MB "
      f"(fp32 {os.path.getsize(SRC)//1024//1024} MB)")

# ---- standalone gate vs fp32 reference waveform ----
d = np.load("qwen3tts_work/ref/codec_equiv_ref.npz")
codes = d["codes"].astype(np.int32)          # [1,16,52]
wav_ref = d["wav"][0, 0]                      # [99840]
Tr = codes.shape[-1]
buf = np.zeros((1, 16, T), np.int32)
buf[0, :, :Tr] = codes[0]


def run(path):
    it = Interpreter(model_path=path, num_threads=4)
    sig = it.get_signature_runner()
    out = sig(args_0=buf)
    return list(out.values())[0][0, 0][: Tr * 1920]


wav_fp32 = run(SRC)
wav_q = run(DST)
c_vs_ref = float(np.corrcoef(wav_q, wav_ref)[0, 1])
c_vs_fp32 = float(np.corrcoef(wav_q, wav_fp32)[0, 1])
print(f"{MODE} wav corr vs fp32-ref {c_vs_ref:.6f} / vs fp32-tflite {c_vs_fp32:.6f}, "
      f"max|d| {np.abs(wav_q - wav_ref).max():.4f}, peak {np.abs(wav_q).max():.3f}")
import soundfile as sf  # noqa: E402

sf.write(f"qwen3tts_work/ref/codec_{MODE}.wav", wav_q, 24000)
print("CODEC_QUANT_DONE")
