"""fp16 (FLOAT_CASTING) / FC-int8 variants of the codec encoder + code-match gate vs oracle reference codes.
  ~/venvs/lt094dev/bin/python3 audio8_tts_work/quantize_encoder.py <src.tflite> <fp16|drq8fc>"""
import os, sys, time
import numpy as np, soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import ai_edge_quantizer.recipe as aqr
from ai_edge_quantizer import quantizer
from ai_edge_litert.interpreter import Interpreter
src, mode = sys.argv[1], sys.argv[2]; dst = src.replace("_fp32", f"_{mode}")
qt = quantizer.Quantizer(src)
if mode == "fp16":
    qt.load_quantization_recipe([{"regex": ".*", "operation": "*", "algorithm_key": aqr.AlgorithmName.FLOAT_CASTING,
        "op_config": {"weight_tensor_config": {"num_bits": 16, "symmetric": True, "granularity": "TENSORWISE", "dtype": "FLOAT"},
                      "compute_precision": "FLOAT", "explicit_dequantize": True, "skip_checks": False, "min_weight_elements": 0}}])
elif mode == "drq8fc":
    qt.load_quantization_recipe(aqr.dynamic_wi8c_afp32(operation_name=aqr.TFLOperationName.FULLY_CONNECTED))
qt.quantize().export_model(dst)
print(f"{mode}: {os.path.getsize(src)/1e6:.0f} MB -> {dst} {os.path.getsize(dst)/1e6:.0f} MB")
run = Interpreter(model_path=dst, num_threads=8).get_signature_runner("encode")
N = run.get_input_details()["audio"]["shape"][-1]
for key in C.REFS:
    a, sr = sf.read(f"{C.FIX}/ref_{key}_44k.wav", dtype="float32")
    buf = np.zeros((1, 1, N), np.float32); buf[0, 0, : len(a)] = a
    t1 = time.perf_counter(); codes = run(audio=buf)["codes"][0]; dt = time.perf_counter() - t1
    oc = np.load(f"{C.FIX}/ref_{key}_codes.npy"); n = oc.shape[1]
    per_cb = (codes[:, :n] != oc).sum(1)
    print(f"ref {key}: codes match {(codes[:, :n] == oc).mean():.4f} ({(codes[:, :n] != oc).sum()} of {oc.size} differ; per codebook {per_cb.tolist()}) | {dt*1e3:.0f} ms")
