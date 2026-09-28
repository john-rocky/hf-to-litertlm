"""Post-hoc quantization of the codec decoder + gate vs the oracle waveform.
  modes: drq8 (dynamic int8 CONV/FC), fp16 (FLOAT_CASTING weights)
  ~/venvs/lt094dev/bin/python3 audio8_tts_work/quantize_codec.py <src.tflite> <mode>
"""
import os, sys, time, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import ai_edge_quantizer.recipe as aqr
from ai_edge_quantizer import quantizer
from ai_edge_litert.interpreter import Interpreter

src, mode = sys.argv[1], sys.argv[2]
dst = src.replace("_fp32", f"_{mode}")
t0 = time.time()
qt = quantizer.Quantizer(src)
if mode == "drq8":
    # per-op-type channelwise dynamic int8 (the generic "*" rule trips aeq on 4-D CONV_2D weights here)
    rules = []
    for opn in (aqr.TFLOperationName.FULLY_CONNECTED, aqr.TFLOperationName.CONV_2D, aqr.TFLOperationName.DEPTHWISE_CONV_2D):
        rules.extend(aqr.dynamic_wi8c_afp32(operation_name=opn))
    qt.load_quantization_recipe(rules)
elif mode == "drq8x":
    # like drq8 but skip the one conv whose weight exceeds aeq's 32 MiB chunked path (per-channel 4-D bug)
    rules = []
    for opn in (aqr.TFLOperationName.FULLY_CONNECTED, aqr.TFLOperationName.CONV_2D, aqr.TFLOperationName.DEPTHWISE_CONV_2D):
        for r in aqr.dynamic_wi8c_afp32(operation_name=opn):
            r = dict(r); r["regex"] = r"^(?!.*ArkttsDecoder_dec/torch\.nn\.modules\.container\.Sequential_model/arktts_codec_vendor\.ArkttsCausalConv1d_0/).*"
            rules.append(r)
    qt.load_quantization_recipe(rules)
elif mode == "drq8fc":
    qt.load_quantization_recipe(aqr.dynamic_wi8c_afp32(operation_name=aqr.TFLOperationName.FULLY_CONNECTED))
elif mode == "fp16":
    qt.load_quantization_recipe([{"regex": ".*", "operation": "*", "algorithm_key": aqr.AlgorithmName.FLOAT_CASTING,
        "op_config": {"weight_tensor_config": {"num_bits": 16, "symmetric": True, "granularity": "TENSORWISE", "dtype": "FLOAT"},
                      "compute_precision": "FLOAT", "explicit_dequantize": True, "skip_checks": False, "min_weight_elements": 0}}])
else:
    raise SystemExit(mode)
qt.quantize().export_model(dst)
print(f"{mode}: {os.path.getsize(src)/1e6:.0f} MB -> {dst} {os.path.getsize(dst)/1e6:.0f} MB in {time.time()-t0:.0f}s", flush=True)

it = Interpreter(model_path=dst, num_threads=8); run = it.get_signature_runner("decode")
T = run.get_input_details()["codes"]["shape"][-1]
worst_corr, worst_d, dts = 1.0, 0.0, []
for f in sorted(glob.glob(f"{C.OUT}/oracle/*.npz"))[:4]:
    d = np.load(f); codes, wav = d["codes"], d["wav"]; n = min(T, codes.shape[1])
    buf = np.zeros((1, C.NUM_CB, T), np.int32); buf[0, :, :n] = codes[:, :n]
    t1 = time.perf_counter(); out = run(codes=buf)["wav"][0, 0]; dts.append(time.perf_counter() - t1)
    ref, got = wav[: n * C.FRAME], out[: n * C.FRAME]
    worst_corr = min(worst_corr, np.corrcoef(got, ref)[0, 1]); worst_d = max(worst_d, np.abs(got - ref).max())
print(f"{os.path.basename(dst)}: vs oracle wav (4 cases, {T} frames): min corr {worst_corr:.6f} max|d| {worst_d:.3e} | invoke {np.median(dts)*1e3:.0f} ms median (8 thr)")
