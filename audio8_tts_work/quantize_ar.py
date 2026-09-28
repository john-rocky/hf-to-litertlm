"""Post-hoc weight quantization of the AR graphs with ai_edge_quantizer.

  drq8   : dynamic int8 per-channel FULLY_CONNECTED + int8 EMBEDDING_LOOKUP   (house dense recipe)
  bo4    : blockwise-32 OCTAV int4 FULLY_CONNECTED + int8 EMBEDDING_LOOKUP    (house int4 recipe)
  bo4_128: blockwise-128 variant
The tied lm_head slice is a FULLY_CONNECTED like every other projection (covered by the FC rule).
Attention BATCH_MATMULs are activation x activation and are left in fp32.

  ~/venvs/lt094dev/bin/python3 audio8_tts_work/quantize_ar.py <src.tflite> <mode> [dst.tflite]
"""
import copy, os, sys, time
import ai_edge_quantizer.recipe as aqr
from ai_edge_quantizer import quantizer

src, mode = sys.argv[1], sys.argv[2]
dst = sys.argv[3] if len(sys.argv) > 3 else src.replace("_fp32", f"_{mode}")

fc8 = copy.deepcopy(aqr.dynamic_wi8_afp32()[0])          # regex .* / FULLY_CONNECTED, per-channel int8
emb8 = copy.deepcopy(fc8); emb8["operation"] = "EMBEDDING_LOOKUP"
if mode == "drq8":
    recipe = [fc8, emb8]
elif mode == "drq8fc":   # FC int8, embedding tables fp32 (sensitivity probe)
    recipe = [fc8]
elif mode.startswith("bo4"):
    r = copy.deepcopy(aqr.dynamic_wi4_afp32()[0])
    r["algorithm_key"] = aqr.AlgorithmName.OCTAV
    r["op_config"]["weight_tensor_config"]["granularity"] = "BLOCKWISE_128" if mode.endswith("128") else "BLOCKWISE_32"
    recipe = [r, emb8]
elif mode == "fp16":
    recipe = [{"regex": ".*", "operation": "*", "algorithm_key": aqr.AlgorithmName.FLOAT_CASTING,
               "op_config": {"weight_tensor_config": {"num_bits": 16, "symmetric": True, "granularity": "TENSORWISE", "dtype": "FLOAT"},
                             "compute_precision": "FLOAT", "explicit_dequantize": True, "skip_checks": False, "min_weight_elements": 0}}]
else:
    raise SystemExit(f"unknown mode {mode}")
t0 = time.time()
qt = quantizer.Quantizer(src)
qt.load_quantization_recipe(recipe)
res = qt.quantize()
res.export_model(dst)
print(f"{mode}: {src} ({os.path.getsize(src)/1e6:.0f} MB) -> {dst} ({os.path.getsize(dst)/1e6:.0f} MB) in {time.time()-t0:.0f}s")
