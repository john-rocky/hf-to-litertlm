"""The shipped weight form of an exported float32 file: FULLY_CONNECTED weights stored as float16 (ai-edge-quantizer).

    python conversion/quantize_dec.py --src exports/L128_w32.tflite --out mnn-pocket-0.3b-dec_L128_fp16w.tflite

The recipe: weight-only 16-bit FLOAT_CASTING on FULLY_CONNECTED, channelwise, explicit DEQUANTIZE, float compute
(every Linear of the backbone; the attention BATCH_MATMULs have no weight). No calibration. Checks: 196 FULLY_CONNECTED
ops, all with FLOAT16 weights, no custom op, the op histogram unchanged except DEQUANTIZE. Prints the output's bytes and
SHA-256 and writes <out>.quant.json next to it (recipe, seconds, checks)."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quant_scan import quant_scan  # noqa: E402

FC_COUNT = 28 * 7


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="the float32 file export_dec.py wrote")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    from ai_edge_quantizer import qtyping, quantizer, recipe_manager
    from ai_edge_quantizer.algorithm_manager import AlgorithmName
    rm = recipe_manager.RecipeManager()
    rm.add_weight_only_config(regex=".*", operation_name=qtyping.TFLOperationName.FULLY_CONNECTED, num_bits=16,
                              granularity=qtyping.QuantGranularity.CHANNELWISE,
                              algorithm_key=AlgorithmName.FLOAT_CASTING)
    recipe = rm.get_quantization_recipe()
    t0 = time.perf_counter()
    qt = quantizer.Quantizer(str(a.src), recipe)
    if qt.need_calibration or rm.need_calibration():
        raise SystemExit("the recipe needs calibration")
    result = qt.quantize()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    result.export_model(str(a.out))
    seconds = round(time.perf_counter() - t0, 1)
    del result, qt
    out_scan, src_scan = quant_scan(a.out), quant_scan(a.src)
    no_deq = lambda h: {k: v for k, v in h.items() if k not in ("DEQUANTIZE", "QUANTIZE")}
    checks = {"fc_count": out_scan["fully_connected"] == FC_COUNT,
              "fc_weights_all_float16": out_scan["fc_weight_source_dtype"] == {"FLOAT16": FC_COUNT},
              "no_custom_op": out_scan["custom_op_count"] == 0,
              "op_histogram_unchanged_except_dequantize":
                  no_deq(out_scan["op_histogram"]) == no_deq(src_scan["op_histogram"])}
    doc = {"input_sha256": sha256_file(a.src), "output": a.out.name, "bytes": a.out.stat().st_size,
           "sha256": sha256_file(a.out), "seconds": seconds, "recipe": recipe, "checks": checks,
           "versions": {p: importlib.metadata.version(p) for p in ("ai-edge-quantizer", "ai-edge-litert")}}
    a.out.with_name(a.out.name + ".quant.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(f"{a.out}: {doc['bytes']:,} bytes, SHA-256 {doc['sha256']}")
    if not all(checks.values()):
        raise SystemExit("checks failed: " + json.dumps(checks))


if __name__ == "__main__":
    main()
