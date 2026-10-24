#!/usr/bin/env python3
"""Round 2: weight forms of the WeVisOne-VL-2B decoder and EMBEDDER from the fp32 export (ai-edge-quantizer 0.10.0).

  wi8     dynamic_wi8_afp32 (int8 channelwise weights, run-time int8 activations on the CPU) on PREFILL_DECODE and
          on the EMBEDDER (its EMBEDDING_LOOKUP table)
  wi4b32  dynamic_wi4b32_afp32 (int4 blockwise-32, symmetric) on PREFILL_DECODE only, with clamp-before-quantize: every
          blockwise int4 scale floored at 2^-14 (memory blockwise-zero-scale-patch-trap; the wrapper is the one of
          aa_mobile_b2_work/nemotron_nano_9b_v2/tools/quantize_clamped.py); the int4 bundle reuses the wi8 EMBEDDER
Each output gets the scale scan (zero and subnormal fp16 block scales; XNNPACK checks std::isnormal).
This is what export_hf's own quantize_model() does after the fp32 export (same Quantizer calls), run separately so
one fp32 export serves every form.

    ~/venvs/lt0100dev/bin/python -I tools/wv_quant_decoder.py wi8|wi4b32 [--src out/r2/dec_fp32] [--dst out/r2/dec_<form>]
"""
import argparse
import atexit
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

K = Path(__file__).resolve().parents[1]
FLOOR = np.float32(2.0 ** -14)
COUNT = {"blockwise_int4_calls": 0, "clamped_scales": 0}


def install_floor():
    from ai_edge_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor as uqt
    orig = uqt.tensor_zp_scale_from_min_max

    def clamped(min_value, max_value, num_bits, symmetric, granularity, clipping_values=None, **kw):
        # ai-edge-quantizer 0.10.0 added `signed` (keyword); pass any keyword through unchanged
        zp, scale = orig(min_value, max_value, num_bits, symmetric, granularity, clipping_values, **kw)
        if uqt.is_blockwise(granularity) and num_bits == 4:
            assert symmetric, "asymmetric blockwise int4 is not covered by the floor"
            COUNT["blockwise_int4_calls"] += 1
            s = np.asarray(scale, dtype=np.float32)
            n = int((s < FLOOR).sum())
            if n:
                COUNT["clamped_scales"] += n
                scale = np.maximum(s, FLOOR)
        return zp, scale
    uqt.tensor_zp_scale_from_min_max = clamped
    atexit.register(lambda: print("FLOOR", COUNT, flush=True))


def scan(path):
    """Blockwise scale census of one tflite: zero, subnormal (0 < s < 2^-14), min."""
    from ai_edge_litert import schema_py_generated as schema
    data = Path(path).read_bytes()
    view = memoryview(data)
    fb = schema.Model.GetRootAsModel(data, 0)
    tiny = np.float16(2.0 ** -14)
    seen, n_t, n_zero, n_sub, n_sc, gmin = set(), 0, 0, 0, 0, None
    for si in range(fb.SubgraphsLength()):
        sg = fb.Subgraphs(si)
        for i in range(sg.TensorsLength()):
            t = sg.Tensors(i)
            q = t.Quantization()
            if q is None or q.DetailsType() != schema.QuantizationDetails.BlockwiseQuantization:
                continue
            bq = schema.BlockwiseQuantization()
            tab = q.Details()
            bq.Init(tab.Bytes, tab.Pos)
            st = sg.Tensors(bq.Scales())
            bi = st.Buffer()
            if bi in seen:
                continue
            seen.add(bi)
            buf = fb.Buffers(bi)
            if buf.DataLength():
                off, size = buf._tab.Vector(buf._tab.Offset(4)), buf.DataLength()
            else:
                off, size = buf.Offset(), buf.Size()
            sc = np.frombuffer(bytes(view[off:off + size]), dtype=np.float16)
            n_t += 1
            n_zero += int((sc == 0).sum())
            n_sub += int(((sc > 0) & (sc < tiny)).sum())
            n_sc += sc.size
            gmin = float(sc.min()) if gmin is None else min(gmin, float(sc.min()))
    return {"blockwise_tensors": n_t, "scales": n_sc, "zero": n_zero, "subnormal": n_sub, "min_scale": gmin,
            "all_normal": n_zero == 0 and n_sub == 0}


def sha256_file(p):
    with open(p, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


FP16_CFG = {"weight_tensor_config": {"num_bits": 16, "symmetric": True, "granularity": "TENSORWISE", "dtype": "FLOAT"},
            "compute_precision": "FLOAT", "explicit_dequantize": False}


def recipe_of(name):
    """Named aeq recipes plus the round-2 mixes (quant lever (a) / (b)):
      fp16_all     float casting (fp16 weights) on every supported op (the vision fp16 recipe of northmv / r2)
      wi8_head32   dynamic_wi8_afp32, then the lm_head FULLY_CONNECTED left float32 (no_quantize)
      wi8_head16   dynamic_wi8_afp32, then the lm_head FULLY_CONNECTED as fp16 weights (float casting)
    A later entry wins in aeq (recipe_manager.get_quantization_configs: the latest matching config). The lm_head FC is
    found by its output tensor `decode_logits_output` (the decode signature's logits; the prefill signatures carry no
    lm_head): its name holds no "lm_head", so a regex on that word matches nothing (first try 20:54, all 197 INT8)."""
    import copy
    import ai_edge_quantizer.recipe as recipe_lib
    if name == "fp16_all":
        base = copy.deepcopy(recipe_lib.dynamic_wi8_afp32()[0])
        base.update({"algorithm_key": "float_casting", "op_config": dict(FP16_CFG)})
        return [base]
    if name in ("wi8_head32", "wi8_head16"):
        r = copy.deepcopy(recipe_lib.dynamic_wi8_afp32())
        if name == "wi8_head32":
            r.append({"regex": "decode_logits_output", "operation": "FULLY_CONNECTED", "algorithm_key": "no_quantize",
                      "op_config": {}})
        else:
            r.append({"regex": "decode_logits_output", "operation": "FULLY_CONNECTED", "algorithm_key": "float_casting",
                      "op_config": dict(FP16_CFG)})
        return r
    return getattr(recipe_lib, name)()


def quantize(src, dst, recipe_name):
    from ai_edge_quantizer import quantizer
    t0 = time.time()
    qt = quantizer.Quantizer(str(src))
    qt.load_quantization_recipe(recipe_of(recipe_name))
    qt.quantize().export_model(str(dst), overwrite=True)
    return round(time.time() - t0, 1)


def fc_dtypes(path):
    """FULLY_CONNECTED / EMBEDDING_LOOKUP weight dtypes of the decode signature (tools/wv_census.py)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("wv_census", Path(__file__).resolve().parent / "wv_census.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    c = m.census(path)
    return {s["signature"]: s["weight_dtypes"] for s in c["subgraphs"] if s["signature"] in ("decode", "embedder",
                                                                                            "prefill_256")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("form", choices=("wi8", "wi4b32", "fp16", "mix32", "mix16"))
    ap.add_argument("--src", default="out/r2/dec_fp32")
    ap.add_argument("--dst", default=None)
    a = ap.parse_args()
    src, dst = K / a.src, K / (a.dst or f"out/r2/dec_{a.form}")
    dst.mkdir(parents=True, exist_ok=True)
    rec = {"form": a.form, "src": str(src), "files": {}}
    import importlib.metadata as md
    rec["ai_edge_quantizer"] = md.version("ai-edge-quantizer")
    if a.form == "wi8":
        jobs = [("model.tflite", "prefill_decode_wi8.tflite", "dynamic_wi8_afp32"),
                ("embedder.tflite", "embedder_wi8.tflite", "dynamic_wi8_afp32")]
    elif a.form == "wi4b32":
        install_floor()
        jobs = [("model.tflite", "prefill_decode_wi4b32.tflite", "dynamic_wi4b32_afp32")]
    elif a.form == "fp16":
        jobs = [("model.tflite", "prefill_decode_fp16.tflite", "fp16_all"),
                ("embedder.tflite", "embedder_fp16.tflite", "fp16_all")]
    elif a.form == "mix32":                          # the EMBEDDER stays the fp32 export (out/r2/dec_fp32)
        jobs = [("model.tflite", "prefill_decode_wi8_head32.tflite", "wi8_head32")]
    else:
        jobs = [("model.tflite", "prefill_decode_wi8_head16.tflite", "wi8_head16"),
                ("embedder.tflite", "embedder_fp16.tflite", "fp16_all")]
    for s, d, recipe in jobs:
        out = dst / d
        if not out.exists():
            sec = quantize(src / s, out, recipe)
            print(f"{d}: {out.stat().st_size} B in {sec} s", flush=True)
        else:
            sec = None
        rec["files"][d] = {"from": s, "recipe": recipe, "seconds": sec, "bytes": out.stat().st_size,
                           "sha256": sha256_file(out), "scale_scan": scan(out), "weight_dtypes": fc_dtypes(out)}
        print(d, json.dumps(rec["files"][d]["weight_dtypes"]), flush=True)
        print(d, json.dumps(rec["files"][d]["scale_scan"]), flush=True)
    rec["floor"] = dict(COUNT)
    (dst / "quant_record.json").write_text(json.dumps(rec, indent=1) + "\n")
    print("QUANT_DONE", a.form, json.dumps({k: v["bytes"] for k, v in rec["files"].items()}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
