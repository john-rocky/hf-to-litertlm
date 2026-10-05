"""Storage variant V2 (fp16 FULLY_CONNECTED weights by FLOAT_CASTING + dynamic int8 embedding table) of a shared-state
fp32 file, with the recipe of quantize_kev.py (imported unchanged: build_recipe, quant_scan). r14_export_shared.py and
r15_export_shared.py import build_recipe_v2_keep_rope and fc_weights from here.

    python quantize_shared_state.py --Ls 128 --Lq 64 [--variant v2]

Input  = exports/kev08b_sharedstate_Ls<Ls>_Lq<Lq>_fp32.tflite (never touched).
Output = exports/kev08b_sharedstate_Ls<Ls>_Lq<Lq>_<v2_fp16fc_i8emb>.tflite + results/quant_sharedstate_Ls<Ls>_Lq<Lq>_<..>.json.
The recipe adds one rule after the V2 ones (later rules win in ai-edge-quantizer): FULLY_CONNECTED ops whose scope
matches `RopeSelect` (the question step's RoPE table, r14_shared_state.RopeSelect) are NO_QUANTIZE, so the cos / sin
table stays float32 and the one-hot selection stays exact.
Checks (json first, then asserted): every FULLY_CONNECTED reads FLOAT16 weights (through DEQUANTIZE) except the RopeSelect
ones (FLOAT32, at least one), every EMBEDDING_LOOKUP table is INT8, BATCH_MATMUL constant operands stay FLOAT32 (zero
initial states), CUSTOM 0, op histogram unchanged except DEQUANTIZE. The size is compared with the row-form V2 file of
the same family to see whether the two signatures share the weight buffers after quantization."""
import argparse
import importlib.metadata
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quantize_kev import EXPECT, VARIANTS, quant_scan  # noqa: E402
from r2_common import K, Clock, dump_json, sha256_file  # noqa: E402


ROPE_SCOPE = "RopeSelect"


def build_recipe_v2_keep_rope():
    """quantize_kev.build_recipe("v2") + NO_QUANTIZE for the RopeSelect FULLY_CONNECTED."""
    from ai_edge_quantizer import qtyping, recipe_manager
    from ai_edge_quantizer.algorithm_manager import AlgorithmName
    G = qtyping.QuantGranularity
    OP = qtyping.TFLOperationName
    rm = recipe_manager.RecipeManager()
    rm.add_weight_only_config(regex=".*", operation_name=OP.FULLY_CONNECTED, num_bits=16,
                              granularity=G.CHANNELWISE, algorithm_key=AlgorithmName.FLOAT_CASTING)
    rm.add_dynamic_config(regex=".*", operation_name=OP.EMBEDDING_LOOKUP, num_bits=8, granularity=G.CHANNELWISE)
    rm.add_quantization_config(regex=ROPE_SCOPE, operation_name=OP.FULLY_CONNECTED,
                               algorithm_key=AlgorithmName.NO_QUANTIZE)
    return rm.get_quantization_recipe(), rm.need_calibration()


def fc_weights(path):
    """[(output tensor name, weight source dtype)] of every FULLY_CONNECTED (through a DEQUANTIZE producer)."""
    import mmap
    from ai_edge_litert import schema_py_generated as schema
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    model = schema.Model.GetRootAs(mm, 0)
    names = {v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int)}
    types = {v: k for k, v in vars(schema.TensorType).items() if isinstance(v, int)}
    codes = [names.get(max(model.OperatorCodes(i).BuiltinCode(), model.OperatorCodes(i).DeprecatedBuiltinCode()))
             for i in range(model.OperatorCodesLength())]
    out = []
    for gi in range(model.SubgraphsLength()):
        g = model.Subgraphs(gi)
        ops = [g.Operators(o) for o in range(g.OperatorsLength())]
        producer = {int(t): op for op in ops for t in (op.OutputsAsNumpy() if op.OutputsLength() else [])}
        for op in ops:
            if codes[op.OpcodeIndex()] != "FULLY_CONNECTED":
                continue
            w = int(op.InputsAsNumpy()[1])
            src = w
            if w in producer and codes[producer[w].OpcodeIndex()] == "DEQUANTIZE":
                src = int(producer[w].InputsAsNumpy()[0])
            out.append((g.Tensors(int(op.OutputsAsNumpy()[0])).Name().decode(), types.get(g.Tensors(src).Type())))
    mm.close()
    f.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ls", type=int, required=True)
    ap.add_argument("--Lq", type=int, required=True)
    ap.add_argument("--variant", choices=["v2"], default="v2")
    ap.add_argument("--only", choices=["state", "question"], default="",
                    help="the single-signature file of export_shared_state.py --only")
    a = ap.parse_args()
    name = VARIANTS[a.variant]
    tag = f"Ls{a.Ls}_Lq{a.Lq}" + (f"_{a.only}only" if a.only else "")
    src = K / f"exports/kev08b_sharedstate_{tag}_fp32.tflite"
    out = K / f"exports/kev08b_sharedstate_{tag}_{name}.tflite"
    res = K / f"results/quant_sharedstate_{tag}_{name}.json"
    assert src.exists(), src
    assert not out.exists() and not res.exists(), f"refusing to overwrite {out.name} / {res.name}"
    clock = Clock()
    started = clock.stamp()
    recipe, need_cal = build_recipe_v2_keep_rope()
    from ai_edge_quantizer import quantizer
    t0 = time.perf_counter()
    qt = quantizer.Quantizer(str(src), recipe)
    load_s = time.perf_counter() - t0
    if qt.need_calibration or need_cal:
        raise SystemExit(f"STOP: recipe {name} needs calibration")
    t1 = time.perf_counter()
    result = qt.quantize()
    quant_s = time.perf_counter() - t1
    t2 = time.perf_counter()
    result.export_model(str(out))
    export_s = time.perf_counter() - t2
    import resource
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    del result, qt
    scan_q, scan_src = quant_scan(out), quant_scan(src)
    exp = EXPECT[a.variant]
    emb = scan_q["embedding_lookup"]
    n_fc = scan_src["fully_connected"]
    fcs = fc_weights(out)
    rope = [(n, d) for n, d in fcs if ROPE_SCOPE in n]
    other = [(n, d) for n, d in fcs if ROPE_SCOPE not in n]
    row_v2 = K / f"exports/kev08b_rowprefill_L512_{name}.tflite"
    checks = {
        f"fc_count_unchanged_{n_fc}": scan_q["fully_connected"] == n_fc,
        "fc_weight_all_" + exp["fc_weight"] + "_except_rope": bool(other) and all(d == exp["fc_weight"] for _, d in other),
        "rope_fc_FLOAT32": (bool(rope) or a.only == "state") and all(d == "FLOAT32" for _, d in rope),
        "embedding_lookup_tables_" + exp["table"]: bool(emb) and all(e["table_source_dtype"] == exp["table"] for e in emb),
        "batch_matmul_constants_FLOAT32_only": set(scan_q["batch_matmul_constant_operand_dtype"]) <= {"FLOAT32"},
        "custom_0": scan_q["custom_op_count"] == 0,
        "op_histogram_unchanged_except_DEQUANTIZE": {k: v for k, v in scan_q["op_histogram"].items() if k != "DEQUANTIZE"}
        == {k: v for k, v in scan_src["op_histogram"].items() if k != "DEQUANTIZE"},
    }
    doc = {
        "step": f"quantize the shared-state pair {tag} -> {name}", "started_at": started,
        "seconds_wall": clock.seconds(),
        "input": {"file": str(src.relative_to(K)), "bytes": src.stat().st_size},
        "output": {"file": str(out.relative_to(K)), "bytes": out.stat().st_size, "sha256": sha256_file(out)},
        "row_form_same_variant": {"file": str(row_v2.relative_to(K)), "bytes": row_v2.stat().st_size}
        if row_v2.exists() else None,
        "ai_edge_quantizer": importlib.metadata.version("ai-edge-quantizer"),
        "ai_edge_litert": importlib.metadata.version("ai-edge-litert"),
        "recipe": recipe, "need_calibration": bool(need_cal),
        "seconds": {"load": round(load_s, 1), "quantize": round(quant_s, 1), "export": round(export_s, 1)},
        "peak_rss_bytes_getrusage": int(peak_rss), "scan": scan_q,
        "rope_fully_connected": [{"output": n, "weight_dtype": d} for n, d in rope],
        "scan_fp32_input": {k: scan_src[k] for k in ("operator_count", "op_histogram", "fully_connected",
                                                     "fc_weight_source_dtype", "dequantize_count", "embedding_lookup",
                                                     "batch_matmul_constant_operand_dtype", "constant_bytes_by_dtype")},
        "checks": checks, "checks_pass": all(checks.values()),
    }
    dump_json(res, doc)
    print(json.dumps({k: doc[k] for k in ("output", "row_form_same_variant", "seconds", "checks", "checks_pass")}, indent=1))
    print(json.dumps({k: scan_q[k] for k in ("fully_connected", "fc_weight_source_dtype", "dequantize_count",
                                             "batch_matmul_constant_operand_dtype", "constant_bytes_by_dtype")}, indent=1))
    print(json.dumps([{k: e[k] for k in ("table_source_dtype", "table_shape", "table_bytes", "table_source_bytes")}
                      for e in emb], indent=1))
    assert doc["checks_pass"], checks


if __name__ == "__main__":
    main()
