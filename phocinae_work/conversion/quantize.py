"""Step 5: storage variants of an fp32 graph with ai-edge-quantizer (weights only; activations stay fp32).

    python quantize.py --src exports/phocinae_L512_ids_fp32.tflite --variant v2_fp16fc_i8emb

variants (the act head stays fp32):
  v2_fp16fc_i8emb   FLOAT_CASTING 16-bit FULLY_CONNECTED (channelwise) + dynamic int8 EMBEDDING_LOOKUP (channelwise) — kev's V2
  v3_fp16fc_fp16emb FLOAT_CASTING 16-bit on FULLY_CONNECTED and EMBEDDING_LOOKUP (the GPU's answer to an fp16 in-graph table)
  v1_wi8fc_i8emb    dynamic int8 FULLY_CONNECTED + dynamic int8 EMBEDDING_LOOKUP (channelwise)
  v5_i8emb          dynamic int8 EMBEDDING_LOOKUP only (fp32 FC: isolates the table's share of the residual)
  wfp16             FLOAT_CASTING 16-bit FULLY_CONNECTED only (the embeds / host-lookup form; the table ships as fp16 rows)
Output: exports/<src stem with fp32 -> variant>.tflite + results/quant_<stem>.json (recipe, bytes, sha256, op histogram, the
weight dtypes found by the static scan). Never overwrites.
"""
import argparse
import collections
import importlib.metadata
import json
import mmap
import time
from pathlib import Path

from common import W, dump_json, sha256_file
from tflite_scan import scan

VARIANTS = ["v2_fp16fc_i8emb", "v3_fp16fc_fp16emb", "v1_wi8fc_i8emb", "v5_i8emb", "wfp16"]


def build_recipe(variant):
    from ai_edge_quantizer import qtyping, recipe_manager
    from ai_edge_quantizer.algorithm_manager import AlgorithmName
    G, OP = qtyping.QuantGranularity, qtyping.TFLOperationName
    rm = recipe_manager.RecipeManager()
    fp16 = dict(num_bits=16, granularity=G.CHANNELWISE, algorithm_key=AlgorithmName.FLOAT_CASTING)
    if variant == "v2_fp16fc_i8emb":
        rm.add_weight_only_config(regex=".*", operation_name=OP.FULLY_CONNECTED, **fp16)
        rm.add_dynamic_config(regex=".*", operation_name=OP.EMBEDDING_LOOKUP, num_bits=8, granularity=G.CHANNELWISE)
    elif variant == "v3_fp16fc_fp16emb":
        for op in (OP.FULLY_CONNECTED, OP.EMBEDDING_LOOKUP):
            rm.add_weight_only_config(regex=".*", operation_name=op, **fp16)
    elif variant == "v1_wi8fc_i8emb":
        rm.add_dynamic_config(regex=".*", operation_name=OP.FULLY_CONNECTED, num_bits=8)
        rm.add_dynamic_config(regex=".*", operation_name=OP.EMBEDDING_LOOKUP, num_bits=8, granularity=G.CHANNELWISE)
    elif variant == "v5_i8emb":
        rm.add_dynamic_config(regex=".*", operation_name=OP.EMBEDDING_LOOKUP, num_bits=8, granularity=G.CHANNELWISE)
    elif variant == "wfp16":
        rm.add_weight_only_config(regex=".*", operation_name=OP.FULLY_CONNECTED, **fp16)
    else:
        raise SystemExit(variant)
    return rm.get_quantization_recipe(), rm.need_calibration()


def weight_dtypes(path):
    """FC weight dtypes (through a DEQUANTIZE producer) and the EMBEDDING_LOOKUP table dtype, from the flatbuffer."""
    from ai_edge_litert import schema_py_generated as schema
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        model = schema.Model.GetRootAs(mm, 0)
        op_names = {v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int)}
        type_names = {v: k for k, v in vars(schema.TensorType).items() if isinstance(v, int)}
        g = model.Subgraphs(0)
        producer = {}
        ops = []
        for oi in range(g.OperatorsLength()):
            op = g.Operators(oi)
            name = op_names.get(model.OperatorCodes(op.OpcodeIndex()).BuiltinCode(), "?")
            ins = [op.Inputs(k) for k in range(op.InputsLength())]
            outs = [op.Outputs(k) for k in range(op.OutputsLength())]
            ops.append((name, ins))
            for o in outs:
                producer[o] = (name, ins)

        def dtype_of(ti):
            t = g.Tensors(ti)
            return type_names.get(t.Type(), str(t.Type()))

        fc = collections.Counter()
        table = None
        for name, ins in ops:
            if name == "FULLY_CONNECTED":
                w = ins[1]
                d = dtype_of(w)
                if producer.get(w, ("",))[0] == "DEQUANTIZE":
                    d = dtype_of(producer[w][1][0]) + "->DEQUANTIZE"
                fc[d] += 1
            if name == "EMBEDDING_LOOKUP":
                w = ins[1]
                d = dtype_of(w)
                if producer.get(w, ("",))[0] == "DEQUANTIZE":
                    d = dtype_of(producer[w][1][0]) + "->DEQUANTIZE"
                table = d
        mm.close()
    return dict(fully_connected=dict(fc), embedding_table=table)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    a = ap.parse_args()
    src = Path(a.src) if Path(a.src).is_absolute() else W / a.src
    assert src.exists() and "fp32" in src.name, src
    dst = src.with_name(src.name.replace("fp32", a.variant))
    assert not dst.exists(), f"never overwrite {dst}"
    out = W / "results" / f"quant_{dst.stem}.json"
    t0 = time.time()
    recipe, need_cal = build_recipe(a.variant)
    from ai_edge_quantizer import quantizer
    qt = quantizer.Quantizer(str(src), recipe)
    if qt.need_calibration:
        raise SystemExit(f"STOP: recipe {a.variant} needs calibration")
    result = qt.quantize()
    result.export_model(str(dst))
    rec = dict(src=str(src.relative_to(W)), dst=str(dst.relative_to(W)), variant=a.variant, recipe=recipe, need_calibration=bool(need_cal),
               seconds=round(time.time() - t0, 1), src_bytes=src.stat().st_size, bytes=dst.stat().st_size, sha256=sha256_file(dst),
               weight_dtypes=weight_dtypes(dst), scan=scan(str(dst), with_sha=False),
               versions={p: importlib.metadata.version(p) for p in ("ai-edge-litert", "litert-torch")},
               quantizer=str(Path(quantizer.__file__).parent.parent / "ai_edge_quantizer_nightly-0.10.0.dev20260914.dist-info"))
    dump_json(out, rec)
    print(json.dumps({k: rec[k] for k in ("dst", "bytes", "sha256", "seconds", "weight_dtypes")}, indent=1, default=str))
    print("ops:", json.dumps(rec["scan"]["op_histogram"], sort_keys=True))


if __name__ == "__main__":
    main()
