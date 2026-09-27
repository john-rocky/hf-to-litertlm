"""(a2) of the round-2 row-1 split: FULLY_CONNECTED-only dynamic int8 on the UNQUANTISED LM export, embedder left fp32.

Input : out/lm_fp32/unpack/*prefill_decode*.tflite (export_simple_template.py ... NONE, same env as the house export)
        out/lm_fp32/unpack/*embedder*.tflite (fp32, used as is)
Output: out/lm_fp32_int8fc/prefill_decode_int8fc.tflite + a symlink to the fp32 embedder (build_bundle.py --lm reads
        this dir) and lm_fp32_int8fc_report.json.
Recipe: RecipeManager.add_dynamic_config(regex ".*", FULLY_CONNECTED, 8 bits) = symmetric per-channel int8 weights,
        compute INTEGER, no explicit dequantize (the per-op config of the house `dynamic_wi8_afp32`, which applies it to
        every supported op, the embedder's EMBEDDING_LOOKUP included).
Audit : weight dtype of every FC op; and the FC weight buffers + scales compared with the house int8 export
        (out/lm_int8/unpack): if they are byte-identical, the house bundle and this one differ ONLY in the embedder.
Run with ~/venvs/ltconv040dev/bin/python (ai-edge-quantizer 0.8.0 = the version that made the house int8 export).
"""
import collections
import glob
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "out", "lm_fp32", "unpack")
HOUSE = os.path.join(HERE, "out", "lm_int8", "unpack")
DST = os.path.join(HERE, "out", "lm_fp32_int8fc")


def one(d, key):
    c = [p for p in glob.glob(os.path.join(d, "*.tflite")) if key in os.path.basename(p)]
    assert len(c) == 1, (d, key, c)
    return c[0]


def fc_weights(path):
    """[(subgraph, op index, weight tensor name, dtype, shape, sha256(data), sha256(scales))] for every FC op."""
    import numpy as np
    from ai_edge_quantizer.utils import tfl_flatbuffer_utils as fu
    raw = open(path, "rb").read()
    m = fu.read_model(bytearray(raw))
    names = {}
    for i, oc in enumerate(m.operatorCodes):
        code = max(oc.builtinCode, getattr(oc, "deprecatedBuiltinCode", 0) or 0)
        names[i] = code
    FC = 9  # tflite BuiltinOperator.FULLY_CONNECTED
    TT = {0: "float32", 1: "float16", 9: "int8", 17: "int4"}
    rows = []
    for si, sg in enumerate(m.subgraphs):
        for oi, op in enumerate(sg.operators):
            if names[op.opcodeIndex] != FC:
                continue
            t = sg.tensors[op.inputs[1]]
            buf = m.buffers[t.buffer]
            if buf.data is not None:
                data = bytes(buf.data)
            elif getattr(buf, "offset", 0):  # >2 GB layout: buffer stored after the flatbuffer
                data = raw[buf.offset:buf.offset + buf.size]
            else:
                data = b""
            q = t.quantization
            scales = np.asarray(q.scale, np.float32).tobytes() if (q is not None and q.scale is not None) else b""
            rows.append({"subgraph": si, "op": oi, "name": t.name.decode() if isinstance(t.name, bytes) else t.name,
                         "dtype": TT.get(t.type, str(t.type)), "shape": [int(x) for x in t.shape],
                         "data_sha": hashlib.sha256(data).hexdigest()[:16], "data_len": len(data),
                         "scale_sha": hashlib.sha256(scales).hexdigest()[:16], "buffer": int(t.buffer)})
    return rows


def main():
    from ai_edge_quantizer import quantizer, recipe_manager, qtyping
    from importlib.metadata import version
    os.makedirs(DST, exist_ok=True)
    pd = one(SRC, "prefill_decode")
    emb = one(SRC, "embedder")
    out = os.path.join(DST, "prefill_decode_int8fc.tflite")
    if os.path.exists(out):
        os.remove(out)
    rm = recipe_manager.RecipeManager()
    rm.add_dynamic_config(regex=".*", operation_name=qtyping.TFLOperationName.FULLY_CONNECTED, num_bits=8)
    recipe = rm.get_quantization_recipe()
    qt = quantizer.Quantizer(pd, recipe)
    assert not qt.need_calibration
    t0 = time.time()
    qt.quantize().export_model(out)
    dt = time.time() - t0
    link = os.path.join(DST, "embedder_fp32.tflite")
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(os.path.relpath(emb, DST), link)
    print(f"{out} {os.path.getsize(out):,} B in {dt:.0f}s | embedder {emb} {os.path.getsize(emb):,} B", flush=True)

    mine = fc_weights(out)
    house = fc_weights(one(HOUSE, "prefill_decode"))
    dt_mine = collections.Counter(r["dtype"] for r in mine)
    dt_house = collections.Counter(r["dtype"] for r in house)
    # compare per unique weight buffer (signatures share weights)
    ub_mine = {r["buffer"]: (r["name"], r["shape"], r["dtype"], r["data_sha"], r["scale_sha"]) for r in mine}
    ub_house = {r["buffer"]: (r["name"], r["shape"], r["dtype"], r["data_sha"], r["scale_sha"]) for r in house}
    sig_mine = collections.Counter((v[1].__repr__(), v[2], v[3], v[4]) for v in ub_mine.values())
    sig_house = collections.Counter((v[1].__repr__(), v[2], v[3], v[4]) for v in ub_house.values())
    rep = {"recipe": recipe, "aeq_version": version("ai-edge-quantizer"), "quantize_s": round(dt, 1),
           "input_prefill_decode": os.path.relpath(pd, HERE), "input_bytes": os.path.getsize(pd),
           "output": os.path.relpath(out, HERE), "output_bytes": os.path.getsize(out),
           "embedder": os.path.relpath(emb, HERE), "embedder_bytes": os.path.getsize(emb),
           "fc_ops": len(mine), "fc_weight_dtypes": dict(dt_mine), "unique_fc_weight_buffers": len(ub_mine),
           "house_fc_ops": len(house), "house_fc_weight_dtypes": dict(dt_house), "house_unique_fc_weight_buffers": len(ub_house),
           "fc_weights_and_scales_identical_to_house": sig_mine == sig_house,
           "only_in_mine": [list(k) for k in (sig_mine - sig_house)][:10],
           "only_in_house": [list(k) for k in (sig_house - sig_mine)][:10]}
    with open(os.path.join(HERE, "lm_fp32_int8fc_report.json"), "w") as f:
        json.dump(rep, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in rep.items() if k != "recipe"}, indent=1, default=str), flush=True)
    print("QUANT_LM_DONE", flush=True)


if __name__ == "__main__":
    main()
