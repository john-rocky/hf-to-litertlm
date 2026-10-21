#!/usr/bin/env python3
"""Round 2: storage forms of a float32 row graph with ai-edge-quantizer 0.9.0 (lt094dev), FULLY_CONNECTED only.

    TMPDIR=<K>/tmp ~/venvs/lt094dev/bin/python scripts/r2_quant.py --src out/r1/clef_rowprefill_L1024_fp32.tflite \
        --form v2_fp16fc
        -> out/r2/clef_rowprefill_L1024_v2_fp16fc.tflite + results/r2_quant_L1024_v2_fp16fc.json (refuses to overwrite)

form (the FC lines of kev_work/scripts/quantize_kev.py `build_recipe`; this graph has no embedding table):
  v2_fp16fc    weight-only 16-bit FLOAT_CASTING on FULLY_CONNECTED, CHANNELWISE (= v2's FC line): fp16 weight
               constants -> DEQUANTIZE -> float32 FC
  v1_dyni8fc   dynamic int8 on FULLY_CONNECTED (= v1's FC line, default granularity): int8 weight constants, the
               activations quantized per call by the runtime
The output name = the input stem without `_fp32` + `_<form>` in out/r2 (`..._L256_ns_fp32` -> `..._L256_ns_v2_fp16fc`).
quantize(serialize_to_path=<out>.part) writes into an mmapped file (the input is mmapped too); the scan runs on the
.part file, which is renamed to the output name only when every check passes (a failed form stays as .part, named in
the json). Recorded: seconds (load / quantize / scan / sha256), memory (ru_maxrss, phys_footprint and its lifetime
maximum), the recipe, the static scan of input and output (op histogram, FC weight source dtype through a DEQUANTIZE
producer, DEQUANTIZE in -> out dtypes, FC input dtypes, tensor dtype histogram, constant bytes by dtype, INT64 /
rank > 4 / CUSTOM / BROADCAST_TO from scripts/tflite_scan.py), sha256 and bytes.
Acceptance (pre-registered in the round record):
  v2_fp16fc   FC count = the input's (296), every FC weight FLOAT16 through DEQUANTIZE (FLOAT16 -> FLOAT32), FC
              inputs FLOAT32, DEQUANTIZE = the FC count, INT8 / INT4 tensors 0, op histogram = the input's +
              DEQUANTIZE, BROADCAST_TO / INT64 / CUSTOM / rank > 4 = 0, bytes in [FC fp32 weight bytes / 2 + the other
              constant bytes, that * 1.01 + 64 MiB]
  v1_dyni8fc  FC count = the input's, every FC weight an INT8 constant (no DEQUANTIZE), op histogram = the input's,
              BROADCAST_TO / INT64 / CUSTOM / rank > 4 = 0 (size recorded)"""
import argparse
import collections
import hashlib
import importlib.metadata
import json
import mmap
import os
import resource
import sys
import time
from pathlib import Path

K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "scripts"))
from clef_export import memory  # noqa: E402
from tflite_scan import scan  # noqa: E402

FORMS = ("v2_fp16fc", "v1_dyni8fc")


def build_recipe(form):
    """The FC line of kev_work/scripts/quantize_kev.py build_recipe v2 / v1 (copied, nothing else)."""
    from ai_edge_quantizer import qtyping, recipe_manager
    from ai_edge_quantizer.algorithm_manager import AlgorithmName

    G = qtyping.QuantGranularity
    OP = qtyping.TFLOperationName
    rm = recipe_manager.RecipeManager()
    if form == "v2_fp16fc":
        rm.add_weight_only_config(regex=".*", operation_name=OP.FULLY_CONNECTED, num_bits=16,
                                  granularity=G.CHANNELWISE, algorithm_key=AlgorithmName.FLOAT_CASTING)
    elif form == "v1_dyni8fc":
        rm.add_dynamic_config(regex=".*", operation_name=OP.FULLY_CONNECTED, num_bits=8)
    else:
        raise ValueError(form)
    return rm.get_quantization_recipe(), rm.need_calibration()


def fc_scan(path):
    """kev_work/scripts/quantize_kev.py quant_scan (copied; the EMBEDDING_LOOKUP part dropped, FC input dtypes and the
    FC weight buffer bytes added): weight dtypes of FC through a DEQUANTIZE producer, DEQUANTIZE in -> out, constants."""
    from ai_edge_litert import schema_py_generated as schema
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    model = schema.Model.GetRootAs(mm, 0)
    op_names = {v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int)}
    type_names = {v: k for k, v in vars(schema.TensorType).items() if isinstance(v, int)}
    codes = []
    for i in range(model.OperatorCodesLength()):
        c = model.OperatorCodes(i)
        name = op_names.get(max(c.BuiltinCode(), c.DeprecatedBuiltinCode()), "UNKNOWN")
        if name == "CUSTOM":
            name = "CUSTOM:" + (c.CustomCode().decode() if c.CustomCode() else "")
        codes.append(name)
    hist, fc_weight, fc_weight_direct, fc_quant_axis = (collections.Counter() for _ in range(4))
    fc_in, deq_in_out, tensor_dtypes, const_bytes_by_dtype = (collections.Counter() for _ in range(4))
    fc_weight_buffers, fc_rows = {}, []
    for gi in range(model.SubgraphsLength()):
        g = model.Subgraphs(gi)
        tensors = [g.Tensors(t) for t in range(g.TensorsLength())]
        ttype = lambda t: type_names.get(tensors[t].Type(), str(tensors[t].Type()))
        shape = lambda t: tensors[t].ShapeAsNumpy().tolist() if tensors[t].ShapeLength() else []
        ops = [g.Operators(o) for o in range(g.OperatorsLength())]
        producer = {}
        for oi, op in enumerate(ops):
            for t in (op.OutputsAsNumpy().tolist() if op.OutputsLength() else []):
                producer[int(t)] = oi
        graph_inputs = set(g.InputsAsNumpy().tolist()) if g.InputsLength() else set()
        is_const = lambda t: t >= 0 and t not in producer and t not in graph_inputs

        def nbytes(t):
            b = model.Buffers(tensors[t].Buffer())
            return b.DataLength() if b.DataLength() > 0 else (b.Size() if b.Offset() > 1 else 0)

        def quant(t):
            q = tensors[t].Quantization()
            if q is None or q.ScaleLength() == 0:
                return None
            return {"scales": q.ScaleLength(), "quantized_dimension": q.QuantizedDimension()}

        for ti in range(len(tensors)):
            tensor_dtypes[ttype(ti)] += 1
            if is_const(ti):
                const_bytes_by_dtype[ttype(ti)] += nbytes(ti)
        for oi, op in enumerate(ops):
            name = codes[op.OpcodeIndex()]
            hist[name] += 1
            ins = op.InputsAsNumpy().tolist() if op.InputsLength() else []
            if name == "DEQUANTIZE":
                deq_in_out[f"{ttype(ins[0])}->{ttype(op.OutputsAsNumpy()[0])}"] += 1
            if name == "FULLY_CONNECTED":
                w = ins[1]
                direct = ttype(w) if is_const(w) else None
                src, via, wsrc = direct, ("constant" if is_const(w) else "activation"), w
                if not is_const(w) and w in producer and codes[ops[producer[w]].OpcodeIndex()] == "DEQUANTIZE":
                    wsrc = int(ops[producer[w]].InputsAsNumpy()[0])
                    src, via = ttype(wsrc), "DEQUANTIZE(" + ttype(wsrc) + ")"
                fc_weight[src] += 1
                fc_weight_direct[str(direct)] += 1
                fc_in[ttype(ins[0])] += 1
                if is_const(wsrc):
                    fc_weight_buffers[tensors[wsrc].Buffer()] = nbytes(wsrc)
                q = quant(w) if is_const(w) else None
                if q:
                    fc_quant_axis[f"scales={q['scales']} axis={q['quantized_dimension']}"] += 1
                if len(fc_rows) < 5:
                    fc_rows.append({"op_index": oi, "out": tensors[op.OutputsAsNumpy()[0]].Name().decode(),
                                    "weight_shape": shape(w), "weight_source_dtype": src, "weight_via": via})
    mm.close()
    f.close()
    return {"operator_count": sum(hist.values()), "op_histogram": dict(sorted(hist.items())),
            "fully_connected": sum(fc_weight.values()), "fc_weight_source_dtype": dict(fc_weight),
            "fc_weight_tensor_dtype_direct": dict(fc_weight_direct), "fc_input_dtype": dict(fc_in),
            "fc_weight_quant_layout": dict(fc_quant_axis), "fc_weight_buffers": len(fc_weight_buffers),
            "fc_weight_buffer_bytes": int(sum(fc_weight_buffers.values())),
            "dequantize_count": hist.get("DEQUANTIZE", 0), "dequantize_in_out": dict(deq_in_out),
            "custom_op_count": sum(v for k, v in hist.items() if k.startswith("CUSTOM")),
            "tensor_dtype_histogram": dict(sorted(tensor_dtypes.items())),
            "constant_bytes_by_dtype": dict(sorted(const_bytes_by_dtype.items())),
            "constant_bytes": int(sum(const_bytes_by_dtype.values())), "fc_rows_first5": fc_rows}


def shared_weight_check(src, qpath) -> list:
    """The FC weight buffers that more than one FC reads in the float32 input (decided 2026-10-20 19:3x: the
    Gated DeltaNet kernel's constant matrices, shared across the 24 layers): which FCs read them (matched by the FC
    output tensor name), shape, values, and whether the quantized file's FLOAT16 copy read back as float32 is
    bit-identical to the float32 buffer."""
    import numpy as np
    from ai_edge_litert import schema_py_generated as schema

    def load(path):
        f = open(path, "rb")
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        m = schema.Model.GetRootAs(mm, 0)
        g = m.Subgraphs(0)
        names = {v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int)}
        tnames = {v: k for k, v in vars(schema.TensorType).items() if isinstance(v, int)}
        codes = [names.get(max(m.OperatorCodes(i).BuiltinCode(), m.OperatorCodes(i).DeprecatedBuiltinCode()))
                 for i in range(m.OperatorCodesLength())]
        tensors = [g.Tensors(t) for t in range(g.TensorsLength())]
        ops = [g.Operators(o) for o in range(g.OperatorsLength())]
        producer = {int(t): oi for oi, op in enumerate(ops) for t in (op.OutputsAsNumpy().tolist() if op.OutputsLength() else [])}

        def data(t):
            b = m.Buffers(tensors[t].Buffer())
            if b.DataLength() > 0:
                return b.DataAsNumpy().tobytes()
            if b.Offset() > 1:
                return bytes(mm[b.Offset(): b.Offset() + b.Size()])
            return b""
        fcs = {}
        for op in ops:
            if codes[op.OpcodeIndex()] != "FULLY_CONNECTED":
                continue
            w = int(op.InputsAsNumpy()[1])
            if w in producer and codes[ops[producer[w]].OpcodeIndex()] == "DEQUANTIZE":
                w = int(ops[producer[w]].InputsAsNumpy()[0])
            fcs[tensors[op.OutputsAsNumpy()[0]].Name().decode()] = (w, tensors[w].Buffer(), tnames.get(tensors[w].Type()),
                                                                      tensors[w].ShapeAsNumpy().tolist())
        return f, mm, data, fcs

    fa, ma, da, fca = load(src)
    fb, mb, db, fcb = load(qpath)
    readers = {}
    for out, (w, buf, dt, shp) in fca.items():
        readers.setdefault(buf, []).append(out)
    rows = []
    for buf, outs in readers.items():
        if len(outs) < 2:
            continue
        w, _, dt, shp = fca[outs[0]]
        v32 = np.frombuffer(da(w), dtype=np.float32)
        q = [fcb[o] for o in outs]
        q_bufs = sorted({x[1] for x in q})
        q_dt = sorted({x[2] for x in q})
        w2 = q[0][0]
        raw = db(w2)
        back = (np.frombuffer(raw, dtype=np.float16).astype(np.float32) if q_dt == ["FLOAT16"]
                else np.frombuffer(raw, dtype=np.float32))
        uniq = np.unique(v32)
        rows.append({"buffer_in_fp32_file": int(buf), "readers": len(outs), "reader_examples": outs[:3], "shape": shp,
                     "dtype_in": dt, "dtype_out": q_dt, "buffers_out": len(q_bufs), "values_unique": int(uniq.size),
                     "values_first": [float(x) for x in uniq[:8]],
                     "bit_equal_after_fp16_to_fp32": bool(back.shape == v32.shape and np.array_equal(back.view(np.uint32),
                                                                                                     v32.view(np.uint32)))})
    for f, mm in ((fa, ma), (fb, mb)):
        mm.close()
        f.close()
    return rows


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 24):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="K-relative float32 .tflite")
    ap.add_argument("--form", choices=FORMS, required=True)
    ap.add_argument("--recheck", action="store_true",
                    help="no quantization: re-scan the .part of an earlier attempt whose json was moved to "
                         "<json stem>.attempt1.json (its quantize seconds / memory are carried over)")
    a = ap.parse_args()
    src = K / a.src
    stem = src.stem
    assert stem.endswith("_fp32"), stem
    base = stem[: -len("_fp32")]
    L = int(base.split("_L")[1].split("_")[0])
    out = K / f"out/r2/{base}_{a.form}.tflite"
    part = out.with_name(out.name + ".part")
    res = K / f"results/r2_quant_{base.replace('clef_rowprefill_', '')}_{a.form}.json"
    prev_path = res.with_name(res.stem + ".attempt1.json")
    assert src.exists(), src
    for p in ((out, res) if a.recheck else (out, part, res)):
        assert not p.exists(), f"refusing to overwrite {p}"
    out.parent.mkdir(parents=True, exist_ok=True)
    t_wall = time.perf_counter()
    doc = {"what": f"round 2: {a.form} of {a.src}", "L": L, "form": a.form, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "input": {"file": a.src, "bytes": src.stat().st_size},
           "versions": {p: importlib.metadata.version(p) for p in ("ai-edge-quantizer", "ai-edge-litert")},
           "memory_start": memory()}
    recipe, need_cal = build_recipe(a.form)
    doc["recipe"], doc["need_calibration"] = recipe, bool(need_cal)
    if a.recheck:
        prev = json.loads(prev_path.read_text())
        assert part.exists() and prev["output"]["file"] == str(part.relative_to(K)), (part, prev["output"])
        for k in ("load_seconds", "quantize_seconds", "memory_after_quantize", "peak_rss_bytes_getrusage", "started_at"):
            doc[k if k != "started_at" else "quantized_at"] = prev.get(k)
        doc["recheck_of"] = {"json": str(prev_path.relative_to(K)), "failing_then": prev["failing"],
                             "sha256_then": prev["output"]["sha256"], "bytes_then": prev["output"]["bytes"]}
    else:
        from ai_edge_quantizer import quantizer
        t0 = time.perf_counter()
        qt = quantizer.Quantizer(str(src), recipe)
        doc["load_seconds"] = round(time.perf_counter() - t0, 1)
        if qt.need_calibration or need_cal:
            raise SystemExit(f"STOP: recipe {a.form} needs calibration")
        t1 = time.perf_counter()
        result = qt.quantize(serialize_to_path=str(part))
        doc["quantize_seconds"] = round(time.perf_counter() - t1, 1)
        doc["memory_after_quantize"] = memory()
        del result, qt
        doc["peak_rss_bytes_getrusage"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps({"quantized": a.form, "load_s": doc["load_seconds"], "quantize_s": doc["quantize_seconds"],
                      "memory": doc["memory_after_quantize"], "recheck": a.recheck}), flush=True)
    t2 = time.perf_counter()
    sq, ss = fc_scan(part), fc_scan(src)
    gq = scan(part, with_sha=False)
    doc["scan_seconds"] = round(time.perf_counter() - t2, 1)
    t3 = time.perf_counter()
    sha = sha256_file(part)
    doc["sha256_seconds"] = round(time.perf_counter() - t3, 1)
    nbytes = part.stat().st_size
    nfc = ss["fully_connected"]
    other_const = ss["constant_bytes"] - ss["fc_weight_buffer_bytes"]
    hist_wo_dq = {k: v for k, v in sq["op_histogram"].items() if k != "DEQUANTIZE"}
    common = {"broadcast_to_0": gq["forbidden_counts"].get("BROADCAST_TO", 0) == 0, "int64_0": gq["int64_tensor_count"] == 0,
              "custom_0": sq["custom_op_count"] == 0, "rank_gt4_0": gq["rank_gt4_tensor_count"] == 0,
              f"fc_count_{nfc}_as_input": sq["fully_connected"] == nfc}
    if a.form == "v2_fp16fc":
        lo = ss["fc_weight_buffer_bytes"] // 2 + other_const
        hi = int(lo * 1.01) + 64 * 2 ** 20
        doc["expected_bytes_range"] = [lo, hi]
        checks = {**common,
                  "fc_weight_all_FLOAT16_via_DEQUANTIZE": sq["fc_weight_source_dtype"] == {"FLOAT16": nfc}
                  and sq["fc_weight_tensor_dtype_direct"] == {"None": nfc},
                  "dequantize_one_per_fc_weight_tensor_FLOAT16_to_FLOAT32":
                      sq["dequantize_in_out"] == {"FLOAT16->FLOAT32": ss["fc_weight_buffers"]}
                      and sq["fc_weight_buffers"] == ss["fc_weight_buffers"],
                  "fc_inputs_FLOAT32": sq["fc_input_dtype"] == {"FLOAT32": nfc},
                  "int8_int4_tensors_0": sq["tensor_dtype_histogram"].get("INT8", 0) == 0
                  and sq["tensor_dtype_histogram"].get("INT4", 0) == 0,
                  "op_histogram_input_plus_DEQUANTIZE": hist_wo_dq == ss["op_histogram"]
                      and sq["dequantize_count"] == ss["fc_weight_buffers"],
                  "size_as_expected": lo <= nbytes <= hi}
        shared = shared_weight_check(src, part)
        doc["shared_fc_weights"] = shared
        checks["shared_kernel_matrices_bit_equal_fp16_to_fp32"] = bool(shared) and all(
            x["bit_equal_after_fp16_to_fp32"] for x in shared)
        if a.recheck:
            checks["sha256_unchanged_since_attempt1"] = sha == doc["recheck_of"]["sha256_then"]
    else:
        checks = {**common,
                  "fc_weight_all_INT8_constant": sq["fc_weight_source_dtype"] == {"INT8": nfc}
                  and sq["fc_weight_tensor_dtype_direct"] == {"INT8": nfc},
                  "dequantize_0": sq["dequantize_count"] == 0,
                  "op_histogram_unchanged": sq["op_histogram"] == ss["op_histogram"]}
    ok = all(checks.values())
    if ok:
        os.replace(part, out)
    doc.update(status="ACCEPTED" if ok else "ACCEPTANCE_FAIL", checks=checks, failing=[k for k, v in checks.items() if not v],
               output={"file": str((out if ok else part).relative_to(K)), "bytes": nbytes, "sha256": sha},
               fc_weight_buffer_bytes_input=ss["fc_weight_buffer_bytes"], other_constant_bytes_input=other_const,
               scan=sq, scan_input={k: ss[k] for k in ("operator_count", "op_histogram", "fully_connected",
                                                       "fc_weight_source_dtype", "fc_weight_buffers", "fc_weight_buffer_bytes",
                                                       "constant_bytes_by_dtype", "constant_bytes")},
               forbidden_counts=gq["forbidden_counts"], int64_tensor_count=gq["int64_tensor_count"],
               rank_gt4_tensor_count=gq["rank_gt4_tensor_count"], tensor_rank_histogram=gq["tensor_rank_histogram"],
               signatures=gq["signatures"], memory_end=memory(), seconds_wall=round(time.perf_counter() - t_wall, 1),
               finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    res.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: doc[k] for k in ("status", "failing", "output", "load_seconds", "quantize_seconds", "seconds_wall")},
                     indent=1), flush=True)
    print(json.dumps({k: sq[k] for k in ("fc_weight_source_dtype", "dequantize_in_out", "fc_weight_quant_layout",
                                         "constant_bytes_by_dtype")}), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
