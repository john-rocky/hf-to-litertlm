"""The storage forms of a float32 graph with ai-edge-quantizer (no calibration), and the flatbuffer scans of the
acceptance. The memory-mapped
file is read through the schema only, so a > 2 GiB file is fine.

  fp16fc    weight-only 16-bit FLOAT_CASTING on FULLY_CONNECTED (channelwise).
  dyni8fc   dynamic int8 on FULLY_CONNECTED.
  wi8fc     weight-only int8 on FULLY_CONNECTED.
  i4block   weight-only int4 (BLOCKWISE 32 by default) on FULLY_CONNECTED."""
from __future__ import annotations

import collections
import json
import mmap
from pathlib import Path


def recipe(form: str):
    from ai_edge_quantizer import qtyping, recipe_manager
    from ai_edge_quantizer.algorithm_manager import AlgorithmName

    G = qtyping.QuantGranularity
    OP = qtyping.TFLOperationName
    rm = recipe_manager.RecipeManager()
    if form == "fp16fc":
        rm.add_weight_only_config(regex=".*", operation_name=OP.FULLY_CONNECTED, num_bits=16, granularity=G.CHANNELWISE,
                                  algorithm_key=AlgorithmName.FLOAT_CASTING)
    elif form == "dyni8fc":
        rm.add_dynamic_config(regex=".*", operation_name=OP.FULLY_CONNECTED, num_bits=8)
    elif form == "wi8fc":
        rm.add_weight_only_config(regex=".*", operation_name=OP.FULLY_CONNECTED, num_bits=8, granularity=G.CHANNELWISE)
    elif form == "i4dyn32":
        rm.add_dynamic_config(regex=".*", operation_name=OP.FULLY_CONNECTED, num_bits=4, granularity=G.BLOCKWISE_32)
    elif form == "i4wo32":
        rm.add_weight_only_config(regex=".*", operation_name=OP.FULLY_CONNECTED, num_bits=4, granularity=G.BLOCKWISE_32)
    else:
        raise ValueError(form)
    return rm.get_quantization_recipe()


def quantize(src, dst, form: str) -> dict:
    from ai_edge_quantizer import quantizer

    assert not Path(dst).exists(), f"refusing to overwrite {dst}"
    r = recipe(form)
    qt = quantizer.Quantizer(str(src), r)
    if qt.need_calibration:
        raise SystemExit(f"STOP: recipe {form} needs calibration")
    part = str(dst) + ".part"
    qt.quantize(serialize_to_path=part)
    Path(part).replace(dst)
    return json.loads(json.dumps(r, default=str))


def fp16fc(src, dst) -> dict:
    return quantize(src, dst, "fp16fc")


def fc_scan(path):
    """Weight dtypes of FC through a DEQUANTIZE producer, DEQUANTIZE in -> out, constants by dtype, distinct FC weight buffers."""
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
    hist, fc_weight, fc_weight_direct = (collections.Counter() for _ in range(3))
    fc_in, deq_in_out, tensor_dtypes, const_bytes_by_dtype = (collections.Counter() for _ in range(4))
    fc_weight_buffers = {}
    for gi in range(model.SubgraphsLength()):
        g = model.Subgraphs(gi)
        tensors = [g.Tensors(t) for t in range(g.TensorsLength())]
        ttype = lambda t: type_names.get(tensors[t].Type(), str(tensors[t].Type()))   # noqa: E731
        ops = [g.Operators(o) for o in range(g.OperatorsLength())]
        producer = {}
        for oi, op in enumerate(ops):
            for t in (op.OutputsAsNumpy().tolist() if op.OutputsLength() else []):
                producer[int(t)] = oi
        graph_inputs = set(g.InputsAsNumpy().tolist()) if g.InputsLength() else set()
        is_const = lambda t: t >= 0 and t not in producer and t not in graph_inputs   # noqa: E731

        def nbytes(t):
            b = model.Buffers(tensors[t].Buffer())
            return b.DataLength() if b.DataLength() > 0 else (b.Size() if b.Offset() > 1 else 0)

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
                src, wsrc = direct, w
                if not is_const(w) and w in producer and codes[ops[producer[w]].OpcodeIndex()] == "DEQUANTIZE":
                    wsrc = int(ops[producer[w]].InputsAsNumpy()[0])
                    src = ttype(wsrc)
                fc_weight[src] += 1
                fc_weight_direct[str(direct)] += 1
                fc_in[ttype(ins[0])] += 1
                if is_const(wsrc):
                    fc_weight_buffers[(gi, tensors[wsrc].Buffer())] = nbytes(wsrc)
    mm.close()
    f.close()
    return {"operator_count": sum(hist.values()), "op_histogram": dict(sorted(hist.items())),
            "fully_connected": sum(fc_weight.values()), "fc_weight_source_dtype": dict(fc_weight),
            "fc_weight_tensor_dtype_direct": dict(fc_weight_direct), "fc_input_dtype": dict(fc_in),
            "fc_weight_buffers": len(fc_weight_buffers), "fc_weight_buffer_bytes": int(sum(fc_weight_buffers.values())),
            "dequantize_count": hist.get("DEQUANTIZE", 0), "dequantize_in_out": dict(deq_in_out),
            "custom_op_count": sum(v for k, v in hist.items() if k.startswith("CUSTOM")),
            "tensor_dtype_histogram": dict(sorted(tensor_dtypes.items())),
            "constant_bytes_by_dtype": dict(sorted(const_bytes_by_dtype.items())),
            "constant_bytes": int(sum(const_bytes_by_dtype.values()))}


def general_scan(path) -> dict:
    """Signatures, INT64 and rank > 4 tensors, all-constant-input ops, the largest activations, and the two table checks."""
    from ai_edge_litert import schema_py_generated as schema
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    model = schema.Model.GetRootAs(mm, 0)
    op_names = {v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int)}
    type_names = {v: k for k, v in vars(schema.TensorType).items() if isinstance(v, int)}
    size = {"FLOAT32": 4, "FLOAT16": 2, "INT32": 4, "INT64": 8, "INT8": 1, "UINT8": 1, "BOOL": 1, "INT16": 2, "INT4": 1}
    int64 = rank_gt4 = all_const = vocab = head = 0
    all_const_names = collections.Counter()
    hist = collections.Counter()
    acts = []
    fc_out_rank, tensors_by_rank = collections.Counter(), collections.Counter()
    for gi in range(model.SubgraphsLength()):
        g = model.Subgraphs(gi)
        produced = set()
        ops = [g.Operators(o) for o in range(g.OperatorsLength())]
        for op in ops:
            if op.OutputsLength():
                produced.update(int(x) for x in op.OutputsAsNumpy())
        graph_inputs = {int(x) for x in g.InputsAsNumpy()} if g.InputsLength() else set()
        is_const = lambda t: t >= 0 and t not in produced and t not in graph_inputs   # noqa: E731
        for ti in range(g.TensorsLength()):
            t = g.Tensors(ti)
            shape = t.ShapeAsNumpy().tolist() if t.ShapeLength() else []
            dt = type_names.get(t.Type())
            int64 += int(dt == "INT64")
            rank_gt4 += int(len(shape) > 4)
            tensors_by_rank[len(shape)] += 1
            vocab += int(248320 in shape)
            if is_const(ti):
                head += int(shape == [2, 2048])
            else:
                n = size.get(dt, 4)
                for d in shape:
                    n *= max(int(d), 1)
                acts.append((n, shape, dt))
        for op in ops:
            c = model.OperatorCodes(op.OpcodeIndex())
            name = op_names.get(max(c.BuiltinCode(), c.DeprecatedBuiltinCode()), "?")
            hist[name] += 1
            ins = [int(x) for x in op.InputsAsNumpy()] if op.InputsLength() else []
            real = [t for t in ins if t >= 0]
            if real and all(is_const(t) for t in real):
                all_const += 1
                all_const_names[name] += 1
            if name == "FULLY_CONNECTED":
                fc_out_rank[g.Tensors(op.Outputs(0)).ShapeLength()] += 1
    sigs = []
    for i in range(model.SignatureDefsLength()):
        sd = model.SignatureDefs(i)
        sg = model.Subgraphs(sd.SubgraphIndex())

        def io(n, get):
            r = []
            for j in range(n):
                tm = get(j)
                t = sg.Tensors(tm.TensorIndex())
                r.append({"name": tm.Name().decode(), "shape": t.ShapeAsNumpy().tolist() if t.ShapeLength() else [], "dtype": type_names.get(t.Type())})
            return r
        sigs.append({"key": sd.SignatureKey().decode(), "inputs": io(sd.InputsLength(), sd.Inputs), "outputs": io(sd.OutputsLength(), sd.Outputs)})
    acts.sort(key=lambda x: -x[0])
    mm.close()
    f.close()
    return {"signatures": sigs, "op_histogram": dict(hist), "int64_tensor_count": int64, "rank_gt4_tensor_count": rank_gt4, "all_constant_input_ops": all_const,
            "all_constant_input_op_names": dict(all_const_names), "tensors_with_vocab_rows": vocab, "constants_shape_2x2048": head,
            "fc_output_rank": {str(k): v for k, v in fc_out_rank.items()}, "tensor_rank_histogram": {str(k): v for k, v in sorted(tensors_by_rank.items())},
            "largest_activations": [{"bytes": b, "shape": s, "dtype": d} for b, s, d in acts[:6]]}


def signature_ok(sigs: list, contract: dict) -> bool:
    if len(sigs) != 1:
        return False
    key = lambda r: sorted((x["name"], list(x["shape"]), x["dtype"]) for x in r)   # noqa: E731
    return key(sigs[0]["inputs"]) == key(contract["inputs"]) and key(sigs[0]["outputs"]) == key(contract["outputs"])
