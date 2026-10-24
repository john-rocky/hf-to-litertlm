"""Op table of a .tflite for the round-1 export probes (question 3), read with ai_edge_litert's interpreter.

Reports: the op histogram; the ops the launch flags (GATHER, GATHER_ND, CUSTOM, FLEX, STABLEHLO, BROADCAST_TO,
RELU_0_TO_1); every tensor of rank > 4; for SLICE / STRIDED_SLICE / PAD / PADV2 / CONCATENATION the count by input
rank and by the axis they act on (innermost or not; PAD's axes from its constant paddings, CONCATENATION's and the
slices' from the input / output shapes); and the reductions (MEAN / SUM) with the length of the reduced axis, the
place where fp16 accumulation can overflow.
"""
import os

import numpy as np

FLAGGED = ("GATHER", "GATHER_ND", "CUSTOM", "FLEX", "STABLEHLO", "BROADCAST_TO", "RELU_0_TO_1")
SHAPE_OPS = ("SLICE", "STRIDED_SLICE", "PAD", "PADV2", "CONCATENATION")


def _axes_changed(a, b):
    if len(a) != len(b):
        return None
    return [i for i, (x, y) in enumerate(zip(a, b)) if x != y]


def subgraph_hist(path):
    """Per-subgraph op histogram, flagged ops and tensors of rank > 4 read from the flatbuffer (the interpreter's op
    list covers subgraph 0 only; a multi-signature graph keeps one subgraph per signature)."""
    from ai_edge_litert import schema_py_generated as fb
    buf = open(path, "rb").read()
    model = fb.ModelT.InitFromObj(fb.Model.GetRootAsModel(buf, 0))
    names = {v: k for k, v in vars(fb.BuiltinOperator).items() if not k.startswith("_")}
    sig_of = {}
    for s in model.signatureDefs or []:
        key = s.signatureKey.decode() if isinstance(s.signatureKey, bytes) else s.signatureKey
        sig_of[s.subgraphIndex] = key
    out = []
    for i, sg in enumerate(model.subgraphs):
        hist = {}
        for op in sg.operators or []:
            oc = model.operatorCodes[op.opcodeIndex]
            code = max(oc.builtinCode, oc.deprecatedBuiltinCode)
            name = names.get(code, str(code))
            if name == "CUSTOM" and oc.customCode:
                name = "CUSTOM:" + (oc.customCode.decode() if isinstance(oc.customCode, bytes) else oc.customCode)
            hist[name] = hist.get(name, 0) + 1
        rank5 = sorted({len(t.shape) for t in sg.tensors or [] if t.shape is not None and len(t.shape) > 4})
        flagged = {n: sum(c for k, c in hist.items() if (k == n or (n in ("CUSTOM", "FLEX", "STABLEHLO")
                                                                     and n in k.upper())))
                   for n in FLAGGED}
        out.append({"subgraph": i, "signature": sig_of.get(i), "ops": sum(hist.values()), "hist": dict(sorted(hist.items())),
                    "flagged": flagged, "ranks_gt4": rank5,
                    "n_tensors_rank_gt4": sum(1 for t in sg.tensors or [] if t.shape is not None and len(t.shape) > 4)})
    return out


def op_table(path, max_list=20):
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
    # without the default XNNPACK delegate: with it, _get_ops_details() lists the delegated nodes, not the graph's ops
    it = Interpreter(model_path=str(path),
                     experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    it.allocate_tensors()
    ops = it._get_ops_details()
    tensors = {t["index"]: t for t in it.get_tensor_details()}
    hist = {}
    for o in ops:
        hist[o["op_name"]] = hist.get(o["op_name"], 0) + 1
    flagged = {}
    for name in FLAGGED:
        if name in ("CUSTOM", "FLEX", "STABLEHLO"):
            flagged[name] = sum(c for k, c in hist.items() if name in k.upper())
        else:
            flagged[name] = hist.get(name, 0)
    rank_gt4 = sorted({(t["name"][:70], tuple(int(v) for v in t["shape"])) for t in tensors.values()
                       if len(t["shape"]) > 4})
    shape_ops = {}
    for o in ops:
        name = o["op_name"]
        if name not in SHAPE_OPS:
            continue
        ins = [tensors[i] for i in o["inputs"] if i in tensors]
        outs = [tensors[i] for i in o["outputs"] if i in tensors]
        if not ins or not outs:
            continue
        in_shape = tuple(int(v) for v in ins[0]["shape"])
        out_shape = tuple(int(v) for v in outs[0]["shape"])
        rank = len(in_shape)
        axes = None
        if name in ("PAD", "PADV2") and len(ins) > 1:
            try:
                pads = np.asarray(it.get_tensor(ins[1]["index"]))
                axes = [i for i in range(pads.shape[0]) if pads[i].any()]
            except ValueError:
                axes = None
        elif name == "CONCATENATION":
            axes = _axes_changed(in_shape, out_shape)
        else:
            axes = _axes_changed(in_shape, out_shape)
        inner = None if axes is None else (axes == [rank - 1] if axes else True)
        rec = shape_ops.setdefault(name, {"count": 0, "by_rank": {}, "innermost": 0, "not_innermost": 0,
                                          "examples": []})
        rec["count"] += 1
        rec["by_rank"][str(rank)] = rec["by_rank"].get(str(rank), 0) + 1
        if inner is True:
            rec["innermost"] += 1
        elif inner is False:
            rec["not_innermost"] += 1
        if len(rec["examples"]) < 6:
            rec["examples"].append({"in": in_shape, "out": out_shape, "axes": axes})
    reductions = []
    for o in ops:
        if o["op_name"] in ("MEAN", "SUM", "REDUCE_MAX", "REDUCE_PROD"):
            ins = [tensors[i] for i in o["inputs"] if i in tensors]
            outs = [tensors[i] for i in o["outputs"] if i in tensors]
            if ins and outs:
                a = tuple(int(v) for v in ins[0]["shape"])
                b = tuple(int(v) for v in outs[0]["shape"])
                axes = None
                if len(ins) > 1:
                    try:
                        axes = [int(x) % len(a) for x in np.asarray(it.get_tensor(ins[1]["index"])).ravel()]
                    except ValueError:
                        axes = None
                if axes is None:   # fall back to the shapes (keepdim reductions only)
                    axes = [i for i in range(len(a)) if len(a) == len(b) and a[i] != b[i]]
                reductions.append({"op": o["op_name"], "in": a, "out": b, "axes": axes,
                                   "reduced_elems": int(np.prod([a[i] for i in axes])) if axes else 1})
    reductions.sort(key=lambda r: -r["reduced_elems"])
    float_rank2 = sum(1 for t in tensors.values() if len(t["shape"]) == 2 and t["dtype"] == np.float32)
    try:
        per_subgraph = subgraph_hist(path)
    except Exception as e:  # noqa: BLE001  (the flatbuffer reader is a cross-check, not the table)
        per_subgraph = f"unread: {type(e).__name__}: {e}"
    return {"file": os.path.basename(str(path)), "bytes": os.path.getsize(path), "ops": len(ops),
            "per_subgraph": per_subgraph,
            "hist": dict(sorted(hist.items())), "flagged": flagged, "n_rank_gt4": len(rank_gt4),
            "rank_gt4": [list(x) for x in rank_gt4[:max_list]], "shape_ops": shape_ops,
            "n_reductions": len(reductions), "largest_reductions": reductions[:8],
            "float32_rank2_tensors": float_rank2,
            "inputs": [(d["name"], tuple(int(v) for v in d["shape"]), str(d["dtype"].__name__))
                       for d in it.get_input_details()],
            "outputs": [(d["name"], tuple(int(v) for v in d["shape"]), str(d["dtype"].__name__))
                        for d in it.get_output_details()]}
