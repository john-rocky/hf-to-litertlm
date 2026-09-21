"""Read all subgraphs and physical buffers without allocating an inference engine."""
import argparse
import json
import mmap
import math
from pathlib import Path

from ai_edge_litert import schema_py_generated as tflite
from bundle_header import read_header

p = argparse.ArgumentParser()
p.add_argument("bundle")
p.add_argument("--variant", required=True, choices=["int8", "mixed_int4"])
p.add_argument("--out", required=True)
a = p.parse_args()
sections, _ = read_header(a.bundle)
section = next(x for x in sections if "tflite" in x["type"].lower())
types = {v: k for k, v in vars(tflite.TensorType).items() if isinstance(v, int)}
vocab, linear, duplicated, conv_weights = [], [], [], []
with open(a.bundle, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
    model = tflite.Model.GetRootAsModel(mm, section["begin"])
    op_types = {i: model.OperatorCodes(i).BuiltinCode() for i in range(model.OperatorCodesLength())}
    for si in range(model.SubgraphsLength()):
        sg = model.Subgraphs(si)
        for ti in range(sg.TensorsLength()):
            tensor = sg.Tensors(ti)
            name = tensor.Name().decode() if tensor.Name() else ""
            shape = [tensor.Shape(k) for k in range(tensor.ShapeLength())]
            if "_duplicated_" in name:
                duplicated.append({"subgraph": si, "tensor": ti, "name": name})
            if shape not in ([248320, 2560], [2560, 9216]):
                continue
            q = tensor.Quantization()
            row = {"subgraph": si, "tensor": ti, "name": name, "shape": shape, "dtype": types[tensor.Type()], "buffer": tensor.Buffer(),
                   "scale_count": q.ScaleLength() if q else 0, "quantized_dimension": q.QuantizedDimension() if q else None}
            if q and q.DetailsType() == tflite.QuantizationDetails.BlockwiseQuantization:
                table = q.Details()
                details = tflite.BlockwiseQuantization()
                details.Init(table.Bytes, table.Pos)
                scales = sg.Tensors(details.Scales())
                scale_shape = [scales.Shape(k) for k in range(scales.ShapeLength())]
                row.update(scale_count=math.prod(scale_shape), block_size=details.BlockSize(),
                           scale_tensor=details.Scales(), scale_shape=scale_shape,
                           scale_dtype=types[scales.Type()], scale_buffer=scales.Buffer())
            (vocab if shape == [248320, 2560] else linear).append(row)
        for oi in range(sg.OperatorsLength()):
            op = sg.Operators(oi)
            if op_types[op.OpcodeIndex()] in (tflite.BuiltinOperator.CONV_2D, tflite.BuiltinOperator.DEPTHWISE_CONV_2D):
                tensor = sg.Tensors(op.Inputs(1))
                conv_weights.append({"subgraph": si, "operator": oi, "weight_buffer": tensor.Buffer(), "dtype": types[tensor.Type()]})
    num_subgraphs = model.SubgraphsLength()
physical = sorted(set(x["buffer"] for x in vocab))
expected_type, expected_scales = ("INT8", 2560) if a.variant == "int8" else ("INT4", 737280)
checks = {"one_physical_vocab_table": len(physical) == 1,
          "vocab_int8_channelwise": bool(vocab) and all(x["dtype"] == "INT8" and x["scale_count"] == 248320 for x in vocab),
          "linear_recipe_matches": bool(linear) and all(x["dtype"] == expected_type and x["scale_count"] == expected_scales for x in linear),
          "no_duplicated_tensors": not duplicated,
          "convolution_weights_float32": bool(conv_weights) and all(x["dtype"] == "FLOAT32" for x in conv_weights)}
if a.variant == "mixed_int4":
    checks["block_size_32"] = bool(linear) and all(x.get("block_size") == 32 for x in linear)
data = {"bundle": a.bundle, "variant": a.variant, "size_bytes": Path(a.bundle).stat().st_size,
        "subgraphs": num_subgraphs, "vocab_physical_buffer_ids": physical, "vocab_references": vocab,
        "linear_example": linear[0] if linear else None, "linear_reference_count": len(linear), "duplicated_tensors": duplicated,
        "convolution_weights": conv_weights, "checks": checks, "verdict": "PASS" if all(checks.values()) else "FINDING"}
Path(a.out).write_text(json.dumps(data, indent=2) + "\n")
print(json.dumps({k: data[k] for k in ["variant", "size_bytes", "subgraphs", "vocab_physical_buffer_ids", "linear_example", "checks", "verdict"]}, indent=2))
