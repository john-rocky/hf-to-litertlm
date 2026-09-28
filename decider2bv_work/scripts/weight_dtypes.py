"""Flatbuffer-only weight-dtype census of a .tflite (no Interpreter, no allocation): which dtype the weights of
FULLY_CONNECTED / CONV_2D / DEPTHWISE_CONV_2D / EMBEDDING_LOOKUP / BATCH_MATMUL are STORED in, following a
DEQUANTIZE producer back to its constant input, counted over unique buffers (the decoder's seven signatures share
their weight buffers), plus the byte total of every constant buffer by dtype. This is the check that a file named
"fp16" really carries fp16 weights (qwen35vl_work/FINDINGS.md: the recipe_manager dynamic-config route wrote an
"fp16" file that was fp32).

    <any venv with ai-edge-litert> python -B scripts/weight_dtypes.py <file.tflite> [...]
"""
import json
import mmap
import os
import sys
from collections import Counter, defaultdict

from ai_edge_litert import schema_py_generated as schema

WEIGHT_OPS = {'FULLY_CONNECTED': 1, 'CONV_2D': 1, 'DEPTHWISE_CONV_2D': 1, 'EMBEDDING_LOOKUP': 1, 'BATCH_MATMUL': 1}


def enum_names(cls):
    return {v: k for k, v in vars(cls).items() if isinstance(v, int)}


def buffer_bytes(model, index):
    buf = model.Buffers(index)
    if buf is None:
        return 0
    if buf.Offset() > 1:                                  # data stored after the flatbuffer (files > 2 GB)
        return int(buf.Size())
    return int(buf.DataLength())


def census(path):
    dtype_names = enum_names(schema.TensorType)
    builtin_names = enum_names(schema.BuiltinOperator)
    with open(path, 'rb') as stream:
        mm = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        model = schema.Model.GetRootAsModel(mm, 0)
        codes = [builtin_names[max(model.OperatorCodes(i).BuiltinCode(), model.OperatorCodes(i).DeprecatedBuiltinCode())]
                 for i in range(model.OperatorCodesLength())]
        per_op = defaultdict(Counter)            # op -> stored dtype -> unique weight buffers
        per_op_bytes = defaultdict(Counter)      # op -> stored dtype -> bytes of those buffers
        via_dequantize = Counter()               # op -> number of op instances whose weight passes through DEQUANTIZE
        instances = Counter()
        seen = defaultdict(set)
        const_by_dtype, const_bytes_by_dtype = Counter(), Counter()
        const_seen = set()
        for s in range(model.SubgraphsLength()):
            sg = model.Subgraphs(s)
            producer = {}
            for j in range(sg.OperatorsLength()):
                op = sg.Operators(j)
                for k in range(op.OutputsLength()):
                    producer[op.Outputs(k)] = op
            for t in range(sg.TensorsLength()):
                tensor = sg.Tensors(t)
                b = tensor.Buffer()
                if b > 0 and b not in const_seen:
                    n = buffer_bytes(model, b)
                    if n:
                        const_seen.add(b)
                        const_by_dtype[dtype_names[tensor.Type()]] += 1
                        const_bytes_by_dtype[dtype_names[tensor.Type()]] += n
            for j in range(sg.OperatorsLength()):
                op = sg.Operators(j)
                name = codes[op.OpcodeIndex()]
                if name not in WEIGHT_OPS or op.InputsLength() <= WEIGHT_OPS[name]:
                    continue
                t = op.Inputs(WEIGHT_OPS[name])
                if t < 0:
                    continue
                instances[name] += 1
                src = producer.get(t)
                if src is not None and codes[src.OpcodeIndex()] == 'DEQUANTIZE':
                    via_dequantize[name] += 1
                    t = src.Inputs(0)
                tensor = sg.Tensors(t)
                b = tensor.Buffer()
                n = buffer_bytes(model, b) if b > 0 else 0
                dt = dtype_names[tensor.Type()] if n else f'{dtype_names[tensor.Type()]} (activation)'
                key = b if n else (s, t)
                if key in seen[name]:
                    continue
                seen[name].add(key)
                per_op[name][dt] += 1
                per_op_bytes[name][dt] += n
        mm.close()
    return dict(path=str(path), bytes=os.path.getsize(path),
                weights_by_op={k: dict(v) for k, v in sorted(per_op.items())},
                weight_bytes_by_op={k: dict(v) for k, v in sorted(per_op_bytes.items())},
                op_instances=dict(sorted(instances.items())), weight_via_dequantize=dict(sorted(via_dequantize.items())),
                constant_buffers_by_dtype=dict(sorted(const_by_dtype.items())),
                constant_bytes_by_dtype=dict(sorted(const_bytes_by_dtype.items())))


if __name__ == '__main__':
    for p in sys.argv[1:]:
        print(json.dumps(census(p), indent=1))
