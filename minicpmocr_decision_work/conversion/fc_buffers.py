"""FC weight storage across the subgraphs of a multi-signature file."""
from __future__ import annotations

import collections
import mmap
import os


def buffers(path):
    """FC weight storage across subgraphs: per subgraph the buffer each FULLY_CONNECTED reads (through a DEQUANTIZE
    when the weight is float16), the distinct ones over the file, their bytes; and every buffer's bytes."""
    from ai_edge_litert import schema_py_generated as schema

    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    model = schema.Model.GetRootAs(mm, 0)
    op_names = {v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int)}
    codes = [op_names.get(max(model.OperatorCodes(i).BuiltinCode(), model.OperatorCodes(i).DeprecatedBuiltinCode()))
             for i in range(model.OperatorCodesLength())]

    def nbytes(bi):
        b = model.Buffers(bi)
        if b is None:
            return 0
        return b.DataLength() if b.DataLength() > 0 else (b.Size() if b.Offset() > 1 else 0)

    per_sg, users = [], collections.defaultdict(set)
    for si in range(model.SubgraphsLength()):
        sg = model.Subgraphs(si)
        producer = {}
        for oi in range(sg.OperatorsLength()):
            op = sg.Operators(oi)
            for k in range(op.OutputsLength()):
                producer[op.Outputs(k)] = op
        fc_bufs = []
        for oi in range(sg.OperatorsLength()):
            op = sg.Operators(oi)
            if codes[op.OpcodeIndex()] != "FULLY_CONNECTED" or op.InputsLength() < 2:
                continue
            ti = op.Inputs(1)
            t = sg.Tensors(ti)
            bi = t.Buffer()
            if nbytes(bi) == 0 and ti in producer and codes[producer[ti].OpcodeIndex()] == "DEQUANTIZE":
                bi = sg.Tensors(producer[ti].Inputs(0)).Buffer()
            if nbytes(bi) == 0:
                continue                      # a runtime operand (the mask's sees_q . sees_k)
            fc_bufs.append(bi)
            users[bi].add(si)
        name = sg.Name().decode() if sg.Name() else str(si)
        per_sg.append({"subgraph": si, "name": name, "fc_weight_buffers": len(fc_bufs),
                       "fc_weight_bytes": sum(nbytes(b) for b in fc_bufs)})
    distinct = sorted(users)
    all_bytes = sum(nbytes(i) for i in range(model.BuffersLength()))
    return {"file_bytes": os.path.getsize(path), "buffers": model.BuffersLength(), "all_buffer_bytes": all_bytes,
            "subgraphs": model.SubgraphsLength(),
            "signatures": [model.SignatureDefs(i).SignatureKey().decode() for i in range(model.SignatureDefsLength())],
            "per_subgraph": per_sg, "fc_weight_buffers_distinct": len(distinct),
            "fc_weight_bytes_distinct": sum(nbytes(b) for b in distinct),
            "fc_weight_buffers_read_by_every_subgraph": sum(1 for b in distinct if len(users[b]) == model.SubgraphsLength()),
            "fc_weight_buffers_read_by_one_subgraph": sum(1 for b in distinct if len(users[b]) == 1)}
