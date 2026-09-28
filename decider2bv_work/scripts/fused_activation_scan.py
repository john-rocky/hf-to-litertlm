"""Fused-activation census of a .tflite (flatbuffer only): for every op whose builtin options carry a
fused_activation_function (ADD, SUB, MUL, DIV, FULLY_CONNECTED, CONV_2D, DEPTHWISE_CONV_2D, CONCATENATION), count
op type x activation per signature. Round 4 reads it to see where the converter put the relu of relu(x) - relu(x - 1)
(standalone RELU vs a RELU fused into the SUB that produces x - 1).

    <venv with ai-edge-litert> python -B scripts/fused_activation_scan.py <file.tflite> [--out results/x.json]
"""
import argparse
import json
import mmap
from collections import Counter, defaultdict

from ai_edge_litert import schema_py_generated as schema

OPTS = {'ADD': schema.AddOptions, 'SUB': schema.SubOptions, 'MUL': schema.MulOptions, 'DIV': schema.DivOptions,
        'FULLY_CONNECTED': schema.FullyConnectedOptions, 'CONV_2D': schema.Conv2DOptions,
        'DEPTHWISE_CONV_2D': schema.DepthwiseConv2DOptions, 'CONCATENATION': schema.ConcatenationOptions}


def enum_names(cls):
    return {v: k for k, v in vars(cls).items() if isinstance(v, int)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    builtin = enum_names(schema.BuiltinOperator)
    act = enum_names(schema.ActivationFunctionType)
    res = defaultdict(Counter)
    with open(args.path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        m = schema.Model.GetRootAsModel(mm, 0)
        codes = [builtin[max(m.OperatorCodes(i).BuiltinCode(), m.OperatorCodes(i).DeprecatedBuiltinCode())] for i in range(m.OperatorCodesLength())]
        sig_of = {m.SignatureDefs(i).SubgraphIndex(): m.SignatureDefs(i).SignatureKey().decode() for i in range(m.SignatureDefsLength())}
        for si in range(m.SubgraphsLength()):
            sg = m.Subgraphs(si)
            for j in range(sg.OperatorsLength()):
                op = sg.Operators(j)
                name = codes[op.OpcodeIndex()]
                if name not in OPTS or op.BuiltinOptions() is None:
                    continue
                o = OPTS[name]()
                t = op.BuiltinOptions()
                o.Init(t.Bytes, t.Pos)
                a = act.get(o.FusedActivationFunction(), '?')
                if a != 'NONE':
                    res[sig_of.get(si, str(si))][f'{name}+{a}'] += 1
        mm.close()
    out = dict(path=args.path, fused_activations_per_signature={k: dict(v) for k, v in sorted(res.items())})
    print(json.dumps(out, indent=1))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
