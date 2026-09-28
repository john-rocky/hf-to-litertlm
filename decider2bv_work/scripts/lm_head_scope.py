"""Find the vocabulary FC (lm_head) of a decoder .tflite and the scope string ai-edge-quantizer 0.8.0 matches recipe
regexes against (tfl_flatbuffer_utils.get_op_scope: the op's output tensor names joined by ';' plus a trailing ';',
matched with re.search), flatbuffer only. For a candidate regex, list every FULLY_CONNECTED op (all subgraphs) whose
scope it matches, so a regex is written against this export's actual names, not a sibling's.

    <venv with ai-edge-litert> python -B scripts/lm_head_scope.py <decoder.tflite> [--regex R] [--vocab 248320]
"""
import argparse
import json
import mmap
import re
from collections import Counter

from ai_edge_litert import schema_py_generated as schema


def enum_names(cls):
    return {v: k for k, v in vars(cls).items() if isinstance(v, int)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('--regex', default=r'^decode_logits_output;$')
    ap.add_argument('--vocab', type=int, default=248320)
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    builtin = enum_names(schema.BuiltinOperator)
    dtypes = enum_names(schema.TensorType)
    with open(args.path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        m = schema.Model.GetRootAsModel(mm, 0)
        codes = [builtin[max(m.OperatorCodes(i).BuiltinCode(), m.OperatorCodes(i).DeprecatedBuiltinCode())] for i in range(m.OperatorCodesLength())]
        sig_of = {}
        sig_outputs = {}
        for i in range(m.SignatureDefsLength()):
            s = m.SignatureDefs(i)
            sig_of[s.SubgraphIndex()] = s.SignatureKey().decode()
            sig_outputs[s.SubgraphIndex()] = {s.Outputs(j).TensorIndex(): s.Outputs(j).Name().decode() for j in range(s.OutputsLength())}
        vocab_fcs, matched, n_fc = [], [], Counter()
        rx = re.compile(args.regex)
        for si in range(m.SubgraphsLength()):
            sg = m.Subgraphs(si)
            name = lambda t: sg.Tensors(t).Name().decode() if sg.Tensors(t).Name() else ''
            producer = {}
            for j in range(sg.OperatorsLength()):
                op = sg.Operators(j)
                for k in range(op.OutputsLength()):
                    producer[op.Outputs(k)] = j
            for j in range(sg.OperatorsLength()):
                op = sg.Operators(j)
                if codes[op.OpcodeIndex()] != 'FULLY_CONNECTED':
                    continue
                n_fc[sig_of.get(si, si)] += 1
                outs = [op.Outputs(k) for k in range(op.OutputsLength()) if op.Outputs(k) != -1]
                scope = ';'.join(n for n in (name(t) for t in outs) if n)
                scope = scope + ';' if scope else scope
                w = op.Inputs(1)
                wt = sg.Tensors(w)
                wshape = [wt.Shape(k) for k in range(wt.ShapeLength())]
                wsrc = w
                if w in producer and codes[sg.Operators(producer[w]).OpcodeIndex()] == 'DEQUANTIZE':
                    wsrc = sg.Operators(producer[w]).Inputs(0)
                rec = dict(signature=sig_of.get(si, f'subgraph{si}'), subgraph=si, op_index=j, scope=scope,
                           output_tensors=outs, output_is_signature_output={t: sig_outputs.get(si, {}).get(t) for t in outs},
                           weight_tensor=name(w), weight_shape=wshape, weight_dtype=dtypes[sg.Tensors(wsrc).Type()],
                           weight_buffer=sg.Tensors(wsrc).Buffer())
                if args.vocab in wshape:
                    vocab_fcs.append(rec)
                if rx.search(scope):
                    matched.append(rec)
        mm.close()
    res = dict(path=args.path, regex=args.regex, fc_ops_per_signature=dict(n_fc), vocab_fcs=vocab_fcs, regex_matches=matched,
               n_regex_matches=len(matched), regex_matches_are_exactly_the_vocab_fcs=sorted((r['subgraph'], r['op_index']) for r in matched)
               == sorted((r['subgraph'], r['op_index']) for r in vocab_fcs))
    print(json.dumps(res, indent=1, default=str))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(res, f, indent=1, default=str)


if __name__ == '__main__':
    main()
