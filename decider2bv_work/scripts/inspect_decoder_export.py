"""Inspect the round-2 decoder export (flatbuffer only): which tflite is the embedder and which the prefill/decode
model, signature list, input names, number of state inputs, output names, op histogram per signature, forbidden ops
(GATHER*, FLEX, CUSTOM, STABLEHLO_*, BROADCAST_TO), max tensor rank, and the ops the derived rotary adds.

The op delta is measured per signature against the same architecture exported earlier with the qwen35 patch's plain
1-D rotary at the same six-signature ladder (qwen35vl 2B decoder, `out/qwen35vl-decoder-l6/model.tflite`, read only;
its own export stack is recorded next to it in export_result.json). A delta confined to the rotary's element-wise op
types is the expected result; anything else is reported as found.

    out/venv-readout/bin/python -B scripts/inspect_decoder_export.py
    out/venv-readout/bin/python -B scripts/inspect_decoder_export.py --export-dir out/decoder_g256_r4_fp32 \
        --result results/decoder_export_r4.json --compare-to results/decoder_export_r2.json          (round 4)

--compare-to adds the per-signature op delta against an earlier inspection of this decoder (round 4: the RELU-free
step against round 2's clamp step), the embedder's byte identity, and the state/IO signature identity.
"""
import argparse
import os
from collections import OrderedDict

from common import ROOT, read_json, write_json
from tflite_scan import scan

OUT = ROOT / 'out/decoder_g256_fp32'
BASELINE = ROOT.parent / 'out/qwen35vl-decoder-l6/model.tflite'
CORE = ('embeddings', 'input_pos', 'mask')


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--export-dir', default=str(OUT))
    ap.add_argument('--result', default='results/decoder_export_r2.json')
    ap.add_argument('--compare-to', default='')
    args = ap.parse_args()
    OUT = (ROOT / args.export_dir).resolve()
    found = {}
    for f in sorted(os.listdir(OUT)):
        if f.endswith('.tflite'):
            s = scan(OUT / f)
            is_dec = any('kv_cache' in n for sig in s['signatures'].values() for n in sig['inputs'])
            found['prefill_decode' if is_dec else 'embedder'] = (f, s)
    assert set(found) == {'prefill_decode', 'embedder'}, list(found)
    res = OrderedDict()
    for role, (f, s) in found.items():
        sigs = s['signatures']
        entry = OrderedDict(path=str((OUT / f).relative_to(ROOT)), bytes=s['bytes'], sha256=s['sha256'],
                            signatures=sorted(sigs), operator_count=s['operator_count'], max_tensor_rank=s['max_tensor_rank'],
                            forbidden=s['forbidden'], forbidden_total=s['forbidden_total'], op_histogram=s['op_histogram'])
        if role == 'prefill_decode':
            per = OrderedDict()
            for k, sig in sorted(sigs.items()):
                states = [n for n in sig['inputs'] if n not in CORE]
                per[k] = OrderedDict(n_inputs=len(sig['inputs']), n_state_inputs=len(states),
                                     core_inputs={n: sig['inputs'][n] for n in CORE if n in sig['inputs']},
                                     n_outputs=len(sig['outputs']), has_logits='logits' in sig['outputs'],
                                     logits=sig['outputs'].get('logits'),
                                     state_dtypes=sorted(set(sig['inputs'][n]['dtype'] for n in states)),
                                     op_histogram=next(sg['op_histogram'] for sg in s['subgraphs'] if sg['index'] == sig['subgraph_index']))
            entry['per_signature'] = per
            dec = sigs['decode']
            entry['state_input_names'] = [n for n in dec['inputs'] if n not in CORE]
            entry['state_shapes'] = OrderedDict((n, dec['inputs'][n]['shape']) for n in entry['state_input_names'])
        else:
            entry['io'] = {k: dict(inputs=v['inputs'], outputs=v['outputs']) for k, v in sigs.items()}
        res[role] = entry

    if BASELINE.exists():
        b = scan(BASELINE, with_sha=False)
        base_per = {k: next(sg['op_histogram'] for sg in b['subgraphs'] if sg['index'] == v['subgraph_index'])
                    for k, v in b['signatures'].items()}
        delta = OrderedDict()
        for k, h in res['prefill_decode']['per_signature'].items():
            if k not in base_per:
                delta[k] = 'signature absent in the baseline'
                continue
            hb = base_per[k]
            d = {op: h['op_histogram'].get(op, 0) - hb.get(op, 0) for op in sorted(set(h['op_histogram']) | set(hb))}
            delta[k] = {op: n for op, n in d.items() if n}
        res['rotary_op_delta_vs_plain_1d'] = OrderedDict(
            baseline=str(BASELINE), baseline_bytes=b['bytes'], baseline_signatures=sorted(b['signatures']),
            baseline_forbidden=b['forbidden'], baseline_state_inputs=len([n for n in b['signatures']['decode']['inputs'] if n not in CORE]),
            delta_per_signature=delta,
            added_op_types=sorted({op for d in delta.values() if isinstance(d, dict) for op, n in d.items() if n > 0}),
            removed_op_types=sorted({op for d in delta.values() if isinstance(d, dict) for op, n in d.items() if n < 0}))
    else:
        res['rotary_op_delta_vs_plain_1d'] = dict(baseline=str(BASELINE), note='baseline not readable (HDD detached?)')
    drv = read_json(str((OUT / 'export_driver.json').relative_to(ROOT)))   # out/ is git-ignored: carry the record here
    drv.pop('traceback', None)
    res['export_driver'] = drv
    if args.compare_to:
        prev = read_json(args.compare_to)
        pp, cp = prev['prefill_decode'], res['prefill_decode']
        delta = OrderedDict()
        for k, h in cp['per_signature'].items():
            hb = pp['per_signature'][k]['op_histogram']
            d = {op: h['op_histogram'].get(op, 0) - hb.get(op, 0) for op in sorted(set(h['op_histogram']) | set(hb))}
            delta[k] = {op: n for op, n in d.items() if n}
        res['compare_to'] = OrderedDict(
            result=args.compare_to, previous_step_form=prev['export_driver'].get('step_form', 'clamp (round 2 default)'),
            step_form=drv.get('step_form'), delta_per_signature=delta,
            added_op_types=sorted({op for d in delta.values() for op, n in d.items() if n > 0}),
            removed_op_types=sorted({op for d in delta.values() for op, n in d.items() if n < 0}),
            operator_count=dict(previous=pp['operator_count'], current=cp['operator_count']),
            relu_0_to_1=dict(previous=pp['op_histogram'].get('RELU_0_TO_1', 0), current=cp['op_histogram'].get('RELU_0_TO_1', 0)),
            forbidden=dict(previous=pp['forbidden'], current=cp['forbidden']),
            signatures_equal=pp['signatures'] == cp['signatures'],
            state_shapes_equal=pp['state_shapes'] == cp['state_shapes'],
            core_inputs_equal=all(pp['per_signature'][k]['core_inputs'] == cp['per_signature'][k]['core_inputs'] for k in cp['per_signature']),
            logits_equal=pp['per_signature']['decode']['logits'] == cp['per_signature']['decode']['logits'],
            embedder_sha256_equal=prev['embedder']['sha256'] == res['embedder']['sha256'],
            embedder_sha256=dict(previous=prev['embedder']['sha256'], current=res['embedder']['sha256']),
            decoder_bytes=dict(previous=pp['bytes'], current=cp['bytes']),
            export_args_equal_except_output_dir={k: v == drv['export_args'].get(k) for k, v in prev['export_driver']['export_args'].items() if k != 'output_dir'},
            clone_equal=prev['export_driver']['clone'] == drv['clone'], patch_sha256_equal=prev['export_driver']['patch_sha256'] == drv['patch_sha256'],
            versions_equal=prev['export_driver']['versions'] == drv['versions'])
        print('vs', args.compare_to, {k: res['compare_to'][k] for k in ('added_op_types', 'removed_op_types', 'relu_0_to_1', 'operator_count',
                                                                          'signatures_equal', 'state_shapes_equal', 'embedder_sha256_equal',
                                                                          'clone_equal', 'versions_equal')})
        print('export args equal', res['compare_to']['export_args_equal_except_output_dir'])
        print('delta per signature', res['compare_to']['delta_per_signature'])
    write_json(args.result, res)
    pd = res['prefill_decode']
    print('signatures', pd['signatures'])
    print('states', {k: v['n_state_inputs'] for k, v in pd['per_signature'].items()}, 'core', pd['per_signature']['decode']['core_inputs'])
    print('forbidden', pd['forbidden'], 'embedder forbidden', res['embedder']['forbidden'], 'max rank', pd['max_tensor_rank'])
    print('op delta', res['rotary_op_delta_vs_plain_1d'].get('delta_per_signature'))


if __name__ == '__main__':
    main()
