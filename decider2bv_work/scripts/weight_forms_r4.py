"""Round 4 aggregate -> results/weight_forms_r4.json (every number the report prints comes from here).

Sources (all written in round 4 unless named): results/mrope_derived_test_r4.json, results/decoder_export_r4.json,
results/lm_head_scope_r4.json, results/graph_parity_r4_fp32.json (+ round 2's results/graph_parity_r2.json for the
slot-by-slot comparison), out/weights_r4/<v>/{decoder,embedder}.quant.json, results/readout_r4_<v>.json (+ round 3's
results/fp16_readout_r3.json for the fp16 comparison), results/bundle_r4_<v>.json, results/runtime_r4.json.

Graph statistics are recomputed here from the per-slot records against the ORIGINAL oracle files (not the readout's
copies): dp = max over options of |p_graph - p_oracle| (unrounded, softmax of the letter logits at T = 1), tie =
oracle top-2 gap <= 1e-4; image arm = 75 slots, no-image arm = 9 slots numbered from 65.

    out/venv-readout/bin/python -B scripts/weight_forms_r4.py
"""
import os
from collections import OrderedDict

import numpy as np

from common import ROOT, TIE_GAP, read_json, write_json

VARIANTS = ('fp16', 'dyn8', 'v7c')
ORACLES = ('fixtures/oracle_fp32.json', 'fixtures/oracle_fp32_v2add.json')


def oracle_slots():
    ref = {}
    for p in ORACLES:
        for f in read_json(p)['forwards']:
            if f['arm'] == 'g256_mrope':
                for s in f['slots']:
                    ref[(f['row_id'], s['k'])] = s
    return ref


def softmax64(z):
    z = np.asarray(z, np.float64)
    e = np.exp(z - z.max())
    return e / e.sum()


def arm_stats(readout, arm, image, ref):
    slots = []
    for r in readout['rows']:
        if 'skipped' in r:
            continue
        arms = r['arms'] if 'arms' in r else {r['arm']: dict(slots=r['slots'])}    # graph_readout / gpu_readout_r4 rows
        if r['image'] != image or arm not in arms:
            continue
        for s in arms[arm]['slots']:
            o = ref[(r['row_id'], s['k'])]
            p = softmax64(s['letter_logits'])
            q = np.asarray(o['probs'], np.float64)
            dp = float(np.abs(p - q).max())
            assert abs(dp - s['dp']) <= 1e-12, (r['row_id'], s['k'], dp, s['dp'])      # readout's own dp = recomputed
            slots.append(dict(row_id=r['row_id'], k=s['k'], family=r['family'], purpose=r['purpose'], dp=dp,
                              argmax=int(p.argmax()), ref_argmax=int(q.argmax()), tie=o['top2_gap'] <= TIE_GAP,
                              ref_top2_gap=o['top2_gap'], probs=p.tolist(), ref_probs=o['probs'],
                              vocab_top1=s['vocab_top1_id'], ref_vocab_top1=o['vocab_top1_id'], logits_sha256=s.get('logits_sha256')))
    dps = np.array([s['dp'] for s in slots])
    nontie = [s for s in slots if not s['tie']]
    flips = [dict(row_id=s['row_id'], k=s['k'], family=s['family'], graph_argmax=s['argmax'], oracle_argmax=s['ref_argmax'],
                  oracle_top2_gap=s['ref_top2_gap'], dp=s['dp'], graph_probs=[round(v, 6) for v in s['probs']],
                  oracle_probs=[round(v, 6) for v in s['ref_probs']], tie=s['tie'])
             for s in slots if s['argmax'] != s['ref_argmax']]
    worst = sorted(slots, key=lambda s: -s['dp'])[:5]
    return OrderedDict(
        n_slots=len(slots), n_tie=len(slots) - len(nontie), nontie_argmax=f"{sum(s['argmax'] == s['ref_argmax'] for s in nontie)}/{len(nontie)}",
        nontie_argmax_all=all(s['argmax'] == s['ref_argmax'] for s in nontie),
        max_dp=float(dps.max()), p95_dp=float(np.percentile(dps, 95)), median_dp=float(np.median(dps)),
        n_dp_gt_1e3=int((dps > 1e-3).sum()), n_dp_gt_2e2=int((dps > 0.02).sum()),
        vocab_top1_agree=f"{sum(s['vocab_top1'] == s['ref_vocab_top1'] for s in slots)}/{len(slots)}",
        flips=flips, worst5=[dict(row_id=s['row_id'], k=s['k'], dp=s['dp'], oracle_top2_gap=s['ref_top2_gap'],
                                  argmax=s['argmax'], ref_argmax=s['ref_argmax']) for s in worst]), slots


def pass_line(img, txt):
    return dict(rule='non-tie argmax 100% AND max|dp| <= 1e-3 (75 image slots + 9 no-image slots from 65)',
                pass_=img['nontie_argmax_all'] and txt['nontie_argmax_all'] and max(img['max_dp'], txt['max_dp']) <= 1e-3,
                max_dp=max(img['max_dp'], txt['max_dp']))


def slotwise(a_rows, a_arm, b_rows, b_arm, image):
    """Slot-by-slot comparison of two readouts of the same rows (letter logits exact, logits hash when both have it)."""
    b = {r['row_id']: r for r in b_rows}
    n = same_letters = same_hash = hashed = 0
    maxd = 0.0
    for r in a_rows:
        if r['image'] != image or a_arm not in r['arms']:
            continue
        rb = b[r['row_id']]
        for sa, sb in zip(r['arms'][a_arm]['slots'], rb['arms'][b_arm]['slots']):
            n += 1
            same_letters += sa['letter_logits'] == sb['letter_logits']
            maxd = max(maxd, float(np.abs(np.asarray(sa['probs']) - np.asarray(sb['probs'])).max()))
            if sa.get('logits_sha256') and sb.get('logits_sha256'):
                hashed += 1
                same_hash += sa['logits_sha256'] == sb['logits_sha256']
    return dict(slots=n, letter_logits_bit_equal=same_letters, logits_sha256_compared=hashed, logits_sha256_equal=same_hash,
                max_abs_dp_between=maxd)


def main():
    ref = oracle_slots()
    res = OrderedDict(status='RUNNING')
    res['unit_test'] = {k: read_json('results/mrope_derived_test_r4.json')[k] for k in ('step_form', 'fixture_summary', 'full_cache', 'step_forms', 'pass')}
    exp = read_json('results/decoder_export_r4.json')
    res['export'] = OrderedDict(
        decoder=dict(path=exp['prefill_decode']['path'], bytes=exp['prefill_decode']['bytes'], sha256=exp['prefill_decode']['sha256'],
                     operator_count=exp['prefill_decode']['operator_count'], forbidden=exp['prefill_decode']['forbidden'],
                     relu_0_to_1=exp['prefill_decode']['op_histogram'].get('RELU_0_TO_1', 0),
                     per_signature_ops={k: v['op_histogram'] for k, v in exp['prefill_decode']['per_signature'].items()}),
        embedder=dict(path=exp['embedder']['path'], bytes=exp['embedder']['bytes'], sha256=exp['embedder']['sha256']),
        compare_to_r2=exp.get('compare_to'), export_driver={k: exp['export_driver'].get(k) for k in
                                                           ('status', 'step_form', 'argv', 'versions', 'clone', 'patch_sha256',
                                                            'mrope_derived_sha256', 'precheck_vs_5170', 'export_args',
                                                            'wall_seconds_contended')})
    res['lm_head_scope'] = read_json('results/lm_head_scope_r4.json')
    g = read_json('results/graph_parity_r4_fp32.json')
    img, _ = arm_stats(g, 'i_tflite_vision', True, ref)
    txt, _ = arm_stats(g, 'text_from_65', False, ref)
    r2 = read_json('results/graph_parity_r2.json')
    res['fp32_graph'] = OrderedDict(readout='results/graph_parity_r4_fp32.json', readout_status=g['status'], image=img, text_from_65=txt,
                                    pass_line=pass_line(img, txt),
                                    vs_r2_graph=dict(image=slotwise(g['rows'], 'i_tflite_vision', r2['rows'], 'i_tflite_vision', True),
                                                     text_from_65=slotwise(g['rows'], 'text_from_65', r2['rows'], 'text_from_65', False)))
    res['variants'] = OrderedDict()
    for v in VARIANTS:
        e = OrderedDict()
        for part in ('decoder', 'embedder'):
            p = ROOT / f'out/weights_r4/{v}/{part}.quant.json'
            if p.exists():
                q = read_json(str(p.relative_to(ROOT)))
                e[part] = dict(path=q['output'], bytes=q['bytes'], sha256=q['sha256'], ratio_vs_fp32=q['ratio_vs_source'],
                               recipe=q['recipe_provenance'], weights_by_op=q['census']['weights_by_op'],
                               weight_bytes_by_op=q['census']['weight_bytes_by_op'], op_instances=q['census']['op_instances'],
                               weight_via_dequantize=q['census']['weight_via_dequantize'],
                               constant_buffers_by_dtype=q['census']['constant_buffers_by_dtype'],
                               constant_bytes_by_dtype=q['census']['constant_bytes_by_dtype'],
                               operator_count=q['operator_count'], op_histogram=q['op_histogram'], forbidden=q['forbidden'])
        rp = ROOT / f'results/readout_r4_{v}.json'
        if rp.exists():
            ro = read_json(f'results/readout_r4_{v}.json')
            vi, _ = arm_stats(ro, v, True, ref)
            vt, _ = arm_stats(ro, 'text_from_65', False, ref)
            e['cpu_graph'] = OrderedDict(readout=f'results/readout_r4_{v}.json', xnn_cache=ro['xnn_cache'], readout_status=ro['status'],
                                         image=vi, text_from_65=vt, pass_line=pass_line(vi, vt) if v == 'fp16' else 'table only (no pass line)',
                                         vision=ro['vision'][v])
            e['cpu_graph']['vs_fp32_graph_r4'] = dict(image=slotwise(ro['rows'], v, g['rows'], 'i_tflite_vision', True),
                                                      text_from_65=slotwise(ro['rows'], 'text_from_65', g['rows'], 'text_from_65', False))
            if v == 'fp16':
                r3 = read_json('results/fp16_readout_r3.json')
                e['cpu_graph']['vs_r3_fp16_graph'] = dict(image=slotwise(ro['rows'], v, r3['rows'], 'A_fp16', True),
                                                          text_from_65=slotwise(ro['rows'], 'text_from_65', r3['rows'], 'text_from_65', False))
        bp = ROOT / f'results/bundle_r4_{v}.json'
        if bp.exists():
            b = read_json(f'results/bundle_r4_{v}.json')
            e['bundle'] = dict(path=b['bundle']['path'], bytes=b['bundle']['bytes'], sha256=b['bundle']['sha256'],
                               pre_executor_metadata=b['pre_executor_metadata'], describe=b['describe'],
                               sections=[dict(section_type=s['section_type'], model_type=s['model_type'], bytes=s['bytes'],
                                              sha256=s.get('sha256'), additional_metadata=s['additional_metadata']) for s in b['sections']],
                               checks=b['checks'], status=b['status'], content_readout=b.get('content_readout'),
                               decoder_signature_ops=b['decoder_signature_ops'])
        gp = {}
        for backend in ('gpu', 'cpu'):
            q = ROOT / f'results/gpu_readout_r4_{v}_{backend}.json'
            if q.exists():
                gr = read_json(f'results/gpu_readout_r4_{v}_{backend}.json')
                gi, _ = arm_stats(gr, v, True, ref)
                gt, _ = arm_stats(gr, 'text_from_65', False, ref)
                vs_cpu = {k: x['vs_cpu_exact_fit'] for k, x in gr['summary'].items()}
                gp[backend] = OrderedDict(readout=f'results/gpu_readout_r4_{v}_{backend}.json', runtime=gr['runtime'], gpu_options=gr['gpu_options'],
                                          decoder_fully_accelerated=gr['decoder_fully_accelerated'], scheme=gr['scheme'],
                                          load_seconds_contended=gr['load_seconds_contended'], all_finite=gr['all_finite'],
                                          vision_hashes_equal_cpu_readout=gr['vision_hashes_equal_cpu_readout'],
                                          image=gi, text_from_65=gt, vs_cpu_exact_fit=vs_cpu)
        if gp:
            if 'gpu' in gp and 'cpu' in gp:
                a_ = {(r['row_id'], s['k']): s['probs'] for r in read_json(gp['gpu']['readout'])['rows'] if 'slots' in r for s in r['slots']}
                b_ = {(r['row_id'], s['k']): s['probs'] for r in read_json(gp['cpu']['readout'])['rows'] if 'slots' in r for s in r['slots']}
                d_ = np.array([float(np.abs(np.asarray(a_[k]) - np.asarray(b_[k])).max()) for k in a_])
                gp['gpu_vs_cpu_same_scheme'] = dict(slots=len(d_), max_dp=float(d_.max()), p95_dp=float(np.percentile(d_, 95)),
                                                    median_dp=float(np.median(d_)), n_gt_1e3=int((d_ > 1e-3).sum()))
            e['padded_readout'] = gp
        res['variants'][v] = e
    if (ROOT / 'results/runtime_r4.json').exists():
        rt = read_json('results/runtime_r4.json')
        res['runtime'] = OrderedDict((v, OrderedDict((leg, dict(summary=lr['summary'],
                                                                rows=[{k: r.get(k) for k in ('row_id', 'status', 'engine_created', 'oracle_n_input_ids',
                                                                                             'runtime_prefill_tokens', 'pass_i', 'first_chunk',
                                                                                             'oracle_top1_id', 'fp32_graph_top1_id', 'variant_graph_top1_id',
                                                                                             'first_equals_oracle_top1', 'first_equals_variant_graph_top1',
                                                                                             'variant_graph_top1_top2_logit_gap', 'validation_error_lines',
                                                                                             'error')} for r in lr['rows']],
                                                                delegate_first_attempt=next((r['delegate'] for r in lr['rows'] if 'delegate' in r), None)))
                                                      for leg, lr in vr['legs'].items()))
                                     for v, vr in rt['variants'].items())
        res['runtime_disk'] = rt.get('disk')
    res['status'] = 'DONE'
    write_json('results/weight_forms_r4.json', res)
    print('WEIGHT_FORMS_R4', res['fp32_graph']['pass_line'], {v: (e.get('cpu_graph', {}).get('image', {}).get('max_dp'),
                                                                   e.get('cpu_graph', {}).get('pass_line')) for v, e in res['variants'].items()})


if __name__ == '__main__':
    main()
