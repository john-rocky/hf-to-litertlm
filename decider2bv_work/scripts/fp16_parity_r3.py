"""Round 3 fp16 parity record -> results/fp16_parity_r3.json.

  - variants: bytes, sha256, recipe provenance and the flatbuffer weight-dtype census of every quantized tflite
    (out/fp16_r3/*.quant.json, written by scripts/quantize_fp16_r3.py);
  - graph parity: the readout's summaries for (A) fp16 vision + fp16 decoder/embedder, (B) fp16 encoder + int8
    adapter + fp16 decoder/embedder, the HF-vision arm (fp16 decoder alone) and both no-image arms, taken from
    results/fp16_readout_r3.json, and the SAME statistics re-computed here from the stored graph probabilities
    against the ORIGINAL oracle files (not the readout's copies of the reference);
  - pass line (A), fixed before the run: slot count equal, non-tie argmax 100 %, max |dp| <= 1e-3
    (image rows vs g256_mrope, no-image rows from position 65). (B) is a table only;
  - informational: the fp16 graph against round 2's fp32 graph slot by slot (the casting's own share).

    out/venv-readout/bin/python -B scripts/fp16_parity_r3.py
"""
import os
import sys
from collections import OrderedDict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import ROOT, TIE_GAP, read_json, write_json, sha256_file  # noqa: E402

PARTS = ('decoder', 'embedder', 'vision_encoder', 'vision_adapter_fp16', 'vision_adapter_int8')
ORACLES = ('fixtures/oracle_fp32.json', 'fixtures/oracle_fp32_v2add.json')
PASS_DP = 1e-3


def stats(dps, agree_nontie, n_nontie):
    dps = np.asarray(dps, np.float64)
    return OrderedDict(n_slots=int(dps.size), nontie_argmax=f'{agree_nontie}/{n_nontie}', max_dp=float(dps.max()),
                       p95_dp=float(np.percentile(dps, 95)), median_dp=float(np.median(dps)), n_dp_gt_1e3=int((dps > PASS_DP).sum()))


def main():
    variants = OrderedDict()
    for part in PARTS:
        q = read_json(f'out/fp16_r3/{part}.quant.json')
        assert q['status'] == 'DONE' and sha256_file(ROOT / q['output']) == q['sha256'], part
        variants[part] = OrderedDict(file=q['output'], bytes=q['bytes'], sha256=q['sha256'], source=q['source'],
                                     source_bytes=q['source_bytes'], source_sha256=q['source_sha256'], ratio=q['ratio_vs_source'],
                                     kind=q['kind'], recipe_provenance=q['recipe_provenance'], recipe=q['recipe'],
                                     quantizer=q['versions'], wall_seconds_contended=q['wall_seconds_contended'],
                                     weights_by_op=q['census']['weights_by_op'], weight_bytes_by_op=q['census']['weight_bytes_by_op'],
                                     weight_via_dequantize=q['census']['weight_via_dequantize'],
                                     constant_bytes_by_dtype=q['census']['constant_bytes_by_dtype'],
                                     source_weights_by_op=q['source_census']['weights_by_op'])
    ro = read_json('results/fp16_readout_r3.json')
    for key, part in (('decoder', 'decoder'), ('embedder', 'embedder')):
        assert ro[key]['sha256'] == variants[part]['sha256'], key
    assert ro['vision']['A_fp16']['encoder']['sha256'] == variants['vision_encoder']['sha256']
    assert ro['vision']['A_fp16']['adapter']['sha256'] == variants['vision_adapter_fp16']['sha256']
    assert ro['vision']['B_fp16enc_int8adp']['adapter']['sha256'] == variants['vision_adapter_int8']['sha256']

    # independent re-computation against the original oracle files
    ref = {}
    for p in ORACLES:
        for f in read_json(p)['forwards']:
            if f['arm'] == 'g256_mrope':
                ref[f['row_id']] = f
    recomputed = OrderedDict()
    per_slot = OrderedDict()
    for arm in ('A_fp16', 'B_fp16enc_int8adp', 'ii_hf_vision', 'text_from_65', 'text_from_0_informational'):
        dps, agree, nontie, top1 = [], 0, 0, 0
        slots_out = []
        for rec in ro['rows']:
            if arm not in rec['arms']:
                continue
            f = ref[rec['row_id']]
            assert len(rec['arms'][arm]['slots']) == len(f['slots'])
            for s, o in zip(rec['arms'][arm]['slots'], f['slots']):
                p = np.asarray(s['probs'], np.float64)
                q = np.asarray(o['probs'], np.float64)
                dp = float(np.abs(p - q).max())
                tie = o['top2_gap'] <= TIE_GAP
                dps.append(dp)
                if not tie:
                    nontie += 1
                    agree += int(p.argmax()) == o['argmax']
                top1 += s['vocab_top1_id'] == o['vocab_top1_id']
                slots_out.append(dict(row_id=rec['row_id'], k=s['k'], family=rec['family'], purpose=rec['purpose'], dp=dp,
                                      argmax=int(p.argmax()), oracle_argmax=o['argmax'], oracle_top2_gap=o['top2_gap'],
                                      probs=[round(v, 6) for v in s['probs']], oracle_probs=[round(v, 6) for v in o['probs']]))
        e = stats(dps, agree, nontie)
        e['vocab_top1_agree'] = f'{top1}/{len(dps)}'
        e['equal_to_readout_summary'] = (abs(e['max_dp'] - ro['summary'][arm]['all']['max_dp']) == 0.0
                                         and abs(e['p95_dp'] - ro['summary'][arm]['all']['p95_dp']) == 0.0)
        recomputed[arm] = e
        per_slot[arm] = sorted(slots_out, key=lambda x: -x['dp'])
    a, t = recomputed['A_fp16'], recomputed['text_from_65']
    slot_count_equal = all(len(rec['arms'][arm]['slots']) == rec['n_slots_oracle'] for rec in ro['rows'] for arm in rec['arms'])
    pass_a = (slot_count_equal and a['nontie_argmax'].split('/')[0] == a['nontie_argmax'].split('/')[1]
              and t['nontie_argmax'].split('/')[0] == t['nontie_argmax'].split('/')[1] and max(a['max_dp'], t['max_dp']) <= PASS_DP)

    # informational: fp16 graph vs round-2 fp32 graph, slot by slot (same rows, same schedule)
    r2 = read_json('results/graph_parity_r2.json')
    r2rows = {r['row_id']: r for r in r2['rows']}
    vs_fp32 = OrderedDict()
    for arm16, arm32 in (('A_fp16', 'i_tflite_vision'), ('ii_hf_vision', 'ii_hf_vision'), ('text_from_65', 'text_from_65'),
                         ('text_from_0_informational', 'text_from_0_informational')):
        d = []
        for rec in ro['rows']:
            if arm16 in rec['arms']:
                for s16, s32 in zip(rec['arms'][arm16]['slots'], r2rows[rec['row_id']]['arms'][arm32]['slots']):
                    d.append(float(np.abs(np.asarray(s16['probs']) - np.asarray(s32['probs'])).max()))
        vs_fp32[f'{arm16}_vs_r2_{arm32}'] = OrderedDict(n_slots=len(d), max_dp=max(d), p95_dp=float(np.percentile(d, 95)),
                                                         median_dp=float(np.median(d)))

    res = OrderedDict(
        status='PASS' if pass_a and ro['all_finite'] else 'FAIL',
        pass_line=dict(rule='(A) slot count equal AND non-tie argmax 100% AND max|dp| <= 1e-3 (image rows vs oracle g256_mrope, '
                            'no-image rows from position 65); (B) table only', A_pass=pass_a, slot_count_equal=slot_count_equal,
                       A_image=f"{a['nontie_argmax']}, max {a['max_dp']:.3e}", no_image_from_65=f"{t['nontie_argmax']}, max {t['max_dp']:.3e}"),
        readout=dict(file='results/fp16_readout_r3.json', sha256=sha256_file(ROOT / 'results/fp16_readout_r3.json'),
                     status=ro['status'], runtime=ro['runtime'], threads=ro['threads'], all_finite=ro['all_finite'],
                     load_seconds_contended=ro['load_seconds_contended'], wall_seconds_contended=ro['wall_seconds_contended'],
                     summary=ro['summary'], pass_line=ro['pass_line']),
        recomputed_vs_original_oracle=recomputed, fp16_vs_fp32_graph_informational=vs_fp32,
        variants=variants,
        worst_slots={arm: v[:10] for arm, v in per_slot.items()},
        B_slots_dp_gt_1e3=[s for s in per_slot['B_fp16enc_int8adp'] if s['dp'] > PASS_DP])
    write_json('results/fp16_parity_r3.json', res)
    print('FP16_PARITY', res['status'], res['pass_line'], flush=True)
    for arm, e in recomputed.items():
        print(f'  {arm:26s}', dict(e))


if __name__ == '__main__':
    main()
