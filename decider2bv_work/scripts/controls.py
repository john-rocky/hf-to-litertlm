"""Controls C1-C7 for round 1, recomputed from the oracle JSON (and the fresh-process rerun JSON for C7).

    python3 scripts/controls.py [--oracle fixtures/oracle_fp32.json] [--rerun results/c7_rerun_oracle.json]
"""
import argparse
from collections import OrderedDict

from common import ROOT, ARMS, read_json, write_json, sha256_file

SLOT_ID, SLOT_TEXT = 318, ' ('


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--oracle', default='fixtures/oracle_fp32.json')
    ap.add_argument('--rerun', default='results/c7_rerun_oracle.json')
    ap.add_argument('--out', default='results/controls_r1.json')
    ap.add_argument('--md', default='rounds/r1_controls.generated.md')
    args = ap.parse_args()
    oracle = read_json(args.oracle)
    assert oracle['status'] == 'DONE'
    fw = oracle['forwards']
    by = {(f['row_id'], f['arm']): f for f in fw}
    rows = list(OrderedDict.fromkeys(f['row_id'] for f in fw))
    author = {r: by[(r, 'author')] for r in rows}
    logits = lambda f: [s['letter_logits_fp32'] for s in f['slots']]
    res = OrderedDict(source=args.oracle, source_sha256=sha256_file(ROOT / args.oracle))

    # C1: no-image rows identical across all five arms (ids and letter logits, exact float equality)
    text_rows = [r for r in rows if author[r]['image_original'] is None]
    c1 = []
    for r in text_rows:
        ref = author[r]
        for arm in ARMS[1:]:
            f = by[(r, arm)]
            c1.append(dict(row_id=r, arm=arm, ids_equal=f['input_ids'] == ref['input_ids'], logits_equal=logits(f) == logits(ref),
                           rope_index_calls=f['positions']['rope_index_calls'], positions_arange=f['positions']['is_arange']))
    res['C1'] = dict(status='PASS' if c1 and all(x['ids_equal'] and x['logits_equal'] for x in c1) else 'FAIL',
                     rows=text_rows, n_comparisons=len(c1), n_equal=sum(x['ids_equal'] and x['logits_equal'] for x in c1),
                     detail=c1)

    # C2: original already GxG -> author == g{G}_mrope bit for bit
    c2 = []
    for r in rows:
        img = author[r]['image_original']
        if img is None or img['width'] != img['height'] or img['width'] not in (256, 512):
            continue
        G = img['width']
        f = by[(r, f'g{G}_mrope')]
        c2.append(dict(row_id=r, G=G, pixel_values_equal=f['pixel_values_sha256'] == author[r]['pixel_values_sha256'],
                       ids_equal=f['input_ids'] == author[r]['input_ids'], logits_equal=logits(f) == logits(author[r]),
                       resized_rgb_equals_original=f['image_input']['sha256_rgb'] == img['sha256_rgb']))
    res['C2'] = dict(status='PASS' if len(c2) >= 2 and all(x['pixel_values_equal'] and x['ids_equal'] and x['logits_equal'] for x in c2)
                     else 'FAIL', detail=c2)

    # C3: the pos1d replacement is what the rotary receives (arange), the M-RoPE arms are not arange
    c3 = []
    for f in fw:
        if f['image_original'] is None:
            continue
        c3.append(dict(row_id=f['row_id'], arm=f['arm'], is_arange=f['positions']['is_arange'],
                       expect_arange=f['pos1d'], rope_index_calls=f['positions']['rope_index_calls']))
    bad = [x for x in c3 if x['is_arange'] != x['expect_arange']]
    # one variable moved: g{G}_mrope and g{G}_pos1d must share token ids and pixel_values exactly
    same_inputs = []
    for r in rows:
        if author[r]['image_original'] is None:
            continue
        for G in (256, 512):
            fb, fc = by[(r, f'g{G}_mrope')], by[(r, f'g{G}_pos1d')]
            same_inputs.append(dict(row_id=r, G=G, ids_equal=fb['input_ids'] == fc['input_ids'],
                                    pixel_values_equal=fb['pixel_values_sha256'] == fc['pixel_values_sha256'],
                                    pos1d_calls=fc['positions']['rope_index_calls'].get('pos1d', 0),
                                    mrope_calls_in_pos1d_arm=fc['positions']['rope_index_calls'].get('mrope', 0)))
    inputs_ok = all(x['ids_equal'] and x['pixel_values_equal'] and x['pos1d_calls'] == 1 and x['mrope_calls_in_pos1d_arm'] == 0
                    for x in same_inputs)
    sample = {arm: by[('game_mario_enemy', arm)]['positions']['probes'] for arm in ARMS if ('game_mario_enemy', arm) in by}
    res['C3'] = dict(status='PASS' if c3 and not bad and inputs_ok else 'FAIL', n_image_forwards=len(c3),
                     b_vs_c_same_ids_and_pixels=f"{sum(x['ids_equal'] and x['pixel_values_equal'] for x in same_inputs)}/{len(same_inputs)}",
                     b_vs_c_inputs_detail=same_inputs,
                     n_pos1d_arange=sum(x['is_arange'] for x in c3 if x['expect_arange']),
                     n_pos1d=sum(x['expect_arange'] for x in c3),
                     n_mrope_not_arange=sum(not x['is_arange'] for x in c3 if not x['expect_arange']),
                     n_mrope=sum(not x['expect_arange'] for x in c3), mismatches=bad, probe_values_game_mario_enemy=sample)

    # C4: processor pixel_values vs hand (x/255 - 0.5)/0.5 on the GxG inputs
    c4 = [dict(row_id=f['row_id'], arm=f['arm'], **f['c4']) for f in fw if 'c4' in f]
    worst64 = max(x['max_abs_vs_float64'] for x in c4)
    worst32 = max(x['max_abs_vs_float32'] for x in c4)
    res['C4'] = dict(status='PASS' if c4 and all(x['shape_match'] for x in c4) and worst64 <= 1e-6 else 'FAIL',
                     n_forwards=len(c4), max_abs_vs_float64_hand=worst64, max_abs_vs_float32_hand=worst32,
                     n_exact_vs_float32_hand=sum(x['max_abs_vs_float32'] == 0.0 for x in c4))

    # C5: slot count, slot token, letters, colour rows
    c5_bad = []
    for f in fw:
        if len(f['slots']) != f['nq'] or len(f['slot_idx']) != f['nq']:
            c5_bad.append((f['row_id'], f['arm'], 'slot count'))
        for s in f['slots']:
            if s['slot_token_id'] != SLOT_ID or s['slot_token_text'] != SLOT_TEXT:
                c5_bad.append((f['row_id'], f['arm'], s['k'], 'slot token'))
    letters_ok = oracle['letters']['texts'] == oracle['letters']['expected'] and len(set(oracle['letters']['ids'])) == 10
    colors = OrderedDict()
    for r in rows:
        if author[r]['family'] == 'color':
            colors[r] = {arm: dict(argmax=by[(r, arm)]['slots'][0]['argmax'], expected=by[(r, arm)]['slots'][0]['expected'],
                                   probs=by[(r, arm)]['slots'][0]['probs']) for arm in ARMS}
    colors_ok = all(v['argmax'] == v['expected'] for c in colors.values() for v in c.values())
    res['C5'] = dict(status='PASS' if not c5_bad and letters_ok and colors_ok and oracle['slot_token']['text'] == SLOT_TEXT else 'FAIL',
                     slot_token=oracle['slot_token'], letters=oracle['letters'],
                     n_forwards=len(fw), n_slots=sum(len(f['slots']) for f in fw), problems=c5_bad, color_rows=colors,
                     note='each letter A-J encodes to exactly one token: asserted by the checkpoint prompt.letter_ids() at model init')

    # C6: grid / image-token count of the author's processor for 256x240 and 224x224 (card: "256x240 = 64 visual tokens")
    c6 = OrderedDict()
    for r in rows:
        img = author[r]['image_original']
        if img is not None:
            c6[r] = dict(original=[img['width'], img['height']], image_grid_thw=author[r]['image_grid_thw'],
                         processor_resized_hw=author[r]['processor_resized_hw'], n_image_tokens=author[r]['n_image_tokens'])
    nes = [v for v in c6.values() if v['original'] == [256, 240]]
    c224 = [v for v in c6.values() if v['original'] == [224, 224]]
    res['C6'] = dict(status='PASS' if nes and all(v['n_image_tokens'] == 64 for v in nes) else 'FAIL',
                     card_claim='256x240 game frame = 64 visual tokens',
                     w256xh240=nes[0] if nes else None, all_256x240_64_tokens=all(v['n_image_tokens'] == 64 for v in nes),
                     w224xh224=c224[0] if c224 else None, all_rows=c6)

    # C7: fresh process rerun of a subset, bit-identical
    rerun_path = ROOT / args.rerun
    if rerun_path.exists():
        rr = read_json(args.rerun)
        c7 = []
        for f in rr['forwards']:
            o = by[(f['row_id'], f['arm'])]
            c7.append(dict(row_id=f['row_id'], arm=f['arm'], ids_equal=f['input_ids'] == o['input_ids'],
                           pixel_values_equal=f.get('pixel_values_sha256') == o.get('pixel_values_sha256'),
                           logits_equal=logits(f) == logits(o), probs_equal=[s['probs'] for s in f['slots']] == [s['probs'] for s in o['slots']]))
        res['C7'] = dict(status='PASS' if c7 and all(x['ids_equal'] and x['pixel_values_equal'] and x['logits_equal'] for x in c7) else 'FAIL',
                         rerun=args.rerun, rerun_sha256=sha256_file(rerun_path), rerun_threads=rr['torch_threads'],
                         oracle_threads=oracle['torch_threads'], n_forwards=len(c7),
                         n_bit_identical=sum(x['ids_equal'] and x['pixel_values_equal'] and x['logits_equal'] for x in c7), detail=c7)
    else:
        res['C7'] = dict(status='NOT RUN', rerun=args.rerun)

    res['summary'] = {k: res[k]['status'] for k in ('C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7')}
    res['kernel_evidence'] = dict(attn_implementation=oracle['runtime']['attn_implementation'],
                                  kernel_call_counts=oracle['runtime']['kernel_call_counts'],
                                  optional_kernel_packages=oracle['runtime']['optional_kernel_packages'],
                                  kernel_functions={k: dict(module=v['module'], qualname=v['qualname'])
                                                    for k, v in oracle['runtime']['kernel_functions'].items()})
    write_json(args.out, res)
    c = res
    md = ['| control | status | measured |', '|---|---|---|',
          f"| C1 no-image rows identical in all 5 arms | {c['C1']['status']} | {c['C1']['n_equal']}/{c['C1']['n_comparisons']} arm pairs bit-identical (ids + letter logits), rows {len(c['C1']['rows'])} |",
          f"| C2 GxG original: author == g{{G}}_mrope | {c['C2']['status']} | " + '; '.join(
              f"{x['row_id']} G={x['G']}: pixel_values {x['pixel_values_equal']}, ids {x['ids_equal']}, logits {x['logits_equal']}" for x in c['C2']['detail']) + ' |',
          f"| C3 rotary receives arange only in pos1d arms | {c['C3']['status']} | pos1d arange {c['C3']['n_pos1d_arange']}/{c['C3']['n_pos1d']}, "
          f"M-RoPE non-arange {c['C3']['n_mrope_not_arange']}/{c['C3']['n_mrope']}, (b) vs (c) same ids+pixels {c['C3']['b_vs_c_same_ids_and_pixels']} |",
          f"| C4 pixel_values vs (x/255-0.5)/0.5 | {c['C4']['status']} | max abs {c['C4']['max_abs_vs_float64_hand']:.3e} vs float64 hand, "
          f"{c['C4']['max_abs_vs_float32_hand']:.3e} vs float32 hand ({c['C4']['n_exact_vs_float32_hand']}/{c['C4']['n_forwards']} exact) |",
          f"| C5 slots / slot token / letters / colour rows | {c['C5']['status']} | {c['C5']['n_slots']} slots in {c['C5']['n_forwards']} forwards, "
          f"slot token {c['C5']['slot_token']['id']} {c['C5']['slot_token']['text']!r}, letters {c['C5']['letters']['ids']}, problems {len(c['C5']['problems'])} |",
          f"| C6 256x240 and 224x224 grid | {c['C6']['status']} | 256x240: grid {c['C6']['w256xh240']['image_grid_thw']} -> {c['C6']['w256xh240']['processor_resized_hw']}, "
          f"{c['C6']['w256xh240']['n_image_tokens']} tokens; 224x224: grid {c['C6']['w224xh224']['image_grid_thw']} -> {c['C6']['w224xh224']['processor_resized_hw']}, "
          f"{c['C6']['w224xh224']['n_image_tokens']} tokens |"]
    if c['C7']['status'] != 'NOT RUN':
        md.append(f"| C7 fresh process rerun bit-identical | {c['C7']['status']} | {c['C7']['n_bit_identical']}/{c['C7']['n_forwards']} forwards "
                  f"(threads {c['C7']['rerun_threads']} = oracle {c['C7']['oracle_threads']}) |")
    else:
        md.append('| C7 fresh process rerun bit-identical | NOT RUN | - |')
    (ROOT / args.md).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / args.md).write_text('\n'.join(md) + '\n')
    print('\n'.join(md))
    for k in ('C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'):
        v = res[k]
        extra = {a: b for a, b in v.items() if a not in ('detail', 'status', 'all_rows', 'color_rows', 'letters', 'probe_values_game_mario_enemy',
                                                           'mismatches', 'problems', 'rows', 'b_vs_c_inputs_detail')}
        print(k, v['status'], extra)


if __name__ == '__main__':
    main()
