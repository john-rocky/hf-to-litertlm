"""Round-2 contract-cost table at G = 256 over fixtures v1 + v2 (no model, no torch).

Reads the committed round-1 oracle (fixtures/oracle_fp32.json, never rewritten) and the round-2 additions
(fixtures/oracle_fp32_v2add.json, arms author / g256_mrope / g256_pos1d only) as one set, then applies the round-1
measurement functions (scripts/compare_arms.py: slot_diffs, summarize) to the 256 arms, for v1 + v2, v1 alone and v2
alone. Fixture adequacy (image slots with reference top-1 < 0.9, target >= 1/3) is reported per set.

    python3 scripts/compare_arms_r2.py
"""
from collections import OrderedDict

import numpy as np

from common import ROOT, TIE_GAP, read_json, write_json, sha256_file
from compare_arms import slot_diffs, summarize, DP_LINE

ORACLES = ('fixtures/oracle_fp32.json', 'fixtures/oracle_fp32_v2add.json')
PAIRS = [('author', 'g256_mrope', 'grid@256'), ('g256_mrope', 'g256_pos1d', 'position@256'),
         ('author', 'g256_pos1d', 'total@256')]
ARMS_256 = ('author', 'g256_mrope', 'g256_pos1d')


def load_union():
    """{(row_id, arm): forward} over both oracle files, the 256 arms only; rows in fixture order; source per row."""
    by, order, source = {}, [], {}
    for path in ORACLES:
        orc = read_json(path)
        assert orc['status'] == 'DONE', (path, orc['status'])
        for f in orc['forwards']:
            if f['arm'] not in ARMS_256:
                continue
            key = (f['row_id'], f['arm'])
            assert key not in by, f'{key} appears in two oracle files'
            by[key] = f
            if f['row_id'] not in source:
                order.append(f['row_id'])
                source[f['row_id']] = 'v1' if path == ORACLES[0] else 'v2'
    for rid in order:
        assert all((rid, a) in by for a in ARMS_256), rid
    return by, order, source


def adequacy(by, rows, arm):
    tops = [s['top1_prob'] for r in rows for s in by[(r, arm)]['slots']]
    return dict(n_slots=len(tops), n_top1_below_0_9=int(sum(t < 0.9 for t in tops)),
                share=float(np.mean([t < 0.9 for t in tops])) if tops else None)


def main():
    by, order, source = load_union()
    image_rows = [r for r in order if by[(r, 'author')]['image_original'] is not None]
    sets = OrderedDict([('v1+v2', image_rows), ('v1', [r for r in image_rows if source[r] == 'v1']),
                        ('v2', [r for r in image_rows if source[r] == 'v2'])])
    res = OrderedDict(sources={p: sha256_file(ROOT / p) for p in ORACLES}, tie_gap=TIE_GAP, dp_line=DP_LINE,
                      dp_definition='per slot: max over the valid options of |p_new - p_base|; p95 = numpy linear percentile',
                      scope='image rows only; 256 arms', n_image_rows={k: len(v) for k, v in sets.items()},
                      comparisons=OrderedDict(), fixture_adequacy=OrderedDict())
    for base, new, name in PAIRS:
        entry = OrderedDict(base=base, new=new)
        for set_name, rows in sets.items():
            slots = slot_diffs(by, rows, base, new)
            e = OrderedDict(all=summarize(slots))
            if set_name == 'v1+v2':
                for key in ('family', 'tier', 'purpose', 'grid_relation'):
                    e[f'by_{key}'] = OrderedDict((v, summarize([s for s in slots if s[key] == v]))
                                                 for v in sorted(set(s[key] for s in slots)))
                e['per_slot'] = [dict(row_id=s['row_id'], k=s['k'], dp=s['dp'], tie=s['tie'], agree=s['agree'],
                                      source=source[s['row_id']]) for s in slots]
            entry[set_name] = e
        res['comparisons'][name] = entry
    for set_name, rows in sets.items():
        res['fixture_adequacy'][set_name] = OrderedDict((arm, adequacy(by, rows, arm)) for arm in ('author', 'g256_mrope'))
    a = res['comparisons']['position@256']['v1+v2']['all']
    res['pos1d_rule_on_v1v2_informational'] = dict(
        rule='(b)->(c): all non-tie argmax agree AND max|dp| <= 0.02 (round-1 rule, decided on v1; recomputed on v1+v2 for the record)',
        nontie_agree=f"{a['argmax_agree_nontie']}/{a['n_nontie']}", max_dp=a['max_dp'],
        pass_=a['argmax_agree_nontie'] == a['n_nontie'] and a['max_dp'] <= DP_LINE)
    write_json('results/contract_cost_r2.json', res)

    L = ['| comparison | set | slots | ties | argmax agree (non-tie) | max abs dp | p95 | median | slots dp>0.02 | flips |',
         '|---|---|---|---|---|---|---|---|---|---|']
    for name, entry in res['comparisons'].items():
        for set_name in sets:
            x = entry[set_name]['all']
            L.append(f"| {name} ({entry['base']}->{entry['new']}) | {set_name} | {x['n_slots']} | {x['n_tie']} | "
                     f"{x['argmax_agree_nontie']}/{x['n_nontie']} | {x['max_dp']:.4f} | {x['p95_dp']:.4f} | {x['median_dp']:.4f} | "
                     f"{x['n_dp_gt_002']} | {len(x['flips'])} |")
    L += ['', 'Flips (v1+v2; base argmax -> new argmax, base top-2 gap):', '']
    for name, entry in res['comparisons'].items():
        for f in entry['v1+v2']['all']['flips']:
            L.append(f"- {name}: {f['row_id']} slot {f['k']}: {f['base_argmax']} -> {f['new_argmax']}, base gap "
                     f"{f['base_top2_gap']:.4f}; base p {[round(v, 4) for v in f['base_probs']]} new p {[round(v, 4) for v in f['new_probs']]}")
    L += ['', 'Fixture adequacy (image slots with reference top-1 < 0.9, target >= 1/3):', '']
    for set_name, d in res['fixture_adequacy'].items():
        L.append(f'- {set_name}: ' + ', '.join(f"{arm} {v['n_top1_below_0_9']}/{v['n_slots']} = {v['share']:.3f}" for arm, v in d.items()))
    (ROOT / 'rounds/r2_contract_tables.generated.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
