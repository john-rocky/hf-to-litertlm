"""Contract-cost comparison, recomputed from the oracle JSON alone (no model, no torch).

Per comparison (base arm -> new arm) over the IMAGE slots: slot count, argmax agreement (all / non-tie),
max / p95 / median of the per-slot max_option |p_new - p_base|, slots above 0.02, and every flip with the base
side's top-2 gap. Tie = base top-2 gap <= 1e-4. Breakdowns by family, tier and purpose. The pre-registered
decision rules (pre-registered, round 1) are evaluated mechanically at the end.

    python3 scripts/compare_arms.py [--oracle fixtures/oracle_fp32.json]
"""
import argparse
from collections import OrderedDict

import numpy as np

from common import ROOT, TIE_GAP, read_json, write_json, sha256_file

PAIRS = [('author', 'g256_mrope', 'grid@256'), ('author', 'g512_mrope', 'grid@512'),
         ('g256_mrope', 'g256_pos1d', 'position@256'), ('g512_mrope', 'g512_pos1d', 'position@512'),
         ('author', 'g256_pos1d', 'total@256'), ('author', 'g512_pos1d', 'total@512')]
DP_LINE = 0.02


def grid_relation(author, G):
    """How the author's dynamic grid compares with the fixed GxG grid (same / fewer / more image tokens at G)."""
    t, h, w = author['image_grid_thw']
    if (h, w) == (G // 16, G // 16):
        return 'same_grid'
    return 'G_fewer_tokens' if (G // 32) ** 2 < author['n_image_tokens'] else 'G_more_tokens'


def slot_diffs(by, row_ids, base, new):
    out = []
    G = int(new[1:4])
    for rid in row_ids:
        fb, fn = by[(rid, base)], by[(rid, new)]
        rel = grid_relation(by[(rid, 'author')], G)
        assert len(fb['slots']) == len(fn['slots']) == fb['nq']
        for sb, sn in zip(fb['slots'], fn['slots']):
            assert sb['k'] == sn['k'] and sb['nopts'] == sn['nopts'] and sb['question'] == sn['question']
            pb, pn = np.array(sb['probs'], dtype=np.float64), np.array(sn['probs'], dtype=np.float64)
            out.append(dict(row_id=rid, k=sb['k'], family=fb['family'], tier=fb['tier'], purpose=fb['purpose'], grid_relation=rel,
                            dp=float(np.abs(pn - pb).max()), tie=sb['top2_gap'] <= TIE_GAP,
                            agree=sb['argmax'] == sn['argmax'], base_argmax=sb['argmax'], new_argmax=sn['argmax'],
                            base_top2_gap=sb['top2_gap'], base_top1_prob=sb['top1_prob'], new_top1_prob=sn['top1_prob'],
                            base_probs=sb['probs'], new_probs=sn['probs']))
    return out


def summarize(slots):
    if not slots:
        return dict(n_slots=0)
    dps = np.array([s['dp'] for s in slots])
    nontie = [s for s in slots if not s['tie']]
    flips = [dict(row_id=s['row_id'], k=s['k'], base_argmax=s['base_argmax'], new_argmax=s['new_argmax'],
                  base_top2_gap=s['base_top2_gap'], tie=s['tie'], base_probs=s['base_probs'], new_probs=s['new_probs'])
             for s in slots if not s['agree']]
    return OrderedDict(
        n_slots=len(slots), n_tie=len(slots) - len(nontie),
        argmax_agree_all=sum(s['agree'] for s in slots), argmax_agree_nontie=sum(s['agree'] for s in nontie),
        n_nontie=len(nontie), max_dp=float(dps.max()), p95_dp=float(np.percentile(dps, 95)),
        median_dp=float(np.median(dps)), n_dp_gt_002=int((dps > DP_LINE).sum()),
        max_dp_slot=next(f"{s['row_id']}#{s['k']}" for s in slots if s['dp'] == dps.max()), flips=flips)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--oracle', default='fixtures/oracle_fp32.json')
    ap.add_argument('--out', default='results/contract_cost_r1.json')
    ap.add_argument('--md', default='rounds/r1_tables.generated.md')
    args = ap.parse_args()
    oracle = read_json(args.oracle)
    assert oracle['status'] == 'DONE', oracle['status']
    by = {(f['row_id'], f['arm']): f for f in oracle['forwards']}
    row_order = list(OrderedDict.fromkeys(f['row_id'] for f in oracle['forwards']))
    image_rows = [r for r in row_order if by[(r, 'author')]['image_original'] is not None]
    meta = {r: by[(r, 'author')] for r in image_rows}

    res = OrderedDict(source=args.oracle, source_sha256=sha256_file(ROOT / args.oracle), tie_gap=TIE_GAP,
                      dp_definition='per slot: max over the valid options of |p_new - p_base|; p95 = numpy linear percentile',
                      scope='image rows only (no-image rows are the C1 control)', n_image_rows=len(image_rows),
                      comparisons=OrderedDict())
    for base, new, name in PAIRS:
        slots = slot_diffs(by, image_rows, base, new)
        entry = OrderedDict(base=base, new=new, all=summarize(slots))
        for key in ('family', 'tier', 'purpose', 'grid_relation'):
            entry[f'by_{key}'] = OrderedDict((v, summarize([s for s in slots if s[key] == v]))
                                             for v in sorted(set(s[key] for s in slots)))
        entry['per_slot'] = [dict(row_id=s['row_id'], k=s['k'], dp=s['dp'], tie=s['tie'], agree=s['agree'])
                             for s in slots]
        res['comparisons'][name] = entry

    # fixture adequacy: share of image slots whose reference top-1 probability is below 0.9
    adequacy = OrderedDict()
    for arm in ('author', 'g256_mrope', 'g512_mrope'):
        tops = [s['top1_prob'] for r in image_rows for s in by[(r, arm)]['slots']]
        adequacy[arm] = dict(n_slots=len(tops), n_top1_below_0_9=int(sum(t < 0.9 for t in tops)),
                             share=float(np.mean([t < 0.9 for t in tops])))
    res['fixture_adequacy'] = adequacy
    # rows whose author grid already equals GxG: are the processor's pixels identical to the PIL-resized ones?
    same = OrderedDict()
    for G in (256, 512):
        rs = [r for r in image_rows if meta[r]['image_grid_thw'] == [1, G // 16, G // 16]]
        same[f'g{G}'] = dict(rows=rs, n_rows=len(rs),
                             n_pixel_values_identical=sum(meta[r]['pixel_values_sha256'] == by[(r, f'g{G}_mrope')]['pixel_values_sha256'] for r in rs),
                             n_letter_logits_identical=sum([s['letter_logits_fp32'] for s in meta[r]['slots']]
                                                           == [s['letter_logits_fp32'] for s in by[(r, f'g{G}_mrope')]['slots']] for r in rs))
    res['author_grid_equals_G'] = same
    res['wall_seconds_contended_median_by_arm'] = OrderedDict(
        (arm, float(np.median([f['wall_seconds_contended'] for f in oracle['forwards'] if f['arm'] == arm])))
        for arm in OrderedDict.fromkeys(f['arm'] for f in oracle['forwards']))

    # pre-registered rules (pre-registered): evaluated mechanically, not tuned
    rules = OrderedDict()
    for G in (256, 512):
        a = res['comparisons'][f'position@{G}']['all']
        rules[f'pos1d_ok@{G}'] = dict(
            rule='(b)->(c): all non-tie argmax agree AND max|dp| <= 0.02',
            nontie_agree=f"{a['argmax_agree_nontie']}/{a['n_nontie']}", max_dp=a['max_dp'],
            pass_=a['argmax_agree_nontie'] == a['n_nontie'] and a['max_dp'] <= DP_LINE)
    sel = [r for r in image_rows if meta[r]['family'] == 'photo'
           or meta[r]['image_original']['width'] != meta[r]['image_original']['height']]
    g = {}
    for G in (256, 512):
        s = slot_diffs(by, sel, 'author', f'g{G}_mrope')
        g[G] = dict(flips_nontie=sorted(f"{x['row_id']}#{x['k']}" for x in s if not x['agree'] and not x['tie']),
                    max_dp=max(x['dp'] for x in s), n_slots=len(s))
    only256 = sorted(set(g[256]['flips_nontie']) - set(g[512]['flips_nontie']))
    rules['grid_prefer_256'] = dict(
        rule='photo or non-square rows: (a)->(b)@256 clearly worse than @512 (a non-tie flip absent at 512, or max|dp| > 2x) -> consider 512',
        rows=sel, n_slots=g[256]['n_slots'], flips_nontie_256=g[256]['flips_nontie'], flips_nontie_512=g[512]['flips_nontie'],
        flips_only_at_256=only256, max_dp_256=g[256]['max_dp'], max_dp_512=g[512]['max_dp'],
        ratio_256_over_512=(g[256]['max_dp'] / g[512]['max_dp']) if g[512]['max_dp'] > 0 else None,
        consider_512=bool(only256) or g[256]['max_dp'] > 2 * g[512]['max_dp'])
    res['rules'] = rules
    write_json(args.out, res)

    # generated markdown tables (copied verbatim into the round report)
    L = ['| comparison | slots | ties | argmax agree (all) | argmax agree (non-tie) | max abs dp | p95 | median | slots dp>0.02 | flips |',
         '|---|---|---|---|---|---|---|---|---|---|']
    for name, e in res['comparisons'].items():
        a = e['all']
        L.append(f"| {name} ({e['base']}->{e['new']}) | {a['n_slots']} | {a['n_tie']} | {a['argmax_agree_all']}/{a['n_slots']} | "
                 f"{a['argmax_agree_nontie']}/{a['n_nontie']} | {a['max_dp']:.4f} | {a['p95_dp']:.4f} | {a['median_dp']:.4f} | "
                 f"{a['n_dp_gt_002']} | {len(a['flips'])} |")
    for key in ('family', 'tier', 'grid_relation'):
        L += ['', f'By {key} (max abs dp / non-tie argmax agree / slots dp>0.02):', '',
              '| comparison | ' + ' | '.join(cols := sorted(set(c for e in res['comparisons'].values() for c in e[f'by_{key}']))) + ' |',
              '|---|' + '---|' * len(cols)]
        for name, e in res['comparisons'].items():
            cells = [(lambda v: f"{v['max_dp']:.4f} / {v['argmax_agree_nontie']}/{v['n_nontie']} / {v['n_dp_gt_002']} (n={v['n_slots']})"
                      if v else '-')(e[f'by_{key}'].get(c)) for c in cols]
            L.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    L += ['', 'Flips (base argmax -> new argmax, base top-2 gap):', '']
    for name, e in res['comparisons'].items():
        for f in e['all']['flips']:
            L.append(f"- {name}: {f['row_id']} slot {f['k']}: {f['base_argmax']} -> {f['new_argmax']}, base gap "
                     f"{f['base_top2_gap']:.4f}{' (tie)' if f['tie'] else ''}; base p {[round(x, 4) for x in f['base_probs']]} "
                     f"new p {[round(x, 4) for x in f['new_probs']]}")
    L += ['', 'Rules:', '']
    for k, v in rules.items():
        L.append(f'- {k}: ' + ', '.join(f'{a}={b}' for a, b in v.items() if a not in ('rows',)))
    L += ['', 'Fixture adequacy (image slots with reference top-1 < 0.9):', '']
    for k, v in adequacy.items():
        L.append(f"- {k}: {v['n_top1_below_0_9']}/{v['n_slots']} = {v['share']:.3f}")
    (ROOT / args.md).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / args.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
