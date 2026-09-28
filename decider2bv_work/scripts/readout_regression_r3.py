"""Round 3: the edited scripts/graph_readout.py, run with NO variant flags (= the round-2 fp32 files and arms) on a few
rows, against round 2's committed results/graph_parity_r2.json, value by value: every slot's probabilities, letter
logits and full-vocabulary top-1, and the per-arm call schedule. Shows that the round-3 edit (paths as flags, extra
hashes) left the numbers of the default path unchanged -> results/readout_regression_r3.json.

    out/venv-readout/bin/python -B scripts/graph_readout.py --rows synth_teal,text_finance --out out/r3_regression_readout.json
    out/venv-readout/bin/python -B scripts/readout_regression_r3.py
"""
from common import read_json, write_json, sha256_file, ROOT


def main():
    new, old = read_json('out/r3_regression_readout.json'), read_json('results/graph_parity_r2.json')
    old_rows = {r['row_id']: r for r in old['rows']}
    out, n, same = [], 0, 0
    for r in new['rows']:
        o = old_rows[r['row_id']]
        for arm, v in r['arms'].items():
            ov = o['arms'][arm]
            for s, os_ in zip(v['slots'], ov['slots']):
                n += 1
                eq = dict(probs=s['probs'] == os_['probs'], letter_logits=s['letter_logits'] == os_['letter_logits'],
                          vocab_top1_id=s['vocab_top1_id'] == os_['vocab_top1_id'], calls=v['calls'] == ov['calls'])
                same += all(eq.values())
                out.append(dict(row_id=r['row_id'], arm=arm, k=s['k'], **eq))
    res = dict(status='PASS' if n and same == n and new['decoder'] == old['decoder'] and new['embedder'] == old['embedder'] else 'FAIL',
               new=dict(file='out/r3_regression_readout.json', status=new['status'], decoder=new['decoder']['sha256'],
                        vision=new['vision']), old=dict(file='results/graph_parity_r2.json', sha256=sha256_file(ROOT / 'results/graph_parity_r2.json')),
               rows=[r['row_id'] for r in new['rows']], slots_compared=n, slots_identical=same, slots=out)
    write_json('results/readout_regression_r3.json', res)
    print('READOUT_REGRESSION', res['status'], f'{same}/{n}')


if __name__ == '__main__':
    main()
