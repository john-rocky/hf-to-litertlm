"""Round 6: run the card's Python usage exactly as printed (image path + one question), plus a text-only request and
a two-question request through the same public API, on one bundle. Writes results/r6/usage_test_<name>.json.

    out/venv-ref/bin/python -B scripts/r6_usage_test.py --bundle out/bundle_r4/v7c/decider-2b-vision_v7c.litertlm
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT / 'publish/reference'))
    from decider_litert import DeciderLiteRT
    fx = {r['id']: r for r in json.loads((ROOT / 'publish/reference/fixtures/fixtures.json').read_text())['rows']}
    d = DeciderLiteRT(args.bundle, cache_dir=str(ROOT / 'out/ref_cache'))
    pong = fx['game_pong_atari_up']
    out = dict(bundle=Path(args.bundle).name)
    out['image_one_question'] = d.decide(str(ROOT / 'publish/reference/fixtures' / pong['image']['file']), pong['context'],
                                         [{'question': q['question'], 'options': q['options']} for q in pong['questions']])
    txt = fx['text_billing_multi']
    out['text_only_three_questions'] = d.decide(None, txt['context'], [{'question': q['question'], 'options': q['options']}
                                                                       for q in txt['questions']])
    two = fx['game_breakout_nes_multi']
    out['image_two_questions'] = d.decide(str(ROOT / 'publish/reference/fixtures' / two['image']['file']), two['context'],
                                          [{'question': q['question'], 'options': q['options']} for q in two['questions']])
    ref = {k: [s['probs'] for s in fx[r]['oracle']['g256_mrope']['slots']]
           for k, r in (('image_one_question', 'game_pong_atari_up'), ('text_only_three_questions', 'text_billing_multi'),
                        ('image_two_questions', 'game_breakout_nes_multi'))}
    out['max_abs_dp_vs_upstream'] = {k: max(abs(a - b) for ans, up in zip(out[k], ref[k]) for a, b in zip(ans['probs'].values(), up))
                                     for k in ref}
    name = Path(args.bundle).stem.split('_')[-1]
    (ROOT / f'results/r6/usage_test_{name}.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out['max_abs_dp_vs_upstream']), [a['choice'] for a in out['image_one_question']],
          [a['choice'] for a in out['text_only_three_questions']], [a['choice'] for a in out['image_two_questions']])


if __name__ == '__main__':
    main()
