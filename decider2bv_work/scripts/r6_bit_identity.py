"""Round 6: is the public reference readout (publish/reference/check_fixtures.py --out) bit-identical to round 4's
CPU graph readout of the same weight form (results/readout_r4_<v>.json, graph_readout.py on the separate tflites)?

Per public row and slot: the letter logits (float32 values, exact equality), the full-vocabulary logits sha256, the
row's text-embedding sha256 and, on image rows, the injected vision-embedding sha256 (round 4 arm <v>; text-only rows
arm text_from_65). Also re-derives argmax / max |dp| / p95 vs the fp32 oracle (g256_mrope) from the reference's own
probabilities, so the card numbers come from the reference output.

    python3 -B scripts/r6_bit_identity.py --variant fp16 --check results/r6/check_fp16_cpu.json
        -> results/r6/bit_identity_<v>.json
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True)
    ap.add_argument('--check', required=True)
    args = ap.parse_args()
    v = args.variant
    chk = json.loads((ROOT / args.check).read_text())
    r4 = json.loads((ROOT / f'results/readout_r4_{v}.json').read_text())
    r4rows = {r['row_id']: r for r in r4['rows']}
    rows, diffs = [], []
    n_slots = n_ll = n_sha = 0
    for row in chk['rows']:
        ref = r4rows[row['row_id']]
        arm = v if row['image'] else 'text_from_65'
        rslots = ref['arms'][arm]['slots']
        assert len(rslots) == len(row['slots']), row['row_id']
        text_eq = row['text_embeddings_sha256'] == ref['text_embeddings_sha256']
        vis_eq = (row['vision_sha256'] == ref['vision_sha256'].get(v)) if row['image'] else None
        srecs = []
        for s, t in zip(row['slots'], rslots):
            ll_eq = s['letter_logits'] == t['letter_logits']          # both are float32 values stored as JSON floats
            sha_eq = s['logits_sha256'] == t['logits_sha256']
            n_slots += 1
            n_ll += ll_eq
            n_sha += sha_eq
            if not (ll_eq and sha_eq):
                diffs.append(dict(row_id=row['row_id'], k=s['k'], letter_logits_equal=ll_eq, logits_sha256_equal=sha_eq,
                                  ref_letter_logits=t['letter_logits'], got_letter_logits=s['letter_logits']))
            srecs.append(dict(k=s['k'], letter_logits_equal=ll_eq, logits_sha256_equal=sha_eq))
        rows.append(dict(row_id=row['row_id'], image=row['image'], text_embeddings_equal=text_eq, vision_embeddings_equal=vis_eq,
                         slots=srecs))
    out = dict(variant=v, check=args.check, check_bundle=chk['bundle'], round4_readout=f'results/readout_r4_{v}.json',
               round4_decoder=r4['decoder'], round4_embedder=r4['embedder'], round4_vision=r4['vision'][v],
               check_sections=chk['sections'], rows_compared=len(rows), slots_compared=n_slots,
               letter_logits_bit_identical=n_ll, full_logits_sha256_identical=n_sha,
               text_embeddings_identical=sum(r['text_embeddings_equal'] for r in rows),
               vision_embeddings_identical=sum(bool(r['vision_embeddings_equal']) for r in rows if r['image']),
               n_image_rows=sum(r['image'] for r in rows), differing_slots=diffs,
               section_sha256_equal_round4=dict(
                   prefill_decode=chk['sections']['prefill_decode']['sha256'] == r4['decoder']['sha256'],
                   embedder=chk['sections']['embedder']['sha256'] == r4['embedder']['sha256'],
                   vision_encoder=chk['sections']['vision_encoder']['sha256'] == r4['vision'][v]['encoder']['sha256'],
                   vision_adapter=chk['sections']['vision_adapter']['sha256'] == r4['vision'][v]['adapter']['sha256']),
               summary_from_check=chk['summary'], rows=rows)
    out['bit_identical'] = (n_ll == n_sha == n_slots and out['text_embeddings_identical'] == len(rows)
                            and out['vision_embeddings_identical'] == out['n_image_rows'])
    dst = ROOT / f'results/r6/bit_identity_{v}.json'
    dst.write_text(json.dumps(out, indent=1) + '\n')
    print('BIT_IDENTITY', v, 'PASS' if out['bit_identical'] else 'FAIL', f"slots {n_ll}/{n_slots} letter logits, {n_sha}/{n_slots} full sha,",
          f"text {out['text_embeddings_identical']}/{len(rows)}, vision {out['vision_embeddings_identical']}/{out['n_image_rows']},",
          'sections', out['section_sha256_equal_round4'])


if __name__ == '__main__':
    main()
