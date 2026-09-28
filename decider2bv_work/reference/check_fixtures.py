"""Check a decider-2b-vision .litertlm against the upstream float32 values in fixtures/fixtures.json.

    python -B reference/check_fixtures.py --bundle decider-2b-vision_fp16-int8vocab.litertlm [--backend cpu|gpu] [--out check.json]

For every published row this rebuilds the request the way decider_litert.py does for any user (original image ->
256 x 256 PIL BICUBIC; upstream build() -> decode -> encode), asserts that the 256 x 256 pixels, the token ids and
the answer slots equal the stored upstream ones, reads the letter probabilities from the bundle and compares them
with the upstream arm `g256_mrope` (the same 256 x 256 input). Reported per group (image rows, text-only rows):
argmax agreement on non-tie slots (tie = upstream top-2 gap <= 1e-4), max and p95 of |dp| (per slot the largest
absolute probability difference over its options). Two more comparisons are printed: the `author` arm (upstream on
the original image, dynamic resolution) adds the cost of the fixed 256 x 256 input; the `g256_pos1d` arm (upstream
with plain 1-D positions) is the contract the bundle must NOT reproduce where the two upstream arms differ.
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from decider_litert import DeciderLiteRT, image_256, diagnostics_to_stderr     # noqa: E402

FIXTURES = HERE / 'fixtures/fixtures.json'
ARMS = ('g256_mrope', 'author', 'g256_pos1d')


def rgb_sha(im):
    im = im.convert('RGB')
    return hashlib.sha256(f'{im.width}x{im.height}:'.encode() + im.tobytes()).hexdigest()


def summarize(slots, arm):
    if not slots:
        return dict(n_slots=0)
    dps = np.array([s[arm]['dp'] for s in slots])
    nontie = [s for s in slots if not s[arm]['tie']]
    return dict(n_slots=len(slots), n_tie=len(slots) - len(nontie), argmax_agree_nontie=sum(s[arm]['agree'] for s in nontie),
                n_nontie=len(nontie), max_dp=float(dps.max()), p95_dp=float(np.percentile(dps, 95)),
                median_dp=float(np.median(dps)), n_dp_gt_0_02=int((dps > 0.02).sum()),
                max_dp_slot=next(f"{s['row_id']}#{s['k']}" for s in slots if s[arm]['dp'] == dps.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--backend', choices=('cpu', 'gpu'), default='cpu')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--cache-dir', default=None)
    ap.add_argument('--rows', default='', help='comma-separated row ids (default: all)')
    ap.add_argument('--out', default=None, help='write the per-slot results as JSON')
    args = ap.parse_args()
    fx = json.loads(FIXTURES.read_text())
    tie_gap = fx['tie_gap']
    t0 = time.monotonic()
    with diagnostics_to_stderr():
        d = DeciderLiteRT(args.bundle, backend=args.backend, threads=args.threads, cache_dir=args.cache_dir)
    rows = [r for r in fx['rows'] if not args.rows or r['id'] in args.rows.split(',')]
    res = dict(bundle=dict(name=Path(args.bundle).name, bytes=d.record['bundle_bytes'], sha256=d.record['bundle_sha256']),
               sections={k: dict(bytes=v['bytes'], sha256=v['sha256']) for k, v in d.record['files'].items()},
               backend=args.backend, threads=args.threads if args.backend == 'cpu' else None,
               scheme='exact-fit prefill, one decode per slot' if args.backend == 'cpu' else 'one padded prefill chunk per slot',
               python=platform.python_version(), machine=platform.machine(), load_seconds=d.load_seconds,
               fixtures_sha256=hashlib.sha256(FIXTURES.read_bytes()).hexdigest(), rows=[])
    for r in rows:
        image = None
        if r['image'] is not None:
            image = Image.open(HERE / 'fixtures' / r['image']['file'])
            assert rgb_sha(image) == r['image']['sha256_rgb'], r['id']
            assert rgb_sha(image_256(image)) == r['image_256']['sha256_rgb'], f"{r['id']}: 256 x 256 pixels differ"
        ref = r['oracle']['g256_mrope']
        with diagnostics_to_stderr():
            out = d.readout(image, r['context'], r['questions'])
        assert out['ids'] == ref['input_ids'], f"{r['id']}: token ids differ from upstream"
        assert out['slot_idx'] == ref['slot_idx'], f"{r['id']}: answer slots differ from upstream"
        assert out['first_pos'] == r['first_position'], r['id']
        rec = dict(row_id=r['id'], family=r['family'], image=image is not None, n_tokens=len(out['ids']),
                   text_embeddings_sha256=out['text_embeddings_sha256'], vision_sha256=out['vision_sha256'],
                   wall_seconds=out['wall_seconds'], slots=[])
        for k, s in enumerate(out['slots']):
            p = np.asarray(s['probs'])
            srec = dict(k=k, nopts=s['nopts'], letter_logits=s['letter_logits'], probs=s['probs'],
                        logits_sha256=s['logits_sha256'], vocab_top1_id=s['vocab_top1_id'], argmax=int(p.argmax()))
            for arm in ARMS:
                a = r['oracle'][arm]['slots'][k]
                q = np.asarray(a['probs'], np.float64)
                srec[arm] = dict(dp=float(np.abs(p - q).max()), agree=int(p.argmax()) == a['argmax'],
                                 tie=a['top2_gap'] <= tie_gap, ref_argmax=a['argmax'])
            rec['slots'].append(srec)
        res['rows'].append(rec)
        print(f"{r['id']:34s} " + ' '.join(f"k{s['k']}:{'ABCDEFGHIJ'[s['argmax']]} dp={s['g256_mrope']['dp']:.2e}"
                                          f"{'' if s['g256_mrope']['agree'] else '!'}" for s in rec['slots'])
              + f"  {out['wall_seconds']:.1f}s", flush=True)
    res['summary'] = {}
    for group, flag in (('image', True), ('text_only', False)):
        sl = [dict(s, row_id=x['row_id']) for x in res['rows'] if x['image'] == flag for s in x['slots']]
        res['summary'][group] = {arm: summarize(sl, arm) for arm in ARMS}
    res['wall_seconds'] = time.monotonic() - t0
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1) + '\n')
    for group, v in res['summary'].items():
        for arm, s in v.items():
            if s['n_slots']:
                print(f"{group:9s} vs {arm:10s} slots {s['n_slots']:3d}  argmax {s['argmax_agree_nontie']}/{s['n_nontie']} non-tie  "
                      f"max|dp| {s['max_dp']:.3g}  p95 {s['p95_dp']:.3g}  ({s['max_dp_slot']})")


if __name__ == '__main__':
    main()
