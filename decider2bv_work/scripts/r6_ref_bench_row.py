"""Round 6: the public reference readout (publish/reference/decider_litert.py, CompiledModel CPU) timed for ONE decision
in a fresh process: constructor wall (bundle hash + section cache lookup + model loads; the section folder and the
decoder's XNNPACK weight cache already exist = a warm start), then the wall of the first decide() on the fixture row,
then a second identical decide() in the same process (informational). Run under /usr/bin/time -l by
scripts/r6_mac_bench.py.

    out/venv-ref/bin/python -B scripts/r6_ref_bench_row.py --bundle B --out J [--row game_pong_atari_up]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'publish/reference'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--row', default='game_pong_atari_up')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--cache-dir', default=str(ROOT / 'out/ref_cache'))
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    t0 = time.monotonic()
    from PIL import Image
    from decider_litert import DeciderLiteRT
    fx = json.loads((ROOT / 'publish/reference/fixtures/fixtures.json').read_text())
    row = next(r for r in fx['rows'] if r['id'] == args.row)
    image = Image.open(ROOT / 'publish/reference/fixtures' / row['image']['file'])
    image.load()
    t1 = time.monotonic()
    d = DeciderLiteRT(args.bundle, threads=args.threads, cache_dir=args.cache_dir)
    t2 = time.monotonic()
    first = d.decide(image, row['context'], row['questions'])
    t3 = time.monotonic()
    second = d.decide(image, row['context'], row['questions'])
    t4 = time.monotonic()
    rec = dict(status='DONE', pid=os.getpid(), bundle=os.path.relpath(Path(args.bundle).resolve(), ROOT), row=args.row,
               threads=args.threads, started=time.strftime('%Y-%m-%d %H:%M:%S %Z'),
               import_and_image_s=t1 - t0, constructor_wall_s=t2 - t1, model_load_s=d.load_seconds,
               first_decide_wall_s=t3 - t2, second_decide_wall_s=t4 - t3, answer=first, second_answer=second,
               same_answer=first == second, process_wall_s=time.monotonic() - t0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rec, indent=1) + '\n')
    print('REF_BENCH_ROW', rec['first_decide_wall_s'], first[0]['choice'], flush=True)


if __name__ == '__main__':
    main()
