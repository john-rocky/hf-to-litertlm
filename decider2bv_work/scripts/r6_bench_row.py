"""Round 6: ONE decision through the released LiteRT-LM 0.17.1 Python API in a fresh process, timed the way a user
feels it. Run under /usr/bin/time -l by scripts/r6_mac_bench.py (peak RSS), stdin /dev/null.

  engine    Engine(bundle, backend, vision_backend = same, max_num_tokens 4096, max_num_images 1,
            cache_dir ':nocache' (= litert-lm --cache no), enable_benchmark) -> wall of the constructor
  request   the public fixture row's image_256 PNG + its text built by the reference code (publish/reference,
            upstream build() -> decode) with the bundle's own tokenizer; one Conversation, greedy (top_k 1),
            max_output_tokens 1; wall from send to the first streamed text (ttft_wall_s) and to the end
  runtime   get_benchmark_info(): init, TTFT, prefill token count and tok/s
The first token must be the expected letter (the fp32 upstream full-vocabulary top-1 of that row, stored in the
public fixtures); the driver also counts 'Validation error' lines in this process's stderr.

    out/venv-readout/bin/python -B scripts/r6_bench_row.py --bundle B --backend cpu|gpu --row game_pong_atari_up --out J
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'publish/reference'))
from bundle_cache import read_tokenizer_json, sha256                  # noqa: E402
from decider_litert import HFTokenizer, request_text, N_IMG           # noqa: E402

LETTERS = 'ABCDEFGHIJ'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--backend', choices=('cpu', 'gpu'), required=True)
    ap.add_argument('--row', default='game_pong_atari_up')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--cache-dir', default=':nocache')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    t_start = time.monotonic()
    rec = dict(status='RUNNING', pid=os.getpid(), bundle=os.path.relpath(Path(args.bundle).resolve(), ROOT),
               backend=args.backend, threads=args.threads if args.backend == 'cpu' else None, cache_dir=args.cache_dir,
               row=args.row, started=time.strftime('%Y-%m-%d %H:%M:%S %Z'))
    try:
        import importlib.metadata as md
        import litert_lm
        from litert_lm import Backend, Engine, SamplerConfig
        rec['versions'] = {p: md.version(p) for p in ('litert-lm', 'litert-lm-api')}
        fx = json.loads((ROOT / 'publish/reference/fixtures/fixtures.json').read_text())
        row = next(r for r in fx['rows'] if r['id'] == args.row)
        assert row['image'] is not None and len(row['questions']) == 1
        ref = row['oracle']['g256_mrope']
        expected = LETTERS[fx['letters']['ids'].index(ref['slots'][0]['vocab_top1_id'])]
        tok = HFTokenizer(json_text=read_tokenizer_json(args.bundle))
        text, nopts, _ = request_text(tok, row['context'], row['questions'])
        assert tok.encode(text, add_special_tokens=True) == ref['input_ids'][N_IMG + 2:], 'text ids differ from upstream'
        png = ROOT / 'publish/reference/fixtures' / row['image_256']['file']
        assert sha256(png) == row['image_256']['sha256']
        rec.update(expected_letter=expected, prompt_tokens_upstream=len(ref['input_ids']), image_256=str(png.relative_to(ROOT)))

        backend = Backend.CPU(thread_count=args.threads) if args.backend == 'cpu' else Backend.GPU()
        t0 = time.monotonic()
        engine = Engine(args.bundle, backend=backend, vision_backend=backend, max_num_tokens=4096, max_num_images=1,
                        cache_dir=args.cache_dir, enable_benchmark=True)
        rec['engine_create_wall_s'] = time.monotonic() - t0
        conv = engine.create_conversation(sampler_config=SamplerConfig(top_k=1), max_output_tokens=1)
        message = dict(role='user', content=[dict(type='image', path=str(png)), dict(type='text', text=text)])
        chunks, times = [], []
        t1 = time.monotonic()
        for m in conv.send_message_async(message):
            c = m.get('content', '')
            if isinstance(c, list):
                c = ''.join(p.get('text', '') for p in c if isinstance(p, dict))
            chunks.append(c)
            times.append(time.monotonic() - t1)
        rec['request_wall_s'] = time.monotonic() - t1
        first_i = next((i for i, c in enumerate(chunks) if c), None)
        rec['ttft_wall_s'] = times[first_i] if first_i is not None else None
        rec['response'] = ''.join(chunks)
        rec['first_token_ok'] = rec['response'].strip() == expected
        b = conv.get_benchmark_info()
        rec['benchmark'] = dict(init_time_in_second=b.init_time_in_second, time_to_first_token_in_second=b.time_to_first_token_in_second,
                                last_prefill_token_count=b.last_prefill_token_count,
                                last_prefill_tokens_per_second=b.last_prefill_tokens_per_second,
                                last_decode_token_count=b.last_decode_token_count,
                                last_decode_tokens_per_second=b.last_decode_tokens_per_second)
        conv.close()
        engine.close()
        rec['status'] = 'DONE'
    except BaseException as e:  # noqa: BLE001
        rec['status'] = 'ERROR'
        rec['error'] = f'{type(e).__name__}: {e}'
        rec['traceback'] = traceback.format_exc()
        print(rec['traceback'], file=sys.stderr, flush=True)
    rec['process_wall_s'] = time.monotonic() - t_start
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rec, indent=1) + '\n')
    print('BENCH_ROW', args.backend, rec['status'], repr(rec.get('response')), rec.get('ttft_wall_s'), flush=True)
    sys.exit(0 if rec['status'] == 'DONE' else 1)


if __name__ == '__main__':
    main()
