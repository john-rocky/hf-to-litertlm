"""Round 3: ONE row through the released LiteRT-LM 0.17.1 Python API (PyPI, out/venv-readout) in a fresh process.

Engine(bundle, backend, vision_backend, max_num_tokens 4096, max_num_images 1, cache_dir = one dir per bundle,
enable_benchmark) -> one Conversation (sampler top_k 1 = greedy, max_output_tokens 3) -> one user message:
  image rows : content [image part (the oracle's own 256 x 256 PNG, out/fixtures_resized/g256/<row>.png),
                        text part (the oracle's input_ids after <|vision_end|>, decoded)]
  no-image   : content [text part (the oracle's input_ids, decoded)]
Records raw facts only; the judgement against the graph readout is made by scripts/runtime_r3.py:
  the runtime's own tokenization of the text part, its render of the message (render_message_to_string, before
  sending), every streamed text chunk in order, get_benchmark_info() (prefill / decode token counts, TTFT, init),
  token_count after the turn (includes the generated tokens; informational), wall times.
One process per row: running-state models must not share an engine across conversations
(memory engine-crossconv-dedup-linear-state). stdin is expected to be /dev/null (driver).

    out/venv-readout/bin/python -B scripts/runtime_row_r3.py --row ROW --backend cpu|gpu [--vision-backend cpu|gpu]
        --cache-dir DIR --out results/runtime_r3_rows/<backend>/<row>.json [--bundle out/bundle_r4/<v>/<name>.litertlm]

Round 4 passes --bundle (default: the round-3 fp16 bundle); nothing else changes.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import ROOT, read_json, sha256_file                      # noqa: E402
from r3_rows import IMG_RENDER, oracle_rows                           # noqa: E402

BUNDLE = ROOT / 'out/bundle/decider-2b-vision_fp16.litertlm'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--row', required=True)
    ap.add_argument('--backend', choices=('cpu', 'gpu'), required=True)
    ap.add_argument('--vision-backend', choices=('cpu', 'gpu'), default=None)
    ap.add_argument('--cache-dir', required=True)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--max-output-tokens', type=int, default=3)
    ap.add_argument('--out', required=True)
    ap.add_argument('--bundle', default=str(BUNDLE))
    args = ap.parse_args()
    bundle = (ROOT / args.bundle).resolve()
    t0 = time.monotonic()
    rec = dict(status='RUNNING', row_id=args.row, backend=args.backend, vision_backend=args.vision_backend or args.backend,
               bundle=dict(path=os.path.relpath(bundle, ROOT), bytes=os.path.getsize(bundle)), cache_dir=args.cache_dir, pid=os.getpid(),
               max_output_tokens=args.max_output_tokens, sampler=dict(top_k=1))
    try:
        import importlib.metadata as md
        import litert_lm
        from litert_lm import Backend, Engine, SamplerConfig
        from tokenizers import Tokenizer
        rec['versions'] = {p: md.version(p) for p in ('litert-lm', 'litert-lm-api')}
        rec['litert_lm_module'] = os.path.dirname(litert_lm.__file__)
        row = oracle_rows(read_json)[args.row]
        tk = Tokenizer.from_file(str(ROOT / 'out/src/decider-2b-vision/tokenizer.json'))
        text = tk.decode(row['text_ids'], skip_special_tokens=False)
        content = [dict(type='text', text=text)]
        expected_render = text
        if row['image']:
            png = ROOT / f'out/fixtures_resized/g256/{args.row}.png'
            rec['image'] = dict(path=os.path.relpath(png, ROOT), sha256=sha256_file(png),
                                oracle_sha256=row['forward']['image_input']['sha256_file'])
            assert rec['image']['sha256'] == rec['image']['oracle_sha256']
            content = [dict(type='image', path=str(png))] + content
            expected_render = IMG_RENDER + text
        rec['text'] = text
        rec['oracle_n_input_ids'] = len(row['input_ids'])
        rec['oracle_text_ids'] = row['text_ids']

        def backend(name):
            return Backend.CPU(thread_count=args.threads) if name == 'cpu' else Backend.GPU()

        os.makedirs(args.cache_dir, exist_ok=True)
        t1 = time.monotonic()
        engine = Engine(str(bundle), backend=backend(args.backend), vision_backend=backend(rec['vision_backend']),
                        max_num_tokens=4096, max_num_images=1, cache_dir=args.cache_dir, enable_benchmark=True)
        rec['engine_create_seconds'] = time.monotonic() - t1
        rec['engine_bos_token_id'] = engine.bos_token_id
        rec['engine_eos_token_ids'] = engine.eos_token_ids
        rt_ids = engine.tokenize(text)
        rec['runtime_text_ids'] = rt_ids
        rec['runtime_text_ids_equal_oracle'] = rt_ids == row['text_ids']
        conv = engine.create_conversation(sampler_config=SamplerConfig(top_k=1), max_output_tokens=args.max_output_tokens)
        message = dict(role='user', content=content)
        rec['message'] = message
        rendered = conv.render_message_to_string(message)
        rec['runtime_render'] = rendered
        rec['runtime_render_equal_expected'] = rendered == expected_render
        rec['token_count_before'] = conv.token_count
        chunks, times = [], []
        t2 = time.monotonic()
        for m in conv.send_message_async(message, max_output_tokens=args.max_output_tokens):
            c = m.get('content', '')
            if isinstance(c, list):
                c = ''.join(p.get('text', '') for p in c if isinstance(p, dict))
            chunks.append(c)
            times.append(time.monotonic() - t2)
        rec['chunks'] = chunks
        rec['chunk_seconds'] = times
        rec['response_text'] = ''.join(chunks)
        first = next((c for c in chunks if c), None)
        rec['first_chunk'] = first
        rec['first_chunk_runtime_ids'] = engine.tokenize(first) if first else None
        b = conv.get_benchmark_info()
        rec['benchmark'] = dict(init_time_in_second=b.init_time_in_second, time_to_first_token_in_second=b.time_to_first_token_in_second,
                                last_prefill_token_count=b.last_prefill_token_count,
                                last_prefill_tokens_per_second=b.last_prefill_tokens_per_second,
                                last_decode_token_count=b.last_decode_token_count,
                                last_decode_tokens_per_second=b.last_decode_tokens_per_second)
        rec['token_count_after'] = conv.token_count
        conv.close()
        engine.close()
        rec['status'] = 'DONE'
    except BaseException as e:  # noqa: BLE001
        rec['status'] = 'ERROR'
        rec['error_type'] = type(e).__name__
        rec['error'] = str(e)
        rec['traceback'] = traceback.format_exc()
        print(rec['traceback'], file=sys.stderr, flush=True)
    rec['wall_seconds'] = time.monotonic() - t0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
    print('RUNTIME_ROW', args.row, args.backend, rec['status'], repr(rec.get('first_chunk')),
          rec.get('benchmark', {}).get('last_prefill_token_count'), f"{rec['wall_seconds']:.0f}s", flush=True)
    sys.exit(0 if rec['status'] == 'DONE' else 1)


if __name__ == '__main__':
    main()
