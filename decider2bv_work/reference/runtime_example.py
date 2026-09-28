"""The LiteRT-LM runtime path: one image and one question in, the answer letter out.

    pip install litert-lm==0.17.1 tokenizers pillow
    python -B reference/runtime_example.py --bundle decider-2b-vision_fp16-int8vocab.litertlm --image frame.png \
        --context "You play Pong (Atari) and control the right paddle. ..." \
        --question "What should you do right now?" --options "move paddle up" "move paddle down" "stay"

Scope of this path (what was checked): the image comes first, there is exactly ONE question, and the answer is the
first generated token with greedy decoding. The runtime returns text, not option probabilities, and its greedy token
is the most likely token over the whole vocabulary, not only over the option letters (the two agree on every
single-question published fixture row, but that is not guaranteed). For probabilities, several questions in one
request, or a text-only request, use decider_litert.py.

The text is built with the upstream prompt code (vendored), and the image is resized to 256 x 256 with PIL BICUBIC
before it is handed to the runtime, as in the conversion checks.
"""
import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bundle_cache import read_tokenizer_json                           # noqa: E402
from decider_litert import HFTokenizer, request_text, image_256        # noqa: E402

LETTERS = 'ABCDEFGHIJ'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--image', required=True)
    ap.add_argument('--context', required=True)
    ap.add_argument('--question', required=True)
    ap.add_argument('--options', nargs='+', required=True)
    ap.add_argument('--backend', choices=('cpu', 'gpu'), default='cpu')
    ap.add_argument('--cache-dir', default=None, help='runtime cache folder (default: next to the bundle); ":nocache" for none')
    args = ap.parse_args()
    import litert_lm
    from litert_lm import Backend, Engine, SamplerConfig

    tok = HFTokenizer(json_text=read_tokenizer_json(args.bundle))
    text, nopts, _ = request_text(tok, args.context, [dict(question=args.question, options=args.options)])
    with tempfile.TemporaryDirectory() as tmp:
        png = os.path.join(tmp, 'image_256.png')
        image_256(args.image).save(png, format='PNG')
        backend = Backend.CPU() if args.backend == 'cpu' else Backend.GPU()
        t0 = time.monotonic()
        engine = Engine(args.bundle, backend=backend, vision_backend=backend, max_num_tokens=4096, max_num_images=1,
                        cache_dir=args.cache_dir)
        t1 = time.monotonic()
        conv = engine.create_conversation(sampler_config=SamplerConfig(top_k=1), max_output_tokens=1)
        message = dict(role='user', content=[dict(type='image', path=png), dict(type='text', text=text)])
        first = ''
        for chunk in conv.send_message_async(message):
            c = chunk.get('content', '')
            if isinstance(c, list):
                c = ''.join(p.get('text', '') for p in c if isinstance(p, dict))
            first += c
        t2 = time.monotonic()
        conv.close()
        engine.close()
    letter = first.strip()
    if letter in LETTERS[:nopts[0]]:
        print(f'answer: ({letter}) {args.options[LETTERS.index(letter)]}')
    else:
        print(f'the greedy token {first!r} is not one of the option letters; read probabilities with decider_litert.py')
    print(f'engine {t1 - t0:.1f} s, request {t2 - t1:.2f} s', file=sys.stderr)


if __name__ == '__main__':
    main()
