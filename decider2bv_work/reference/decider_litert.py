"""decider-2b-vision on LiteRT: option probabilities for lettered questions about one image, read from the bundle.

    python -B reference/decider_litert.py --bundle decider-2b-vision_fp16-int8vocab.litertlm --request request.json
    request.json = {"image": "frame.png" or null, "context": "...",
                    "questions": [{"question": "...", "options": ["...", "..."]}, ...]}

Python:
    from decider_litert import DeciderLiteRT
    d = DeciderLiteRT('decider-2b-vision_fp16-int8vocab.litertlm')
    d.decide(image_or_None, context, [{'question': ..., 'options': [...]}])
    -> [{'choice': ..., 'confidence': ..., 'probs': {option: p}}, ...]

What it computes (the contract the conversion was checked against, upstream decider/vision.py prepare + slot_logits):
  image     RGB, resized to 256 x 256 with PIL BICUBIC, [1, 256, 256, 3] float in [0, 1] -> vision encoder -> adapter
            -> 64 embeddings (the encoder applies (x - 0.5) / 0.5 itself).
  text      upstream build() (vendored, option order kept) -> decode -> encode again, as upstream prepare() does;
            ids = <|vision_start|> + 64 x <|image_pad|> + <|vision_end|> + text (no BOS, no role markers).
  slots     upstream rule: a " (" token after ":" with "Answer" within the 5 tokens before it; one per question.
  positions image requests 0, 1, 2, ...; text-only requests start at 65. The decoder derives the upstream 3-channel
            positions from this 1-D position in the graph; from 65 on all channels move by the same constant, so a
            text-only request sees the same relative positions as upstream.
  readout   state zeroed per request; prefill with the largest prefill signatures that fit exactly (no padding);
            at each slot, decode that token once and read the letter logits A.. up to the option count; softmax at
            T = 1.
The default backend is CPU (the checked path). backend='gpu' runs the decoder on the GPU with one padded prefill
chunk per slot (fresh state per slot, slots up to 1024 tokens); the embedder and the vision model stay on the CPU.
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tokenizers import Tokenizer
from ai_edge_litert.compiled_model import CompiledModel
from ai_edge_litert.hardware_accelerator import HardwareAccelerator
from ai_edge_litert.options import Options, CpuOptions

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bundle_cache import unpack_bundle                                  # noqa: E402
from decider_vendored.infer import Example, Q                          # noqa: E402
from decider_vendored.prompt import build, letter_ids, MAX_OPTIONS     # noqa: E402

G = 256
N_IMG = 64
TEXT_START = N_IMG + 1
LADDER = [1024, 256, 64, 16, 4, 1]
MAX_CTX_TOKENS = 1536                   # upstream prepare() default
VSTART, IMAGE_PAD, VEND = '<|vision_start|>', '<|image_pad|>', '<|vision_end|>'


class _NoShuffle:
    """Keeps the option order as given (decider/vision.py _NoShuffle)."""

    def shuffle(self, x):
        pass

    def sample(self, xs, k):
        return xs[:k]


class HFTokenizer:
    """The two tokenizer calls the vendored prompt code makes, on the bundle's own tokenizer.json."""

    def __init__(self, path=None, json_text=None):
        self.inner = Tokenizer.from_str(json_text) if json_text is not None else Tokenizer.from_file(str(path))

    def encode(self, text, add_special_tokens=False):
        return self.inner.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids):
        return self.inner.decode(ids, skip_special_tokens=False)

    def token_id(self, token):
        i = self.inner.token_to_id(token)
        assert i is not None, token
        return i


def request_text(tok, context, questions, max_ctx_tokens=MAX_CTX_TOKENS):
    """The request text after the image, as upstream prepare() makes it: build() with the option order kept, then
    decode. Returns (text, option counts, Q list)."""
    qs = [q if isinstance(q, Q) else Q(q['question'], list(q['options']), 0) for q in questions]
    if not qs:
        raise ValueError('at least one question is required')
    for q in qs:
        if not 2 <= len(q.options) <= MAX_OPTIONS:
            raise ValueError(f'2..{MAX_OPTIONS} options required, got {len(q.options)}')
    b = build(Example(context, qs), tok, _NoShuffle(), max_ctx_tokens=max_ctx_tokens)
    return tok.decode(b['ids']), b['nopts'], qs


def to_pil(x):
    """decider/vision.py to_pil, plus a file path."""
    if isinstance(x, Image.Image):
        return x.convert('RGB')
    if isinstance(x, (bytes, bytearray)):
        return Image.open(io.BytesIO(x)).convert('RGB')
    if isinstance(x, (str, Path)):
        return Image.open(x).convert('RGB')
    return Image.fromarray(x).convert('RGB')


def image_256(image):
    return to_pil(image).resize((G, G), Image.BICUBIC)


def pixels01(im256):
    assert im256.size == (G, G) and im256.mode == 'RGB'
    return (np.asarray(im256, dtype=np.float32) / np.float32(255.0))[None]     # [1, 256, 256, 3]


def softmax64(z):
    z = np.asarray(z, np.float64)
    e = np.exp(z - z.max())
    return e / e.sum()


def _sha(arr):
    return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.float32).tobytes()).hexdigest()


@contextlib.contextmanager
def diagnostics_to_stderr():
    """Keep native runtime diagnostics out of stdout."""
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        with contextlib.redirect_stdout(sys.stderr):
            yield
    finally:
        sys.stderr.flush()
        os.dup2(saved, 1)
        os.close(saved)


def _run(model, key, inputs, outputs, feeds):
    ibuf, obuf = {}, {}
    try:
        for name, arr in feeds.items():
            assert list(arr.shape) == list(inputs[name]['shape']), (key, name, arr.shape, inputs[name]['shape'])
            b = model.create_input_buffer_by_name(key, name)
            ibuf[name] = b
            b.write(np.ascontiguousarray(arr))
        obuf = {name: model.create_output_buffer_by_name(key, name) for name in outputs}
        model.run_by_name(key, ibuf, obuf)
        return {name: np.array(obuf[name].read(int(np.prod(d['shape'])), d['dtype']), dtype=d['dtype'], copy=True).reshape(d['shape'])
                for name, d in outputs.items()}
    finally:
        for b in [*ibuf.values(), *obuf.values()]:
            b.destroy()


class _Single:
    """A one-signature model (embedder, vision encoder, vision adapter) on the CPU."""

    def __init__(self, path, threads):
        self.model = CompiledModel.from_file(str(path), options=Options(
            hardware_accelerators=HardwareAccelerator.CPU, cpu_options=CpuOptions(num_threads=threads)))
        sigs = self.model.get_signature_list()
        assert len(sigs) == 1, sigs
        self.key = next(iter(sigs))
        self.inp = self.model.get_input_tensor_details(self.key)
        self.out = self.model.get_output_tensor_details(self.key)
        assert len(self.inp) == 1 and len(self.out) == 1

    def __call__(self, x):
        (iname, idet), = self.inp.items()
        (oname, odet), = self.out.items()
        x = np.ascontiguousarray(x, dtype=idet['dtype']).reshape(idet['shape'])
        return _run(self.model, self.key, self.inp, self.out, {iname: x})[oname]


class _Decoder:
    """prefill_1024 / 256 / 64 / 16 / 4 / 1 + decode, 48 state tensors carried between calls."""

    def __init__(self, path, backend, threads, weight_cache):
        if backend == 'gpu':
            from ai_edge_litert.gpu_options import GpuOptions
            opts = Options(hardware_accelerators=HardwareAccelerator.GPU, gpu_options=GpuOptions(enforce_f32=True))
        else:
            cpu = dict(num_threads=threads)
            if weight_cache:
                cpu['xnnpack_weight_cache_path'] = str(weight_cache)
            opts = Options(hardware_accelerators=HardwareAccelerator.CPU, cpu_options=CpuOptions(**cpu))
        t0 = time.monotonic()
        self.model = CompiledModel.from_file(str(path), options=opts)
        self.load_seconds = time.monotonic() - t0
        self.backend = backend
        self.signatures = sorted(self.model.get_signature_list())
        assert set(self.signatures) == {'decode', *(f'prefill_{n}' for n in LADDER)}, self.signatures
        self.inputs = {k: self.model.get_input_tensor_details(k) for k in self.signatures}
        self.outputs = {k: self.model.get_output_tensor_details(k) for k in self.signatures}
        self.states = {n: d for n, d in self.inputs['decode'].items() if n not in ('embeddings', 'input_pos', 'mask')}
        assert len(self.states) == 48, len(self.states)
        self.cache_length = int(self.inputs['decode']['mask']['shape'][-1])

    def zero_states(self):
        return {n: np.zeros(d['shape'], dtype=d['dtype']) for n, d in self.states.items()}

    def step(self, key, emb, positions, first_pos, states):
        inp = self.inputs[key]
        cols = np.arange(self.cache_length)[None, :]
        allowed = (cols <= positions[:, None]) & (cols >= first_pos)
        mask = np.where(allowed, np.float32(0.0), np.float32(-1e30)).reshape(inp['mask']['shape'])
        feeds = dict(states, embeddings=np.ascontiguousarray(emb, dtype=np.float32).reshape(inp['embeddings']['shape']),
                     input_pos=positions.astype(inp['input_pos']['dtype']).reshape(inp['input_pos']['shape']), mask=mask)
        vals = _run(self.model, key, inp, self.outputs[key], feeds)
        finite = all(np.isfinite(v).all() for v in vals.values())
        return {n: vals[n] for n in self.states}, vals.get('logits'), finite

    @staticmethod
    def chunk_plan(start, stop):
        """Largest prefill signatures that fit exactly into [start, stop) (no padding)."""
        plan, p = [], start
        for length in LADDER:
            while stop - p >= length:
                plan.append((f'prefill_{length}', p, length))
                p += length
        assert p == stop
        return plan

    def slot_logits_exact(self, emb, slots, first_pos):
        """CPU contract: one pass over the request, exact-fit prefill chunks, one decode per slot."""
        S = emb.shape[0]
        pos = np.arange(first_pos, first_pos + S, dtype=np.int64)
        assert pos[-1] < self.cache_length, f'request of {S} tokens exceeds the {self.cache_length}-token cache'
        states = self.zero_states()
        cur, out, finite = 0, [], True
        for s in slots:
            for key, a, n in self.chunk_plan(cur, s):
                states, lg, ok = self.step(key, emb[a:a + n], pos[a:a + n], first_pos, states)
                finite &= ok
            states, lg, ok = self.step('decode', emb[s:s + 1], pos[s:s + 1], first_pos, states)
            finite &= ok
            out.append(lg.reshape(-1))
            cur = s + 1
        return out, finite

    def slot_logits_padded(self, emb, slots, first_pos):
        """GPU rule: per slot, a fresh state, ONE padded prefill chunk of the tokens before it, then one decode.
        Pad rows: embedding 0, position 0, mask -1e30 everywhere."""
        out, finite = [], True
        cols = np.arange(self.cache_length)[None, :]
        for s in slots:
            if s > max(LADDER):
                raise ValueError(f'slot at token {s}: the GPU rule covers slots up to {max(LADDER)} tokens; use the CPU')
            states = self.zero_states()
            if s > 0:
                size = min(x for x in LADDER if x >= s)
                key = f'prefill_{size}'
                inp = self.inputs[key]
                e = np.zeros((size, emb.shape[1]), np.float32)
                e[:s] = emb[:s]
                pos = np.zeros(size, np.int64)
                pos[:s] = np.arange(first_pos, first_pos + s)
                allowed = (cols <= pos[:, None]) & (cols >= first_pos) & (np.arange(size) < s)[:, None]
                mask = np.where(allowed, np.float32(0.0), np.float32(-1e30)).reshape(inp['mask']['shape'])
                vals = _run(self.model, key, inp, self.outputs[key], dict(
                    states, embeddings=e.reshape(inp['embeddings']['shape']),
                    input_pos=pos.astype(inp['input_pos']['dtype']).reshape(inp['input_pos']['shape']), mask=mask))
                finite &= all(np.isfinite(v).all() for v in vals.values())
                states = {n: vals[n] for n in self.states}
            p1 = np.array([first_pos + s], np.int64)
            states, lg, ok = self.step('decode', emb[s:s + 1], p1, first_pos, states)
            finite &= ok
            out.append(lg.reshape(-1))
        return out, finite


class DeciderLiteRT:
    def __init__(self, bundle, backend='cpu', threads=8, cache_dir=None, weight_cache=True):
        assert backend in ('cpu', 'gpu'), backend
        self.bundle = Path(bundle)
        self.folder, self.paths, self.record = unpack_bundle(bundle, cache_dir)
        self.tok = HFTokenizer(self.paths['tokenizer'])
        self.letters = letter_ids(self.tok)
        self.slot_tok = self.tok.encode(' (')[-1]
        self.colon = self.tok.encode(':')[-1]
        self.answer_tok = self.tok.encode('Answer')[0]
        self.vstart, self.pad, self.vend = (self.tok.token_id(t) for t in (VSTART, IMAGE_PAD, VEND))
        t0 = time.monotonic()
        self.embedder = _Single(self.paths['embedder'], threads)
        self.encoder = _Single(self.paths['vision_encoder'], threads)
        self.adapter = _Single(self.paths['vision_adapter'], threads)
        cache = self.folder / 'prefill_decode.xnnpack_cache' if (weight_cache and backend == 'cpu') else None
        self.decoder = _Decoder(self.paths['prefill_decode'], backend, threads, cache)
        self.load_seconds = time.monotonic() - t0
        self.backend = backend
        self._emb_cache = {}

    def prepare(self, image, context, questions, max_ctx_tokens=MAX_CTX_TOKENS):
        """Token ids, answer slots and option counts, as upstream prepare() builds them for one request."""
        txt, nopts, qs = request_text(self.tok, context, questions, max_ctx_tokens)
        prefix = VSTART + IMAGE_PAD * N_IMG + VEND if image is not None else ''
        ids = self.tok.encode(prefix + txt, add_special_tokens=True)
        n_pad = ids.count(self.pad)
        if image is not None:
            assert ids[:N_IMG + 2] == [self.vstart] + [self.pad] * N_IMG + [self.vend] and n_pad == N_IMG, 'image block'
        else:
            assert n_pad == 0 and self.vstart not in ids, 'the text contains image tokens'
        slots = [i for i in range(2, len(ids))
                 if ids[i] == self.slot_tok and ids[i - 1] == self.colon and self.answer_tok in ids[max(0, i - 5):i]]
        assert len(slots) == len(qs), (len(slots), len(qs))
        return dict(ids=ids, slot_idx=slots, nopts=nopts, text=txt, first_pos=0 if image is not None else TEXT_START,
                    options=[q.options for q in qs])

    def _embed(self, ids):
        rows = []
        for t in ids:
            if t not in self._emb_cache:
                self._emb_cache[t] = self.embedder(np.array([[t]], dtype=np.int32)).reshape(-1).copy()
            rows.append(self._emb_cache[t])
        return np.stack(rows)

    def vision(self, image):
        return self.adapter(self.encoder(pixels01(image_256(image)))).reshape(N_IMG, -1)

    def readout(self, image, context, questions, max_ctx_tokens=MAX_CTX_TOKENS):
        """Everything decide() uses, plus hashes for bit-level comparisons."""
        t0 = time.monotonic()
        prep = self.prepare(image, context, questions, max_ctx_tokens)
        emb = self._embed(prep['ids'])
        text_sha = _sha(emb)
        vis_sha = None
        if image is not None:
            vis = self.vision(image)
            vis_sha = _sha(vis)
            emb = emb.copy()
            emb[1:1 + N_IMG] = vis
        run = self.decoder.slot_logits_padded if self.backend == 'gpu' else self.decoder.slot_logits_exact
        logits, finite = run(emb, prep['slot_idx'], prep['first_pos'])
        if not finite:
            raise RuntimeError('non-finite model output')
        slots = []
        for lg, n in zip(logits, prep['nopts']):
            letters = lg[self.letters[:n]]
            slots.append(dict(nopts=n, letter_logits=[float(x) for x in letters], probs=softmax64(letters).tolist(),
                              logits_sha256=_sha(lg), vocab_top1_id=int(lg.argmax())))
        return dict(prep, slots=slots, text_embeddings_sha256=text_sha, vision_sha256=vis_sha,
                    wall_seconds=time.monotonic() - t0)

    def decide(self, image, context, questions, max_ctx_tokens=MAX_CTX_TOKENS):
        r = self.readout(image, context, questions, max_ctx_tokens)
        out = []
        for opts, s in zip(r['options'], r['slots']):
            p = s['probs']
            j = int(np.argmax(p))
            out.append(dict(choice=opts[j], confidence=p[j], probs={o: pi for o, pi in zip(opts, p)}))
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--request', required=True, help='JSON: {"image": path or null, "context": str, "questions": [...]}')
    ap.add_argument('--backend', choices=('cpu', 'gpu'), default='cpu')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--cache-dir', default=None, help='default: <repo>/.cache/readout (or $DECIDER_LITERT_CACHE)')
    ap.add_argument('--no-weight-cache', action='store_true', help='do not write the CPU weight cache next to the sections')
    args = ap.parse_args()
    req = json.loads(Path(args.request).read_text())
    image = req.get('image')
    if image is not None:
        image = (Path(args.request).parent / image) if not os.path.isabs(image) else Path(image)
    with diagnostics_to_stderr():
        d = DeciderLiteRT(args.bundle, backend=args.backend, threads=args.threads, cache_dir=args.cache_dir,
                          weight_cache=not args.no_weight_cache)
        answers = d.decide(image, req['context'], req['questions'])
    print(json.dumps(dict(model=Path(args.bundle).name, backend=args.backend, answers=answers), ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
