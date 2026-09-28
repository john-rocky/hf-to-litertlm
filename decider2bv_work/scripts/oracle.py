"""HF fp32 CPU oracle for decider-2b-vision, three arms (round 1: contract cost before any conversion).

The readout is the checkpoint's own decider/vision.py (VisionDecisionModel.prepare + slot_logits), imported
from the snapshot, untouched. One row = one forward (batch 1, no padding). Arms:

  author      original image -> prepare() (the processor's dynamic resolution), full M-RoPE
  g{G}_mrope  original image -> PIL BICUBIC GxG -> prepare(), full M-RoPE
  g{G}_pos1d  same pixels as g{G}_mrope; Qwen3_5Model.get_rope_index replaced by arange(S) on all three
              channels with rope delta 0 (the only producer of text positions on this path: forward ->
              compute_3d_position_ids -> self.get_rope_index; read in transformers 5.17.0 modeling_qwen3_5.py)

Instrumentation only observes: a forward-pre-hook on the text rotary embedding records the position ids it
receives, a forward hook on the backbone keeps last_hidden_state for the full-vocab top-1 at each slot, and the
gated-delta / conv module globals are wrapped with counting delegates.

    out/venv-oracle/bin/python -B scripts/oracle.py [--threads 8] [--rows a,b] [--out fixtures/oracle_fp32.json]
    round 2 (added rows only): ... --fixtures fixtures/fixtures_v2.json --rows <added ids> --arms author,g256_mrope,g256_pos1d --out fixtures/oracle_fp32_v2add.json
"""
import os
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
import argparse
import functools
import importlib.util
import inspect
import platform
import sys
import time
import traceback
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from common import (ROOT, SNAPSHOT, REVISION, MODEL_SHA256, ARMS, TIE_GAP, write_json, read_json,
                    sha256_file, sha256_rgb, sha256_bytes)

sys.path.insert(0, str(SNAPSHOT))
import decider.vision as decider_vision                                    # noqa: E402  (checkpoint's own package)
from decider.vision import VisionDecisionModel                             # noqa: E402
from decider.infer import Example, Q                                       # noqa: E402
from decider.prompt import LETTERS, MAX_OPTIONS                            # noqa: E402
import transformers                                                        # noqa: E402
from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling       # noqa: E402

IMAGE_ID, VSTART_ID, VEND_ID = 248056, 248053, 248054
KERNELS = ['torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule', 'causal_conv1d_fn', 'causal_conv1d_update']


def hand_pixel_values(image, patch=16, merge=2, temporal=2):
    """(x/255 - 0.5)/0.5 in the processor's patch order (Qwen2VLImageProcessor.patchify), temporal duplicated."""
    x = np.asarray(image.convert('RGB'), dtype=np.float64)
    x = (x / 255.0 - 0.5) / 0.5
    H, W, C = x.shape
    gh, gw = H // patch, W // patch
    x = x.transpose(2, 0, 1).reshape(C, gh // merge, merge, patch, gw // merge, merge, patch)
    x = x.transpose(1, 4, 2, 5, 0, 3, 6)                                   # gh/m, gw/m, m, m, C, p, p
    x = np.repeat(x[:, :, :, :, :, None, :, :], temporal, axis=5)          # gh/m, gw/m, m, m, C, T, p, p
    return x.reshape(gh * gw, C * temporal * patch * patch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--rows', default='')
    ap.add_argument('--arms', default=','.join(ARMS))
    ap.add_argument('--out', default='fixtures/oracle_fp32.json')
    ap.add_argument('--fixtures', default='fixtures/fixtures.json')   # round 2: fixtures/fixtures_v2.json for the added rows
    args = ap.parse_args()
    start = time.monotonic()
    torch.set_num_threads(args.threads)
    assert torch.get_num_threads() == args.threads
    arms = args.arms.split(',')
    assert all(a in ARMS for a in arms), arms

    fixtures = read_json(args.fixtures)
    rows = fixtures['rows']
    if args.rows:
        want = args.rows.split(',')
        rows = [r for r in rows if r['id'] in want]
        assert len(rows) == len(want), (want, [r['id'] for r in rows])

    load_start = time.monotonic()
    m = VisionDecisionModel(str(SNAPSHOT), dtype=torch.float32, grad_ckpt=False).eval()
    load_seconds = time.monotonic() - load_start
    assert all(p.dtype == torch.float32 and p.device.type == 'cpu' for p in m.parameters())
    tok = m.tok
    backbone = m.lm.model

    counters, kernel_shapes, kernel_functions = Counter(), Counter(), {}
    for name in KERNELS:
        original = getattr(modeling, name)
        fn = inspect.unwrap(original)
        kernel_functions[name] = dict(module=fn.__module__, qualname=fn.__qualname__, file=inspect.getsourcefile(fn),
                                      first_line=inspect.getsourcelines(fn)[1], wrapped_by_hub_decorator=original is not fn)

        def instrument(fn, key):
            @functools.wraps(fn)
            def tracked(*a, **kw):
                counters[key] += 1
                first = a[0] if a else next(iter(kw.values()))
                if isinstance(first, torch.Tensor):
                    kernel_shapes[f'{key}:{list(first.shape)}:{first.dtype}:{first.device}'] += 1
                return fn(*a, **kw)
            return tracked
        setattr(modeling, name, instrument(original, name))

    captured = {}
    backbone.language_model.rotary_emb.register_forward_pre_hook(
        lambda mod, a: captured.__setitem__('pos', a[1].detach().clone()))
    backbone.register_forward_hook(lambda mod, a, out: captured.__setitem__('h', out.last_hidden_state.detach()))

    orig_rope = backbone.get_rope_index                                    # bound method of Qwen3_5Model
    rope_calls = Counter()

    def mrope_delegate(*a, **kw):
        rope_calls['mrope'] += 1
        return orig_rope(*a, **kw)

    def pos1d_rope_index(*a, **kw):
        rope_calls['pos1d'] += 1
        pos3, delta = orig_rope(*a, **kw)                                  # [3, B, S]
        S = pos3.shape[-1]
        seq = torch.arange(S, device=pos3.device, dtype=pos3.dtype).view(1, 1, S).expand_as(pos3).contiguous()
        return seq, torch.zeros_like(delta)

    letters = m.letters.tolist()
    result = dict(
        status='RUNNING', model='Mapika/decider-2b-vision', revision=REVISION, model_sha256=MODEL_SHA256,
        dtype='float32', device=f'{platform.machine()} CPU', torch_threads=args.threads,
        readout='checkpoint decider/vision.py VisionDecisionModel.prepare + slot_logits (untouched), T=1',
        fixture_path=args.fixtures, fixture_sha256=sha256_file(ROOT / args.fixtures), arms=arms, tie_gap=TIE_GAP,
        letters=dict(ids=letters, texts=[tok.decode([i]) for i in letters], expected=list(LETTERS)),
        slot_token=dict(id=m.slot_tok, text=tok.decode([m.slot_tok])), colon_token=m.colon, answer_token=m.answer_tok,
        processor=dict(processor_class=type(m.proc).__name__, image_processor_class=type(m.proc.image_processor).__name__,
                       image_processor_module=type(m.proc.image_processor).__module__,
                       tokenizer_class=type(tok).__name__),
        decider_files={f: sha256_file(SNAPSHOT / 'decider' / f) for f in ('vision.py', 'prompt.py', 'infer.py', 'model.py')},
        decider_vision_module=decider_vision.__file__, load_seconds_contended=load_seconds, forwards=[])
    out_path = args.out

    for row in rows:
        src = None
        if row['image'] is not None:
            src = Image.open(ROOT / row['image']['path'])
            assert sha256_rgb(src) == row['image']['sha256_rgb'], row['id']
            src = src.convert('RGB')
        ex = Example(row['context'], [Q(q['text'], list(q['options']), 0) for q in row['questions']])
        for arm in arms:
            G = None if arm == 'author' else int(arm[1:4])
            pos1d = arm.endswith('pos1d')
            image, image_input = None, None
            if src is not None:
                if G is None:
                    image = src
                else:
                    image = src.resize((G, G), Image.BICUBIC)
                    path = ROOT / f'out/fixtures_resized/g{G}/{row["id"]}.png'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if not path.exists():
                        image.save(path, format='PNG')
                    assert sha256_rgb(Image.open(path)) == sha256_rgb(image), f'{path}: stored pixels differ'
                    image_input = dict(path=str(path.relative_to(ROOT)), width=G, height=G,
                                       sha256_file=sha256_file(path), sha256_rgb=sha256_rgb(image),
                                       resample='PIL.Image.BICUBIC')
            inp = m.prepare([(image, ex)])
            ids = inp['input_ids'][0].tolist()
            backbone.get_rope_index = pos1d_rope_index if pos1d else mrope_delegate
            before = dict(rope_calls)
            captured.clear()
            t0 = time.perf_counter()
            try:
                with torch.no_grad():
                    lg = m.slot_logits(inp)
            finally:
                del backbone.get_rope_index                                # restore the class method
            wall = time.perf_counter() - t0
            rope_used = {k: rope_calls[k] - before.get(k, 0) for k in rope_calls}
            h = captured['h'][0]
            slots = inp['slot_idx'].tolist()
            with torch.no_grad():
                full = F.linear(h[slots], m.lm.lm_head.weight).float()
            slot_recs = []
            for k, (s, q) in enumerate(zip(slots, row['questions'])):
                n = int(inp['nopts'][k])
                raw = lg[k, :n]
                probs = torch.softmax(lg[k], -1)[:n]
                arr = probs.double().numpy()
                order = np.sort(arr)
                gap = float(order[-1] - order[-2])
                top1 = int(full[k].argmax())
                slot_recs.append(dict(
                    k=k, question=q['text'], options=q['options'], expected=q['expected'], nopts=n, slot_index=s,
                    slot_token_id=ids[s], slot_token_text=tok.decode([ids[s]]),
                    letter_logits_fp32=[float(x) for x in raw], probs=[float(x) for x in probs],
                    argmax=int(arr.argmax()), top1_prob=float(order[-1]), top2_gap=gap, tie=gap <= TIE_GAP,
                    vocab_top1_id=top1, vocab_top1_text=tok.decode([top1]), vocab_top1_is_valid_letter=top1 in letters[:n],
                    letter_vs_full_vocab_maxabs=float((full[k, letters][:n] - raw).abs().max())))
            pos = captured['pos']                                          # [3, B, S] as the rotary received it
            S = len(ids)
            arange = torch.arange(S, dtype=pos.dtype).view(1, 1, S).expand_as(pos)
            img_idx = [i for i, t in enumerate(ids) if t == IMAGE_ID]
            probe = dict(first_token=0, first_slot=slots[0], last_token=S - 1)
            if img_idx:
                probe.update(vision_start=ids.index(VSTART_ID), first_image=img_idx[0], last_image=img_idx[-1],
                             after_image=img_idx[-1] + 1)
            positions = dict(shape=list(pos.shape), is_arange=bool(torch.equal(pos, arange)),
                             probes={k: dict(index=i, token_id=ids[i], thw=[int(v) for v in pos[:, 0, i]]) for k, i in probe.items()},
                             rope_index_calls=rope_used,
                             rope_deltas=None if backbone.rope_deltas is None else backbone.rope_deltas.flatten().tolist())
            rec = dict(row_id=row['id'], arm=arm, G=G, pos1d=pos1d, family=row['family'], tier=row['tier'],
                       purpose=row['purpose'], image_original=row['image'], image_input=image_input,
                       input_keys=sorted(k for k in inp.keys()), n_tokens=S, input_ids=ids,
                       n_image_tokens=len(img_idx), slot_idx=slots, nq=len(row['questions']), slots=slot_recs,
                       positions=positions, wall_seconds_contended=wall)
            if 'image_grid_thw' in inp:
                thw = inp['image_grid_thw'][0].tolist()
                pv = inp['pixel_values']
                rec.update(image_grid_thw=thw, processor_resized_hw=[thw[1] * 16, thw[2] * 16],
                           pixel_values_shape=list(pv.shape), pixel_values_dtype=str(pv.dtype),
                           pixel_values_sha256=sha256_bytes(pv.contiguous().numpy().tobytes()))
                assert len(img_idx) == thw[0] * thw[1] * thw[2] // 4
                if G is not None:
                    hand = hand_pixel_values(image)
                    got = pv.double().numpy()
                    rec['c4'] = dict(shape_match=list(hand.shape) == list(got.shape),
                                     max_abs_vs_float64=float(np.abs(hand - got).max()),
                                     max_abs_vs_float32=float(np.abs(hand.astype(np.float32).astype(np.float64) - got).max()))
            result['forwards'].append(rec)
            top = [(sr['argmax'], round(sr['top1_prob'], 4)) for sr in slot_recs]
            print(f"{row['id']:28s} {arm:11s} S={S:4d} img={len(img_idx):3d} grid={rec.get('image_grid_thw')} "
                  f"arange={positions['is_arange']} {top} {wall:6.1f}s", flush=True)
        write_json(out_path, result)

    assert counters['torch_chunk_gated_delta_rule'] > 0, 'kernel instrumentation observed no gated-delta call'
    result.update(
        status='DONE', wall_seconds_contended=time.monotonic() - start,
        runtime=dict(python=sys.version, torch=torch.__version__, transformers=transformers.__version__,
                     torch_threads=torch.get_num_threads(),
                     attn_implementation=dict(top=m.lm.config._attn_implementation,
                                              text=m.lm.config.text_config._attn_implementation,
                                              vision=m.lm.config.vision_config._attn_implementation),
                     kernel_functions=kernel_functions, kernel_call_counts=dict(counters),
                     kernel_shapes=dict(kernel_shapes), rope_index_calls=dict(rope_calls),
                     optional_kernel_packages={p: importlib.util.find_spec(p) is not None
                                               for p in ('kernels', 'fla', 'causal_conv1d')},
                     model_parameters=sum(p.numel() for p in m.parameters())))
    write_json(out_path, result)
    print('ORACLE DONE', out_path, len(result['forwards']), 'forwards', f"{result['wall_seconds_contended']:.0f}s", flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        (ROOT / 'logs').mkdir(exist_ok=True)
        (ROOT / 'logs/oracle.traceback.txt').write_text(traceback.format_exc())
        raise
