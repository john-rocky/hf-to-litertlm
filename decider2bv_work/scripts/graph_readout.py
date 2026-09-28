"""Graph readout, round 2: the G = 256 fp32 LiteRT graph (vision tflites + the derived-M-RoPE decoder) against the
fp32 HF oracle's letter probabilities, one row at a time, CompiledModel CPU, fixed threads.

Per row (state zeroed at the start of every row):
  ids       = the oracle's own input_ids (g256_mrope forward for image rows; the same ids in every arm for no-image
              rows, C1). Token embeddings come from the embedder tflite, one token per call.
  image     = the 64 <|image_pad|> slots (positions 1..64) are replaced, in order, by
              (i)  the vision tflites on the same 256 PNG (encoder -> adapter), or
              (ii) transformers 5.17.0's vision output for that PNG (out/hf_vision_g256/, isolates the decoder).
  positions = the runtime contract: a 1-D run 0, 1, 2, ... for image rows. No-image rows start at 65 (= N + 1):
              the derived rotary maps p >= 65 to (p - 56) on all three channels, a constant shift of RoPE positions
              (relative positions unchanged); the mask then allows only the written slots (65 onwards).
  schedule  = for each answer slot (the " (" token, in order): prefill the tokens before it with the largest
              ladder signatures that fit exactly (no padding), decode the slot token once -> logits, then continue
              prefilling after it. Letter logits = logits[letter ids][:nopts], softmax at T = 1.
  mask      = float32, 0 where first_position <= cache slot <= own position, -1e30 elsewhere.

Compared with oracle g256_mrope (image rows) and the oracle's no-image rows (identical in every arm): slot count,
argmax on non-tie slots (tie = oracle top-2 gap <= 1e-4), max |dp| and p95 over slots (dp = max over the options),
full-vocabulary top-1 id. Pre-registered pass line (pre-registered): slot count equal, non-tie argmax 100 %, max |dp|
<= 1e-3 over all v1 + v2 image rows and all no-image rows. Falsification control: the same graph probabilities vs
the oracle's g256_pos1d arm (the 1-D position contract), which the graph must NOT reproduce where they differ.
Informational extra: the no-image rows again with positions from 0 (what a text-only prompt would see if the
runtime did not offset it).

Round 3 (variants): --decoder / --embedder / --vision NAME=ENC:ADP (repeatable) / --xnn-cache point the same
readout at other files (fp16 variants, or the tflites unpacked from a bundle); with no such flags it reads the
round-2 fp32 files exactly as before. The arithmetic is unchanged; each slot additionally records the sha256 of its
full float32 logits vector and the vocabulary top-2 (for bit-identity checks and first-token margins), and each
row-arm the sha256 of the injected vision embeddings. --gate-arm names the image arm the pass line is read from.

    out/venv-readout/bin/python -B -u scripts/graph_readout.py [--threads 8] [--rows a,b]
    out/venv-readout/bin/python -B -u scripts/graph_readout.py --decoder D --embedder E --xnn-cache C \
        --vision A_fp16=ENC:ADP [--vision B_...=ENC:ADP] [--no-hf-arm] --gate-arm A_fp16 --out results/x.json
"""
import argparse
import hashlib
import os
import time
from collections import OrderedDict

import numpy as np
from ai_edge_litert.compiled_model import CompiledModel
from ai_edge_litert.hardware_accelerator import HardwareAccelerator
from ai_edge_litert.options import Options, CpuOptions

from common import ROOT, TIE_GAP, read_json, write_json, sha256_file
from tflite_scan import scan
from vision_parity_g256 import Single, load01

G = 256
N_IMG = 64
TEXT_START = N_IMG + 1
LADDER = [1024, 256, 64, 16, 4, 1]
IMAGE_ID, VSTART_ID, VEND_ID = 248056, 248053, 248054
ORACLES = ('fixtures/oracle_fp32.json', 'fixtures/oracle_fp32_v2add.json')
PASS_DP = 1e-3


def chunk_plan(start, stop):
    """Largest ladder signatures that fit exactly into ids[start:stop] (no padding)."""
    plan, p = [], start
    for length in LADDER:
        while stop - p >= length:
            plan.append((f'prefill_{length}', p, length))
            p += length
    assert p == stop
    return plan


def softmax64(z):
    z = np.asarray(z, np.float64)
    e = np.exp(z - z.max())
    return e / e.sum()


class Decoder:
    def __init__(self, path, threads, cache_path):
        t0 = time.monotonic()
        self.model = CompiledModel.from_file(str(path), options=Options(
            hardware_accelerators=HardwareAccelerator.CPU,
            cpu_options=CpuOptions(num_threads=threads, xnnpack_weight_cache_path=str(cache_path))))
        self.load_seconds = time.monotonic() - t0
        self.signatures = sorted(self.model.get_signature_list())
        assert set(self.signatures) == {'decode', *(f'prefill_{n}' for n in LADDER)}, self.signatures
        self.inputs = {k: self.model.get_input_tensor_details(k) for k in self.signatures}
        self.outputs = {k: self.model.get_output_tensor_details(k) for k in self.signatures}
        self.states = {n: d for n, d in self.inputs['decode'].items() if n not in ('embeddings', 'input_pos', 'mask')}
        for key in self.signatures:
            assert set(self.inputs[key]) == {'embeddings', 'input_pos', 'mask', *self.states}, key
            assert set(self.outputs[key]) == set(self.states) | ({'logits'} if key == 'decode' else set()), key
        self.cache_length = int(self.inputs['decode']['mask']['shape'][-1])

    def zero_states(self):
        return {n: np.zeros(d['shape'], dtype=d['dtype']) for n, d in self.states.items()}

    def step(self, key, emb, positions, first_pos, states):
        inp, out = self.inputs[key], self.outputs[key]
        allowed = (np.arange(self.cache_length)[None, :] <= positions[:, None]) & (np.arange(self.cache_length)[None, :] >= first_pos)
        mask = np.where(allowed, np.float32(0.0), np.float32(-1e30)).reshape(inp['mask']['shape'])
        feeds = dict(states, embeddings=np.ascontiguousarray(emb, dtype=np.float32).reshape(inp['embeddings']['shape']),
                     input_pos=positions.astype(inp['input_pos']['dtype']).reshape(inp['input_pos']['shape']), mask=mask)
        ibuf, obuf = {}, {}
        try:
            for name, arr in feeds.items():
                assert list(arr.shape) == list(inp[name]['shape']), (key, name, arr.shape, inp[name]['shape'])
                b = self.model.create_input_buffer_by_name(key, name)
                ibuf[name] = b
                b.write(np.ascontiguousarray(arr))
            obuf = {name: self.model.create_output_buffer_by_name(key, name) for name in out}
            self.model.run_by_name(key, ibuf, obuf)
            vals = {name: np.array(obuf[name].read(int(np.prod(d['shape'])), d['dtype']), dtype=d['dtype'], copy=True).reshape(d['shape'])
                    for name, d in out.items()}
        finally:
            for b in [*ibuf.values(), *obuf.values()]:
                b.destroy()
        finite = all(np.isfinite(v).all() for v in vals.values())
        return {n: vals[n] for n in self.states}, vals.get('logits'), finite


class Embedder:
    def __init__(self, path, threads):
        self.single = Single(path, threads)
        self.cache = {}

    def __call__(self, ids):
        rows = []
        for t in ids:
            if t not in self.cache:
                self.cache[t] = self.single(np.array([[t]], dtype=np.int32)).reshape(-1).copy()
            rows.append(self.cache[t])
        return np.stack(rows)                                                   # [S, 2048]


def run_row(dec, ids, emb, slot_idx, first_pos):
    """Prefill/decode schedule over one row; returns per-slot logits and the call log."""
    S = len(ids)
    pos = np.arange(first_pos, first_pos + S, dtype=np.int64)
    assert pos[-1] < dec.cache_length
    states = dec.zero_states()
    cur, out, calls, finite = 0, [], [], True
    for s in slot_idx:
        for key, a, n in chunk_plan(cur, s):
            states, lg, ok = dec.step(key, emb[a:a + n], pos[a:a + n], first_pos, states)
            assert lg is None
            finite &= ok
            calls.append(key)
        states, lg, ok = dec.step('decode', emb[s:s + 1], pos[s:s + 1], first_pos, states)
        finite &= ok
        calls.append('decode')
        out.append(lg.reshape(-1))
        cur = s + 1
    return out, calls, finite


def compare(slot_logits, ref_forward, letter_ids):
    recs = []
    for lg, sl in zip(slot_logits, ref_forward['slots']):
        n = sl['nopts']
        letters = lg[letter_ids[:n]].astype(np.float64)
        p = softmax64(letters)
        ref = np.asarray(sl['probs'], np.float64)
        dp = np.abs(p - ref)
        second = lg.astype(np.float32, copy=True)
        second[int(lg.argmax())] = -np.inf
        top2 = (int(lg.argmax()), int(second.argmax()))
        recs.append(dict(k=sl['k'], nopts=n, probs=p.tolist(), ref_probs=sl['probs'], dp=float(dp.max()),
                         argmax=int(p.argmax()), ref_argmax=sl['argmax'], tie=sl['top2_gap'] <= TIE_GAP,
                         ref_top2_gap=sl['top2_gap'], agree=int(p.argmax()) == sl['argmax'],
                         letter_logits=letters.tolist(), ref_letter_logits=sl['letter_logits_fp32'],
                         letter_logit_max_abs=float(np.abs(letters - np.asarray(sl['letter_logits_fp32'])).max()),
                         vocab_top1_id=int(lg.argmax()), ref_vocab_top1_id=sl['vocab_top1_id'],
                         vocab_top1_agree=int(lg.argmax()) == sl['vocab_top1_id'],
                         vocab_top2_id=int(top2[1]), vocab_top1_top2_logit_gap=float(lg[top2[0]]) - float(lg[top2[1]]),
                         logits_sha256=hashlib.sha256(np.ascontiguousarray(lg, dtype=np.float32).tobytes()).hexdigest()))
    return recs


def summary(slots):
    if not slots:
        return dict(n_slots=0)
    dps = np.array([s['dp'] for s in slots])
    nontie = [s for s in slots if not s['tie']]
    return OrderedDict(n_slots=len(slots), n_tie=len(slots) - len(nontie),
                       argmax_agree_nontie=sum(s['agree'] for s in nontie), n_nontie=len(nontie),
                       max_dp=float(dps.max()), p95_dp=float(np.percentile(dps, 95)), median_dp=float(np.median(dps)),
                       n_dp_gt_1e3=int((dps > PASS_DP).sum()),
                       vocab_top1_agree=sum(s['vocab_top1_agree'] for s in slots),
                       max_letter_logit_abs=max(s['letter_logit_max_abs'] for s in slots),
                       max_dp_slot=next(f"{s['row_id']}#{s['k']}" for s in slots if s['dp'] == dps.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--rows', default='')
    ap.add_argument('--out', default='results/graph_parity_r2.json')
    ap.add_argument('--decoder', default='', help='prefill_decode tflite (default: the round-2 fp32 export)')
    ap.add_argument('--embedder', default='', help='embedder tflite (default: the round-2 fp32 export)')
    ap.add_argument('--vision', action='append', default=[], help='NAME=ENCODER.tflite:ADAPTER.tflite, repeatable '
                    '(default: i_tflite_vision = the round-2 fp32 vision tflites)')
    ap.add_argument('--no-hf-arm', action='store_true', help='skip the ii_hf_vision (HF vision + this decoder) arm')
    ap.add_argument('--xnn-cache', default='out/xnn_cache/decoder_g256_fp32.xnnpack_cache')
    ap.add_argument('--gate-arm', default='i_tflite_vision', help='image arm the pass line / status is read from')
    args = ap.parse_args()
    t0 = time.monotonic()
    insp = read_json('results/decoder_export_r2.json')
    if args.decoder or args.embedder:
        assert args.decoder and args.embedder, 'pass --decoder and --embedder together'
        files = {}
        for key, p in (('prefill_decode', args.decoder), ('embedder', args.embedder)):
            p = (ROOT / p).resolve()
            files[key] = dict(path=os.path.relpath(p, ROOT), bytes=os.path.getsize(p), sha256=sha256_file(p))
        dec_rec, emb_rec = files['prefill_decode'], files['embedder']
    else:
        dec_rec, emb_rec = insp['prefill_decode'], insp['embedder']
        assert sha256_file(ROOT / dec_rec['path']) == dec_rec['sha256'] and sha256_file(ROOT / emb_rec['path']) == emb_rec['sha256']
    dec_path, emb_path = ROOT / dec_rec['path'], ROOT / emb_rec['path']
    vision_specs = OrderedDict()
    for spec in args.vision or [f'i_tflite_vision=out/vision_g{G}/vision_encoder.tflite:out/vision_g{G}/vision_adapter.tflite']:
        name, paths = spec.split('=', 1)
        enc_p, adp_p = ((ROOT / p).resolve() for p in paths.split(':'))
        vision_specs[name] = dict(encoder=dict(path=os.path.relpath(enc_p, ROOT), bytes=os.path.getsize(enc_p), sha256=sha256_file(enc_p)),
                                  adapter=dict(path=os.path.relpath(adp_p, ROOT), bytes=os.path.getsize(adp_p), sha256=sha256_file(adp_p)))
    assert args.gate_arm in vision_specs or (args.gate_arm == 'ii_hf_vision' and not args.no_hf_arm), args.gate_arm
    xnn_cache = ROOT / args.xnn_cache
    xnn_cache.parent.mkdir(parents=True, exist_ok=True)
    dec = Decoder(dec_path, args.threads, xnn_cache)
    embed = Embedder(emb_path, args.threads)
    vision = OrderedDict((name, (Single(ROOT / v['encoder']['path'], args.threads), Single(ROOT / v['adapter']['path'], args.threads)))
                         for name, v in vision_specs.items())
    print('loaded decoder', f'{dec.load_seconds:.0f}s', dec.signatures, len(dec.states), 'states', flush=True)

    fwd = {}
    letter_ids = None
    for path in ORACLES:
        orc = read_json(path)
        letter_ids = letter_ids or orc['letters']['ids']
        assert orc['letters']['ids'] == letter_ids
        for f in orc['forwards']:
            fwd[(f['row_id'], f['arm'])] = f
    fx = read_json('fixtures/fixtures_v2.json')
    rows = [r for r in fx['rows'] if not args.rows or r['id'] in args.rows.split(',')]
    res = OrderedDict(status='RUNNING', runtime='ai-edge-litert CompiledModel CPU', threads=args.threads,
                      decoder=dec_rec, embedder=emb_rec, vision=vision_specs, hf_arm=not args.no_hf_arm,
                      gate_arm=args.gate_arm, xnn_cache=args.xnn_cache,
                      oracles={p: sha256_file(ROOT / p) for p in ORACLES}, fixtures_v2_sha256=sha256_file(ROOT / 'fixtures/fixtures_v2.json'),
                      schedule=dict(ladder=LADDER, image_first_pos=0, text_only_first_pos=TEXT_START, softmax_T=1.0,
                                    mask='0 on [first_pos, own pos], -1e30 elsewhere', state='zero per row'),
                      load_seconds_contended=dec.load_seconds, rows=[])
    for r in rows:
        rid = r['id']
        image = r['image'] is not None
        ref = fwd[(rid, 'g256_mrope')]
        ids = ref['input_ids']
        slot_idx = ref['slot_idx']
        assert len(slot_idx) == len(r['questions']) == len(ref['slots'])
        emb_text = embed(ids)
        arms = {}
        vis_sha = {}
        if image:
            assert ids[0] == VSTART_ID and ids[1:1 + N_IMG] == [IMAGE_ID] * N_IMG and ids[1 + N_IMG] == VEND_ID, rid
            png = ROOT / f'out/fixtures_resized/g{G}/{rid}.png'
            assert sha256_file(png) == ref['image_input']['sha256_file'], rid
            vis_arms = [(name, vadp(venc(load01(png))).reshape(N_IMG, -1)) for name, (venc, vadp) in vision.items()]
            if not args.no_hf_arm:
                vis_arms.append(('ii_hf_vision', np.load(ROOT / f'out/hf_vision_g{G}/{rid}.npy').reshape(N_IMG, -1)))
            for arm, vis in vis_arms:
                vis_sha[arm] = hashlib.sha256(np.ascontiguousarray(vis, dtype=np.float32).tobytes()).hexdigest()
                e = emb_text.copy()
                e[1:1 + N_IMG] = vis
                arms[arm] = (e, 0)
        else:
            arms['text_from_65'] = (emb_text, TEXT_START)
            arms['text_from_0_informational'] = (emb_text, 0)
        rec = OrderedDict(row_id=rid, family=r['family'], tier=r['tier'], purpose=r['purpose'],
                          source='v2' if r.get('added_in') == 'v2' else 'v1', image=image, n_tokens=len(ids),
                          slot_idx=slot_idx, n_slots_oracle=len(ref['slots']),
                          text_embeddings_sha256=hashlib.sha256(np.ascontiguousarray(emb_text, dtype=np.float32).tobytes()).hexdigest(),
                          vision_sha256=vis_sha, arms=OrderedDict())
        for arm, (e, first_pos) in arms.items():
            t1 = time.monotonic()
            logits, calls, finite = run_row(dec, ids, e, slot_idx, first_pos)
            slots = compare(logits, ref, letter_ids)
            if image:
                pos1d = fwd[(rid, 'g256_pos1d')]
                for s, sp in zip(slots, pos1d['slots']):
                    pp = np.asarray(sp['probs'], np.float64)
                    s['pos1d_probs'] = sp['probs']
                    s['dp_vs_pos1d'] = float(np.abs(np.asarray(s['probs']) - pp).max())
                    s['agree_pos1d'] = s['argmax'] == sp['argmax']
                    s['mrope_vs_pos1d_dp'] = float(np.abs(np.asarray(s['ref_probs']) - pp).max())
            rec['arms'][arm] = OrderedDict(first_pos=first_pos, n_slots_graph=len(slots), finite=bool(finite),
                                           calls=dict((k, calls.count(k)) for k in sorted(set(calls))),
                                           wall_seconds_contended=time.monotonic() - t1, slots=slots)
            print(f"{rid:34s} {arm:26s} " + ' '.join(f"k{s['k']}:dp={s['dp']:.2e}{'' if s['agree'] else '!'}" for s in slots)
                  + f"  {time.monotonic() - t1:.1f}s", flush=True)
        res['rows'].append(rec)
        write_json(args.out, res)

    def collect(arm_names, image_flag):
        out = []
        for rec in res['rows']:
            if rec['image'] != image_flag:
                continue
            for arm in arm_names:
                if arm in rec['arms']:
                    for s in rec['arms'][arm]['slots']:
                        out.append(dict(s, row_id=rec['row_id'], source=rec['source'], family=rec['family'], tier=rec['tier']))
        return out

    res['summary'] = OrderedDict()
    image_arms = list(vision_specs) + ([] if args.no_hf_arm else ['ii_hf_vision'])
    for name, arm_names, flag in ([(a, [a], True) for a in image_arms] +
                                  [('text_from_65', ['text_from_65'], False),
                                   ('text_from_0_informational', ['text_from_0_informational'], False)]):
        sl = collect(arm_names, flag)
        if not sl:
            res['summary'][name] = OrderedDict(all=summary(sl))
            continue
        entry = OrderedDict(all=summary(sl))
        if flag:
            entry['by_source'] = OrderedDict((v, summary([s for s in sl if s['source'] == v])) for v in ('v1', 'v2'))
            entry['by_family'] = OrderedDict((v, summary([s for s in sl if s['family'] == v])) for v in sorted(set(s['family'] for s in sl)))
            entry['by_tier'] = OrderedDict((v, summary([s for s in sl if s['tier'] == v])) for v in sorted(set(s['tier'] for s in sl)))
            dps = np.array([s['dp_vs_pos1d'] for s in sl])
            ref_gap = np.array([s['mrope_vs_pos1d_dp'] for s in sl])
            entry['falsification_vs_pos1d'] = OrderedDict(
                max_dp_vs_pos1d=float(dps.max()), p95_dp_vs_pos1d=float(np.percentile(dps, 95)),
                argmax_agree_pos1d=sum(s['agree_pos1d'] for s in sl), n_slots=len(sl),
                slots_where_oracle_arms_differ_gt_1e2=[
                    dict(row_id=s['row_id'], k=s['k'], graph=[round(v, 6) for v in s['probs']],
                         mrope=[round(v, 6) for v in s['ref_probs']], pos1d=[round(v, 6) for v in s['pos1d_probs']],
                         dp_vs_mrope=s['dp'], dp_vs_pos1d=s['dp_vs_pos1d'], graph_argmax=s['argmax'],
                         mrope_argmax=s['ref_argmax'], pos1d_agree=s['agree_pos1d'])
                    for s in sl if s['mrope_vs_pos1d_dp'] > 1e-2],
                n_slots_closer_to_mrope=int(sum(s['dp'] < s['dp_vs_pos1d'] for s in sl if s['mrope_vs_pos1d_dp'] > 1e-2)),
                n_slots_oracle_arms_differ_gt_1e2=int((ref_gap > 1e-2).sum()))
        res['summary'][name] = entry

    def verdict(a, t):
        empty = dict(argmax_agree_nontie=0, n_nontie=0, max_dp=0.0)      # a --rows subset without image or text rows
        a, t = (x if x.get('n_slots') else empty for x in (a, t))
        ok_slots = all(len(rec['arms'][arm]['slots']) == rec['n_slots_oracle'] for rec in res['rows'] for arm in rec['arms'])
        return dict(slot_count_equal=ok_slots,
                    nontie_argmax=f"{a['argmax_agree_nontie']}/{a['n_nontie']} + {t['argmax_agree_nontie']}/{t['n_nontie']}",
                    max_dp=max(a['max_dp'], t['max_dp']),
                    pass_=ok_slots and a['argmax_agree_nontie'] == a['n_nontie'] and t['argmax_agree_nontie'] == t['n_nontie']
                    and max(a['max_dp'], t['max_dp']) <= PASS_DP)
    t = res['summary']['text_from_65']['all']
    res['pass_line'] = dict(rule='slot count equal AND non-tie argmax 100% AND max|dp| <= 1e-3 over all v1+v2 image rows and all no-image rows (from 65)',
                            **{a: verdict(res['summary'][a]['all'], t) for a in image_arms})
    all_finite = all(rec['arms'][arm]['finite'] for rec in res['rows'] for arm in rec['arms'])
    res['all_finite'] = all_finite
    res['status'] = 'PASS' if res['pass_line'][args.gate_arm]['pass_'] and all_finite and not args.rows else (
        'PARTIAL_ROWS' if args.rows else 'FAIL')
    res['wall_seconds_contended'] = time.monotonic() - t0
    write_json(args.out, res)
    print('GRAPH_READOUT', res['status'], {k: v for k, v in res['pass_line'].items() if k != 'rule'}, flush=True)


if __name__ == '__main__':
    main()
