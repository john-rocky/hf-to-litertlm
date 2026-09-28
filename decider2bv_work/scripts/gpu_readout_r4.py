"""Round 4, step 7: the probabilities of one weight form with the DECODER on ai-edge-litert 2.2.0 CompiledModel GPU
(GpuOptions(enforce_f32=True), the bundle's fp32 activation preference) against the same file on CPU, with ONE padded
prefill chunk per slot - the decider-0.8b GPU rule (decider_work/FINDINGS.md, GPU follow-up: multi-chunk prefill
continuation is not trusted on the GPU; one padded chunk reproduced the CPU to 9.8e-6 there).

Per answer slot s of a row (a fresh zero state for every slot): prefill ids[:s] as ONE chunk of the smallest ladder
signature >= s, then decode ids[s] at position first_pos + s -> logits (letters at T = 1, as graph_readout.py).
  valid rows  : the row's embeddings, positions first_pos .. first_pos + s - 1, mask 0 on [first_pos, own pos]
  pad rows    : embedding 0, position 0, mask -1e30 on every column. Position 0 after a valid position is
                non-increasing, so the export's pad guard (valid = p[i] > p[i-1], position monotonicity; the
                externalized-embedder form of the qwen35 patch) marks every pad row invalid. The attention KV cache
                is written by DYNAMIC_UPDATE_SLICE at input_pos[0] (12 per signature), so pad K/V land in the slots
                after the valid ones; the decode then writes slot s itself and its mask stops at s.
The embedder and the vision encoder + adapter run on CPU from the same files as the CPU readout (their output hashes
are recorded and compared), so only the decoder's backend and the feeding scheme move. `--backend cpu` with the same
padded scheme is the control that separates the scheme from the backend.

    out/venv-readout/bin/python -B -u scripts/gpu_readout_r4.py --variant fp16 --backend gpu|cpu [--rows a,b]
        -> results/gpu_readout_r4_<variant>_<backend>.json
"""
import argparse
import hashlib
import os
import time
from collections import OrderedDict

import numpy as np
from ai_edge_litert.compiled_model import CompiledModel
from ai_edge_litert.gpu_options import GpuOptions
from ai_edge_litert.hardware_accelerator import HardwareAccelerator
from ai_edge_litert.options import Options, CpuOptions

from common import ROOT, read_json, write_json, sha256_file
from graph_readout import Embedder, compare, summary, LADDER, N_IMG, TEXT_START, IMAGE_ID, VSTART_ID, VEND_ID, ORACLES, G
from vision_parity_g256 import Single, load01


class PaddedDecoder:
    def __init__(self, path, backend, threads, cache_path):
        t0 = time.monotonic()
        if backend == 'gpu':
            opts = Options(hardware_accelerators=HardwareAccelerator.GPU, gpu_options=GpuOptions(enforce_f32=True))
        else:
            opts = Options(hardware_accelerators=HardwareAccelerator.CPU,
                           cpu_options=CpuOptions(num_threads=threads, xnnpack_weight_cache_path=str(cache_path)))
        self.model = CompiledModel.from_file(str(path), options=opts)
        self.load_seconds = time.monotonic() - t0
        self.fully_accelerated = bool(self.model.is_fully_accelerated())
        self.signatures = sorted(self.model.get_signature_list())
        self.inputs = {k: self.model.get_input_tensor_details(k) for k in self.signatures}
        self.outputs = {k: self.model.get_output_tensor_details(k) for k in self.signatures}
        self.states = {n: d for n, d in self.inputs['decode'].items() if n not in ('embeddings', 'input_pos', 'mask')}
        self.cache_length = int(self.inputs['decode']['mask']['shape'][-1])

    def run(self, key, feeds):
        inp, out = self.inputs[key], self.outputs[key]
        ibuf, obuf = {}, {}
        try:
            for name, arr in feeds.items():
                assert list(arr.shape) == list(inp[name]['shape']), (key, name, arr.shape, inp[name]['shape'])
                b = self.model.create_input_buffer_by_name(key, name)
                ibuf[name] = b
                b.write(np.ascontiguousarray(arr))
            obuf = {name: self.model.create_output_buffer_by_name(key, name) for name in out}
            self.model.run_by_name(key, ibuf, obuf)
            return {name: np.array(obuf[name].read(int(np.prod(d['shape'])), d['dtype']), dtype=d['dtype'], copy=True).reshape(d['shape'])
                    for name, d in out.items()}
        finally:
            for b in [*ibuf.values(), *obuf.values()]:
                b.destroy()

    def slot_logits(self, emb, s, first_pos):
        """One padded prefill of emb[:s] + one decode of emb[s]; returns (logits, plan, finite)."""
        states = {n: np.zeros(d['shape'], dtype=d['dtype']) for n, d in self.states.items()}
        size = min(x for x in LADDER if x >= s)
        key = f'prefill_{size}'
        inp = self.inputs[key]
        e = np.zeros((size, emb.shape[1]), np.float32)
        e[:s] = emb[:s]
        pos = np.zeros(size, np.int64)
        pos[:s] = np.arange(first_pos, first_pos + s)
        cols = np.arange(self.cache_length)[None, :]
        allowed = (cols <= pos[:, None]) & (cols >= first_pos) & (np.arange(size) < s)[:, None]
        mask = np.where(allowed, np.float32(0.0), np.float32(-1e30)).reshape(inp['mask']['shape'])
        vals = self.run(key, dict(states, embeddings=e.reshape(inp['embeddings']['shape']),
                                  input_pos=pos.astype(inp['input_pos']['dtype']).reshape(inp['input_pos']['shape']), mask=mask))
        finite = all(np.isfinite(v).all() for v in vals.values())
        states = {n: vals[n] for n in self.states}
        dinp = self.inputs['decode']
        p1 = np.array([first_pos + s])
        dmask = np.where((cols <= p1[:, None]) & (cols >= first_pos), np.float32(0.0), np.float32(-1e30)).reshape(dinp['mask']['shape'])
        vals = self.run('decode', dict(states, embeddings=emb[s:s + 1].astype(np.float32).reshape(dinp['embeddings']['shape']),
                                       input_pos=p1.astype(dinp['input_pos']['dtype']).reshape(dinp['input_pos']['shape']), mask=dmask))
        finite &= all(np.isfinite(v).all() for v in vals.values())
        return vals['logits'].reshape(-1), dict(signature=key, valid=s, pad=size - s), bool(finite)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True)
    ap.add_argument('--backend', choices=('gpu', 'cpu'), required=True)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--rows', default='')
    args = ap.parse_args()
    v = args.variant
    t0 = time.monotonic()
    cpu_ro = read_json(f'results/readout_r4_{v}.json')                     # the variant's CPU exact-fit readout
    files = dict(decoder=cpu_ro['decoder'], embedder=cpu_ro['embedder'], vision=cpu_ro['vision'][v])
    for k, rec in (('decoder', files['decoder']), ('embedder', files['embedder']), ('encoder', files['vision']['encoder']),
                   ('adapter', files['vision']['adapter'])):
        assert sha256_file(ROOT / rec['path']) == rec['sha256'], k
    dec = PaddedDecoder(ROOT / files['decoder']['path'], args.backend, args.threads, ROOT / cpu_ro['xnn_cache'])
    print('loaded', args.backend, f'{dec.load_seconds:.0f}s', 'fully_accelerated', dec.fully_accelerated, flush=True)
    embed = Embedder(ROOT / files['embedder']['path'], args.threads)
    venc, vadp = Single(ROOT / files['vision']['encoder']['path'], args.threads), Single(ROOT / files['vision']['adapter']['path'], args.threads)
    fwd, letter_ids = {}, None
    for path in ORACLES:
        orc = read_json(path)
        letter_ids = letter_ids or orc['letters']['ids']
        for f in orc['forwards']:
            fwd[(f['row_id'], f['arm'])] = f
    cpu_rows = {r['row_id']: r for r in cpu_ro['rows']}
    fx = read_json('fixtures/fixtures_v2.json')
    rows = [r for r in fx['rows'] if not args.rows or r['id'] in args.rows.split(',')]
    res = OrderedDict(status='RUNNING', variant=v, backend=args.backend, runtime='ai-edge-litert 2.2.0 CompiledModel',
                      gpu_options=dict(enforce_f32=True) if args.backend == 'gpu' else None,
                      cpu_threads=args.threads if args.backend == 'cpu' else None, decoder_fully_accelerated=dec.fully_accelerated,
                      load_seconds_contended=dec.load_seconds, files=files, cpu_exact_fit_readout=f'results/readout_r4_{v}.json',
                      scheme='one padded prefill chunk per slot (fresh zero state), pads: embedding 0 / position 0 / mask -1e30; then one decode',
                      embedder_vision_backend='cpu', rows=[])
    for r in rows:
        rid = r['id']
        image = r['image'] is not None
        ref = fwd[(rid, 'g256_mrope')]
        ids, slot_idx = ref['input_ids'], ref['slot_idx']
        if max(slot_idx) > max(LADDER):
            res['rows'].append(dict(row_id=rid, skipped='a slot beyond the largest prefill signature'))
            continue
        emb = embed(ids)
        vis_sha = None
        if image:
            assert ids[0] == VSTART_ID and ids[1:1 + N_IMG] == [IMAGE_ID] * N_IMG and ids[1 + N_IMG] == VEND_ID, rid
            png = ROOT / f'out/fixtures_resized/g{G}/{rid}.png'
            assert sha256_file(png) == ref['image_input']['sha256_file'], rid
            vis = vadp(venc(load01(png))).reshape(N_IMG, -1)
            vis_sha = hashlib.sha256(np.ascontiguousarray(vis, dtype=np.float32).tobytes()).hexdigest()
            emb = emb.copy()
            emb[1:1 + N_IMG] = vis
        arm, first_pos = (v, 0) if image else ('text_from_65', TEXT_START)
        t1 = time.monotonic()
        logits, plans, finite = [], [], True
        for s in slot_idx:
            lg, plan, ok = dec.slot_logits(emb, s, first_pos)
            logits.append(lg)
            plans.append(plan)
            finite &= ok
        slots = compare(logits, ref, letter_ids)
        cslots = cpu_rows[rid]['arms'][arm]['slots']
        for s, c in zip(slots, cslots):
            s['cpu_exact_fit_probs'] = c['probs']
            s['dp_vs_cpu_exact_fit'] = float(np.abs(np.asarray(s['probs']) - np.asarray(c['probs'])).max())
            s['argmax_vs_cpu_exact_fit'] = s['argmax'] == c['argmax']
        res['rows'].append(OrderedDict(row_id=rid, family=r['family'], tier=r['tier'], purpose=r['purpose'], image=image, n_tokens=len(ids),
                                       arm=arm, first_pos=first_pos, vision_sha256=vis_sha,
                                       vision_sha256_equals_cpu_readout=(vis_sha == cpu_rows[rid]['vision_sha256'].get(v)) if image else None,
                                       plans=plans, finite=finite, wall_seconds_contended=time.monotonic() - t1, slots=slots))
        print(f"{rid:34s} {arm:14s} " + ' '.join(f"k{s['k']}:dp={s['dp']:.2e}/cpu={s['dp_vs_cpu_exact_fit']:.1e}" for s in slots)
              + f"  {time.monotonic() - t1:.1f}s", flush=True)
        write_json(f'results/gpu_readout_r4_{v}_{args.backend}.json', res)
    res['summary'] = OrderedDict()
    for name, flag in (('image', True), ('text_from_65', False)):
        sl = [dict(s, row_id=rec['row_id']) for rec in res['rows'] if 'slots' in rec and rec['image'] == flag for s in rec['slots']]
        if sl:
            d = np.array([s['dp_vs_cpu_exact_fit'] for s in sl])
            res['summary'][name] = OrderedDict(vs_oracle=summary(sl), vs_cpu_exact_fit=dict(
                max_dp=float(d.max()), p95_dp=float(np.percentile(d, 95)), median_dp=float(np.median(d)),
                n_gt_1e3=int((d > 1e-3).sum()), argmax_equal=sum(s['argmax_vs_cpu_exact_fit'] for s in sl), n=len(sl)))
    res['all_finite'] = all(rec.get('finite', True) for rec in res['rows'])
    res['vision_hashes_equal_cpu_readout'] = all(rec['vision_sha256_equals_cpu_readout'] for rec in res['rows'] if rec.get('image'))
    res['status'] = 'DONE'
    res['wall_seconds_contended'] = time.monotonic() - t0
    write_json(f'results/gpu_readout_r4_{v}_{args.backend}.json', res)
    print('GPU_READOUT_R4', v, args.backend, {k: (x['vs_oracle']['max_dp'], x['vs_oracle']['argmax_agree_nontie'], x['vs_oracle']['n_nontie'],
                                                 x['vs_cpu_exact_fit']['max_dp']) for k, x in res['summary'].items()}, flush=True)


if __name__ == '__main__':
    main()
