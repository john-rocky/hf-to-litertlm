"""Vision parity at G = 256: the fp32 tflites (encoder -> adapter, CompiledModel CPU) vs transformers 5.17.0's
model.model.visual(pixel_values, grid_thw).pooler_output (dumped by scripts/dump_hf_vision_g256.py), per fixture
image (v1 + v2), plus the flatbuffer op tables of both tflites (GATHER / FLEX / CUSTOM must be
absent). The graph input is the oracle's own GxG PNG as [1, 256, 256, 3] float in [0, 1] (the fast_vlm contract:
the encoder applies (x - 0.5) / 0.5 itself).

    out/venv-readout/bin/python -B scripts/vision_parity_g256.py [--threads 8]
"""
import argparse
import time

import numpy as np
from PIL import Image
from ai_edge_litert.compiled_model import CompiledModel
from ai_edge_litert.hardware_accelerator import HardwareAccelerator
from ai_edge_litert.options import Options, CpuOptions

from common import ROOT, read_json, write_json, sha256_file
from tflite_scan import scan

G = 256


class Single:
    """One-signature tflite on CompiledModel CPU with named buffers."""

    def __init__(self, path, threads):
        self.model = CompiledModel.from_file(str(path), options=Options(
            hardware_accelerators=HardwareAccelerator.CPU, cpu_options=CpuOptions(num_threads=threads)))
        sigs = self.model.get_signature_list()
        assert len(sigs) == 1, sigs
        self.key = next(iter(sigs))
        self.inp = self.model.get_input_tensor_details(self.key)
        self.out = self.model.get_output_tensor_details(self.key)
        assert len(self.inp) == 1 and len(self.out) == 1, (self.inp, self.out)

    def __call__(self, x):
        (iname, idet), = self.inp.items()
        (oname, odet), = self.out.items()
        x = np.ascontiguousarray(x, dtype=idet['dtype']).reshape(idet['shape'])
        ib = self.model.create_input_buffer_by_name(self.key, iname)
        ob = self.model.create_output_buffer_by_name(self.key, oname)
        try:
            ib.write(x)
            self.model.run_by_name(self.key, {iname: ib}, {oname: ob})
            y = np.array(ob.read(int(np.prod(odet['shape'])), odet['dtype']), dtype=odet['dtype'], copy=True)
        finally:
            ib.destroy()
            ob.destroy()
        return y.reshape(odet['shape'])


def load01(path):
    im = Image.open(path).convert('RGB')
    assert im.size == (G, G)
    return (np.asarray(im, dtype=np.float32) / np.float32(255.0))[None]     # [1,G,G,3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threads', type=int, default=8)
    args = ap.parse_args()
    t0 = time.monotonic()
    vdir = ROOT / f'out/vision_g{G}'
    enc_p, adp_p = vdir / 'vision_encoder.tflite', vdir / 'vision_adapter.tflite'
    conv = read_json(f'out/vision_g{G}/result.json')
    assert conv['ok'] and conv['files']['vision_encoder.tflite']['sha256'] == sha256_file(enc_p)
    assert conv['files']['vision_adapter.tflite']['sha256'] == sha256_file(adp_p)
    ops = {'vision_encoder': scan(enc_p, with_sha=False), 'vision_adapter': scan(adp_p, with_sha=False)}
    for v in ops.values():
        v.pop('subgraphs')
    enc, adp = Single(enc_p, args.threads), Single(adp_p, args.threads)
    hf = read_json(f'out/hf_vision_g{G}/index.json')
    rows = []
    for r in hf['rows']:
        ref = np.load(ROOT / r['npy']).astype(np.float64)                   # [64, 2048]
        assert sha256_file(ROOT / r['npy']) == r['npy_sha256']
        feats = enc(load01(ROOT / f'out/fixtures_resized/g{G}/{r["row_id"]}.png'))
        got = adp(feats).reshape(ref.shape).astype(np.float64)
        d = np.abs(got - ref)
        rows.append(dict(row_id=r['row_id'], tier=r['tier'], corr=float(np.corrcoef(got.ravel(), ref.ravel())[0, 1]),
                         max_abs_diff=float(d.max()), mean_abs_diff=float(d.mean()), ref_absmax=float(np.abs(ref).max()),
                         rel_max=float(d.max() / np.abs(ref).max()),
                         token_cos_min=float(np.min(np.sum(got * ref, 1) / (np.linalg.norm(got, axis=1) * np.linalg.norm(ref, axis=1))))))
        print(r['row_id'], f"corr {rows[-1]['corr']:.10f} max|d| {rows[-1]['max_abs_diff']:.3e}", flush=True)
    res = dict(reference=f"transformers {hf['transformers']} model.model.visual(...).pooler_output (out/hf_vision_g{G}/)",
               runtime='ai-edge-litert CompiledModel CPU', threads=args.threads, G=G,
               tflites={k: dict(sha256=conv['files'][f'{k}.tflite']['sha256'], bytes=conv['files'][f'{k}.tflite']['bytes'])
                        for k in ('vision_encoder', 'vision_adapter')},
               op_tables=ops, n_rows=len(rows),
               corr_min=min(x['corr'] for x in rows), max_abs_diff_max=max(x['max_abs_diff'] for x in rows),
               rel_max_max=max(x['rel_max'] for x in rows), token_cos_min=min(x['token_cos_min'] for x in rows),
               forbidden_ops_total=sum(v['forbidden_total'] for v in ops.values()),
               wall_seconds_contended=time.monotonic() - t0, rows=rows)
    write_json('results/vision_parity_r2.json', res)
    print('VISION_PARITY rows', res['n_rows'], 'corr_min', res['corr_min'], 'max_abs_diff_max', res['max_abs_diff_max'],
          'forbidden', res['forbidden_ops_total'], flush=True)


if __name__ == '__main__':
    main()
