"""Dump transformers 5.17.0's vision output for every fixture image (v1 + v2) at G = 256: the parity reference for
the vision tflites and the image embeddings of the decoder-only readout arm (ii).

Per image row: the GxG PNG (out/fixtures_resized/g256/<row>.png, the oracle's own g256 input) -> the checkpoint's
processor (the same object VisionDecisionModel.prepare uses) -> pixel_values [256, 1536], grid [1, 16, 16] ->
model.model.visual(pixel_values, grid_thw).pooler_output [64, 2048] fp32. The pixel_values sha256 is checked against
the one the oracle recorded for that row's g256_mrope forward, and the output is checked against
get_image_features() (the call the oracle's forward makes before scattering into the image slots).

    out/venv-oracle/bin/python -B scripts/dump_hf_vision_g256.py [--threads 8]
"""
import os
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
import argparse
import sys
import time

import numpy as np
import torch
import transformers
from PIL import Image

from common import ROOT, SNAPSHOT, read_json, write_json, sha256_file, sha256_bytes, sha256_rgb

sys.path.insert(0, str(SNAPSHOT))
from decider.vision import VisionDecisionModel                             # noqa: E402

G = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threads', type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.monotonic()
    fx = read_json('fixtures/fixtures_v2.json')
    oracle = {}
    for path in ('fixtures/oracle_fp32.json', 'fixtures/oracle_fp32_v2add.json'):
        for f in read_json(path)['forwards']:
            if f['arm'] == f'g{G}_mrope' and f['n_image_tokens']:
                oracle[f['row_id']] = f
    m = VisionDecisionModel(str(SNAPSHOT), dtype=torch.float32, grad_ckpt=False).eval()
    visual = m.lm.model.visual
    out_dir = ROOT / f'out/hf_vision_g{G}'
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in fx['rows']:
        if r['image'] is None:
            continue
        rid = r['id']
        png = ROOT / f'out/fixtures_resized/g{G}/{rid}.png'
        pil = Image.open(png).convert('RGB')
        o = oracle[rid]
        assert sha256_rgb(pil) == o['image_input']['sha256_rgb'], rid
        pp = m.proc.image_processor(images=[pil], return_tensors='pt')
        pv, grid = pp['pixel_values'], pp['image_grid_thw']
        assert grid.tolist() == [[1, G // 16, G // 16]], (rid, grid)
        pv_sha = sha256_bytes(pv.contiguous().numpy().tobytes())
        with torch.no_grad():
            emb = visual(pv, grid_thw=grid).pooler_output.float()          # [64, 2048]
            gif = m.lm.model.get_image_features(pv, grid).pooler_output[0].float()
        arr = emb.numpy()
        np.save(out_dir / f'{rid}.npy', arr)
        rows.append(dict(row_id=rid, tier=r['tier'], shape=list(arr.shape), npy=f'out/hf_vision_g{G}/{rid}.npy',
                         npy_sha256=sha256_file(out_dir / f'{rid}.npy'), pixel_values_sha256=pv_sha,
                         pixel_values_equal_oracle=pv_sha == o['pixel_values_sha256'],
                         equals_get_image_features=bool(torch.equal(emb, gif)), absmax=float(np.abs(arr).max())))
        print(rid, arr.shape, 'pv==oracle', rows[-1]['pixel_values_equal_oracle'], 'gif', rows[-1]['equals_get_image_features'], flush=True)
    res = dict(transformers=transformers.__version__, torch=torch.__version__, torch_threads=args.threads, G=G,
               snapshot=str(SNAPSHOT.relative_to(ROOT)), n_rows=len(rows),
               n_pixel_values_equal_oracle=sum(x['pixel_values_equal_oracle'] for x in rows),
               n_equals_get_image_features=sum(x['equals_get_image_features'] for x in rows),
               wall_seconds_contended=time.monotonic() - t0, rows=rows)
    write_json(f'out/hf_vision_g{G}/index.json', res)
    print('HF_VISION_DUMP', res['n_rows'], res['n_pixel_values_equal_oracle'], res['n_equals_get_image_features'], flush=True)
    assert res['n_pixel_values_equal_oracle'] == res['n_rows'] == res['n_equals_get_image_features']


if __name__ == '__main__':
    main()
