"""Save transformers 5.17.0's position-embedding interpolation taps for a static GxG image, raster order.

The vision export runs in a transformers 5.14.1 venv whose helper (get_vision_bilinear_indices_and_weights) picks
the same taps as 5.17.0's get_vision_interpolation_indices_and_weights(mode='bilinear', align_corners=True) but
rounds the weights differently (max |dw| 3.6e-6 at G=256, measured in round 2). The parity reference is 5.17.0,
so the converter consumes these taps instead of recomputing them.

    out/venv-oracle/bin/python -B scripts/vision_interp_5170.py [--img 256]
"""
import argparse

import numpy as np
import torch
import transformers
from transformers.models.qwen3_5 import modeling_qwen3_5
from transformers.vision_utils import get_vision_interpolation_indices_and_weights

from common import ROOT, SNAPSHOT, read_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img', type=int, default=256)
    args = ap.parse_args()
    cfg = read_json(SNAPSHOT.relative_to(ROOT) / 'config.json')['vision_config']
    side = int(cfg['num_position_embeddings'] ** 0.5)
    grid = args.img // cfg['patch_size']
    # the model sets these two attributes in Qwen3_5VisionModel.__init__; read them from the installed source
    src = open(modeling_qwen3_5.__file__).read()
    assert 'self.interpolation_align_corners = True' in src and 'self.interpolation_mode = "bilinear"' in src
    idx, w = get_vision_interpolation_indices_and_weights(torch.tensor([[1, grid, grid]]), num_grid_per_side=side,
                                                          mode='bilinear', align_corners=True, spatial_merge_size=1)
    out = ROOT / f'out/vision_g{args.img}/interp_5170_g{args.img}.npz'
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, indices=idx.numpy(), weights=w.numpy(), grid=np.array([1, grid, grid]),
             num_grid_per_side=np.array(side), transformers=np.array(transformers.__version__))
    print(out, idx.shape, w.shape, transformers.__version__)


if __name__ == '__main__':
    main()
