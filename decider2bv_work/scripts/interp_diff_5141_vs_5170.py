"""Record the difference between transformers 5.14.1's and 5.17.0's position-embedding interpolation at G = 256
(same bilinear taps, differently rounded weights), which is why the vision export consumes 5.17.0's taps.

    PYTHONDONTWRITEBYTECODE=1 $PY_VISION -B scripts/interp_diff_5141_vs_5170.py
"""
import numpy as np
import torch
import transformers
from transformers.vision_utils import get_vision_bilinear_indices_and_weights

from common import ROOT, write_json, sha256_file


def main():
    npz = ROOT / 'out/vision_g256/interp_5170_g256.npz'
    tap = np.load(npz)
    idx14, w14 = get_vision_bilinear_indices_and_weights(torch.tensor([[1, 16, 16]]), num_grid_per_side=48, spatial_merge_size=1)
    idx14, w14 = idx14.numpy().T, w14.numpy().T                        # [4, N] -> [N, 4]
    same_taps = bool(np.array_equal(idx14.astype(np.int64), tap['indices']))
    dw = np.abs(w14 - tap['weights'])
    torch.manual_seed(0)
    table = torch.randn(48 * 48, 1024)
    a = (table[torch.from_numpy(idx14.astype(np.int64))] * torch.from_numpy(w14)[:, :, None]).sum(1)
    b = (table[torch.from_numpy(tap['indices'])] * torch.from_numpy(tap['weights'])[:, :, None]).sum(1)
    res = dict(transformers_export=transformers.__version__, transformers_reference=str(tap['transformers']),
               npz=str(npz.relative_to(ROOT)), npz_sha256=sha256_file(npz), same_taps_same_order=same_taps,
               max_abs_weight_diff=float(dw.max()), n_weights_different=int((dw > 0).sum()), n_weights=int(dw.size),
               resampled_random_table_max_abs_diff=float((a - b).abs().max()))
    write_json('results/vision_interp_diff.json', res)
    print(res)


if __name__ == '__main__':
    main()
