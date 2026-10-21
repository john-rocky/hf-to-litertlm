#!/usr/bin/env python3
"""Round 1: the NumPy host pieces (scripts/clef_host.py) against torch / transformers 5.14.1 (lt094dev), no model load.

    ~/venvs/lt094dev/bin/python scripts/host_selftest.py       -> results/host_selftest.json (refuses to overwrite)

1. inv_freq: clef_host.inv_freq() vs Qwen3_5TextRotaryEmbedding.compute_default_rope_parameters(text_config).
2. M-RoPE tables vs the stock Qwen3_5TextRotaryEmbedding.forward on the same planes: text positions 0..1023 (the row
   graph's L1024 table) and an image-shaped record (the Core AI closed form: prefix P = 36, merged grid 8 x 8, the
   positions after the image shifted by N - max(H, W)) padded to 1024; max |diff|, count of differing elements,
   largest difference in ulps.
3. Embedding rows: EmbedTable.rows(ids) (bf16 bits widened in NumPy) vs safetensors' torch slice .to(float32), bit
   for bit, for 4,096 ids (every id of the first 4,096 rows' pattern plus the special ids)."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "scripts"))
import clef_host as H  # noqa: E402

SNAP = K / "hf/hub/models--Cloudflare--clef-flash/snapshots/f0b7d3cd65296abfe459e6c228b618f5d8b26d2f"


def ulps(a: np.ndarray, b: np.ndarray) -> int:
    ia, ib = a.astype(np.float32).view(np.int32).astype(np.int64), b.astype(np.float32).view(np.int32).astype(np.int64)
    return int(np.abs(ia - ib).max())


def image_planes(P: int = 36, H_: int = 8, W: int = 8, n_after: int = 300) -> np.ndarray:
    N = H_ * W
    T = P + 1 + N + 1 + n_after
    pos = np.broadcast_to(np.arange(T, dtype=np.int64), (3, T)).copy()
    k = np.arange(N)
    pos[0, P + 1:P + 1 + N] = P + 1
    pos[1, P + 1:P + 1 + N] = P + 1 + k // W
    pos[2, P + 1:P + 1 + N] = P + 1 + k % W
    pos[:, P + N + 1:] = np.arange(P + N + 1, T) - (N - max(H_, W))
    return pos


def main() -> int:
    out = K / "results/host_selftest.json"
    assert not out.exists(), f"refusing to overwrite {out}"
    from transformers import AutoConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    cfg = AutoConfig.from_pretrained(str(SNAP)).text_config
    res = {"transformers": __import__("transformers").__version__, "torch": torch.__version__}
    tf_freq, scaling = M.Qwen3_5TextRotaryEmbedding.compute_default_rope_parameters(cfg)
    np_freq = H.inv_freq()
    res["inv_freq"] = {"n": int(np_freq.size), "bit_equal": bool(np.array_equal(tf_freq.numpy(), np_freq)),
                       "max_ulps": ulps(tf_freq.numpy(), np_freq), "attention_scaling": scaling,
                       "rope_parameters": cfg.rope_parameters}
    rot = M.Qwen3_5TextRotaryEmbedding(cfg)
    L = 1024
    for name, pos3 in (("text_L1024", H.text_positions(L)), ("image_g256_like_padded_L1024", H.padded_positions(image_planes(), L))):
        cos_np, sin_np = H.mrope_tables(pos3)
        with torch.no_grad():
            cos_t, sin_t = rot(torch.zeros(1, L, 8), torch.from_numpy(pos3)[:, None, :])
        cos_t, sin_t = cos_t.numpy(), sin_t.numpy()
        res[name] = {"shape": list(cos_np.shape), "torch_shape": list(cos_t.shape),
                     "cos_max_abs": float(np.abs(cos_np - cos_t).max()), "sin_max_abs": float(np.abs(sin_np - sin_t).max()),
                     "cos_differing": int((cos_np != cos_t).sum()), "sin_differing": int((sin_np != sin_t).sum()),
                     "cos_max_ulps": ulps(cos_np, cos_t), "sin_max_ulps": ulps(sin_np, sin_t), "elements": int(cos_np.size)}
    table = H.EmbedTable(SNAP)
    ids = np.unique(np.concatenate([np.arange(0, 248320, 61), [0, 1, 248044, 248046, 248053, 248054, 248056, 248319]]))
    from safetensors import safe_open
    with safe_open(str(SNAP / table.info["shard"]), framework="pt") as f:
        ref = f.get_slice(H.EMBED_KEY)
        want = torch.stack([ref[int(i):int(i) + 1][0] for i in ids]).to(torch.float32).numpy()
    got = table.rows(ids)
    res["embed_rows"] = {**table.info, "ids_checked": int(ids.size), "bit_equal": bool(np.array_equal(got.view(np.uint32), want.view(np.uint32))),
                         "row_absmax": float(np.abs(got).max())}
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
