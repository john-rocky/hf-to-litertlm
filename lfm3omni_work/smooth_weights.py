#!/usr/bin/env python3
"""smooth_weights.py SCALES.npz HF_DIR OUT.safetensors - LFM3-Omni-2B's language-model weights with per-channel scales
folded in before int8 quantization (the SmoothQuant form). The decoder computes the same function in float32: each
scale multiplies one input channel of a linear layer and divides, by the same value, the output that feeds that channel.

SCALES.npz holds one float32 vector per `<layer>:<site>`, applied in the file's order:
  w2        column j of feed_forward.w2 * s_j, row j of feed_forward.w3 / s_j
  out_proj  column j of conv.out_proj * s_j, row 2048 + j of conv.in_proj / s_j (the second of its three chunks, which
            multiplies the convolution's output)
  opnorm    operator_norm weight j / s_j, column j * s_j of what it feeds: conv.in_proj in a convolution layer,
            self_attn.{q,k,v,gate}_proj in an attention layer
  ffnnorm   ffn_norm weight j / s_j, column j * s_j of feed_forward.w1 and feed_forward.w3
The scales were fitted as s_j = amax_j^0.65 / wmax_j^0.35, clipped to [1e-4, 1e4]: amax_j = the largest |input_j| of
the site over 16 calibration sequences (text, pictures and audio, none of them a test prompt), wmax_j = the largest |W[:, j]|
over the weight matrices that read channel j. Every changed tensor is written in float32 (the
scaled values are not bfloat16 numbers), every other tensor keeps its bfloat16 bytes; the decoder export reads both as
float32. HF_DIR = the source checkout (config.json, model.safetensors).
"""
import json
import pathlib
import sys

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

PRE = "model.language_model.layers."


def main():
  if len(sys.argv) != 4:
    print(__doc__, file=sys.stderr)
    return 2
  scales_path, hf_dir, out_path = (pathlib.Path(p) for p in sys.argv[1:])
  sc = np.load(scales_path)
  layer_types = json.load(open(hf_dir / "config.json"))["text_config"]["layer_types"]
  with safe_open(str(hf_dir / "model.safetensors"), "pt") as f:
    tensors = {k: f.get_tensor(k) for k in f.keys()}
  touched = {}

  def get(key):
    if key not in touched:
      touched[key] = tensors[key].float().clone()
    return touched[key]

  for item in sc.files:
    l, site = item.split(":")
    s = torch.from_numpy(sc[item]).float()
    p = f"{PRE}{l}."
    conv = layer_types[int(l)] == "conv"
    if site == "w2":
      get(p + "feed_forward.w2.weight").mul_(s[None, :])
      get(p + "feed_forward.w3.weight").div_(s[:, None])
    elif site == "out_proj":
      get(p + "conv.out_proj.weight").mul_(s[None, :])
      get(p + "conv.in_proj.weight")[2048:4096].div_(s[:, None])
    elif site == "opnorm":
      get(p + "operator_norm.weight").div_(s)
      fed = ["conv.in_proj"] if conv else ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                                           "self_attn.gate_proj"]
      for n in fed:
        get(p + n + ".weight").mul_(s[None, :])
    elif site == "ffnnorm":
      get(p + "ffn_norm.weight").div_(s)
      for n in ("feed_forward.w1", "feed_forward.w3"):
        get(p + n + ".weight").mul_(s[None, :])
    else:
      raise ValueError(site)
  out = {k: (touched[k].contiguous() if k in touched else v.contiguous()) for k, v in tensors.items()}
  out_path.parent.mkdir(parents=True, exist_ok=True)
  tmp = out_path.with_name(out_path.name + ".part")
  save_file(out, str(tmp), metadata={"format": "pt"})
  tmp.rename(out_path)
  print(f"rewrote {len(touched)} tensors from {len(sc.files)} scale vectors -> {out_path} ({out_path.stat().st_size} B)",
        flush=True)
  return 0


if __name__ == "__main__":
  sys.exit(main())
