#!/usr/bin/env python3
"""r2 step 2: HF reference for the MiniCPM-V-4.7 vision path on the 448x448 fixtures.

  python vision_ref.py --src SRC --fixtures W/fixtures \
      --out W/ref --stages W/stages        (transformers 5.19.0; W = a work directory)

The real MiniCPMV4_7ForConditionalGeneration (fp32, eager attention). Per fixture PNG
(<name>_448.png, already 448x448, so the processor takes the single-thumbnail path):
  pixel_values  [1,3,14,14336]  processor strip (MiniCPMV4_6ImageProcessor via AutoProcessor)
  target_sizes  [[32,32]]
  enc_last_hidden [1,256,1152]  get_image_features(...).last_hidden_state
                                (= vision_tower(use_vit_merger=True) -> post_layernorm)
  feat          [64,1024]       get_image_features(...).pooler_output[0]
  pos_ids       [1024]          get_vision_nearest_position_ids(target_sizes, 70)
  window_index  [1024], cu_window_seqlens [257]  vit_merger.get_window_index(target_sizes)
  seconds       wall time of get_image_features
-> OUT/<name>.npz (small, the parity reference) and STAGES/<name>.npz (per-stage hooks:
embeddings, every encoder layer output, vit_merger output; debugging only, deletable).
A second direct call vision_tower(pixel_values, target_sizes, use_vit_merger=True) is
checked against enc_last_hidden (max|d| printed).
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import transformers
from PIL import Image
from transformers import AutoProcessor, MiniCPMV4_7ForConditionalGeneration
from transformers.vision_utils import get_vision_nearest_position_ids


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--src", required=True)
  ap.add_argument("--fixtures", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--stages", required=True)
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)
  os.makedirs(args.stages, exist_ok=True)

  t0 = time.time()
  proc = AutoProcessor.from_pretrained(args.src)
  model = MiniCPMV4_7ForConditionalGeneration.from_pretrained(
      args.src, dtype=torch.float32, attn_implementation="eager").eval()
  load_s = time.time() - t0
  vt = model.model.vision_tower
  cfg = model.config
  assert cfg.downsample_mode == "16x" and vt.config.insert_layer_id == 6, (cfg.downsample_mode, vt.config.insert_layer_id)
  print(f"loaded in {load_s:.1f} s; image processor {type(proc.image_processor).__name__}; "
        f"vision attn {vt.config._attn_implementation}", flush=True)

  taps = {}

  def hook(name):
    def fn(_m, _i, out):
      o = out[0] if isinstance(out, tuple) else out
      taps[name] = o.detach().float().cpu().numpy().copy()
    return fn

  handles = [vt.embeddings.register_forward_hook(hook("embeddings")),
             vt.vit_merger.register_forward_hook(hook("vit_merger"))]
  for i, layer in enumerate(vt.encoder.layers):
    handles.append(layer.register_forward_hook(hook(f"layer{i:02d}")))

  summary = {"transformers": transformers.__version__, "torch": torch.__version__,
             "src": args.src, "load_seconds": round(load_s, 1),
             "image_processor_class": type(proc.image_processor).__name__,
             "vision_attn_implementation": vt.config._attn_implementation, "fixtures": []}
  for path in sorted(glob.glob(os.path.join(args.fixtures, "*_448.png"))):
    name = os.path.basename(path)[: -len("_448.png")]
    img = Image.open(path).convert("RGB")
    pp = proc.image_processor(images=[img], return_tensors="pt")
    pv, ts = pp["pixel_values"], pp["target_sizes"]
    assert list(pv.shape) == [1, 3, 14, 14336], pv.shape
    assert ts.tolist() == [[32, 32]], ts
    taps.clear()
    t1 = time.time()
    with torch.no_grad():
      out = model.model.get_image_features(pv, ts)
    sec = time.time() - t1
    enc = out.last_hidden_state.float().numpy()
    feat = out.pooler_output[0].float().numpy()
    stage = dict(taps)
    with torch.no_grad():
      enc2 = vt(pv, target_sizes=ts, use_vit_merger=True).last_hidden_state.float().numpy()
    pos_ids = get_vision_nearest_position_ids(ts, vt.embeddings.num_patches_per_side).numpy()
    widx, wcu, _ = vt.vit_merger.get_window_index(ts)
    np.savez(os.path.join(args.out, f"{name}.npz"),
             pixel_values=pv.float().numpy(), target_sizes=ts.numpy(),
             enc_last_hidden=enc, feat=feat, pos_ids=pos_ids,
             window_index=widx.numpy(), cu_window_seqlens=wcu.numpy(),
             seconds=np.array(sec))
    np.savez(os.path.join(args.stages, f"{name}.npz"), **stage)
    row = {"name": name, "pixel_values_shape": list(pv.shape), "target_sizes": ts.tolist(),
           "enc_last_hidden_shape": list(enc.shape), "feat_shape": list(feat.shape),
           "enc_absmax": float(np.abs(enc).max()), "feat_absmax": float(np.abs(feat).max()),
           "direct_vision_tower_maxdiff": float(np.abs(enc - enc2).max()),
           "pos_ids_head": pos_ids[:8].tolist(), "pos_ids_unique": int(len(np.unique(pos_ids))),
           "window_index_head": widx[:8].tolist(), "n_windows": int(len(wcu) - 1),
           "stage_keys": sorted(stage), "seconds": round(sec, 2)}
    summary["fixtures"].append(row)
    print(json.dumps(row), flush=True)
  for h in handles:
    h.remove()
  with open(os.path.join(args.out, "ref_summary.json"), "w") as f:
    json.dump(summary, f, indent=1)
    f.write("\n")
  print("REF_DONE", flush=True)


if __name__ == "__main__":
  main()
