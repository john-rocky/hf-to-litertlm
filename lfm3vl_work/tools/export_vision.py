#!/usr/bin/env python3
"""export_vision.py --src SRC --out DIR [--g 512] - the LFM3-VL-2B vision encoder and projector (tools/lfm3_vision.py,
re-written for one fixed G x G picture) converted with litert-torch, in the two forms the published files carry:

  DIR/vision_encoder_g<G>_v2.tflite       fp32 weights            -> the VISION_ENCODER section
  DIR/vision_adapter_g<G>_v2_fp16.tflite  fp16 weights            -> the VISION_ADAPTER section

export_tower also writes the fp32 adapter and the fp16 encoder (the fp16 forms are derived from the fp32 graphs);
the published files do not use them. SRC = the source snapshot (config.json, model.safetensors). The graph takes the
runtime's float(u8) / 255 input back to the exact u8 and normalises it as the source processor does (input_norm "u8").
Writes DIR/export_vision.json (versions, seconds, sha256 of every file).

--form u8c (the input step centred before the FLOOR) and --form fs --ranges RANGES (u8c plus power-of-two norm scales
from lfm3_vision.fp16_safe_scales over RANGES, the output of tower_cuts.py ranges) write the other two forms of
docs/vlm_fixed_grid_tower.md under the same file names, for tower_tfl.py and tower_gpu.py; the default is --form u8.
They refuse a DIR that already holds vision graphs: export_tower keeps a file that exists, and convert_lfm3_vl.sh
packs whatever OUT/tower holds, so each form needs a folder of its own.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lfm3_vision as lv  # noqa: E402


def sha256(p):
  h = hashlib.sha256()
  with open(p, "rb") as f:
    for c in iter(lambda: f.read(1 << 24), b""):
      h.update(c)
  return h.hexdigest()


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--src", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--g", type=int, default=512)
  ap.add_argument("--form", default="u8", choices=("u8", "u8c", "fs"))
  ap.add_argument("--ranges", default=None, help="tower_cuts.py ranges output (--form fs)")
  a = ap.parse_args()
  if a.form == "fs" and not a.ranges:
    sys.exit("--form fs needs --ranges (the output of tower_cuts.py ranges)")
  if a.form == "fs":
    rms_s, ln_s = lv.fp16_safe_scales(json.load(open(a.ranges))["sites"])
    opts = lv.Opts(input_norm="u8c", rms_s=rms_s, ln_s=ln_s)
  else:
    opts = lv.Opts(input_norm=a.form)
  torch.set_grad_enabled(False)
  src, out = os.path.abspath(a.src), os.path.abspath(a.out)
  if a.form != "u8" and os.path.isdir(out):
    held = sorted(f for f in os.listdir(out) if f.startswith("vision_") and f.endswith(".tflite"))
    if held:
      sys.exit(f"--form {a.form} needs a folder of its own: {out} already holds {held[0]} (export_tower keeps a file "
               "that exists, and convert_lfm3_vl.sh packs OUT/tower)")
  t0 = time.time()
  enc, adp, info = lv.build_tower(os.path.join(src, "model.safetensors"), os.path.join(src, "config.json"), a.g,
                                  opts=opts)
  build_s = round(time.time() - t0, 1)
  t0 = time.time()
  paths = lv.export_tower(enc, adp, out, forms=("fp32", "fp16"), tag=f"g{a.g}_v2")
  export_s = round(time.time() - t0, 1)
  import importlib.metadata as md
  rec = {"g": a.g, "build_s": build_s, "export_s": export_s,
         "info": {k: v for k, v in info.items() if k != "layer_types"},
         "versions": {p: md.version(p) for p in ("torch", "litert-torch", "litert-converter", "ai-edge-litert",
                                                 "ai-edge-quantizer")},
         "files": {os.path.basename(p): {"bytes": os.path.getsize(p), "sha256": sha256(p)}
                   for v in paths.values() for p in v.values()}}
  if a.form != "u8":
    rec["form"] = {"form": a.form, "ranges": a.ranges and os.path.abspath(a.ranges), "rms_s": opts.rms_s, "ln_s": opts.ln_s}
  json.dump(rec, open(os.path.join(out, "export_vision.json"), "w"), indent=1)
  print("EXPORT_VISION_DONE", json.dumps(rec["files"]), flush=True)
  return 0


if __name__ == "__main__":
  sys.exit(main())
