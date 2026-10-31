#!/usr/bin/env python3
"""export_vision.py --src SRC --out DIR [--g 512] - the LFM3-VL-2B vision encoder and projector (tools/lfm3_vision.py,
re-written for one fixed G x G picture) converted with litert-torch, in the two forms the published files carry:

  DIR/vision_encoder_g<G>_v2.tflite       fp32 weights            -> the VISION_ENCODER section
  DIR/vision_adapter_g<G>_v2_fp16.tflite  fp16 weights            -> the VISION_ADAPTER section

export_tower also writes the fp32 adapter and the fp16 encoder (the fp16 forms are derived from the fp32 graphs);
the published files do not use them. SRC = the source snapshot (config.json, model.safetensors). The graph takes the
runtime's float(u8) / 255 input back to the exact u8 and normalises it as the source processor does (input_norm "u8").
Writes DIR/export_vision.json (versions, seconds, sha256 of every file).
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
  a = ap.parse_args()
  torch.set_grad_enabled(False)
  src, out = os.path.abspath(a.src), os.path.abspath(a.out)
  t0 = time.time()
  enc, adp, info = lv.build_tower(os.path.join(src, "model.safetensors"), os.path.join(src, "config.json"), a.g,
                                  opts=lv.Opts(input_norm="u8"))
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
  json.dump(rec, open(os.path.join(out, "export_vision.json"), "w"), indent=1)
  print("EXPORT_VISION_DONE", json.dumps(rec["files"]), flush=True)
  return 0


if __name__ == "__main__":
  sys.exit(main())
