#!/usr/bin/env python3
"""export_vision.py WEIGHTS CONFIG_JSON OUT_DIR - LFM3-Omni-2B's vision tower and projector at a fixed 512 x 512 grid
(256 image tokens), through LFM3-VL-2B's re-written tower (lfm3vl_work/tools/lfm3_vision.py: the 2D rotary tables and
positions as constants, the runtime's u8 input taken back exactly, raster order with the projector's 2 x 2 merge in the
source order). The default prefixes `model.vision_tower.` / `model.multi_modal_projector.` are Omni's names too.

Writes OUT_DIR/vision_encoder_g512.tflite (float32), vision_adapter_g512.tflite (float32) and
vision_adapter_g512_fp16.tflite (float16 weights); the .litertlm takes the float32 encoder and the float16 adapter.
lfm3_vision.py is loaded by path under the module name `lfm3_peer_vision`, which appears in the graphs' tensor names.
"""
import importlib.util
import pathlib
import sys
import time

R = pathlib.Path(__file__).resolve().parents[1]
G = 512


def load_by_path(name, path):
  spec = importlib.util.spec_from_file_location(name, path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def main():
  if len(sys.argv) != 4:
    print(__doc__, file=sys.stderr)
    return 2
  weights, config, out = sys.argv[1:]
  lv = load_by_path("lfm3_peer_vision", R / "lfm3vl_work" / "tools" / "lfm3_vision.py")
  t0 = time.time()
  enc, adp, _info = lv.build_tower(str(pathlib.Path(weights).resolve()), str(pathlib.Path(config).resolve()), G)
  paths = lv.export_tower(enc, adp, str(pathlib.Path(out).resolve()), forms=("fp32", "fp16"), tag="g512")
  for p in sorted(pathlib.Path(out).glob("vision_*.tflite")):
    print(f"exported {p} ({p.stat().st_size} B)", flush=True)
  print(f"{len(paths) if hasattr(paths, '__len__') else ''} files in {time.time() - t0:.0f} s", flush=True)
  return 0


if __name__ == "__main__":
  sys.exit(main())
