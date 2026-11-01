#!/usr/bin/env python3
"""tower_gpu.py - the LFM3-VL vision graphs on the Mac GPU (ai-edge-litert CompiledModel, Metal), GPU only, at fp32
and at fp16 precision, against the source model's image features.

  python -I tower_gpu.py --tower-dir OUTDIR/tower --ref-dir REF/ref_npy --out tower_gpu_g512.json [--g 512]
      [--forms fp32,fp16] [--fixtures lfm3vl_work/fixtures]

One fresh process per graph form x precision (fp32 = GpuOptions(enforce_f32=True), fp16 = the default). The accelerator
set is the GPU alone: with GPU + CPU a graph the delegate only partly takes still builds, so a GPU-only build is the
test of full delegation. The delegate's "not supported" lines are kept. When both graphs build: on every fixture
picture the encoder against HF `hid`, the adapter on HF's own `hid` against HF `feat`, and the GPU encoder -> GPU
adapter against HF `feat` (per-token cosine minimum, max |d| / max |ref|). Inputs as tower_tfl.py.
Exit 0 = every cell built and its outputs are finite (accuracy is reported, not judged: compare the fp16 column with
the fp32 one).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

CHILD = r'''
import json, os, sys
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, sys.argv[1])
import lfm3_vision as lv
from ai_edge_litert.compiled_model import CompiledModel, GpuOptions, HardwareAccelerator, Options
_, _, fxdir, ref_dir, g, enc_p, adp_p, prec = sys.argv
g = int(g)
class G:
  def __init__(self, path):
    self.m = CompiledModel.from_file(path, options=Options(hardware_accelerators=HardwareAccelerator.GPU,
                                                           gpu_options=GpuOptions(enforce_f32=prec == "fp32")))
    self.key = next(iter(self.m.get_signature_list()))
    self.i = self.m.get_input_tensor_details(self.key); self.o = self.m.get_output_tensor_details(self.key)
    self.ib = {n: self.m.create_input_buffer_by_name(self.key, n) for n in self.i}
    self.ob = {n: self.m.create_output_buffer_by_name(self.key, n) for n in self.o}
  def __call__(self, x):
    (n,), (on,) = self.i, self.o
    self.ib[n].write(np.ascontiguousarray(x, dtype=self.i[n]["dtype"]).reshape(self.i[n]["shape"]))
    self.m.run_by_name(self.key, self.ib, self.ob)
    d = self.o[on]
    return np.array(self.ob[on].read(int(np.prod(d["shape"])), d["dtype"]), copy=True).reshape(d["shape"])
def met(a, r):
  a = np.asarray(a, np.float64).reshape(-1, r.shape[-1]); r = np.asarray(r, np.float64).reshape(a.shape)
  c = (a * r).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(r, axis=-1) + 1e-30)
  return float(c.min()), float(np.abs(a - r).max() / np.abs(r).max()), bool(np.isfinite(a).all())
fx = json.load(open(os.path.join(fxdir, "fixtures.json")))
gh = g // 16
enc = G(enc_p); adp = G(adp_p)
print("GPU_ONLY_BUILT", flush=True)
rows = {}
for iid, rec in fx["images"].items():
  x = (np.asarray(Image.open(os.path.join(fxdir, rec["g%d" % g])).convert("RGB"), dtype=np.float32) / 255.0)[None]
  hid_ref = np.load(os.path.join(ref_dir, iid + ".hid.npy")); feat_ref = np.load(os.path.join(ref_dir, iid + ".feat.npy"))
  e = enc(x)
  hid = lv.raster_to_merge(torch.from_numpy(e), gh, gh)[0].numpy()
  a_on_hf = adp(lv.merge_to_raster(torch.from_numpy(hid_ref)[None], gh, gh).numpy())
  rows[iid] = {"encoder_vs_hid": met(hid, hid_ref), "adapter_on_hf_hid": met(a_on_hf, feat_ref), "end2end": met(adp(e), feat_ref)}
out = {"rows": rows}
for k in ("encoder_vs_hid", "adapter_on_hf_hid", "end2end"):
  out[k + "_cos_min"] = min(v[k][0] for v in rows.values())
  out[k + "_worst_picture"] = min(rows, key=lambda i: rows[i][k][0])
  out[k + "_rel_max"] = max(v[k][1] for v in rows.values())
  out[k + "_finite"] = all(v[k][2] for v in rows.values())
print("RESULT " + json.dumps(out), flush=True)
'''


def run_child(fxdir, ref_dir, g, enc, adp, prec):
  t0 = time.time()
  p = subprocess.run([sys.executable, "-I", "-c", CHILD, HERE, fxdir, ref_dir, str(g), enc, adp, prec],
                     capture_output=True, text=True, timeout=3600, stdin=subprocess.DEVNULL)
  raw = (p.stdout + p.stderr).splitlines()
  lines, inside = [], False
  for ln in raw:
    if "not supported" in ln.lower():
      inside = True
    if inside or re.search(r"GPU_ONLY_BUILT|Some ops are not accelerated|RuntimeError|Error", ln):
      lines.append(ln.strip()[:300])
    if "operations will run" in ln:
      inside = False
  res = next((json.loads(ln[len("RESULT "):]) for ln in raw if ln.startswith("RESULT ")), None)
  return {"rc": p.returncode, "s": round(time.time() - t0, 1), "built": "GPU_ONLY_BUILT" in p.stdout, "result": res,
          "lines": lines[:40]}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--tower-dir", required=True)
  ap.add_argument("--ref-dir", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--g", type=int, default=512)
  ap.add_argument("--forms", default="fp32,fp16")
  ap.add_argument("--fixtures", default=os.path.join(HERE, "..", "fixtures"))
  args = ap.parse_args()
  fxdir = os.path.abspath(args.fixtures)
  ref_dir = os.path.join(os.path.abspath(args.ref_dir), f"g{args.g}")
  res, ok = {"g": args.g}, True
  for form in args.forms.split(","):
    t = "" if form == "fp32" else f"_{form}"
    enc, adp = (os.path.join(args.tower_dir, f"vision_{k}_g{args.g}_v2{t}.tflite") for k in ("encoder", "adapter"))
    for prec in ("fp32", "fp16"):
      r = run_child(fxdir, ref_dir, args.g, enc, adp, prec)
      res[f"{form}|gpu|{prec}"] = r
      rr = r["result"] or {}
      ok &= r["built"] and bool(rr.get("end2end_finite"))
      print(f"G{args.g} weights {form}, GPU precision {prec}: built={r['built']} rc={r['rc']} "
            f"end2end cos_min {rr.get('end2end_cos_min')} ({rr.get('end2end_worst_picture')}) "
            f"encoder {rr.get('encoder_vs_hid_cos_min')} adapter on HF hid {rr.get('adapter_on_hf_hid_cos_min')} "
            f"finite {rr.get('end2end_finite')}" + (f" | {r['lines'][:3]}" if not r["built"] else ""), flush=True)
      json.dump(res, open(args.out, "w"), indent=1)
  print("TOWER_GPU_DONE", args.g, "OK" if ok else "NOT_BUILT_OR_NONFINITE", flush=True)
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
