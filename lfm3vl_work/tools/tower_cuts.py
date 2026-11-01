#!/usr/bin/env python3
"""tower_cuts.py - find where the LFM3-VL vision encoder breaks at the GPU's fp16 precision: graphs cut after each
stage, run on Metal at fp32 and at fp16 precision; and the value ranges that pick fp16-safe norm scales.

  python -I tower_cuts.py ranges --src SRC --out ranges.json [--g 512] [--fixtures lfm3vl_work/fixtures]
      host torch fp32 (lfm3_vision.py, the default graph) over the fixture pictures: per norm input the absmax, the
      largest row sum of x^2 (LayerNorm: of (x - mean)^2), the largest x^2, the smallest / median / largest row mean
      of x^2, the share of elements whose square is below fp16's smallest normal 2^-14 and below 2^-25 (rounds to 0 in
      fp16); the absmax of the other intermediates (every Linear output, the MLP product, the conv mixer's input and
      output, the attention scores, the patch conv output, the encoder and adapter outputs).
      lfm3_vision.fp16_safe_scales() reads its "sites" to pick the power-of-two norm scales of the `fs` form.
  python -I tower_cuts.py cuts --src SRC --out cuts.json [--form v2|fs --ranges ranges.json] [--only embed,norm,L0]
      [--g 512] [--keep DIR]
      the encoder cut after the patch embedding ("embed"), after its norm ("norm") and after each layer i ("L<i>",
      i = 0..13), each converted with litert_torch.convert (fp32 graph); each graph on Metal (ai-edge-litert
      CompiledModel, GPU only) at fp32 precision and at fp16 precision, a fresh process per graph x precision; per-token
      cosine of the fp16 output against the fp32 output, and of each against host torch fp32 at the same cut, over the
      fixture pictures. The first cut whose fp16 column drops while the fp32 column stays at 1 is where fp16 breaks.
      --form fs = the u8c input step (centred before the FLOOR) + fp16-safe norm scales from --ranges. Graphs go to a
      temporary folder and are removed after their cut unless --keep DIR is given.
SRC = the source snapshot (config.json, model.safetensors). Needs torch, litert-torch 0.10.0, ai-edge-litert 2.3.0.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lfm3_vision as lv  # noqa: E402

F16_MIN_NORMAL = 2.0 ** -14
F16_ZERO = 2.0 ** -25
CUTS = ["embed", "norm"] + [f"L{i}" for i in range(14)]


def fixture_paths(fxdir, g):
  fx = json.load(open(os.path.join(fxdir, "fixtures.json")))
  return {iid: os.path.join(fxdir, rec[f"g{g}"]) for iid, rec in fx["images"].items()}


def images(fxdir, g):
  return {iid: torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0)[None]
          for iid, p in fixture_paths(fxdir, g).items()}


def tower(src, g, opts=None):
  return lv.build_tower(os.path.join(src, "model.safetensors"), os.path.join(src, "config.json"), g, opts=opts)


# --------------------------------------------------------------------------------------------------- ranges
class Stats:
  def __init__(self):
    self.s = {}

  def norm(self, key, x, centre=False):
    r = x.detach().double().reshape(-1, x.shape[-1])
    if centre:
      r = r - r.mean(-1, keepdim=True)
    sq = r * r
    m = sq.mean(-1)
    cur = {"absmax": float(r.abs().max()), "max_row_sum_sq": float(sq.sum(-1).max()), "max_x_sq": float(sq.max()),
           "min_row_mean_sq": float(m.min()), "median_row_mean_sq": float(m.median()), "max_row_mean_sq": float(m.max()),
           "share_sq_below_f16_normal": float((sq < F16_MIN_NORMAL).double().mean()),
           "share_sq_to_zero_f16": float((sq < F16_ZERO).double().mean()), "dim": r.shape[-1]}
    self._merge(key, cur)

  def plain(self, key, x):
    r = x.detach().double()
    a = r.abs()
    cur = {"absmax": float(a.max()), "share_below_f16_normal": float(((a < F16_MIN_NORMAL) & (a > 0)).double().mean())}
    self._merge(key, cur)

  def _merge(self, key, cur):
    old = self.s.get(key)
    if old is None:
      self.s[key] = {**cur, "n_images": 1}
      return
    for k, v in cur.items():
      if k.startswith(("absmax", "max_")):
        old[k] = max(old[k], v)
      elif k.startswith("min_"):
        old[k] = min(old[k], v)
      elif k.startswith(("share_", "median_")):
        old[k] = (old[k] * old["n_images"] + v) / (old["n_images"] + 1)   # mean over pictures
    old["n_images"] += 1


def run_ranges(a):
  torch.set_num_threads(6)
  enc, adp, info = tower(a.src, a.g)
  st = Stats()
  for name, m in list(enc.named_modules()) + [("projector." + n, mm) for n, mm in adp.named_modules()]:
    if isinstance(m, lv.RMSNorm):
      m.register_forward_pre_hook(lambda mod, args, name=name: st.norm(name, args[0]))
    elif isinstance(m, torch.nn.LayerNorm):
      m.register_forward_pre_hook(lambda mod, args, name=name: st.norm(name, args[0], centre=True))
    elif isinstance(m, torch.nn.Linear):
      m.register_forward_hook(lambda mod, args, out, name=name: st.plain(name + ".out", out))
      if name.endswith((".w2", "out_proj", "linear_2")):
        m.register_forward_pre_hook(lambda mod, args, name=name: st.plain(name + ".in", args[0]))
    elif isinstance(m, torch.nn.Conv2d) and name.endswith("spatial_conv"):
      m.register_forward_pre_hook(lambda mod, args, name=name: st.plain(name + ".in", args[0]))
      m.register_forward_hook(lambda mod, args, out, name=name: st.plain(name + ".out", out))
    elif isinstance(m, torch.nn.Conv2d) and name == "patch_conv":
      m.register_forward_hook(lambda mod, args, out, name=name: st.plain(name + ".out", out))
    elif isinstance(m, lv.Attention):
      def att(mod, args, out, name=name):
        h = args[0]
        n, hd = mod.n, mod.hd
        q = mod._rope(mod.q_layernorm(mod.q_proj(h).reshape(1, n, mod.h, hd)))
        k = mod._rope(mod.k_layernorm(mod.k_proj(h).reshape(1, n, mod.kvh, hd)))
        q = q.permute(0, 2, 1, 3).reshape(1, mod.kvh, mod.rep * n, hd)
        st.plain(name + ".scores", torch.matmul(q, k.permute(0, 2, 1, 3).transpose(-2, -1)))
      m.register_forward_hook(att)
  feats = {}
  t0 = time.time()
  with torch.no_grad():
    for iid, x in images(a.fixtures, a.g).items():
      e = enc(x)
      st.plain("encoder.out", e)
      f = adp(e)
      st.plain("adapter.out", f)
      feats[iid] = float(f.abs().max())
  out = {"g": a.g, "seconds": round(time.time() - t0, 1), "torch": torch.__version__,
         "info": {k: info[k] for k in ("g", "grid", "n_patches", "n_tokens")},
         "fp16": {"min_normal": F16_MIN_NORMAL, "max": 65504.0}, "norm_eps": {"vision": 1e-5, "projector": 1e-6},
         "adapter_out_absmax_per_image": feats, "sites": st.s}
  json.dump(out, open(a.out, "w"), indent=1)
  rms, ln = lv.fp16_safe_scales(st.s)
  print("fp16-safe scales (headroom 4): " + ", ".join(f"{k} 2^{int(np.log2(v))}" for k, v in rms.items())
        + f", projector.layer_norm 2^{int(np.log2(ln))}", flush=True)
  print("RANGES_DONE", a.out, out["seconds"], "s", flush=True)
  return 0


# ----------------------------------------------------------------------------------------------------- cuts
class Cut(torch.nn.Module):
  """The encoder up to a cut: "embed" (the patch conv output), "norm" (after embedding_norm) or layer i (int)."""

  def __init__(self, enc, stop):
    super().__init__()
    self.enc, self.stop = enc, stop

  def forward(self, images_):
    e = self.enc
    x = e.embed(images_)
    if self.stop == "embed":
      return x
    x = e.embedding_norm(x, e.opts.s("embedding_norm"))
    if self.stop == "norm":
      return x
    for i in range(self.stop + 1):
      x = e.layers[i](x, e.opts, f"layers.{i}")
    return x


def stop_of(name):
  return name if name in ("embed", "norm") else int(name[1:])


CHILD = r'''
import json, os, sys
import numpy as np
from PIL import Image
from ai_edge_litert.compiled_model import CompiledModel, GpuOptions, HardwareAccelerator, Options
path, prec, out_dir, img_json = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
m = CompiledModel.from_file(path, options=Options(hardware_accelerators=HardwareAccelerator.GPU,
                                                  gpu_options=GpuOptions(enforce_f32=prec == "fp32")))
print("GPU_ONLY_BUILT", flush=True)
key = next(iter(m.get_signature_list()))
i, o = m.get_input_tensor_details(key), m.get_output_tensor_details(key)
ib = {n: m.create_input_buffer_by_name(key, n) for n in i}
ob = {n: m.create_output_buffer_by_name(key, n) for n in o}
(n,), (on,) = i, o
for iid, p in json.loads(img_json).items():
  x = (np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0)[None]
  ib[n].write(np.ascontiguousarray(x, dtype=i[n]["dtype"]).reshape(i[n]["shape"]))
  m.run_by_name(key, ib, ob)
  d = o[on]
  y = np.array(ob[on].read(int(np.prod(d["shape"])), d["dtype"]), copy=True).reshape(d["shape"])
  np.save(os.path.join(out_dir, iid + ".npy"), y)
print("CHILD_DONE", flush=True)
'''


def tok_metrics(a, r):
  a = np.asarray(a, np.float64).reshape(-1, r.shape[-1])
  r = np.asarray(r, np.float64).reshape(a.shape)
  c = (a * r).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(r, axis=-1) + 1e-30)
  return {"cos_min": float(c.min()), "cos_mean": float(c.mean()), "rel_max": float(np.abs(a - r).max() / np.abs(r).max()),
          "finite": bool(np.isfinite(a).all()), "argmin_token": int(c.argmin())}


def gpu_child(path, prec, out_dir, img_paths):
  os.makedirs(out_dir, exist_ok=True)
  p = subprocess.run([sys.executable, "-I", "-c", CHILD, path, prec, out_dir, json.dumps(img_paths)],
                     capture_output=True, text=True, timeout=3600, stdin=subprocess.DEVNULL)
  lines = [ln.strip()[:300] for ln in (p.stdout + p.stderr).splitlines()
           if "not supported" in ln.lower() or re.search(r"rror|GPU_ONLY_BUILT", ln)]
  return {"rc": p.returncode, "built": "GPU_ONLY_BUILT" in p.stdout, "done": "CHILD_DONE" in p.stdout,
          "lines": lines[:20]}


def summarize(per_image):
  out = {}
  for k in next(iter(per_image.values())):
    vals = [v[k] for v in per_image.values()]
    out[k] = {"cos_min": min(v["cos_min"] for v in vals), "cos_mean_mean": float(np.mean([v["cos_mean"] for v in vals])),
              "rel_max": max(v["rel_max"] for v in vals), "finite": all(v["finite"] for v in vals),
              "worst_image": min(per_image, key=lambda iid: per_image[iid][k]["cos_min"])}
  return out


def form_opts(form, ranges):
  if form == "v2":
    return lv.Opts()
  rms, ln = lv.fp16_safe_scales(json.load(open(ranges))["sites"])
  return lv.Opts(input_norm="u8c", rms_s=rms, ln_s=ln)


def run_cuts(a):
  import litert_torch
  if a.form == "fs" and not a.ranges:
    sys.exit("--form fs needs --ranges (the output of the ranges command)")
  torch.set_num_threads(6)
  enc, _adp, _info = tower(a.src, a.g, form_opts(a.form, a.ranges))
  imgs = images(a.fixtures, a.g)
  img_paths = fixture_paths(a.fixtures, a.g)
  first = next(iter(imgs))
  names = a.only.split(",") if a.only else CUTS
  work = tempfile.mkdtemp(prefix="tower_cuts_")
  gdir = a.keep or work
  os.makedirs(gdir, exist_ok=True)
  res = json.load(open(a.out)) if os.path.exists(a.out) else {
      "g": a.g, "form": a.form, "rule": "per-token cosine over the fixture pictures; Metal GPU-only CompiledModel, a "
                                       "fresh process per graph x precision (fp32 = enforce_f32, fp16 = the default)",
      "cuts": {}}
  for name in names:
    cut = Cut(enc, stop_of(name)).eval()
    with torch.no_grad():
      ref = {iid: cut(x).numpy() for iid, x in imgs.items()}
    tfl = os.path.join(gdir, f"cut_{name}.tflite")
    t0 = time.time()
    if not os.path.exists(tfl):
      litert_torch.convert(cut, (imgs[first],)).export(tfl)
    rec = {"tflite_bytes": os.path.getsize(tfl), "convert_s": round(time.time() - t0, 1)}
    outs = {}
    for prec in ("fp32", "fp16"):
      od = os.path.join(work, "run", name, prec)
      rec[prec] = gpu_child(tfl, prec, od, img_paths)
      outs[prec] = {iid: np.load(os.path.join(od, iid + ".npy")) for iid in img_paths
                    if os.path.exists(os.path.join(od, iid + ".npy"))}
    if len(outs["fp32"]) == len(outs["fp16"]) == len(img_paths):
      per = {iid: {"fp32_vs_torch": tok_metrics(outs["fp32"][iid], ref[iid]),
                   "fp16_vs_torch": tok_metrics(outs["fp16"][iid], ref[iid]),
                   "fp16_vs_fp32": tok_metrics(outs["fp16"][iid], outs["fp32"][iid])} for iid in img_paths}
      rec["per_image"] = per
      rec["summary"] = summarize(per)
      rec["ref_absmax"] = max(float(np.abs(r).max()) for r in ref.values())
    res["cuts"][name] = rec
    json.dump(res, open(a.out, "w"), indent=1)
    s = rec.get("summary", {})
    print(f"cut {name}: built {rec['fp32']['built']}/{rec['fp16']['built']} | "
          + " | ".join(f"{k} min {v['cos_min']:.7f} mean {v['cos_mean_mean']:.7f} ({v['worst_image']})" for k, v in s.items()),
          flush=True)
    shutil.rmtree(os.path.join(work, "run", name), ignore_errors=True)
    if not a.keep:
      os.remove(tfl)
  shutil.rmtree(work, ignore_errors=True)
  print("CUTS_DONE", a.out, flush=True)
  return 0


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("cmd", choices=("ranges", "cuts"))
  ap.add_argument("--src", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--g", type=int, default=512)
  ap.add_argument("--fixtures", default=os.path.join(HERE, "..", "fixtures"))
  ap.add_argument("--form", default="v2", choices=("v2", "fs"))
  ap.add_argument("--ranges", default=None)
  ap.add_argument("--only", default="")
  ap.add_argument("--keep", default=None)
  a = ap.parse_args()
  a.fixtures = os.path.abspath(a.fixtures)
  return {"ranges": run_ranges, "cuts": run_cuts}[a.cmd](a)


if __name__ == "__main__":
  sys.exit(main())
