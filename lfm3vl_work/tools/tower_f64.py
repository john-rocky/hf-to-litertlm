#!/usr/bin/env python3
"""tower_f64.py - a float64 test of the LFM3-VL vision tower re-write (lfm3_vision.py) against the source model's own
modules, to tell a re-write error from float32 rounding.

  HF_HUB_OFFLINE=1 python -I tower_f64.py --src SRC --ref-dir REF/ref_npy --out tower_f64_v2.json [--variant v2]
      [--images main|all] [--cross] [--fixtures lfm3vl_work/fixtures]

SRC = the source snapshot (its folder name must be a valid Python identifier, e.g. convert_lfm3_vl.sh's OUTDIR/src:
the remote code is imported from it as a package); needs transformers 5.21.0 and torch. REF-DIR = hf_ref.py --npy's
ref_npy folder.
Truth = the source's own modules (the snapshot's remote code, real weights) in float64 with their fp32 casts removed in
this process only (Lfm3VlRMSNorm's .to(float32), apply_rotary_pos_emb_vision's .float(), the rotary's fp32 inv_freq /
positions; attention = torch sdpa, which keeps float64), fed float64 pixel values of the PNG (the processor's
(u8 / 255 - 0.5) / 0.5 and merge-block patchify with T = 2, in float64).
Per case:
  (a) HF fp32 vs truth   = the reference of hf_ref.py --npy (sdpa, fp32) for the tower cases; F.linear in fp32 on
                           the processor's fp32 pixel_values for the patch-embedding cases
  (b) re-write fp32 vs truth (lfm3_vision.py fp32 on the PNG / 255; --variant v1 = Opts(input_norm="unit"), the
      first form (x - 0.5) * 2; v2 = the default input step, the exact u8 back and (u8 - 127.5) / 127.5)
  (c) re-write float64 (the same variant) vs HF float64 (the same float64 input; RoPE tables, inv_freq and the T = 2
      fold of the patchifier weight in float64 on both)
  --images all: every fixture picture x G, tower and patch embedding, with the other variant's fp32 beside (b);
      these rows go to "extra".
  --cross: HF fp32 (its own unpatched code) fed v1's input step ((u8 / 255 - 0.5) * 2 in fp32, patchified) vs
      truth, nasa_radar at both G: if the v1 excess comes from the input step, HF's own fp32 shows it too.
Metrics: rel_rms = ||x - t||_F / ||t||_F, max|d|, max|d| / max|t|, per-token cos min to the truth.
Verdict on the main cases (nasa_radar G512 / G768 hid + feat, syn_bar G512 / syn_table G768 patch embedding):
(c) max|d| <= 1e-10 and rel_rms(b) <= 2 x rel_rms(a) -> PASS; (c) > 1e-10 = a re-write error.
Exit 0 = all PASS, 1 = (c) open, 2 = FAIL ((b) more than twice as far from the truth as HF's own fp32).
"""
import argparse
import importlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lfm3_vision as lv  # noqa: E402

cfgm = mm = None          # the snapshot's configuration_lfm3_vl / modeling_lfm3_vl, imported in main()
WEIGHTS = CONFIG = None
C_BAR = 1e-10
NORM = {"v1": "unit", "v2": "u8"}
SUP_TOWER = {(512, "nasa_radar"), (768, "nasa_radar")}
SUP_PATCH = {(512, "syn_bar"), (768, "syn_table")}


def patchify(x, g):
  """[G, G, 3] normalised pixels -> [n, 1536] merge-block order, (C, T, ph, pw) per patch (the processor's layout)."""
  x = x.transpose(2, 0, 1)
  gh = g // 16
  p = x.reshape(3, gh // 2, 2, 16, gh // 2, 2, 16).transpose(1, 4, 2, 5, 0, 3, 6)
  p = np.repeat(p[:, :, :, :, :, None], 2, axis=5)
  return np.ascontiguousarray(p.reshape(gh * gh, 3 * 2 * 16 * 16))


def patch_hf_float64():
  """Remove the remote code's fp32 casts (this process only)."""
  def rms_forward(self, x):
    return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))

  def rope_apply(q, k, cos, sin):
    cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
    return q * cos + mm.rotate_half(q) * sin, k * cos + mm.rotate_half(k) * sin

  def rotary_forward(self, x, position_ids):
    spatial_dim = self.config.hidden_size // self.config.num_attention_heads // 2
    base = self.config.rope_parameters["rope_theta"]
    inv = 1.0 / (base ** (torch.arange(0, spatial_dim, 2, dtype=torch.float64) / spatial_dim))
    freqs = position_ids[..., None].to(torch.float64) * inv
    return self.recomposition_frequencies(freqs.cos()), self.recomposition_frequencies(freqs.sin())

  mm.Lfm3VlRMSNorm.forward = rms_forward
  mm.apply_rotary_pos_emb_vision = rope_apply
  mm.Lfm3VlVisionRotaryEmbedding.forward = rotary_forward


def raw(key, dtype):
  from safetensors import safe_open
  with safe_open(WEIGHTS, framework="pt") as f:
    return f.get_tensor(key).to(dtype)


def hf_modules(dtype):
  full = json.load(open(CONFIG))
  cfg = cfgm.Lfm3VlConfig(**{k: v for k, v in full.items() if k not in ("auto_map", "architectures")})
  cfg.vision_config._attn_implementation = "sdpa"
  tower = mm.Lfm3VlVisionModel(cfg.vision_config)
  proj = mm.Lfm3VlMultiModalProjector(cfg)
  from safetensors import safe_open
  t_sd, p_sd = {}, {}
  with safe_open(WEIGHTS, framework="pt") as f:
    for k in f.keys():
      if k.startswith("model.vision_tower."):
        t_sd[k[len("model.vision_tower."):]] = f.get_tensor(k).to(dtype)
      elif k.startswith("model.multi_modal_projector."):
        p_sd[k[len("model.multi_modal_projector."):]] = f.get_tensor(k).to(dtype)
  tower, proj = tower.to(dtype), proj.to(dtype)
  tower.load_state_dict(t_sd, strict=True)
  proj.load_state_dict(p_sd, strict=True)
  return tower.eval(), proj.eval()


def mine_float64(g, input_norm):
  enc, adp, _ = lv.build_tower(WEIGHTS, CONFIG, g, dtype=torch.float64, opts=lv.Opts(input_norm=input_norm))
  cfg = lv.read_config(CONFIG)
  w = raw("model.vision_tower.patchifier.weight", torch.float64)        # the T = 2 fold in float64, not fp32
  enc.patch_conv.weight.copy_(w.reshape(w.shape[0], cfg["in_channels"], cfg["temporal_patch_size"],
                                        cfg["patch_size"], cfg["patch_size"]).sum(2))
  hd = cfg["hidden_size"] // cfg["num_attention_heads"]
  spatial_dim = hd // 2
  inv = 1.0 / (cfg["rope_theta"] ** (torch.arange(0, spatial_dim, 2, dtype=torch.float64) / spatial_dim))
  hh, ww = torch.meshgrid(torch.arange(enc.gh), torch.arange(enc.gw), indexing="ij")
  pos = torch.stack([hh.flatten(), ww.flatten()], dim=-1).to(torch.float64)
  freqs = pos[..., None] * inv
  tabs = []
  for f in (freqs.cos(), freqs.sin()):
    fhw = torch.cat([f[:, 0], f[:, 1]], dim=-1)
    tabs.append(torch.cat([fhw, fhw], dim=-1).reshape(1, enc.n, 1, hd))
  for layer in enc.layers:
    if layer.kind == "full_attention":
      layer.self_attn.cos.copy_(tabs[0])
      layer.self_attn.sin.copy_(tabs[1])
  return enc, adp


def metrics(x, t):
  t = np.asarray(t, np.float64)
  t = t.reshape(-1, t.shape[-1])
  x = np.asarray(x, np.float64).reshape(t.shape)
  cos = (x * t).sum(-1) / (np.linalg.norm(x, axis=-1) * np.linalg.norm(t, axis=-1) + 1e-300)
  d = np.abs(x - t)
  return {"rel_rms": float(np.linalg.norm(x - t) / np.linalg.norm(t)), "maxdiff": float(d.max()),
          "max_over_absmax": float(d.max() / np.abs(t).max()), "cos_min": float(cos.min()),
          "worst_cos_token": int(cos.argmin()), "maxdiff_token": int(np.unravel_index(d.argmax(), d.shape)[0])}


def judge(case, ma, mb, mc, extra=None, quiet=False):
  ok = mc["maxdiff"] <= C_BAR and mb["rel_rms"] <= 2 * ma["rel_rms"]
  row = {"case": case, "a_hf_fp32_vs_truth": ma, "b_mine_fp32_vs_truth": mb, "c_mine_f64_vs_hf_f64": mc,
         "ratio_b_over_a": mb["rel_rms"] / ma["rel_rms"], "c_open": mc["maxdiff"] > C_BAR, "pass": ok}
  row.update(extra or {})
  more = "".join(f" | {k.split('_vs_')[0]} {v['rel_rms'] / ma['rel_rms']:.2f}x" for k, v in (extra or {}).items()
                 if isinstance(v, dict) and "rel_rms" in v)
  print(f"{'  extra ' if quiet else ''}{case}: (a) rel_rms {ma['rel_rms']:.3e} max {ma['maxdiff']:.2e} cos "
        f"{ma['cos_min']:.8f} | (b) rel_rms {mb['rel_rms']:.3e} max {mb['maxdiff']:.2e} cos {mb['cos_min']:.8f} | b/a "
        f"{row['ratio_b_over_a']:.2f} | (c) max {mc['maxdiff']:.2e} rel {mc['max_over_absmax']:.1e}{more} -> "
        f"{'PASS' if ok else 'FAIL'}", flush=True)
  return row


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--src", required=True)
  ap.add_argument("--ref-dir", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--fixtures", default=os.path.join(HERE, "..", "fixtures"))
  ap.add_argument("--variant", choices=("v1", "v2"), default="v2")
  ap.add_argument("--images", choices=("main", "all"), default="main")
  ap.add_argument("--cross", action="store_true")
  a = ap.parse_args()
  global cfgm, mm, WEIGHTS, CONFIG
  src = os.path.abspath(a.src)
  sys.path.insert(0, os.path.dirname(src))
  pkg = os.path.basename(src)
  cfgm = importlib.import_module(f"{pkg}.configuration_lfm3_vl")
  mm = importlib.import_module(f"{pkg}.modeling_lfm3_vl")
  WEIGHTS, CONFIG = os.path.join(src, "model.safetensors"), os.path.join(src, "config.json")
  fxdir = os.path.abspath(a.fixtures)
  torch.set_grad_enabled(False)
  torch.set_num_threads(8)
  t0 = time.time()
  other = "v2" if a.variant == "v1" else "v1"
  fx = json.load(open(os.path.join(fxdir, "fixtures.json")))
  png = lambda iid, g: np.asarray(Image.open(os.path.join(fxdir, fx["images"][iid][f"g{g}"])).convert("RGB"))  # noqa: E731
  res = {"variant": a.variant, "images": a.images, "cross": a.cross, "rule": __doc__.split("Per case:")[1].strip(),
         "c_bar": C_BAR, "cases": [], "extra": []}
  cross = {}
  if a.cross:                                       # HF fp32 with its own code first, before the cast patch
    hf32_t, hf32_p = hf_modules(torch.float32)
    for g in (512, 768):
      gh = g // 16
      x = (png("nasa_radar", g).astype(np.float32) / np.float32(255.0) - np.float32(0.5)) * np.float32(2.0)
      h = hf32_t(torch.from_numpy(patchify(x, g)), grid_thw=torch.tensor([[1, gh, gh]])).last_hidden_state
      cross[g] = (h.numpy(), hf32_p(h).numpy())
    del hf32_t, hf32_p
  patch_hf_float64()
  hf64_t, hf64_p = hf_modules(torch.float64)
  w32 = raw("model.vision_tower.patchifier.weight", torch.float32)
  for g in (512, 768):
    gh = g // 16
    thw = torch.tensor([[1, gh, gh]])
    enc64, adp64 = mine_float64(g, NORM[a.variant])
    m32 = {v: lv.build_tower(WEIGHTS, CONFIG, g, opts=lv.Opts(input_norm=NORM[v]))[:2]
           for v in ((a.variant, other) if a.images == "all" else (a.variant,))}
    ref = os.path.join(a.ref_dir, f"g{g}")
    for iid in sorted(fx["images"]):
      do_t = (g, iid) in SUP_TOWER or a.images == "all"
      do_p = (g, iid) in SUP_PATCH or a.images == "all"
      if not (do_t or do_p):
        continue
      img8 = png(iid, g)
      img64 = img8.astype(np.float64) / 255.0
      pv64 = torch.from_numpy(patchify((img64 - 0.5) / 0.5, g))
      x32 = torch.from_numpy((img8.astype(np.float32) / np.float32(255.0))[None])
      x64 = torch.from_numpy(img64[None])
      if do_t:
        t_hid = hf64_t(pv64, grid_thw=thw).last_hidden_state
        t_feat = hf64_p(t_hid).numpy()
        t_hid = t_hid.numpy()
        c_r = enc64(x64)
        c_hid, c_feat = lv.raster_to_merge(c_r, gh, gh)[0].numpy(), adp64(c_r)[0].numpy()
        outs = {}
        for v, (e, ad) in m32.items():
          hr = e(x32)
          outs[v] = (lv.raster_to_merge(hr, gh, gh)[0].numpy(), ad(hr)[0].numpy())
        a_ref = (np.load(os.path.join(ref, f"{iid}.hid.npy")), np.load(os.path.join(ref, f"{iid}.feat.npy")))
        for i, (what, c_, t_) in enumerate((("hid", c_hid, t_hid), ("feat", c_feat, t_feat))):
          extra = {}
          if other in outs:
            extra[f"{other}_fp32_vs_truth"] = metrics(outs[other][i], t_)
          if iid == "nasa_radar" and g in cross:
            extra["hf_fp32_on_v1_input_vs_truth"] = metrics(cross[g][i], t_)
          sup = (g, iid) in SUP_TOWER
          row = judge(f"{iid} G{g} {what}", metrics(a_ref[i], t_), metrics(outs[a.variant][i], t_),
                      metrics(c_, t_), extra, quiet=not sup)
          res["cases" if sup else "extra"].append(row)
      if do_p:
        truth = lv.merge_to_raster(F.linear(pv64, w32.double())[None], gh, gh)[0].numpy()
        pv32 = torch.from_numpy(np.load(os.path.join(ref, f"{iid}.pv.npy")))
        a_ = lv.merge_to_raster(F.linear(pv32, w32)[None], gh, gh)[0].numpy()                  # HF's own fp32
        b_ = {v: e.embed(x32)[0].numpy() for v, (e, _ad) in m32.items()}
        c_ = enc64.embed(x64)[0].numpy()
        extra = {"truth_absmax": float(np.abs(truth).max())}
        if other in b_:
          extra[f"{other}_fp32_vs_truth"] = metrics(b_[other], truth)
        sup = (g, iid) in SUP_PATCH
        row = judge(f"{iid} G{g} patch_embed", metrics(a_, truth), metrics(b_[a.variant], truth), metrics(c_, truth),
                    extra, quiet=not sup)
        res["cases" if sup else "extra"].append(row)
    del enc64, adp64, m32
  ok_all = all(r["pass"] for r in res["cases"])
  c_open = any(r["c_open"] for r in res["cases"])
  if res["extra"]:
    rows = res["cases"] + res["extra"]
    summ = {}
    for v, key in ((a.variant, None), (other, f"{other}_fp32_vs_truth")):
      ratios = [((r[key]["rel_rms"] if key else r["b_mine_fp32_vs_truth"]["rel_rms"]) / r["a_hf_fp32_vs_truth"]["rel_rms"],
                 r["case"]) for r in rows if key is None or key in r]
      if ratios:
        mx = max(ratios)
        summ[v] = {"rows": len(ratios), "over_2x": sorted(c for q, c in ratios if q > 2), "max_ratio": [mx[0], mx[1]]}
    summ["c_max_over_all_rows"] = max(r["c_mine_f64_vs_hf_f64"]["maxdiff"] for r in rows)
    res["all_rows_summary"] = summ
    print("ALL_ROWS " + json.dumps(summ), flush=True)
  res.update({"all_pass": ok_all, "c_open": c_open, "seconds": round(time.time() - t0, 1), "torch": torch.__version__})
  json.dump(res, open(a.out, "w"), indent=1)
  print("TOWER_F64", "PASS" if ok_all else ("C_OPEN" if c_open else "FAIL"), a.variant, f"{res['seconds']} s", flush=True)
  return 0 if ok_all else (1 if c_open else 2)


if __name__ == "__main__":
  sys.exit(main())
