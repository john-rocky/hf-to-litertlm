#!/usr/bin/env python3
"""r2 step 3: eager parity of the pure-torch re-author (mv47_vision.py) vs HF, fp32.

  python vision_parity_eager.py --src SRC --ref W/ref \
      --fixtures W/fixtures --stages W/stages --out W      (litert-torch 0.9.3 environment)

1. calibration: plain LayerNorm, every LN input absmax over the 3 fixtures + noise
   -> S = 2^ceil(log2(absmax/8)) (>= 1) -> OUT/ln_scales.json
2. parity (fp32) of the EXPORT configuration (fp16-safe LN with those S + LN->FC barrier)
   and of the plain configuration (S = 1, no barrier), per fixture:
     encoder [1,256,1152] vs HF enc_last_hidden, adapter [1,64,1024] vs HF feat:
     per-token cos mean / min, Pearson corr, max|d|; the adapter also on HF's own
     encoder output (isolates the adapter).
   checks: pos_ids == HF pos_ids; window mask == HF window_index/cu_seqlens partition;
   processor strip vs the strip built from the NHWC [0,1] image (max|d| <= 1e-6).
3. per-stage (if STAGES/<name>.npz exists): embeddings, layer00..26, vit_merger
   (HF window order == our raster order of the 16x16 window grid).
-> OUT/parity_eager.json; prints PASS/FAIL against cos min >= 0.9999 and corr >= 0.9999.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mv47_vision as mv  # noqa: E402

BAR = 0.9999
NAMES = ("cats", "kitchen1", "kitchen2", "noise")


def load_img(path):
  a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
  return torch.from_numpy(a).unsqueeze(0)                # [1,448,448,3]


def metrics(got, ref):
  g = np.asarray(got, dtype=np.float64).reshape(-1, np.asarray(ref).shape[-1])
  r = np.asarray(ref, dtype=np.float64).reshape(g.shape)
  cos = (g * r).sum(-1) / (np.linalg.norm(g, axis=-1) * np.linalg.norm(r, axis=-1) + 1e-30)
  return {"cos_mean": float(cos.mean()), "cos_min": float(cos.min()),
          "corr": float(np.corrcoef(g.reshape(-1), r.reshape(-1))[0, 1]),
          "maxdiff": float(np.abs(g - r).max()), "ref_absmax": float(np.abs(r).max())}


def run(enc, adp, x, taps=None, calib=None):
  with torch.no_grad():
    e = enc(x, calib=calib, taps=taps)
    a = adp(e, calib=calib)
  return e, a


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--src", required=True)
  ap.add_argument("--ref", required=True)
  ap.add_argument("--fixtures", required=True)
  ap.add_argument("--stages", default=None)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()
  torch.set_grad_enabled(False)
  res = {"torch": torch.__version__, "bar": BAR, "mask_neg": mv.MASK_NEG}

  t0 = time.time()
  opts = mv.Opts(barrier=False, fp16safe=False)
  enc, adp, info = mv.build(args.src, opts)
  res["load_seconds"] = round(time.time() - t0, 1)
  res["build"] = {k: v for k, v in info.items() if k != "pos_ids"}
  print("built", res["build"], f"{res['load_seconds']} s", flush=True)

  imgs = {n: load_img(os.path.join(args.fixtures, f"{n}_448.png")) for n in NAMES}
  refs = {n: np.load(os.path.join(args.ref, f"{n}.npz")) for n in NAMES}

  # static constants vs HF
  r0 = refs[NAMES[0]]
  res["pos_ids_equal"] = bool(np.array_equal(info["pos_ids"].numpy(), r0["pos_ids"]))
  widx, wcu = r0["window_index"], r0["cu_window_seqlens"]
  win_of = np.empty(mv.N_PATCH, dtype=np.int64)
  for w in range(len(wcu) - 1):
    win_of[widx[wcu[w]:wcu[w + 1]]] = w
  same = win_of[:, None] == win_of[None, :]
  mask = enc.win_mask[0, 0].numpy()
  res["window_mask_matches_hf_partition"] = bool(np.array_equal(mask == 0.0, same))
  res["hf_window_order_is_raster_of_window_grid"] = bool(np.array_equal(
      widx.reshape(mv.WGRID, mv.WGRID, 4)[:, :, 0].reshape(-1),
      ((np.arange(mv.N_WIN) // mv.WGRID) * 2 * mv.GRID + (np.arange(mv.N_WIN) % mv.WGRID) * 2)))
  res["strip_maxdiff"] = {n: float(np.abs(mv.strip_from_image(imgs[n]).numpy() - refs[n]["pixel_values"]).max())
                          for n in NAMES}
  print("pos_ids_equal", res["pos_ids_equal"], "mask==HF partition", res["window_mask_matches_hf_partition"],
        "window order raster", res["hf_window_order_is_raster_of_window_grid"], "strip", res["strip_maxdiff"], flush=True)

  # 1. calibration (plain LN)
  calib = {}
  for n in NAMES:
    run(enc, adp, imgs[n], calib=calib)
  scales = {k: {"absmax": round(v, 4), "S": mv.calib_S(v)} for k, v in calib.items()}
  with open(os.path.join(args.out, "ln_scales.json"), "w") as f:
    json.dump({"rule": "S = 2^max(0, ceil(log2(absmax/8))), absmax over cats/kitchen1/kitchen2/noise 448, plain fp32 LN",
               "n_ln": len(scales), "n_S_gt_1": sum(1 for v in scales.values() if v["S"] > 1.0),
               "scales": scales}, f, indent=1)
    f.write("\n")
  res["ln_n"] = len(scales)
  res["ln_S_gt_1"] = {k: v["S"] for k, v in scales.items() if v["S"] > 1.0}
  res["ln_absmax_max"] = max(calib.values())
  print("LN", len(scales), "S>1:", res["ln_S_gt_1"], "absmax max", round(res["ln_absmax_max"], 1), flush=True)

  # 2. parity: plain and export configuration
  def parity(label):
    out = {}
    for n in NAMES:
      taps = {}
      t1 = time.time()
      e, a = run(enc, adp, imgs[n], taps=taps)
      sec = time.time() - t1
      with torch.no_grad():
        a_on_hf = adp(torch.from_numpy(refs[n]["enc_last_hidden"]))
      row = {"encoder": metrics(e.numpy(), refs[n]["enc_last_hidden"]),
             "adapter_end2end": metrics(a.numpy(), refs[n]["feat"]),
             "adapter_on_hf_encoder": metrics(a_on_hf.numpy(), refs[n]["feat"]),
             "seconds": round(sec, 2)}
      if args.stages and os.path.exists(os.path.join(args.stages, f"{n}.npz")):
        st = np.load(os.path.join(args.stages, f"{n}.npz"))
        row["stages"] = {k: {kk: (round(vv, 8) if "cos" in kk or kk == "corr" else vv)
                             for kk, vv in metrics(taps[k].numpy(), st[k]).items()}
                         for k in sorted(taps) if k in st.files}
      out[n] = row
      print(label, n, "enc", {k: round(v, 7) for k, v in row["encoder"].items()},
            "adp", {k: round(v, 7) for k, v in row["adapter_end2end"].items()}, f"{sec:.1f} s", flush=True)
    return out

  res["plain"] = parity("plain")
  enc.opts = adp.opts = mv.Opts(barrier=True, fp16safe=True, ln_s={k: v["S"] for k, v in scales.items()})
  res["export_config"] = parity("export")

  def worst(block, part, key):
    return min(block[n][part][key] for n in NAMES)

  res["summary"] = {cfg: {"encoder_cos_min": worst(res[cfg], "encoder", "cos_min"),
                          "encoder_corr_min": worst(res[cfg], "encoder", "corr"),
                          "adapter_cos_min": worst(res[cfg], "adapter_end2end", "cos_min"),
                          "adapter_corr_min": worst(res[cfg], "adapter_end2end", "corr"),
                          "encoder_maxdiff_max": max(res[cfg][n]["encoder"]["maxdiff"] for n in NAMES),
                          "adapter_maxdiff_max": max(res[cfg][n]["adapter_end2end"]["maxdiff"] for n in NAMES)}
                    for cfg in ("plain", "export_config")}
  s = res["summary"]["export_config"]
  res["pass"] = bool(min(s["encoder_cos_min"], s["encoder_corr_min"], s["adapter_cos_min"], s["adapter_corr_min"]) >= BAR
                     and res["pos_ids_equal"] and res["window_mask_matches_hf_partition"]
                     and max(res["strip_maxdiff"].values()) <= 1e-6)
  with open(os.path.join(args.out, "parity_eager.json"), "w") as f:
    json.dump(res, f, indent=1)
    f.write("\n")
  print("SUMMARY", json.dumps(res["summary"]), "PASS" if res["pass"] else "FAIL", flush=True)


if __name__ == "__main__":
  main()
