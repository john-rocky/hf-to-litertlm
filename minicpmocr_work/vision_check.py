#!/usr/bin/env python3
"""HF reference of the 4x vision path at GxG and the eager parity of the pure-torch re-author (ocr_vision.py).

  python vision_check.py --snap SNAPSHOT --images A.png B.png ... --out OUT_DIR [--g 448]

For every image (already GxG; the processor must keep it at GxG, checked):
  ref    transformers main: processor(images) -> pixel_values / target_sizes, then
         model.model.get_image_features(...) -> pooler_output[0] [(G/28)^2, 1024] and the vision tower's
         last_hidden_state [1, (G/14)^2, 1152]; also the strip the processor built vs the re-author's strip.
  eager  ocr_vision.build(...) with Opts(barrier=True, fp16safe=False): encoder vs last_hidden_state and adapter vs
         pooler_output (per-token cosine min, Pearson over all values, max |delta|).
  calib  the absmax of every LayerNorm input over all images -> S = 2^ceil(log2(absmax / 8)) -> OUT/ln_scales.json,
         then the fp16-safe module (fp16safe=True, the S table) against the same references.
Writes OUT/vision_check.json and OUT/ref_<stem>.npz (img01 [G,G,3], enc_ref, pooler_ref) for the TFLite parity.
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
import ocr_vision as ov  # noqa: E402


def metrics(got, ref):
    g = got.reshape(-1, got.shape[-1]).double()
    r = ref.reshape(-1, ref.shape[-1]).double()
    cos = torch.nn.functional.cosine_similarity(g, r, dim=-1)
    gc, rc = g.flatten() - g.mean(), r.flatten() - r.mean()
    corr = float((gc @ rc) / (gc.norm() * rc.norm()))
    return {"cos_min": float(cos.min()), "cos_mean": float(cos.mean()), "corr": corr,
            "max_abs": float((g - r).abs().max()), "ref_absmax": float(r.abs().max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--g", type=int, default=448)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)

    from transformers import AutoModelForImageTextToText, AutoProcessor
    import transformers
    processor = AutoProcessor.from_pretrained(args.snap)
    model = AutoModelForImageTextToText.from_pretrained(args.snap, dtype=torch.float32).eval()

    t0 = time.time()
    enc, adp, info = ov.build(args.snap, ov.Opts(barrier=True, fp16safe=False), args.g)
    print("re-author", info, f"{time.time() - t0:.1f}s", flush=True)

    rows, refs, calib = [], {}, {}
    for path in args.images:
        stem = os.path.splitext(os.path.basename(path))[0]
        img = Image.open(path).convert("RGB")
        assert img.size == (args.g, args.g), (path, img.size)
        img01 = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)[None]          # [1,G,G,3]
        feats = processor.image_processor(images=[img], return_tensors="pt")
        pv, ts = feats["pixel_values"], feats["target_sizes"]
        assert ts.tolist() == [[args.g // 14, args.g // 14]], ts.tolist()
        strip_delta = float((pv.float() - ov.strip_from_image(img01)).abs().max())
        out = model.model.get_image_features(pv, ts)
        pooler = out.pooler_output[0].float()                                              # [(G/28)^2,1024]
        last = out.last_hidden_state.float()                                               # [1,(G/14)^2,1152]
        e = enc(img01, calib=calib)
        a = adp(e, calib=calib)
        row = {"image": stem, "strip_max_abs": strip_delta, "n_tokens": int(pooler.shape[0]),
               "enc_eager": metrics(e[0], last[0]), "adp_eager": metrics(a[0], pooler)}
        refs[stem] = (img01[0].numpy(), last[0].numpy(), pooler.numpy())
        rows.append(row)
        print(stem, json.dumps({k: v for k, v in row.items() if k != "image"}), flush=True)

    scales = {k: ov.calib_S(v) for k, v in sorted(calib.items())}
    json.dump({"absmax": calib, "S": scales}, open(os.path.join(args.out, "ln_scales.json"), "w"), indent=1)
    enc16, adp16, _ = ov.build(args.snap, ov.Opts(barrier=True, fp16safe=True, ln_s=scales), args.g)
    for row in rows:
        img01, last, pooler = refs[row["image"]]
        e = enc16(torch.from_numpy(img01)[None])
        a = adp16(e)
        row["enc_fp16safe"] = metrics(e[0], torch.from_numpy(last))
        row["adp_fp16safe"] = metrics(a[0], torch.from_numpy(pooler))
        np.savez(os.path.join(args.out, f"ref_{row['image']}.npz"), img01=img01, enc_ref=last, pooler_ref=pooler)
        print(row["image"], "fp16safe", json.dumps(row["adp_fp16safe"]), flush=True)

    s_hist = {}
    for v in scales.values():
        s_hist[str(int(v))] = s_hist.get(str(int(v)), 0) + 1
    result = {"transformers": transformers.__version__, "torch": torch.__version__, "g": args.g, "info": info,
              "ln_S_histogram": s_hist, "ln_absmax_max": max(calib.values()), "rows": rows,
              "worst": {k: min(r[k]["cos_min"] for r in rows) for k in ("enc_eager", "adp_eager", "enc_fp16safe",
                                                                         "adp_fp16safe")}}
    json.dump(result, open(os.path.join(args.out, "vision_check.json"), "w"), indent=1)
    print("worst cos_min", json.dumps(result["worst"]), "S histogram", s_hist, flush=True)


if __name__ == "__main__":
    main()
