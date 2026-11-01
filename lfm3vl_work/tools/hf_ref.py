#!/usr/bin/env python3
"""hf_ref.py - the Transformers reference of LiquidAI/LFM3-VL-2B on the fixture pictures (CPU), for the vision tower
gates (tower_tfl.py, tower_gpu.py, tower_cuts.py, tower_f64.py) and the teacher tokens of tools/quant_sensitivity.

  python -I hf_ref.py --src SRC --out OUT --arms B512 --dtype fp32 --npy [--only id|think,...] [--threads 8]

SRC = the source snapshot at f83d2b0a (config, weights, tokenizer and the remote code: convert_lfm3_vl.sh's fetch step
writes it to OUTDIR/src); needs transformers 5.21.0 (the model card's floor), torch, Pillow. Set HF_HUB_OFFLINE=1 and
HF_MODULES_CACHE to a folder of your own so the remote code is read from SRC only.
FIXTURES (--fixtures, default lfm3vl_work/fixtures) = fixtures.json and img/ (bash fixtures/fetch.sh <python> write).

Arms (each x think off = enable_thinking False / think on = the template's default, a generation prompt ending in
<think>): N = the row's source picture (<= 1024 px) through the processor as shipped (smart_resize, aspect kept);
B512 / B768 = img/g<G>/<picture>.png (exif_transpose -> RGB -> resize((G, G), BICUBIC), lfm3_preprocess.py), for which
smart_resize is the identity. Greedy (do_sample False, repetition_penalty 1.0), max_new_tokens 64 (off) / 32 (on),
eos 124900. -> OUT/hf_<arm>_<dtype>.json (written after every case; a re-run skips the cases it already has).
--npy (fp32, B512 / B768): per picture, get_image_features() with sdpa (the default) and eager attention:
OUT/ref_npy/g<G>/<picture>.{feat,hid}.npy (sdpa), .{feat,hid}_eager.npy and .pv.npy (the processor's pixel_values);
feat = pooler_output[0] [(G/32)^2, 2048], hid = the tower's last_hidden_state [(G/16)^2, 1024] in the processor's
merge-block order. Summary rows: OUT/hf_tower_g<G>.json.
--only id|think,... runs only those cases of the arms (e.g. syn_shapes_count|off,nasa_truck_loc|on); a value that
names no case (e.g. --only none) writes the --npy arrays and generates nothing.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import transformers
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

HERE = os.path.dirname(os.path.abspath(__file__))
EOS = 124900
IMAGE_TOKEN = 125030
MAX_NEW = {"off": 64, "on": 32}
REV = "f83d2b0a2296aaa277f394990a9b8b949ddf9691"


def sha256(p):
  return hashlib.sha256(open(p, "rb").read()).hexdigest()


def image_path(fxdir, fx, arm, iid):
  key = {"N": "src", "B512": "g512", "B768": "g768"}[arm]
  return os.path.join(fxdir, fx["images"][iid][key])


def cases_for(fx, arm, score):
  out = []
  for think in ("off", "on"):
    if arm == "T0":
      for lab, q, _pat in score.TEXT_ROWS:
        out.append({"id": lab, "image": None, "question": q, "think": think})
    else:
      for r in fx["rows"]:
        out.append({"id": r["id"], "image": r["image"], "question": r["question"], "think": think,
                    "expect": r["expect"]})
  return out


def run_case(model, proc, fxdir, fx, arm, c, score):
  if c["image"] is None:
    msgs = [{"role": "user", "content": [{"type": "text", "text": c["question"]}]}]
  else:
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": c["question"]}]}]
  prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=(c["think"] == "on"))
  rec = {"prompt": prompt}
  if c["image"] is None:
    inputs = proc(text=[prompt], return_tensors="pt")
  else:
    p = image_path(fxdir, fx, arm, c["image"])
    img = Image.open(p).convert("RGB")
    inputs = proc(text=[prompt], images=[img], return_tensors="pt")
    rec.update({"image_file": os.path.relpath(p, fxdir), "image_sha256": sha256(p), "image_size_wh": list(img.size),
                "image_grid_thw": inputs["image_grid_thw"].tolist(),
                "pixel_values_shape": list(inputs["pixel_values"].shape)})
  ids = inputs["input_ids"][0].tolist()
  rec["n_input_ids"] = len(ids)
  rec["n_image_tokens"] = ids.count(IMAGE_TOKEN)
  rec["input_ids_head"] = ids[:8]
  rec["input_ids_tail"] = ids[-8:]
  t0 = time.time()
  with torch.inference_mode():
    out = model.generate(**inputs, do_sample=False, temperature=None, top_p=None, top_k=None,
                         repetition_penalty=1.0, max_new_tokens=MAX_NEW[c["think"]], eos_token_id=EOS,
                         pad_token_id=124893)
  new = out[0, len(ids):].tolist()
  rec["seconds"] = round(time.time() - t0, 2)
  rec["gen_ids"] = new
  rec["n_new"] = len(new)
  rec["ended_on_eos"] = bool(new) and new[-1] == EOS
  rec["text_raw"] = proc.tokenizer.decode(new, skip_special_tokens=False)
  rec["text"] = proc.tokenizer.decode(new, skip_special_tokens=True)
  if c.get("expect") is not None:
    rec["correct"] = score.correct(rec["text_raw"], c["expect"])
  return rec


def tower_refs(model, proc, fxdir, fx, g, out_dir, summary_path):
  os.makedirs(out_dir, exist_ok=True)
  vt = model.model.vision_tower
  rows = []
  for iid in fx["images"]:
    p = os.path.join(fxdir, fx["images"][iid][f"g{g}"])
    img = Image.open(p).convert("RGB")
    pp = proc.image_processor(images=[img], return_tensors="pt")
    pv, thw = pp["pixel_values"], pp["image_grid_thw"]
    res = {}
    for impl in ("sdpa", "eager"):
      vt.config._attn_implementation = impl
      with torch.inference_mode():
        o = model.model.get_image_features(pv, thw, return_dict=True)
      res[impl] = (o.pooler_output[0].float().numpy(), o.last_hidden_state.float().numpy())
    vt.config._attn_implementation = "sdpa"
    np.save(os.path.join(out_dir, f"{iid}.feat.npy"), res["sdpa"][0])
    np.save(os.path.join(out_dir, f"{iid}.hid.npy"), res["sdpa"][1])
    np.save(os.path.join(out_dir, f"{iid}.feat_eager.npy"), res["eager"][0])
    np.save(os.path.join(out_dir, f"{iid}.hid_eager.npy"), res["eager"][1])
    np.save(os.path.join(out_dir, f"{iid}.pv.npy"), pv.float().numpy())

    def m(a, b):
      a64, b64 = a.astype(np.float64), b.astype(np.float64)
      cos = (a64 * b64).sum(-1) / (np.linalg.norm(a64, axis=-1) * np.linalg.norm(b64, axis=-1) + 1e-30)
      return {"cos_min": float(cos.min()), "maxdiff": float(np.abs(a64 - b64).max()),
              "maxdiff_over_absmax": float(np.abs(a64 - b64).max() / np.abs(b64).max())}

    row = {"image": iid, "file": os.path.relpath(p, fxdir), "sha256": sha256(p), "image_grid_thw": thw.tolist(),
           "pixel_values_shape": list(pv.shape), "feat_shape": list(res["sdpa"][0].shape),
           "hid_shape": list(res["sdpa"][1].shape), "feat_absmax": float(np.abs(res["sdpa"][0]).max()),
           "hid_absmax": float(np.abs(res["sdpa"][1]).max()),
           "eager_vs_sdpa_feat": m(res["eager"][0], res["sdpa"][0]),
           "eager_vs_sdpa_hid": m(res["eager"][1], res["sdpa"][1])}
    rows.append(row)
    print("TOWER", g, json.dumps(row), flush=True)
  json.dump({"g": g, "transformers": transformers.__version__, "torch": torch.__version__, "rev": REV,
             "rule": "pooler_output[0] and last_hidden_state of get_image_features, fp32 CPU; sdpa = default",
             "rows": rows}, open(summary_path, "w"), indent=1)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--src", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--fixtures", default=os.path.join(HERE, "..", "fixtures"))
  ap.add_argument("--arms", required=True)
  ap.add_argument("--dtype", choices=("fp32", "bf16"), required=True)
  ap.add_argument("--npy", action="store_true")
  ap.add_argument("--only", default=None)
  ap.add_argument("--threads", type=int, default=8)
  args = ap.parse_args()
  fxdir = os.path.abspath(args.fixtures)
  sys.path.insert(0, fxdir)
  import score  # noqa: E402  (fixtures/score.py)
  torch.set_num_threads(args.threads)
  torch.manual_seed(0)
  os.makedirs(args.out, exist_ok=True)
  fx = json.load(open(os.path.join(fxdir, "fixtures.json")))
  only = set(args.only.split(",")) if args.only else None
  t0 = time.time()
  proc = AutoProcessor.from_pretrained(args.src, trust_remote_code=True)
  model = AutoModelForImageTextToText.from_pretrained(
      args.src, trust_remote_code=True, dtype=torch.float32 if args.dtype == "fp32" else torch.bfloat16).eval()
  load_s = round(time.time() - t0, 1)
  print(f"loaded {args.dtype} in {load_s} s attn={model.config._attn_implementation} "
        f"vision attn={model.model.vision_tower.config._attn_implementation}", flush=True)
  for arm in args.arms.split(","):
    if args.npy and args.dtype == "fp32" and arm in ("B512", "B768"):
      g = int(arm[1:])
      tower_refs(model, proc, fxdir, fx, g, os.path.join(args.out, "ref_npy", f"g{g}"),
                 os.path.join(args.out, f"hf_tower_g{g}.json"))
    path = os.path.join(args.out, f"hf_{arm}_{args.dtype}.json")
    res = json.load(open(path)) if os.path.exists(path) else {
        "arm": arm, "dtype": args.dtype, "rev": REV, "transformers": transformers.__version__,
        "torch": torch.__version__, "threads": args.threads, "max_new_tokens": MAX_NEW, "eos": EOS,
        "rule": "greedy, repetition_penalty 1.0, CPU; think off = enable_thinking False", "cases": []}
    res.setdefault("runs", []).append({"load_s": load_s, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                                       "attn": model.config._attn_implementation})
    done = {(c["id"], c["think"]) for c in res["cases"]}
    for c in cases_for(fx, arm, score):
      if (c["id"], c["think"]) in done or (only is not None and f"{c['id']}|{c['think']}" not in only):
        continue
      rec = run_case(model, proc, fxdir, fx, arm, c, score)
      res["cases"].append({**{k: c[k] for k in ("id", "image", "question", "think")},
                           "expect": c.get("expect"), **rec})
      tmp = path + ".tmp"
      with open(tmp, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
        f.write("\n")
      os.replace(tmp, path)
      print(f"[{arm} {args.dtype} {c['think']} {c['id']}] {rec['seconds']} s in={rec['n_input_ids']} "
            f"img={rec['n_image_tokens']} new={rec['n_new']} eos={rec['ended_on_eos']} "
            f"ok={rec.get('correct')} | {rec['text'][:100]!r}", flush=True)
    print("ARM_DONE", arm, args.dtype, path, flush=True)
  print("HF_REF_DONE", args.arms, args.dtype, flush=True)
  return 0


if __name__ == "__main__":
  sys.exit(main())
