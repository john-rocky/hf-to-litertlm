#!/usr/bin/env python3
"""quant_sens_cases.py - the cases file of tools/quant_sensitivity/teacher_force.py for LFM3-VL-2B: the 23 fixture
questions x thinking off / on, each with its picture's 256 image rows from the vision graphs.

  python -I quant_sens_cases.py --teacher REF/hf_B512_fp32.json [--bf16 REF/hf_B512_bf16.json] --src SRC
      --tower-dir OUTDIR/tower --out-dir DIR [--ref-dir REF/ref_npy] [--fixtures lfm3vl_work/fixtures]

--teacher = hf_ref.py --arms B512 --dtype fp32 (the source model's greedy tokens on the 512 x 512 pictures): each
case's prompt is tokenized with SRC/tokenizer.json and its one <|image|> expanded to the 256 slots the processor
writes; the ids must equal the record's n_input_ids / input_ids_head / input_ids_tail (the command stops otherwise).
Teacher = the record's first 32 generated ids. --bf16 (the same arm in bfloat16) marks the cases whose first 32 tokens
agree between float32 and bfloat16 as `counted` (a case where the source model's own precisions disagree says nothing
about quantization); without it every case is counted.
Image rows = the shipped vision graphs (vision_encoder_g512_v2.tflite, fp32 weights -> vision_adapter_g512_v2_fp16.tflite)
on the picture as the runtime feeds it, float(u8) * (1 / 255), LiteRT CPU Interpreter -> DIR/vision/<picture>.npy
[256, 2048]; with --ref-dir the per-token cosine against the source model's image features is printed.
-> DIR/cases.json.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_ID, N_IMAGE, PAD_ID, EOS = 125030, 256, 124893, 124900


def sha256(p):
  h = hashlib.sha256()
  with open(p, "rb") as f:
    for c in iter(lambda: f.read(1 << 24), b""):
      h.update(c)
  return h.hexdigest()


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--teacher", required=True)
  ap.add_argument("--bf16", default=None)
  ap.add_argument("--src", required=True)
  ap.add_argument("--tower-dir", required=True)
  ap.add_argument("--out-dir", required=True)
  ap.add_argument("--ref-dir", default=None)
  ap.add_argument("--fixtures", default=os.path.join(HERE, "fixtures"))
  ap.add_argument("--threads", type=int, default=8)
  a = ap.parse_args()
  from ai_edge_litert import interpreter as interp
  from tokenizers import Tokenizer
  fxdir = os.path.abspath(a.fixtures)
  sys.path.insert(0, fxdir)
  import score  # noqa: E402  (fixtures/score.py)
  fx = json.load(open(os.path.join(fxdir, "fixtures.json")))
  tok = Tokenizer.from_file(os.path.join(a.src, "tokenizer.json"))
  ref = json.load(open(a.teacher))
  bf16 = {f"{c['id']}|{c['think']}": c for c in json.load(open(a.bf16))["cases"]} if a.bf16 else {}
  cases = []
  for c in ref["cases"]:
    key = f"{c['id']}|{c['think']}"
    head, tail = c["prompt"].split("<|image|>")
    ids = (tok.encode(head, add_special_tokens=False).ids + [IMAGE_ID] * N_IMAGE
           + tok.encode(tail, add_special_tokens=False).ids)
    ok = (len(ids) == c["n_input_ids"] and ids[:len(c["input_ids_head"])] == c["input_ids_head"]
          and ids[-len(c["input_ids_tail"]):] == c["input_ids_tail"] and ids.count(IMAGE_ID) == c["n_image_tokens"])
    if not ok:
      sys.exit(f"STOP: ids differ from the processor's: {key} len {len(ids)} vs {c['n_input_ids']}")
    counted = True
    if bf16:
      f_ids, f_eos = score.split_eos(c["gen_ids"])
      b_ids, b_eos = score.split_eos(bf16[key]["gen_ids"])
      counted = score.same32_eos(f_ids, f_eos, b_ids, b_eos)
    cases.append({"key": key, "group": c["think"], "ids": ids, "teacher": c["gen_ids"][:32],
                  "image_npy": f"vision/{c['image']}.npy", "counted": counted})
  vdir = os.path.join(a.out_dir, "vision")
  os.makedirs(vdir, exist_ok=True)
  enc_p = os.path.join(a.tower_dir, "vision_encoder_g512_v2.tflite")
  adp_p = os.path.join(a.tower_dir, "vision_adapter_g512_v2_fp16.tflite")
  enc = interp.Interpreter(model_path=enc_p, num_threads=a.threads).get_signature_runner()
  adp = interp.Interpreter(model_path=adp_p, num_threads=a.threads).get_signature_runner()
  e_in, a_in = list(enc.get_input_details())[0], list(adp.get_input_details())[0]
  vis = {}
  for img in sorted({c["image_npy"].split("/")[1][:-4] for c in cases}):
    png = os.path.join(fxdir, fx["images"][img]["g512"])
    u8 = np.asarray(Image.open(png).convert("RGB"), dtype=np.uint8)
    x = (u8.astype(np.float32) * np.float32(1.0 / 255.0))[None]
    feat = list(adp(**{a_in: list(enc(**{e_in: x}).values())[0]}).values())[0][0].astype(np.float32)
    assert feat.shape == (N_IMAGE, 2048), feat.shape
    np.save(os.path.join(vdir, f"{img}.npy"), feat)
    vis[img] = {"png_sha256": sha256(png)}
    if a.ref_dir:
      hf = np.load(os.path.join(a.ref_dir, "g512", f"{img}.feat.npy")).reshape(N_IMAGE, -1).astype(np.float64)
      f64 = feat.astype(np.float64)
      cos = (hf * f64).sum(-1) / (np.linalg.norm(hf, axis=-1) * np.linalg.norm(f64, axis=-1))
      vis[img]["cos_min_vs_source"] = float(cos.min())
      print(f"vision {img}: per-token cosine min vs the source model {cos.min():.7f}", flush=True)
  json.dump({"pad_id": PAD_ID, "image_token_id": IMAGE_ID, "teacher": os.path.abspath(a.teacher),
             "vision": {"encoder": sha256(enc_p), "adapter": sha256(adp_p)}, "pictures": vis, "cases": cases},
            open(os.path.join(a.out_dir, "cases.json"), "w"), indent=1)
  n_counted = {g: sum(c["counted"] for c in cases if c["group"] == g) for g in ("off", "on")}
  print(f"CASES_DONE {len(cases)} cases ({n_counted} counted), {len(vis)} pictures -> {a.out_dir}/cases.json", flush=True)
  return 0


if __name__ == "__main__":
  sys.exit(main())
