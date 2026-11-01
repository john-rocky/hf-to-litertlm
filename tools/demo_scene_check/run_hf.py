#!/usr/bin/env python3
"""Run one model's plan with Hugging Face transformers: greedy, float32, one JSON line per case.

  HF_HUB_OFFLINE=1 python tools/demo_scene_check/run_hf.py --model-path <local snapshot or org/name> --name lfm3 \
      --plan DIR/plan/lfm3.json --out DIR/raw/lfm3_mps.jsonl --device mps [--revision <sha>] [--trust-remote-code] \
      [--only ID,ID] [--dry]

Load: AutoProcessor + AutoModelForImageTextToText in float32 (text-only cases go through the same classes). Generate:
do_sample False, repetition_penalty 1.0, no temperature / top-k / top-p, eos = the generation config's eos plus
<|im_end|> when the tokenizer has it. The picture of a case is the file the plan names, relative to the fixtures
directory (arm N = the source picture through the model's own processor unchanged; arm S512 = the picture already
stretched to 512 x 512). Each output line = the case + the rendered prompt, token counts, image tokens, generated ids,
both decodes, seconds, think_tokens (ids up to and including the first thinking close), ended_on_eos. A re-run skips
the keys already in --out. --dry renders every prompt and runs the processor without loading the model.

Run the first three cases on --device cpu and on mps into two files and compare their generated ids before trusting
the faster device for the rest (a device whose ids differ is not used).
"""
import argparse
import hashlib
import json
import os
import sys
import time

import torch
import transformers
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def sha256(p):
  return hashlib.sha256(open(p, "rb").read()).hexdigest()


def eos_ids(model, tok):
  ids = set()
  gc = getattr(model, "generation_config", None) if model is not None else None
  e = getattr(gc, "eos_token_id", None) if gc is not None else None
  if isinstance(e, int):
    ids.add(e)
  elif e:
    ids.update(e)
  i = tok.convert_tokens_to_ids("<|im_end|>")
  if isinstance(i, int) and i is not None and i >= 0 and i != tok.unk_token_id:
    ids.add(i)
  return sorted(ids)


def render(proc, case):
  msgs = []
  if case.get("system"):
    msgs.append({"role": "system", "content": case["system"]})
  content = []
  if case.get("image"):
    content.append({"type": "image"})
  content.append({"type": "text", "text": case["text"]})
  msgs.append({"role": "user", "content": content})
  kw = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": case["think"] == "on"}
  if case.get("tools"):
    kw["tools"] = case["tools"]
  return proc.apply_chat_template(msgs, **kw)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model-path", required=True)
  ap.add_argument("--name", required=True, help="the model name of the plan (written into every record)")
  ap.add_argument("--revision", default=None, help="the Hub commit (loaded for an org/name, recorded for a local dir)")
  ap.add_argument("--trust-remote-code", action="store_true")
  ap.add_argument("--plan", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
  ap.add_argument("--threads", type=int, default=8)
  ap.add_argument("--only", default="")
  ap.add_argument("--dry", action="store_true")
  args = ap.parse_args()
  torch.set_num_threads(args.threads)
  torch.manual_seed(0)
  plan = json.load(open(args.plan))
  if plan.get("model") != args.name:
    raise SystemExit(f"--name {args.name} but the plan is for {plan.get('model')}")
  fx_dir = os.path.dirname(plan["fixtures"])
  cases = plan["cases"]
  if args.only:
    want = set(args.only.split(","))
    cases = [c for c in cases if c["key"] in want or c["id"] in want]
  done = set()
  if os.path.exists(args.out):
    for line in open(args.out):
      if line.strip():
        done.add(json.loads(line)["key"])
  todo = [c for c in cases if c["key"] not in done]
  print(f"plan {args.plan}: {len(cases)} cases, {len(done)} already in {args.out}, {len(todo)} to run", flush=True)
  t0 = time.time()
  kw = {"trust_remote_code": args.trust_remote_code}
  if args.revision and not os.path.isdir(args.model_path):   # a local snapshot: the revision is only recorded
    kw["revision"] = args.revision
  proc = AutoProcessor.from_pretrained(args.model_path, **kw)
  tok = proc.tokenizer
  img_tok = getattr(proc, "image_token", None)
  img_id = tok.convert_tokens_to_ids(img_tok) if img_tok else None
  model = None
  if not args.dry:
    model = AutoModelForImageTextToText.from_pretrained(args.model_path, dtype=torch.float32, **kw).eval().to(args.device)
  load_s = round(time.time() - t0, 1)
  eos = eos_ids(model, tok)
  closes = {c.get("think_close") or "</think>" for c in todo} or {"</think>"}
  if len(closes) != 1:
    raise SystemExit(f"one thinking close per plan: {closes}")
  think_close = tok.convert_tokens_to_ids(closes.pop())
  if not isinstance(think_close, int) or think_close == tok.unk_token_id:
    think_close = None
  print(f"loaded {args.name} on {args.device} in {load_s} s (dry={args.dry}) eos={eos} think_close={think_close} "
        f"image_token={img_tok}:{img_id} transformers={transformers.__version__} torch={torch.__version__}", flush=True)
  os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
  for c in todo:
    prompt = render(proc, c)
    rec = dict(c)
    rec.update({"model": args.name, "model_path": args.model_path, "model_rev": args.revision, "device": args.device,
                "dtype": "fp32", "transformers": transformers.__version__, "torch": torch.__version__,
                "prompt": prompt, "eos_ids": eos, "started": time.strftime("%Y-%m-%d %H:%M:%S")})
    if c.get("image"):
      p = os.path.join(fx_dir, c["image"])
      img = Image.open(p).convert("RGB")
      inputs = proc(text=[prompt], images=[img], return_tensors="pt")
      rec.update({"image_sha256": sha256(p), "image_size_wh": list(img.size)})
      for k in ("image_grid_thw", "pixel_values", "spatial_shapes", "pixel_attention_mask", "image_sizes"):
        if k in inputs:
          v = inputs[k]
          rec[f"{k}_shape"] = list(v.shape)
          if k in ("image_grid_thw", "spatial_shapes"):
            rec[k] = v.tolist()
    else:
      inputs = proc(text=[prompt], return_tensors="pt")
    ids = inputs["input_ids"][0].tolist()
    rec["n_input_ids"] = len(ids)
    rec["n_image_tokens"] = ids.count(img_id) if img_id is not None else None
    if args.dry:
      print(f"[dry {c['key']}] in={len(ids)} img={rec['n_image_tokens']} | {prompt[-160:]!r}", flush=True)
      continue
    inputs = inputs.to(args.device)
    t1 = time.time()
    with torch.inference_mode():
      out = model.generate(**inputs, do_sample=False, temperature=None, top_p=None, top_k=None,
                           repetition_penalty=1.0, max_new_tokens=c["max_new"], eos_token_id=eos,
                           pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else eos[0])
    new = out[0, len(ids):].tolist()
    rec["seconds"] = round(time.time() - t1, 2)
    rec["gen_ids"] = new
    rec["n_new"] = len(new)
    rec["ended_on_eos"] = bool(new) and new[-1] in eos
    rec["think_tokens"] = (new.index(think_close) + 1) if (think_close is not None and think_close in new) else None
    rec["text_raw"] = tok.decode(new, skip_special_tokens=False)
    rec["text"] = tok.decode(new, skip_special_tokens=True)
    rec["load_s"] = load_s
    with open(args.out, "a") as f:
      f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[{args.name} {c['key']}] {rec['seconds']} s in={len(ids)} img={rec['n_image_tokens']} new={len(new)} "
          f"eos={rec['ended_on_eos']} think={rec['think_tokens']} | {rec['text'][-140:]!r}", flush=True)
  print(f"RUN_HF_DONE {args.name} {args.device} {args.out} total {round(time.time() - t0, 1)} s", flush=True)
  return 0


if __name__ == "__main__":
  sys.exit(main())
