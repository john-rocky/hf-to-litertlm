#!/usr/bin/env python3
"""HF fp32 oracle generations for the MiniCPM-V-4.7-1B fast_vlm gate, three arms on the same
fixtures (copied from qwen35vl_work/hf_oracle_gen.py and adapted; the source file is untouched):

  N  native  FIX512 through the processor as shipped (512^2 > 448^2 -> thumbnail + 2 slices,
             196 soft tokens) with canvas M-RoPE.
  F  fix448  FIX448 (W/fixtures, PIL BICUBIC 448^2) through the processor (thumbnail only,
             64 soft tokens) with canvas M-RoPE = the bundle's pixels, HF's positions.
  P  pos1d   F + every position channel = arange and rope_deltas = 0 (model.model.get_rope_index
             replaced) = what the fast_vlm runtime can express (1-D positions from 0, no start
             token). The bundle's reference arm.

Real MiniCPMV4_7ForConditionalGeneration, fp32, eager attention, CPU, greedy, enable_thinking=False
(empty <think> block = the bundle's template). Results are written after every case, and a rerun
skips the cases already in the output file (so one arm can be resumed or split).

    python vl_oracle.py --arm F --src SRC --fix448 W/fixtures --fix512 $FIXTURE_SRC \
        --out r3/oracle/hf_vl_F.json [--only cats_448.png:0] [--threads 4] [--max-new-tokens 64]
"""
import argparse
import hashlib
import json
import os
import time

import torch
import transformers
from PIL import Image
from transformers import AutoProcessor, MiniCPMV4_7ForConditionalGeneration

IMAGE_PAD = 248056
EOS = {248044, 248046}
NAMES = ["cats", "kitchen1", "kitchen2"]
PROMPTS = ["What is in this image?",
           "Describe the main colors in this image.",
           "Where is this scene?"]
ARMS = {
    "N": "native: FIX512 through the processor as shipped (thumbnail + slices), canvas M-RoPE",
    "F": "fix448: FIX448 through the processor (thumbnail only, 64 tokens), canvas M-RoPE",
    "P": "pos1d: FIX448 (64 tokens) + all position channels = arange, rope_deltas = 0 (fast_vlm contract)",
}


def sha256(p):
  return hashlib.sha256(open(p, "rb").read()).hexdigest()


def rope_summary(model, inputs):
  pos, deltas = model.model.get_rope_index(
      inputs["input_ids"], mm_token_type_ids=inputs["mm_token_type_ids"],
      target_sizes=inputs["target_sizes"], attention_mask=inputs["attention_mask"])
  ids = inputs["input_ids"][0]
  p = pos[:, 0, :]
  img = (ids == IMAGE_PAD)
  nz = img.nonzero()
  first_img, last_img = int(nz[0]), int(nz[-1])
  return {
      "rope_deltas": int(deltas[0, 0]),
      "seq_len": int(ids.shape[0]),
      "max_position": int(p.max()),
      "image_token_span": [first_img, last_img],
      "image_t_range": [int(p[0, img].min()), int(p[0, img].max())],
      "image_h_range": [int(p[1, img].min()), int(p[1, img].max())],
      "image_w_range": [int(p[2, img].min()), int(p[2, img].max())],
      "last_token_thw": [int(v) for v in p[:, -1]],
      "positions_equal_arange_all_channels": bool(
          (p == torch.arange(p.shape[1]).expand(3, -1)).all()),
  }


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--arm", required=True, choices=sorted(ARMS))
  ap.add_argument("--src", required=True)
  ap.add_argument("--fix448", required=True)
  ap.add_argument("--fix512", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--only", default=None, help="comma list of <name>:<prompt index>, e.g. cats:0")
  ap.add_argument("--threads", type=int, default=0)
  ap.add_argument("--max-new-tokens", type=int, default=64)
  args = ap.parse_args()
  if args.threads:
    torch.set_num_threads(args.threads)

  cases = [(n, k) for n in NAMES for k in range(len(PROMPTS))]
  if args.only:
    want = {(s.split(":")[0], int(s.split(":")[1])) for s in args.only.split(",")}
    cases = [c for c in cases if c in want]

  res = None
  if os.path.exists(args.out):
    res = json.load(open(args.out))
    assert res["arm"] == args.arm, (res["arm"], args.arm)
  done = {(c["name"], c["prompt_index"]) for c in (res or {}).get("cases", [])}
  todo = [c for c in cases if c not in done]
  print(f"arm {args.arm}: {len(todo)} to run, {len(done)} already in {args.out}", flush=True)
  if not todo:
    return 0

  t_load = time.time()
  proc = AutoProcessor.from_pretrained(args.src)
  model = MiniCPMV4_7ForConditionalGeneration.from_pretrained(
      args.src, dtype=torch.float32, attn_implementation="eager").eval()
  t_load = time.time() - t_load

  if args.arm == "P":
    orig = model.model.get_rope_index

    def seq_rope_index(*a, **kw):
      pos3, delta = orig(*a, **kw)          # pos3 [3, B, S]
      S = pos3.shape[-1]
      seq = torch.arange(S, device=pos3.device).view(1, 1, S).expand_as(pos3)
      return seq.contiguous(), torch.zeros_like(delta)

    model.model.get_rope_index = seq_rope_index

  fixtures = {}
  for n in NAMES:
    p = (os.path.join(args.fix512, f"{n}_512.png") if args.arm == "N"
         else os.path.join(args.fix448, f"{n}_448.png"))
    fixtures[n] = {"path": p, "sha256": sha256(p)}
  if res is None:
    res = {"arm": args.arm, "desc": ARMS[args.arm], "transformers": transformers.__version__,
           "torch": torch.__version__, "src": args.src, "prompts": PROMPTS,
           "max_new_tokens": args.max_new_tokens, "fixtures": fixtures, "cases": []}
  res.setdefault("runs", []).append({"threads": torch.get_num_threads(), "load_s": round(t_load, 1),
                                     "started": time.strftime("%Y-%m-%d %H:%M:%S")})

  for n, k in todo:
    img = Image.open(fixtures[n]["path"]).convert("RGB")
    want = (512, 512) if args.arm == "N" else (448, 448)
    assert img.size == want, (img.size, want)
    msgs = [{"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": PROMPTS[k]}]}]
    prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                      enable_thinking=False)
    inputs = proc(text=[prompt], images=[img], return_tensors="pt")
    ids = inputs["input_ids"][0].tolist()
    t0 = time.time()
    with torch.no_grad():
      out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                           temperature=None, top_p=None, top_k=None)
    new = out[0, len(ids):].tolist()
    case = {
        "name": n, "prompt_index": k, "image": os.path.basename(fixtures[n]["path"]),
        "prompt": PROMPTS[k],
        "input_image_size_wh": list(img.size),
        "pixel_values_shape": list(inputs["pixel_values"].shape),
        "target_sizes": inputs["target_sizes"].tolist(),
        "image_pad_count": ids.count(IMAGE_PAD),
        "seq_len": len(ids),
        "rope": rope_summary(model, inputs),
        "answer": proc.tokenizer.decode(new, skip_special_tokens=True),
        "answer_ids": new,
        "n_new": len(new),
        "ended_on_eos": bool(new) and new[-1] in EOS,
        "seconds": round(time.time() - t0, 1),
    }
    res["cases"].append(case)
    res["cases"].sort(key=lambda c: (NAMES.index(c["name"]), c["prompt_index"]))
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
      json.dump(res, f, indent=1, ensure_ascii=False)
      f.write("\n")
    os.replace(tmp, args.out)
    print(f"[{args.arm} {n} | {PROMPTS[k]}] {case['seconds']} s pads={case['image_pad_count']} "
          f"seq={case['seq_len']} n_new={case['n_new']} eos={case['ended_on_eos']}\n{case['answer']}\n",
          flush=True)
  print("ORACLE_ARM_DONE", args.arm, args.out, flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
