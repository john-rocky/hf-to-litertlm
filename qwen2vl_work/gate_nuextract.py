"""NuExtract-2.0-2B task gate: bundle vs HF fp32 greedy on the model's own
task (template-based structured extraction), text AND image.

Prompt contract: NuExtract's own chat template with its `template` kwarg
renders `# Template:\n{...}\n# Context:\n<doc>`; both sides consume the SAME
rendered string (`litert-lm run --no-template`). For image cases the
NuExtract placeholder `<|image_pad|>` is swapped for the bundle's
`<image_soft_token>` (the fast_vlm runtime splits the prompt on that literal;
the image itself rides `--attachment`).

The HF image reference runs with 1-D sequential positions (all three M-RoPE
streams equal) — the deployed fast_vlm contract, per the base card's M-RoPE
note — and the synthetic image is exactly 672x672 so HF's dynamic-resolution
processor lands on the same 48x48 grid / 576 tokens the static bundle uses.
Image comparison is semantic (parsed JSON fields), not byte parity: the
bundle's int8 vision + int4 decoder and its in-graph preprocessing make
byte-equality the wrong bar; the base ship set the precedent
(device-verified "functionally accurate").

    python qwen2vl_work/gate_nuextract.py <bundle> [--backend cpu]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
MODEL = os.environ.get("MODEL", "numind/NuExtract-2.0-2B")
SOFT = "<image_soft_token>"

TEXT_CASES = [
    # (template, document) — case 0 is the model card's own worked example
    ('{"names": ["string"]}',
     "John went to the restaurant with Mary. James went to the cinema."),
    ('{"invoice": {"number": "string", "total": "string", "due_date": "string"}}',
     "Invoice #INV-2041 issued 2026-08-01. Total amount: $1,530.50, due September 15, 2026."),
    ('{"product": "string", "issues": ["string"]}',
     "The X200 laptop keeps overheating and the fan is loud. Also the battery drains fast."),
]

IMAGE_TEMPLATE = '{"name": "string", "email": "string", "phone": "string"}'
IMAGE_GROUND_TRUTH = {"name": "Alice Tanaka", "email": "alice@example.com",
                      "phone": "+81-90-1234-5678"}


def make_card_image(path):
  from PIL import Image, ImageDraw
  img = Image.new("RGB", (672, 672), "white")
  d = ImageDraw.Draw(img)
  try:
    from PIL import ImageFont
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
    small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
  except Exception:
    font = small = None
  d.text((60, 180), "Alice Tanaka", fill="black", font=font)
  d.text((60, 280), "alice@example.com", fill="black", font=small)
  d.text((60, 360), "+81-90-1234-5678", fill="black", font=small)
  img.save(path)
  return path


def parse_json(text):
  m = re.search(r"\{.*\}", text, re.DOTALL)
  if not m:
    return None
  try:
    return json.loads(m.group(0))
  except json.JSONDecodeError:
    return None


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("bundle")
  ap.add_argument("--backend", default="cpu")
  ap.add_argument("--litert-lm", default="litert-lm")
  ap.add_argument("--out", default=None)
  ap.add_argument("--skip-image", action="store_true")
  args = ap.parse_args()

  import torch
  from transformers import AutoProcessor, AutoTokenizer

  tok = AutoTokenizer.from_pretrained(MODEL)

  def render(document, template, image=False):
    # image case: a bare image mapping — the NuExtract jinja's list branch
    # treats ANY text item (even "") as the input document and then drops the
    # image placeholder (measured: HF raises tokens:0/features:576), while the
    # mapping branch renders image_placeholder unconditionally
    content = {"type": "image"} if image else document
    return tok.apply_chat_template(
        [{"role": "user", "content": content}], template=template,
        tokenize=False, add_generation_prompt=True)

  def run_bundle(prompt, attachment=None):
    cmd = [args.litert_lm, "run", args.bundle, "--no-template",
           "--prompt", prompt, "--backend", args.backend, "--cache", "no",
           "--temperature", "0", "--seed", "0"]
    if attachment:
      cmd += ["--attachment", attachment]
    pr = subprocess.run(cmd, capture_output=True, text=True, timeout=1800,
                        stdin=subprocess.DEVNULL)
    return pr.stdout.strip(), pr.returncode

  # ---- HF references ----
  from transformers import Qwen2VLForConditionalGeneration
  print("loading HF fp32 reference...", flush=True)
  model = Qwen2VLForConditionalGeneration.from_pretrained(
      MODEL, dtype=torch.float32, low_cpu_mem_usage=True).eval()

  def hf_text(prompt):
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    with torch.no_grad():
      out = model.generate(ids, max_new_tokens=256, do_sample=False)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

  rows, match_text = [], 0
  for template, doc in TEXT_CASES:
    prompt = render(doc, template)
    hf = hf_text(prompt)
    got, rc = run_bundle(prompt)
    row = {"mode": "text", "template": template, "doc": doc,
           "hf": hf, "bundle": got, "match_hf": got == hf}
    match_text += row["match_hf"]
    rows.append(row)
    print(f"[text] rc={rc} match={row['match_hf']} bundle: {got[:80]!r}", flush=True)

  image_row = None
  if not args.skip_image:
    img_path = str(HERE / "gate_nuextract_card.png")
    make_card_image(img_path)
    prompt = render("", IMAGE_TEMPLATE, image=True)
    # --attachment is incompatible with --no-template (CLI refuses), so the
    # image case runs through the ENGINE template: prompt = the NuExtract body
    # only; the engine renders ChatML and places the attachment's image item
    # before the text (the bundle's IMG_RENDER carries <image_soft_token>).
    body = f"# Template:\n{IMAGE_TEMPLATE}\n# Context:\n"
    cmd = [args.litert_lm, "run", args.bundle, "--prompt", body,
           "--backend", args.backend, "--cache", "no",
           "--temperature", "0", "--seed", "0", "--attachment", img_path]
    pr = subprocess.run(cmd, capture_output=True, text=True, timeout=1800,
                        stdin=subprocess.DEVNULL)
    got, rc = pr.stdout.strip(), pr.returncode
    bundle_json = parse_json(got)

    # HF 1-D-position reference (the deployed fast_vlm contract)
    from PIL import Image
    processor = AutoProcessor.from_pretrained(MODEL)
    holder = model.model if hasattr(model.model, "get_rope_index") else model

    def seq_rope_index(input_ids=None, image_grid_thw=None, video_grid_thw=None,
                       second_per_grid_ts=None, attention_mask=None, **kw):
      bsz, seqlen = input_ids.shape
      pos = torch.arange(seqlen, dtype=input_ids.dtype).view(1, 1, seqlen).expand(3, bsz, seqlen)
      return pos.contiguous(), torch.zeros(bsz, 1, dtype=input_ids.dtype)

    holder.get_rope_index = seq_rope_index
    if hasattr(model, "rope_deltas"):
      model.rope_deltas = None
    inputs = processor(text=[prompt], images=[Image.open(img_path)],
                       return_tensors="pt")
    with torch.no_grad():
      out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    hf_img = processor.tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    hf_json = parse_json(hf_img)

    image_row = {"mode": "image", "template": IMAGE_TEMPLATE,
                 "ground_truth": IMAGE_GROUND_TRUTH,
                 "hf_1d": hf_img, "bundle": got,
                 "hf_json": hf_json, "bundle_json": bundle_json,
                 "match_hf_json": bundle_json == hf_json,
                 "match_truth": bundle_json == IMAGE_GROUND_TRUTH}
    rows.append(image_row)
    print(f"[image] rc={rc} bundle: {got[:100]!r}", flush=True)
    print(f"[image] hf_1d : {hf_img[:100]!r}", flush=True)

  res = {"bundle": args.bundle, "backend": args.backend,
         "match_text": match_text, "n_text": len(TEXT_CASES),
         "image": image_row, "cases": rows}
  out = args.out or str(HERE / f"gate_nuextract_{args.backend}.json")
  Path(out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
  img_ok = (args.skip_image or (image_row and
            (image_row["match_truth"] or image_row["match_hf_json"])))
  print("GATE_DONE", json.dumps({"match_text": match_text, "n_text": len(TEXT_CASES),
                                 "image_ok": bool(img_ok)}))
  raise SystemExit(0 if match_text == len(TEXT_CASES) and img_ok else 1)


if __name__ == "__main__":
  main()
