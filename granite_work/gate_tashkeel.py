"""Tashkeel-350M task gate: bundle vs HF fp32 greedy on the model's own task
(Arabic diacritization), following the s1-mini precedent — this finetune
diacritizes its input instead of answering questions, so the generic
8-question gate certifies nothing about it.

Both sides render the SAME string with tokenizer.apply_chat_template (the
model card's prompt form: "قم بتشكيل هذا النص :\n" + text); the bundle side
runs `litert-lm run --no-template` so the engine applies no template of its
own. An HF BOS A/B runs first: granite's template has no leading BOS, and at
350M scale a prepended <|end_of_text|> start_token flips the greedy
trajectory (the shipped granite-4.0-h-350m lesson) — the bundle under test
must have start_token dropped.

    python granite_work/gate_tashkeel.py <bundle> [--backend cpu]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import torch

HERE = Path(__file__).parent
MODEL = os.environ.get("MODEL", "Etherll/Tashkeel-350M-v2")
INSTRUCTION = "قم بتشكيل هذا النص :\n"

# (input, model-card expected output or None). Case 0 is the card's own
# worked example; the rest are undiacritized MSA probes checked HF-vs-bundle.
CASES = [
    ("السلام عليكم", "اَلسَلَامُ عَلَيْكُمْ"),
    ("ذهب الولد الى المدرسة صباحا", None),
    ("العلم نور والجهل ظلام", None),
    ("قرات كتابا مفيدا في المكتبة", None),
    ("تطلع الشمس من الشرق وتغرب في الغرب", None),
    ("اللغة العربية من اجمل لغات العالم", None),
    ("يحب الاطفال اللعب في الحديقة", None),
    ("الصبر مفتاح الفرج", None),
    ("كتب الطالب الدرس في دفتره الجديد", None),
    ("السماء صافية والطقس جميل اليوم", None),
]


def render(tok, text):
  messages = [{"role": "user", "content": INSTRUCTION + text}]
  return tok.apply_chat_template(messages, tokenize=False,
                                 add_generation_prompt=True)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("bundle")
  ap.add_argument("--backend", default="cpu")
  ap.add_argument("--litert-lm", default="litert-lm")
  ap.add_argument("--out", default=None)
  ap.add_argument("--skip-hf", action="store_true")
  args = ap.parse_args()

  from transformers import AutoModelForCausalLM, AutoTokenizer
  tok = AutoTokenizer.from_pretrained(MODEL)

  hf_texts, bos_ab = [], None
  if not args.skip_hf:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, low_cpu_mem_usage=True).eval()
    bos_id = tok.convert_tokens_to_ids("<|end_of_text|>")

    def hf_gen(prompt_str, prepend_bos=False):
      ids = tok(prompt_str, return_tensors="pt", add_special_tokens=False).input_ids
      if prepend_bos:
        ids = torch.cat([torch.tensor([[bos_id]]), ids], dim=1)
      with torch.no_grad():
        out = model.generate(ids, max_new_tokens=256, do_sample=False)
      return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

    p0 = render(tok, CASES[0][0])
    a, b = hf_gen(p0, False), hf_gen(p0, True)
    bos_ab = {"no_bos": a, "with_bos": b, "same": a == b}
    print("BOS A/B same:", bos_ab["same"], flush=True)

    for text, _exp in CASES:
      t = hf_gen(render(tok, text))
      hf_texts.append(t)
      print(f"HF: {t[:90]}", flush=True)

  rows, match_hf, match_card, n_card = [], 0, 0, 0
  for i, (text, exp) in enumerate(CASES):
    prompt = render(tok, text)
    pr = subprocess.run(
        [args.litert_lm, "run", args.bundle, "--no-template", "--prompt", prompt,
         "--backend", args.backend, "--cache", "no",
         "--temperature", "0", "--seed", "0"],
        capture_output=True, text=True, timeout=900, stdin=subprocess.DEVNULL)
    got = pr.stdout.strip()
    row = {"in": text, "bundle": got}
    if hf_texts:
      row["hf"] = hf_texts[i]
      row["match_hf"] = got == hf_texts[i]
      match_hf += row["match_hf"]
    if exp is not None:
      n_card += 1
      row["card_expected"] = exp
      row["match_card"] = got == exp
      match_card += row["match_card"]
    rows.append(row)
    print(f"[{i}] rc={pr.returncode} bundle: {got[:90]}", flush=True)

  res = {"bundle": args.bundle, "backend": args.backend, "bos_ab": bos_ab,
         "match_hf": match_hf, "n": len(CASES),
         "match_card": match_card, "n_card": n_card, "cases": rows}
  out = args.out or str(HERE / f"gate_tashkeel_{args.backend}.json")
  Path(out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
  print("GATE_DONE", json.dumps({"match_hf": match_hf, "n": len(CASES),
                                 "match_card": match_card, "n_card": n_card}))
  raise SystemExit(0 if match_hf == len(CASES) else 1)


if __name__ == "__main__":
  main()
