"""s1-mini task gate: bundle vs HF fp32 greedy on the model's own documented
behavior, plus the model card's published expected outputs where they exist.

Prompt contract (model card, REQUIRED): exact system prompt + control line +
raw transcript, ChatML with enable_thinking=False. Both sides render the SAME
string via tokenizer.apply_chat_template; the bundle side runs
`litert-lm run --no-template` so the engine applies no template of its own.

Also runs an HF BOS A/B first: if
prepending <|endoftext|> changes HF outputs, the bundle's declared start_token
matters and must be checked/removed.

    .venv-vl093/bin/python s1mini_work/gate_normalize.py <bundle> [--backend cpu]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import torch

HERE = Path(__file__).parent
ROOT = HERE.parent
MODEL = os.environ.get("MODEL", "superwhisper/s1-mini")

SYSTEM = (
    "You are a text normalizer for speech-to-text transcripts. The input begins "
    "with a control line specifying the styling, structure, and context settings; "
    "clean the transcript to match those settings and output only the cleaned text."
)

REG_INPUT = "hmm im gonna be late theres a cute dog outside i cant just walk past him"
CASES = [
    # (styling, structure, context, transcript, expected_or_None)
    ("casual", "prose", "general", REG_INPUT,
     "hmm im gonna be late. theres a cute dog outside. i cant just walk past him"),
    ("semi-casual", "prose", "general", REG_INPUT,
     "hmm, I'm gonna be late. there's a cute dog outside. I can't just walk past him"),
    ("semi-formal", "prose", "general", REG_INPUT,
     "I'm going to be late. There's a cute dog outside. I can't just walk past him."),
    ("formal", "prose", "general", REG_INPUT,
     "I am going to be late. There is a cute dog outside. I cannot just walk past him."),
    ("semi-formal", "prose", "general",
     "so um i need to like send the the report by uh friday no wait make that thursday",
     "I need to send the report by Thursday."),
    ("semi-formal", "prose", "general",
     "the meeting is at three thirty pm on march fifth in room two oh four", None),
    ("semi-formal", "lists", "general",
     "we need three things for the trip um sunscreen a good map and uh bug spray", None),
    ("semi-formal", "prose", "email",
     "hi sarah um just wanted to follow up on the invoice from last week let me know "
     "when you get a chance thanks alex", None),
    ("formal", "prose", "general",
     "shes gonna call the client at nine am and itll take about forty five minutes", None),
    ("casual", "prose", "general", "um uh hmm", ""),  # pure filler -> empty string
]


def render(tok, transcript, styling, structure, context):
  control = f"[Styling: {styling}] [Structure: {structure}] [Context: {context}]"
  messages = [{"role": "system", "content": SYSTEM},
              {"role": "user", "content": f"{control}\n{transcript}"}]
  return tok.apply_chat_template(messages, tokenize=False,
                                 add_generation_prompt=True, enable_thinking=False)


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

    def hf_gen(prompt_str, prepend_bos=False):
      ids = tok(prompt_str, return_tensors="pt", add_special_tokens=False).input_ids
      if prepend_bos:
        ids = torch.cat([torch.tensor([[151643]]), ids], dim=1)
      with torch.no_grad():
        out = model.generate(ids, max_new_tokens=256, do_sample=False)
      return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

    # BOS A/B on the first case
    p0 = render(tok, CASES[0][3], CASES[0][0], CASES[0][1], CASES[0][2])
    a, b = hf_gen(p0, False), hf_gen(p0, True)
    bos_ab = {"no_bos": a, "with_bos": b, "same": a == b}
    print("BOS A/B same:", bos_ab["same"], flush=True)

    for st, sr, cx, tr, _exp in CASES:
      t = hf_gen(render(tok, tr, st, sr, cx))
      hf_texts.append(t)
      print(f"HF [{st}/{sr}/{cx}] {t[:90]}", flush=True)

  rows, match_hf, match_card, n_card = [], 0, 0, 0
  for i, (st, sr, cx, tr, exp) in enumerate(CASES):
    prompt = render(tok, tr, st, sr, cx)
    pr = subprocess.run(
        [args.litert_lm, "run", args.bundle, "--no-template", "--prompt", prompt,
         "--backend", args.backend, "--cache", "no",
         "--temperature", "0", "--seed", "0"],
        capture_output=True, text=True, timeout=900, stdin=subprocess.DEVNULL)
    got = pr.stdout.strip()
    row = {"styling": st, "structure": sr, "context": cx, "in": tr, "bundle": got}
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
    print(f"[{st}/{sr}/{cx}] rc={pr.returncode} bundle: {got[:90]}", flush=True)

  res = {"bundle": args.bundle, "backend": args.backend, "bos_ab": bos_ab,
         "match_hf": match_hf, "n": len(CASES),
         "match_card": match_card, "n_card": n_card, "cases": rows}
  out = args.out or str(HERE / f"gate_{args.backend}.json")
  Path(out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
  print("GATE_DONE", json.dumps({"match_hf": match_hf, "n": len(CASES),
                                 "match_card": match_card, "n_card": n_card}))


if __name__ == "__main__":
  main()
