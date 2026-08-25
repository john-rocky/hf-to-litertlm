"""Run the 9-case suite (3 fixtures x 3 prompts) through the Qwen3.5 VLM bundle
with `litert-lm run --attachment` and score token-prefix agreement against the
HF fp32 oracles (full M-RoPE and, when present, the POS1D contract ablation).

    suite_gate_vl.py <bundle.litertlm> [--backend cpu|gpu] [--vision-backend cpu|gpu]
                     [--out gate.json] [--litert-lm PATH] [--n-tok 48]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
FIX = ["cats_512.png", "kitchen1_512.png", "kitchen2_512.png"]
PROMPTS = ["What is in this image?",
           "Describe the main colors in this image.",
           "Where is this scene?"]


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("bundle")
  ap.add_argument("--backend", default="cpu")
  ap.add_argument("--vision-backend", default=None)
  ap.add_argument("--out", default=None)
  ap.add_argument("--litert-lm",
                  default=str(ROOT / ".venv-lt016/bin/litert-lm"))
  ap.add_argument("--n-tok", type=int, default=48)
  args = ap.parse_args()

  from transformers import AutoTokenizer
  tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B")

  oracles = {}
  for name, f in (("full", "hf_oracle_full.json"), ("pos1d", "hf_oracle_pos1d.json")):
    p = HERE / f
    if p.exists():
      d = json.loads(p.read_text())
      oracles[name] = {(c["image"], c["prompt"]): c["text"] for c in d["cases"]}

  def ids(t):
    return tok(t, add_special_tokens=False).input_ids[: args.n_tok]

  def score(got, want):
    g, w = ids(got), ids(want)
    L = min(len(g), len(w))
    eq = [a == b for a, b in zip(g[:L], w[:L])]
    first_div = eq.index(False) if False in eq else L
    exact = (first_div == L) and (len(g) >= len(w) or got.strip() == want.strip())
    return {"exact": bool(exact), "first_div": first_div, "n_ref": len(w)}

  rows = []
  exact = {k: 0 for k in oracles}
  for img in FIX:
    for prompt in PROMPTS:
      cmd = [args.litert_lm, "run", args.bundle, "--prompt", prompt,
             "--attachment", str(ROOT / "northmv_work/fixtures" / img),
             "--backend", args.backend, "--cache", "no",
             "--temperature", "0", "--seed", "0"]
      if args.vision_backend:
        cmd += ["--vision-backend", args.vision_backend]
      t0 = time.time()
      pr = subprocess.run(cmd, capture_output=True, text=True, timeout=1800,
                          stdin=subprocess.DEVNULL)
      text = pr.stdout.strip()
      row = {"image": img, "prompt": prompt, "text": text, "rc": pr.returncode,
             "sec": round(time.time() - t0, 1)}
      for k, m in oracles.items():
        s = score(text, m[(img, prompt)])
        row[k] = s
        exact[k] += s["exact"]
      rows.append(row)
      print(f"[{img} | {prompt}] rc={pr.returncode} "
            + " ".join(f"{k}:fd={row[k]['first_div']}" for k in oracles))
      print(text[:200].replace("\n", " ") + ("…" if len(text) > 200 else ""), flush=True)

  res = {"bundle": args.bundle, "backend": args.backend,
         "vision_backend": args.vision_backend,
         "exact": exact, "n": len(rows), "cases": rows}
  out = args.out or str(HERE / f"suite_{args.backend}.json")
  Path(out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
  print("SUITE_DONE", json.dumps(exact))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
