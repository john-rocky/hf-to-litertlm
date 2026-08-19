#!/usr/bin/env python3
"""Run a 9-case fp32 reference suite (3 COCO images x 3 prompts, 512 tile) through a
North-Micro-Vision .litertlm bundle with `litert-lm run` and score it against
  (a) the fp32 HF reference (deepstack + M-RoPE)      -- suite _meta_texts
  (b) the fp32 HF fold1d ablation (what the bundle CAN reproduce: single embedding
      = merger + deepstack sum, 1-D positions)         -- phase0_ablation_results.json

Images are the suite's own PIL-BICUBIC 512x512 resizes (northmv_work/fixtures/*_512.png),
so the runtime's resize is the identity and only normalization/patching is exercised.
Texts are compared token-wise with the HF tokenizer over the first 48 tokens (the
suite's greedy budget; the CLI has no max-new-tokens).

    suite_gate_bundle.py <bundle.litertlm> [--backend cpu|gpu] [--vision-backend cpu|gpu]
                         [--out gate.json] [--litert-lm PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SUITE = Path(os.environ.get("NORTHMV_SUITE", "northmv_work/north_micro_vision_instruct_suite_512.npz"))
ABL = HERE / "phase0_ablation_results.json"
FIX = {0: "cats_512.png", 1: "kitchen1_512.png", 2: "kitchen2_512.png"}
PROMPTS = ["What is in this image?", "Describe the main colors in this image.", "Where is this scene?"]
LLM_DIR = HERE.parent / "src_models/north-micro-vision-llm"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--vision-backend", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--litert-lm", default=os.path.expanduser("~/venvs/lt0160run/bin/litert-lm"))
    ap.add_argument("--n-tok", type=int, default=48)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(LLM_DIR))
    suite = np.load(SUITE)
    ref_texts = [str(t) for t in suite["_meta_texts"]]
    abl = json.loads(ABL.read_text())["fold1d"]["cases"]
    fold1d_texts = [c["text"] for c in abl]

    def ids(t):
        return tok(t, add_special_tokens=False).input_ids[: args.n_tok]

    def score(got, want):
        g, w = ids(got), ids(want)
        L = min(len(g), len(w))
        eq = [a == b for a, b in zip(g[:L], w[:L])]
        first_div = eq.index(False) if False in eq else L
        exact = (first_div == L) and (len(g) >= len(w) or got.strip() == want.strip())
        return {"exact": bool(exact), "first_div": first_div, "n_ref": len(w)}

    rows, ex_ref, ex_f1d, tsec = [], 0, 0, []
    case = 0
    for img_i in range(3):
        for p in PROMPTS:
            cmd = [args.litert_lm, "run", args.bundle, "--prompt", p,
                   "--attachment", str(HERE / "fixtures" / FIX[img_i]),
                   "--backend", args.backend, "--cache", "no", "--temperature", "0", "--seed", "0"]
            if args.vision_backend:
                cmd += ["--vision-backend", args.vision_backend]
            t0 = time.time()
            pr = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, stdin=subprocess.DEVNULL)
            dt = time.time() - t0
            text = pr.stdout.strip()
            s_ref = score(text, ref_texts[case])
            s_f1d = score(text, fold1d_texts[case])
            ex_ref += s_ref["exact"]
            ex_f1d += s_f1d["exact"]
            tsec.append(dt)
            rows.append({"case": case, "img": FIX[img_i], "prompt": p, "text": text, "rc": pr.returncode,
                         "vs_ref": s_ref, "vs_fold1d": s_f1d, "sec": round(dt, 1),
                         "err": pr.stderr.strip()[-300:] if pr.returncode else ""})
            print(f"case {case}: vs_ref {'EXACT' if s_ref['exact'] else 'div@%d' % s_ref['first_div']}"
                  f" | vs_fold1d {'EXACT' if s_f1d['exact'] else 'div@%d' % s_f1d['first_div']} ({dt:.0f}s)"
                  f"\n    got: {text[:200]!r}\n    f1d: {fold1d_texts[case][:200]!r}", flush=True)
            if pr.returncode:
                print("    RC", pr.returncode, pr.stderr.strip()[-300:])
            case += 1
    print(f"SUMMARY exact vs fp32-ref {ex_ref}/9 | vs fold1d {ex_f1d}/9 | "
          f"first-div vs fold1d {[r['vs_fold1d']['first_div'] for r in rows]}")
    res = {"bundle": os.path.basename(args.bundle), "backend": args.backend,
           "vision_backend": args.vision_backend, "exact_vs_ref": ex_ref, "exact_vs_fold1d": ex_f1d,
           "rows": rows}
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
