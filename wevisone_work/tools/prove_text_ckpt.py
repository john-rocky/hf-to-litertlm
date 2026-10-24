#!/usr/bin/env python3
"""Question 5: prove the derived text-tower checkpoint (out/text_ckpt, Qwen3ForCausalLM) against the full VLM.

The 9 text prompts of run_arms.py (the 8 verify_quality questions + France), each rendered with the checkpoint's
chat template and the package's system turn, go through the full Qwen3VLForConditionalGeneration and the derived
Qwen3ForCausalLM, both fp32 with eager attention, one thread. Reported per prompt and overall: max |delta logit|,
argmax agreement at every position. Writes ref/text_ckpt_prove.json.
"""
import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
K = Path(__file__).resolve().parents[1]
SNAP = Path(os.environ.get("WEVISONE_SNAPSHOT", "~/.cache/huggingface/hub/models--tencent--WeVisOne-VL-2B/"
                           "snapshots/3dad9ce8bcb6fbe0aea945bf8860bd2cfc3663bf")).expanduser()
CKPT = K / "out" / "text_ckpt"


def text_rows():
    """The 9 text prompts, as run_arms.py defines them (loaded by path; nothing heavy runs at its import)."""
    spec = importlib.util.spec_from_file_location("run_arms", Path(__file__).resolve().parent / "run_arms.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TEXT_ROWS


def main():
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    import transformers
    tok = AutoTokenizer.from_pretrained(str(SNAP))
    prompts = []
    for rid, q, _ in text_rows():
        msgs = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": q}]
        ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        if isinstance(ids, dict) or hasattr(ids, "keys"):
            ids = ids["input_ids"]
        prompts.append((rid, [int(i) for i in ids]))
    out = {"transformers": transformers.__version__, "torch": torch.__version__, "attn": "eager", "dtype": "float32",
           "prompts": []}
    t0 = time.time()
    full = AutoModelForImageTextToText.from_pretrained(str(SNAP), dtype=torch.float32,
                                                       attn_implementation="eager").eval()
    logits_full = []
    with torch.inference_mode():
        for _, ids in prompts:
            logits_full.append(full(input_ids=torch.tensor([ids])).logits[0].float().numpy())
    del full
    derived = AutoModelForCausalLM.from_pretrained(str(CKPT), dtype=torch.float32, attn_implementation="eager").eval()
    out["derived_class"] = type(derived).__name__
    out["derived_rope"] = getattr(derived.config, "rope_parameters", None)
    worst = 0.0
    agree_all = True
    with torch.inference_mode():
        for (rid, ids), lf in zip(prompts, logits_full):
            ld = derived(input_ids=torch.tensor([ids])).logits[0].float().numpy()
            d = np.abs(lf.astype(np.float64) - ld.astype(np.float64))
            agree = int((lf.argmax(-1) == ld.argmax(-1)).sum())
            rec = {"row": rid, "positions": len(ids), "argmax_agree": agree, "max_abs_delta": float(d.max()),
                   "mean_abs_delta": float(d.mean()), "max_abs_logit": float(np.abs(lf).max())}
            out["prompts"].append(rec)
            worst = max(worst, rec["max_abs_delta"])
            agree_all &= agree == len(ids)
            print(rid, rec, flush=True)
    out["summary"] = {"prompts": len(prompts), "positions": sum(p["positions"] for p in out["prompts"]),
                      "argmax_all_agree": bool(agree_all), "max_abs_delta": worst,
                      "line4": "argmax all positions equal; max |delta| recorded, > 1e-3 -> stop and report",
                      "wall_s": round(time.time() - t0, 1)}
    (K / "ref" / "text_ckpt_prove.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["summary"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
