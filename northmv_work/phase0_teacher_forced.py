#!/usr/bin/env python3
"""Companion to phase0_deepstack_ablation.py: greedy exact-match is a cliff metric once
the MODEL is changed (paraphrase forks look identical to collapse). This scores each
ablation continuously against the fp32 reference continuation, teacher-forced:

  top1   : fraction of the 48 reference tokens the ablated model would also pick greedily
  nll    : mean -log p(reference token) under the ablated model (full = the floor)
  ref_rank_worst : worst rank of a reference token (1 = argmax)

    .venv-vl0930-t515/bin/python northmv_work/phase0_teacher_forced.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

HF_ID = "CohereLabs/North-Micro-Vision-Instruct"
SUITE = Path(os.environ.get("NORTHMV_SUITE", "northmv_work/north_micro_vision_instruct_suite_512.npz"))
IMAGE_TOKEN_ID = 255031
OUT = Path(__file__).parent / "phase0_teacher_forced.json"
MODES = ["full", "drop", "fold", "drop1d", "fold1d"]


def main() -> None:
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(HF_ID, dtype=torch.float32).eval()
    suite = np.load(SUITE)
    n_cases = int(suite["_meta_cases"])
    core, lm, visual = model.model, model.model.language_model, model.model.visual
    orig_ds, orig_vis_fwd, orig_rope_index = lm._deepstack_process, visual.forward, core.get_rope_index

    def fold_vis_fwd(*a, **k):
        out = orig_vis_fwd(*a, **k)
        out.pooler_output = out.pooler_output + sum(list(out.deepstack_features))
        out.deepstack_features = []
        return out

    def seq_rope_index(input_ids, **kw):
        bsz, seqlen = input_ids.shape
        pos = torch.arange(seqlen, dtype=input_ids.dtype).view(1, 1, seqlen).expand(3, bsz, seqlen)
        return pos.contiguous(), torch.zeros(bsz, 1, dtype=input_ids.dtype)

    def set_mode(mode):
        lm._deepstack_process, visual.forward, core.get_rope_index = orig_ds, orig_vis_fwd, orig_rope_index
        if mode.startswith("drop"):
            lm._deepstack_process = lambda h, m, e: h
        elif mode.startswith("fold"):
            visual.forward = fold_vis_fwd
        if mode.endswith("1d"):
            core.get_rope_index = seq_rope_index

    results = {}
    for mode in MODES:
        set_mode(mode)
        rows = []
        for c in range(n_cases):
            ids = torch.from_numpy(suite[f"case{c}_ids"].astype(np.int64))
            ref = torch.from_numpy(suite[f"case{c}_gen"].astype(np.int64))
            full = torch.cat([ids, ref])[None]
            patches = torch.from_numpy(suite[f"case{c}_patches"].astype(np.float32))
            core.rope_deltas = None
            with torch.no_grad():
                logits = model(input_ids=full, pixel_values=patches,
                               image_grid_thw=torch.tensor([[1, 32, 32]]),
                               mm_token_type_ids=(full == IMAGE_TOKEN_ID).to(torch.int32),
                               use_cache=False).logits[0].float()
            pred = logits[len(ids) - 1:-1]                       # predicts ref[0..47]
            lp = torch.log_softmax(pred, -1)
            nll = -lp[torch.arange(len(ref)), ref]
            top1 = (pred.argmax(-1) == ref).float()
            ranks = (pred > pred[torch.arange(len(ref)), ref][:, None]).sum(-1) + 1
            rows.append({"case": c, "top1": float(top1.mean()), "nll": float(nll.mean()),
                         "worst_rank": int(ranks.max()), "n": int(len(ref))})
            print(f"[{mode}] case {c}: top1 {top1.mean():.3f} nll {nll.mean():.3f} "
                  f"worst_rank {int(ranks.max())}", flush=True)
        results[mode] = {"rows": rows,
                         "top1_mean": float(np.mean([r["top1"] for r in rows])),
                         "nll_mean": float(np.mean([r["nll"] for r in rows])),
                         "worst_rank_max": max(r["worst_rank"] for r in rows)}
        print(f"=== [{mode}] top1 {results[mode]['top1_mean']:.3f} nll {results[mode]['nll_mean']:.3f} "
              f"worst_rank {results[mode]['worst_rank_max']}", flush=True)
        OUT.write_text(json.dumps(results, indent=1))
    print("\nSUMMARY (teacher-forced vs fp32 reference continuation)")
    for m, r in results.items():
        print(f"  {m:7s} top1 {r['top1_mean']:.3f}  nll {r['nll_mean']:.3f}  worst_rank {r['worst_rank_max']}")


if __name__ == "__main__":
    main()
