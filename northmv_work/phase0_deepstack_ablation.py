#!/usr/bin/env python3
"""Phase 0 precheck for the North-Micro-Vision -> fast_vlm ride: can the decoder
live WITHOUT per-layer deepstack injection?

The fast_vlm contract carries ONE image embedding (VISION_ENCODER -> VISION_ADAPTER
-> soft-token positions). CohereCompass (= Qwen3-VL structure) additionally adds
three deepstack embeddings [256, 2048] to the residual stream AFTER decoder layers
0/1/2 at the image positions (`CohereCompassTextModel._deepstack_process`). This
runs the HF fp32 model on a 9-case fp32 reference suite (3 COCO images x 3 prompts,
512 tile, 48-token greedy; an .npz with case{i}_ids / case{i}_patches / case{i}_gen /
_meta_texts / _meta_cases, produced by running the HF model's own processor + greedy
generate on the three COCO val2017 images 000000039769/397133/037777 resized to 512x512
BICUBIC — path via NORTHMV_SUITE) under several ablations and compares against the fp32
reference ids stored in the suite:

  full   : untouched model (harness sanity — must reproduce the suite 9/9)
  drop   : deepstack removed (identity)
  fold   : the three deepstack embeddings summed into the layer-0 input image
           embedding (what a single-embedding contract CAN carry)
  drop1d : drop + 1-D sequential positions instead of M-RoPE (the actual
           fast_vlm runtime feed)
  fold1d : fold + 1-D positions

Verdict rule from the handoff: >=7/9 cases token-equivalent -> GO; else PARK.

    .venv-vl0930-t515/bin/python northmv_work/phase0_deepstack_ablation.py [--modes full,drop,...]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

HF_ID = "CohereLabs/North-Micro-Vision-Instruct"
SUITE = Path(os.environ.get("NORTHMV_SUITE", "northmv_work/north_micro_vision_instruct_suite_512.npz"))
IMAGE_TOKEN_ID = 255031
OUT = Path(__file__).parent / "phase0_ablation_results.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="full,drop,fold,drop1d,fold1d")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--cases", default=None, help="comma list of case indices")
    args = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor

    torch.manual_seed(0)
    processor = AutoProcessor.from_pretrained(HF_ID)
    print(f"loading {HF_ID} fp32 ...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(HF_ID, dtype=torch.float32).eval()

    suite = np.load(SUITE)
    n_cases = int(suite["_meta_cases"])
    cases = [int(c) for c in args.cases.split(",")] if args.cases else list(range(n_cases))
    print(f"suite {SUITE.name}: {n_cases} cases, transformers {suite['_meta_transformers']}, "
          f"tile {suite['_meta_tile']}")

    core = model.model  # CohereCompassModel
    lm = core.language_model  # CohereCompassTextModel
    visual = core.visual
    orig_ds = lm._deepstack_process
    orig_vis_fwd = visual.forward
    orig_rope_index = core.get_rope_index

    def fold_vis_fwd(*a, **k):
        out = orig_vis_fwd(*a, **k)
        ds = list(out.deepstack_features)
        out.pooler_output = out.pooler_output + sum(ds)
        out.deepstack_features = []
        return out

    def seq_rope_index(input_ids, mm_token_type_ids=None, image_grid_thw=None,
                       video_grid_thw=None, attention_mask=None, **kw):
        bsz, seqlen = input_ids.shape
        pos = torch.arange(seqlen, dtype=input_ids.dtype, device=input_ids.device)
        pos = pos.view(1, 1, seqlen).expand(3, bsz, seqlen).contiguous()
        deltas = torch.zeros(bsz, 1, dtype=input_ids.dtype, device=input_ids.device)
        return pos, deltas

    def set_mode(mode: str) -> None:
        lm._deepstack_process = orig_ds
        visual.forward = orig_vis_fwd
        core.get_rope_index = orig_rope_index
        if mode.startswith("drop"):
            lm._deepstack_process = lambda h, m, e: h
        elif mode.startswith("fold"):
            visual.forward = fold_vis_fwd
        elif mode != "full":
            raise ValueError(mode)
        if mode.endswith("1d"):
            core.get_rope_index = seq_rope_index

    results: dict[str, dict] = {}
    for mode in args.modes.split(","):
        set_mode(mode)
        per_case = []
        n_exact = 0
        t0 = time.time()
        for c in cases:
            ids = torch.from_numpy(suite[f"case{c}_ids"].astype(np.int64))[None]
            patches = torch.from_numpy(suite[f"case{c}_patches"].astype(np.float32))
            ref = suite[f"case{c}_gen"].astype(np.int64)
            grid = torch.tensor([[1, 32, 32]], dtype=torch.long)
            mm_tt = (ids == IMAGE_TOKEN_ID).to(torch.int32)
            core.rope_deltas = None
            with torch.no_grad():
                gen = model.generate(
                    input_ids=ids, pixel_values=patches, image_grid_thw=grid,
                    mm_token_type_ids=mm_tt, max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            new = gen[0, ids.shape[1]:].numpy().astype(np.int64)
            L = min(len(new), len(ref))
            eq = new[:L] == ref[:L]
            first_div = int(np.argmin(eq)) if not eq.all() else L
            exact = bool(eq.all() and len(new) == len(ref))
            n_exact += exact
            text = processor.decode(new, skip_special_tokens=True)
            ref_text = str(suite["_meta_texts"][c])
            per_case.append({
                "case": c, "exact": exact, "first_div": first_div, "n_gen": int(len(new)),
                "n_ref": int(len(ref)), "text": text, "ref_text": ref_text,
            })
            flag = "EXACT" if exact else f"div@{first_div}"
            print(f"[{mode}] case {c}: {flag}\n    got: {text!r}\n    ref: {ref_text!r}", flush=True)
        dt = time.time() - t0
        results[mode] = {"exact": n_exact, "n": len(cases), "cases": per_case, "sec": dt}
        print(f"=== [{mode}] exact {n_exact}/{len(cases)}  ({dt:.0f}s)", flush=True)
        OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False))

    print("\nSUMMARY")
    for mode, r in results.items():
        divs = [c["first_div"] for c in r["cases"]]
        print(f"  {mode:7s} exact {r['exact']}/{r['n']}  first-div per case {divs}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
