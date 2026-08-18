#!/usr/bin/env python3
"""Catalog-wide audit for the vocab-table duplication defect, from HF metadata alone.

The defect (found twice on 2026-08-17, both times silently): a recipe that asks for int4 on
every FULLY_CONNECTED and int8 on EMBEDDING_LOOKUP describes ONE tensor two ways when the
model ties its embedding and lm_head. ai-edge-quantizer resolves that by copying the vocab
table -- once per prefill signature. Nothing errors. The file is 2-3x the size it should be,
and since decode is memory-bandwidth-bound it is also 2-3x slower to decode than the same
weights need to be.

  granite-4.1-3b     2.79 GB -> 1.01 GB   (12 copies of a 154 MB table)
  Qwen2.5-Coder-1.5B 2.53 GB -> 1.12 GB   (7 copies of a 233 MB table, 65% of the file)

`scripts/check_bundle_sanity.py` detects it on a local file. This finds SUSPECTS across a
whole HF org without downloading anything: file size / parameter count is a strong enough
signal, because the recipe fixes the expected ratio.

    python scripts/audit_catalog_bytes_per_param.py [--author litert-community] [--json out]

Expected bytes/param (weights + embedding table, one copy):
    int4 blockwise  ~0.5-0.75      int8 / wi8  ~1.05-1.35      float16  ~2.0      float32  ~4.0
A file whose name says int4 but that measures >1.0, or says int8 and measures >1.6, is a
suspect: re-export with EXTERNALIZE_EMBEDDER=1 and compare, or run check_bundle_sanity.py.

This reports SUSPECTS, not verdicts. Param counts come from the base model named in the card,
so a wrong/absent base_model makes a row unscorable rather than wrong -- those are listed
separately instead of being silently dropped.
"""
import argparse
import json
import re
import sys

from huggingface_hub import HfApi

# name fragment -> (low, high) plausible bytes/param for a single-copy bundle
RECIPES = [
    (re.compile(r"(int4|q4|4bit|_b4|block32|block128|ternary)", re.I), 0.40, 1.00, "int4"),
    (re.compile(r"(int8|wi8|q8|8bit|_b8)", re.I), 0.90, 1.60, "int8"),
    (re.compile(r"(f32|float32|fp32)", re.I), 3.0, 4.5, "f32"),
    (re.compile(r"(f16|float16|fp16|bf16)", re.I), 1.7, 2.3, "f16"),
]


# An NPU build embeds an AOT-compiled binary for one SoC, so it is legitimately larger than
# the CPU/GPU build of the same weights. Measured 2026-08-17 within single repos, where the
# CPU/GPU build is the floor: qualcomm +52%, mediatek +83% (gemma-3-270m-it); mediatek +49%
# over sm8550 (Gemma3-1B-IT). The excess tracks the VENDOR, which is what a compiled payload
# looks like -- table duplication would instead track the SIGNATURE COUNT. Scoring these
# against a weights-only ratio produces pure false positives, so they are reported apart.
# NB: the SoC codes appear as `_mt6989.` / `_sm8550.`, and `_` is a word character, so a
# leading \b never matches there -- anchor on the separator instead.
NPU_VARIANT = re.compile(
    r"(qualcomm|mediatek|intel|Google_Tensor|_LNL|_PTL"
    r"|(?:^|[_.])(?:sm|mt|qcs)\d{3,5}(?=[_.]|$))", re.I)


def classify(fname):
    for rx, lo, hi, label in RECIPES:
        if rx.search(fname):
            return lo, hi, label
    return None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--author", default="litert-community")
    ap.add_argument("--json")
    ap.add_argument("--limit", type=int, default=1000)
    args = ap.parse_args()

    api = HfApi()
    repos = list(api.list_models(author=args.author, limit=args.limit))
    print(f"{args.author}: {len(repos)} repos; fetching file lists…", flush=True)

    suspects, clean, unscorable, npu = [], [], [], []
    param_cache = {}

    for i, r in enumerate(repos):
        try:
            info = api.model_info(r.id, files_metadata=True)
        except Exception as e:
            unscorable.append((r.id, f"model_info failed: {type(e).__name__}"))
            continue
        bundles = [s for s in info.siblings if s.rfilename.endswith(".litertlm")]
        if not bundles:
            continue

        base = None
        cd = info.card_data.to_dict() if info.card_data else {}
        b = cd.get("base_model")
        if isinstance(b, list):
            b = b[0] if b else None
        base = b

        if not base:
            unscorable.append((r.id, "no base_model in card"))
            continue
        if base not in param_cache:
            try:
                bi = api.model_info(base, files_metadata=False)
                st = getattr(bi, "safetensors", None)
                param_cache[base] = st.total if st and st.total else None
            except Exception:
                param_cache[base] = None
        params = param_cache[base]
        if not params:
            unscorable.append((r.id, f"no param count for base {base}"))
            continue

        for s in bundles:
            size = s.size
            if not size:
                continue
            lo, hi, label = classify(s.rfilename)
            bpp = size / params
            row = dict(repo=r.id, file=s.rfilename, bytes=size, params=params,
                       bytes_per_param=round(bpp, 3), recipe=label or "?")
            if NPU_VARIANT.search(s.rfilename):
                row["npu_variant"] = True
                npu.append(row)
            elif label is None:
                unscorable.append((f"{r.id}/{s.rfilename}",
                                   f"recipe not inferable from name (bytes/param {bpp:.2f})"))
            elif bpp > hi:
                row["expected"] = f"{lo}-{hi}"
                row["excess_x"] = round(bpp / hi, 2)
                suspects.append(row)
            else:
                clean.append(row)

    suspects.sort(key=lambda r: -r["excess_x"])
    print(f"\nscored {len(clean) + len(suspects)} bundles: "
          f"{len(clean)} within expectation, {len(suspects)} SUSPECT, "
          f"{len(npu)} NPU variants (not scored), {len(unscorable)} unscorable\n")

    if suspects:
        print(f"{'bytes/param':>11}  {'x over':>6}  {'size':>9}  repo/file")
        for r in suspects:
            print(f"{r['bytes_per_param']:>11.2f}  {r['excess_x']:>6.2f}  "
                  f"{r['bytes']/1e9:>7.2f}GB  {r['repo']}/{r['file']}  [{r['recipe']}]")
    else:
        print("no suspects")

    if unscorable:
        print(f"\nunscorable ({len(unscorable)}) — listed so they are not mistaken for clean:")
        for rid, why in unscorable[:40]:
            print(f"  {rid}: {why}")
        if len(unscorable) > 40:
            print(f"  … and {len(unscorable) - 40} more")

    if args.json:
        json.dump(dict(suspects=suspects, clean=clean, npu_variants=npu,
                       unscorable=[{"id": a, "why": b} for a, b in unscorable]),
                  open(args.json, "w"), indent=1)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
