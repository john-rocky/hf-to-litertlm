#!/usr/bin/env python3
"""Tokenizer-parity gate for a built .litertlm bundle: added-token specials,
Latin-1 / Extended-A characters, emoji, digits.

Compares what the runtime encodes from the bundle's own tokenizer section
(`litert_lm` `Engine.tokenize`) against the upstream `tokenizer.json` read by
the `tokenizers` library, on:

  1. every added token, alone and mid-string (``a<tok>b``);
  2. a ChatML turn built from the specials, when the vocab has them;
  3. a fixed probe list (Latin-1 / Ext-A words, emoji, CJK, digits, whitespace).

Any id mismatch fails the gate (exit 1). Rows are printed so a failure names
the token or character that split (all rows up to 80, failures only above
that; ``--verbose`` prints everything).

Why this exists. A 2026-08-25 Qwen2-VL derivative bundle (NuExtract-2.0-2B)
answered wrong the moment a prompt carried ``<|im_start|>`` / ``<|im_end|>``.
That was first attributed to the fast_vlm runtime (LiteRT-LM #3348) and turned
out to be the bundle's own tokenizer section: the public
litert-community/Qwen2-VL-2B bundle encodes every special to its single
correct id, on the templated and the raw path alike. The converter-side
mechanism is the BPE->SentencePiece conversion (litert-torch #1205: no byte
fallback, pad/eos typed as the UNK piece, so a special can split into its
spelling and unknown characters can turn INTO it). This gate is what would
have caught that before anything was reported. Run it on every built bundle;
prefer the HF ``tokenizer.json`` path when it fails on an SP section.

Usage:
    python scripts/gate_specials.py <bundle.litertlm> --hf <dir-with-tokenizer.json | HF repo id>
    python scripts/gate_specials.py out/x.litertlm --hf src_models/x-llm --probe 'extra text'

Notes:
  * The reference is the `tokenizers` reading of ``tokenizer.json`` with
    ``add_special_tokens=False``. For the few vocabs where transformers' slow
    class disagrees with `tokenizers` (Nanbeige ``Prepend ▁``, Cohere digit
    split), pass ``--transformers`` to compare against ``AutoTokenizer``
    instead; matching either reading is parity.
  * ``Engine.tokenize`` does not prepend the bundle's start_token (that happens
    at session prefill), so a leading BOS is not expected on either side.
  * Bundle-only literals (e.g. the fast_vlm ``<image_soft_token>``) are not in
    the upstream vocab and are not probed here.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys

PROBES = [
    ("latin1-words", "Zürich, São Paulo, Łódź, Ærø, Reykjavík"),
    ("latin1-solo", "é ñ ü ° × · Ł ß"),
    ("ext-a", "ăŃőűŽ"),
    ("emoji", "😀🎉👍"),
    ("cjk", "日本語のテキストと漢字"),
    ("digits", "1234567890 3.14 2026-09-04"),
    ("whitespace", "  two leading, two trailing  "),
    ("tab-newline", "Tab\tand\nnewline"),
]


def load_reference(src: str, use_transformers: bool):
    if use_transformers:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
        added = {int(i): a.content for i, a in tok.added_tokens_decoder.items()}
        return (lambda s: tok.encode(s, add_special_tokens=False)), added, "transformers.AutoTokenizer"
    from tokenizers import Tokenizer
    path = os.path.join(src, "tokenizer.json") if os.path.isdir(src) else src
    if not os.path.exists(path):
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(src, "tokenizer.json")
    tok = Tokenizer.from_file(path)
    added = {int(i): a.content for i, a in tok.get_added_tokens_decoder().items()}
    return (lambda s: tok.encode(s, add_special_tokens=False).ids), added, f"tokenizers {path}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bundle")
    ap.add_argument("--hf", required=True, help="dir with tokenizer.json, a tokenizer.json path, or an HF repo id")
    ap.add_argument("--transformers", action="store_true", help="reference = transformers AutoTokenizer instead of tokenizers")
    ap.add_argument("--probe", action="append", default=[], help="extra probe text (repeatable)")
    ap.add_argument("--no-added", action="store_true", help="skip the per-added-token rows")
    ap.add_argument("--verbose", action="store_true", help="print every row (default: all rows when <= 80, else failures only)")
    args = ap.parse_args()

    ref_encode, added, ref_name = load_reference(args.hf, args.transformers)
    from litert_lm.engine import Engine

    rows = []  # (label, text)
    if not args.no_added:
        for tid, content in sorted(added.items()):
            rows.append((f"added[{tid}]", content))
            rows.append((f"added[{tid}]-mid", f"a{content}b"))
    names = set(added.values())
    if "<|im_start|>" in names and "<|im_end|>" in names:
        rows.append(("chatml-turn", "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"))
    rows += PROBES
    rows += [(f"probe{i}", p) for i, p in enumerate(args.probe)]

    eng = Engine(args.bundle)
    try:
        results = []
        for label, text in rows:
            ref = list(ref_encode(text))
            got = list(eng.tokenize(text))
            results.append((label, text, ref, got))
    finally:
        eng.close()
        gc.collect()

    print(f"bundle   : {args.bundle}")
    print(f"reference: {ref_name}")
    print(f"{'row':22s} {'ok':4s} {'ref':>4s} {'eng':>4s}  detail")
    fails = 0
    show_all = args.verbose or len(results) <= 80
    for label, text, ref, got in results:
        ok = ref == got
        fails += (not ok)
        if ok and not show_all:
            continue
        detail = ""
        if not ok:
            detail = f"ref={ref[:12]}{'…' if len(ref) > 12 else ''} eng={got[:12]}{'…' if len(got) > 12 else ''}"
        print(f"{label:22s} {'ok' if ok else 'FAIL':4s} {len(ref):4d} {len(got):4d}  {text[:40]!r} {detail}")
    n = len(results)
    if fails:
        print(f"FAIL {fails}/{n} rows differ from the reference — the bundle's tokenizer section does not encode like upstream")
        return 1
    print(f"PASS {n}/{n} rows match the reference")
    return 0


if __name__ == "__main__":
    sys.exit(main())
