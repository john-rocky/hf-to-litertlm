#!/usr/bin/env python3
"""Derive a text-only Qwen3_5ForCausalLM checkpoint from MiniCPM-V-4.7-1B.

  python make_text_ckpt.py SRC TXT [--report report.json]

SRC = the Hub snapshot of openbmb/MiniCPM-V-4.7-1B (MiniCPMV4_7ForConditionalGeneration).
TXT = the new directory (must not exist).

Weights: `model.language_model.X` -> `model.X`, `lm_head.weight` kept as is (untied),
`model.vision_tower.*` and `model.merger.*` dropped; dtype stays as stored (bf16).
Every source key must fall in exactly one of those groups, or the script refuses.
config.json = SRC's text_config promoted to the top level, with
`architectures ["Qwen3_5ForCausalLM"]`, `model_type "qwen3_5_text"`,
`tie_word_embeddings false`, `dtype "bfloat16"` and SRC's `transformers_version`.
tokenizer.json / tokenizer_config.json / chat_template.jinja / generation_config.json
are copied byte for byte.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

from safetensors import safe_open
from safetensors.torch import save_file

COPY = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
        "generation_config.json"]
LM = "model.language_model."
DROP = ("model.vision_tower.", "model.merger.")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("txt")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    if os.path.exists(args.txt):
        sys.exit(f"refusing: {args.txt} exists")

    src_cfg = json.load(open(os.path.join(args.src, "config.json")))
    cfg = dict(src_cfg["text_config"])
    cfg["architectures"] = ["Qwen3_5ForCausalLM"]
    cfg["model_type"] = "qwen3_5_text"
    cfg["tie_word_embeddings"] = False
    cfg["dtype"] = "bfloat16"
    cfg["transformers_version"] = src_cfg["transformers_version"]

    st_path = os.path.join(args.src, "model.safetensors")
    kept, dropped, renamed = {}, {}, []
    with safe_open(st_path, framework="pt") as f:
        src_meta = f.metadata()
        keys = list(f.keys())
        for k in keys:
            if k.startswith(LM):
                new = "model." + k[len(LM):]
                renamed.append((k, new))
                kept[new] = f.get_tensor(k)
            elif k == "lm_head.weight":
                kept[k] = f.get_tensor(k)
            elif k.startswith(DROP):
                group = next(p for p in DROP if k.startswith(p))
                dropped[group] = dropped.get(group, 0) + 1
            else:
                sys.exit(f"refusing: unmapped source key {k!r}")
    if len(kept) != len(renamed) + 1:
        sys.exit("refusing: renamed keys collide")

    os.makedirs(args.txt)
    out_st = os.path.join(args.txt, "model.safetensors")
    save_file(kept, out_st, metadata={"format": "pt"})
    with open(os.path.join(args.txt, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
        fh.write("\n")
    for name in COPY:
        shutil.copyfile(os.path.join(args.src, name), os.path.join(args.txt, name))

    dtypes = {}
    for t in kept.values():
        dtypes[str(t.dtype)] = dtypes.get(str(t.dtype), 0) + 1
    report = {
        "src": os.path.realpath(args.src),
        "txt": os.path.realpath(args.txt),
        "src_safetensors_metadata": src_meta,
        "src_tensors": len(keys),
        "kept_tensors": len(kept),
        "renamed_language_model": len(renamed),
        "lm_head_kept": "lm_head.weight" in kept,
        "dropped": dropped,
        "kept_dtypes": dtypes,
        "kept_params": int(sum(t.numel() for t in kept.values())),
        "shapes": {k: list(kept[k].shape) for k in
                   ("model.embed_tokens.weight", "lm_head.weight", "model.norm.weight")},
        "out_files": {n: {"bytes": os.path.getsize(os.path.join(args.txt, n)),
                          "sha256": sha256(os.path.join(args.txt, n))}
                      for n in ["model.safetensors", "config.json"] + COPY},
        "copied_identical": {n: sha256(os.path.join(args.src, n)) ==
                             sha256(os.path.join(args.txt, n)) for n in COPY},
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.report:
        with open(args.report, "w") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
