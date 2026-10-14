#!/usr/bin/env python3
"""Export a VLM's text tower as a plain CausalLM checkpoint that the LLM rail loads.

    python tools/vlm_text_ckpt/make_text_ckpt.py SRC OUT \
        --prefix model.language_model. --model-type qwen3_5_text --architectures Qwen3_5ForCausalLM \
        [--drop model.vision_tower. model.merger.] [--head lm_head.weight] [--text-config text_config] \
        [--copy tokenizer.json tokenizer_config.json chat_template.jinja generation_config.json] \
        [--report OUT.json] [--prove] [--prove-ids ids.json | --prove-text "..."] [--prove-max-tokens 48]

SRC = a local snapshot of the VLM (config.json + safetensors, one file or a sharded index).
OUT = the new directory (must not exist).

Weights: every tensor under --prefix is kept with the prefix replaced by `model.`; --head is kept
as is (an untied head; when SRC has no such tensor the config keeps tie_word_embeddings = true);
tensors under a --drop prefix are dropped; any other key makes the script refuse, so a new
family's layout is checked, not guessed. Dtypes stay as stored.
config.json = SRC's --text-config promoted to the top level with --architectures, --model-type,
tie_word_embeddings and SRC's transformers_version. The --copy files are copied byte for byte
when SRC has them.

--prove loads the full VLM and the new checkpoint in fp32 with eager attention and compares their
logits on one text-only input (an ids file, or --prove-text through SRC's tokenizer). The remap is
a rename, so equal kernels must give max |delta| 0; differences at the 1e-5 scale come from
different attention kernels (sdpa vs eager), not from the remap.
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import sys

from safetensors import safe_open
from safetensors.torch import save_file


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def shard_files(src):
    index = os.path.join(src, "model.safetensors.index.json")
    if os.path.exists(index):
        names = sorted(set(json.load(open(index))["weight_map"].values()))
        return [os.path.join(src, n) for n in names]
    single = os.path.join(src, "model.safetensors")
    if os.path.exists(single):
        return [single]
    files = sorted(glob.glob(os.path.join(src, "*.safetensors")))
    if not files:
        sys.exit(f"refusing: no safetensors under {src}")
    return files


def derive(args):
    src_cfg = json.load(open(os.path.join(args.src, "config.json")))
    if args.text_config not in src_cfg:
        sys.exit(f"refusing: config.json has no {args.text_config!r} (keys: {sorted(src_cfg)})")
    cfg = dict(src_cfg[args.text_config])
    cfg["architectures"] = [args.architectures]
    cfg["model_type"] = args.model_type
    if "transformers_version" in src_cfg:
        cfg["transformers_version"] = src_cfg["transformers_version"]

    kept, dropped, renamed, src_meta, n_src = {}, {}, [], {}, 0
    for path in shard_files(args.src):
        with safe_open(path, framework="pt") as f:
            src_meta = f.metadata() or src_meta
            for k in f.keys():
                n_src += 1
                if k.startswith(args.prefix):
                    new = "model." + k[len(args.prefix):]
                    renamed.append((k, new))
                    kept[new] = f.get_tensor(k)
                elif k == args.head:
                    kept[k] = f.get_tensor(k)
                elif any(k.startswith(p) for p in args.drop):
                    group = next(p for p in args.drop if k.startswith(p))
                    dropped[group] = dropped.get(group, 0) + 1
                else:
                    sys.exit(f"refusing: unmapped source key {k!r} (not under --prefix, --head or --drop)")
    head_kept = args.head in kept
    if len(kept) != len(renamed) + (1 if head_kept else 0):
        sys.exit("refusing: renamed keys collide")
    cfg["tie_word_embeddings"] = not head_kept
    dtypes = {}
    for t in kept.values():
        dtypes[str(t.dtype)] = dtypes.get(str(t.dtype), 0) + 1
    if len(dtypes) == 1:
        cfg["dtype"] = next(iter(dtypes)).replace("torch.", "")

    os.makedirs(args.out)
    out_st = os.path.join(args.out, "model.safetensors")
    save_file(kept, out_st, metadata={"format": "pt"})
    with open(os.path.join(args.out, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
        fh.write("\n")
    copied = []
    for name in args.copy:
        if os.path.exists(os.path.join(args.src, name)):
            shutil.copyfile(os.path.join(args.src, name), os.path.join(args.out, name))
            copied.append(name)

    report = {
        "src": os.path.realpath(args.src),
        "out": os.path.realpath(args.out),
        "prefix": args.prefix, "head": args.head, "drop": args.drop,
        "src_safetensors_metadata": src_meta,
        "src_tensors": n_src,
        "kept_tensors": len(kept),
        "renamed_under_prefix": len(renamed),
        "head_kept": head_kept,
        "dropped": dropped,
        "kept_dtypes": dtypes,
        "kept_params": int(sum(t.numel() for t in kept.values())),
        "config": {k: cfg[k] for k in ("architectures", "model_type", "tie_word_embeddings") if k in cfg},
        "out_files": {n: {"bytes": os.path.getsize(os.path.join(args.out, n)),
                          "sha256": sha256(os.path.join(args.out, n))}
                      for n in ["model.safetensors", "config.json"] + copied},
        "copied_identical": {n: sha256(os.path.join(args.src, n)) == sha256(os.path.join(args.out, n))
                             for n in copied},
    }
    return report


def compare(a, b):
    import numpy as np
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {
        "max_abs_delta": float(d.max()),
        "mean_abs_delta": float(d.mean()),
        "argmax_agree": int((a.argmax(-1) == b.argmax(-1)).sum()),
        "positions": int(a.shape[0]),
        "max_abs_logit": float(np.abs(a).max()),
    }


def prove(args):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.prove_ids:
        ids = json.load(open(args.prove_ids))
    else:
        tok = AutoTokenizer.from_pretrained(args.src, trust_remote_code=args.trust_remote_code)
        ids = tok(args.prove_text).input_ids
    ids = [int(i) for i in ids][: args.prove_max_tokens]
    x = torch.tensor([ids], dtype=torch.long)

    if args.full_class:
        cls = getattr(transformers, args.full_class)
    else:
        from transformers import AutoModelForImageTextToText as cls
    full = cls.from_pretrained(args.src, dtype=torch.float32, attn_implementation="eager",
                               trust_remote_code=args.trust_remote_code).eval()
    with torch.no_grad():
        pt_full = full(input_ids=x).logits[0].float().numpy()
    full_info = {"class": type(full).__name__,
                 "attn_implementation": getattr(full.config, "_attn_implementation", None),
                 "param_dtypes": sorted({str(p.dtype) for p in full.parameters()})}
    del full

    derived = AutoModelForCausalLM.from_pretrained(args.out, dtype=torch.float32,
                                                   attn_implementation="eager").eval()
    with torch.no_grad():
        pt_derived = derived(input_ids=x).logits[0].float().numpy()
    derived_info = {"class": type(derived).__name__,
                    "attn_implementation": getattr(derived.config, "_attn_implementation", None)}
    del derived

    cmp = compare(pt_full, pt_derived)
    ok = cmp["max_abs_delta"] <= args.prove_bar and cmp["argmax_agree"] == cmp["positions"]
    return {
        "transformers": transformers.__version__, "torch": torch.__version__,
        "ids_len": len(ids), "ids": ids,
        "full_model": full_info, "derived_model": derived_info,
        "full_vs_derived_eager": cmp,
        "bar": f"max_abs_delta <= {args.prove_bar} and argmax_agree == positions",
        "pass": bool(ok),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--prefix", required=True, help="tensor prefix of the text tower, e.g. model.language_model.")
    ap.add_argument("--model-type", required=True, help="model_type of the derived config, e.g. qwen3_5_text")
    ap.add_argument("--architectures", required=True, help="architecture class name, e.g. Qwen3_5ForCausalLM")
    ap.add_argument("--drop", nargs="*", default=["model.vision_tower.", "model.merger."],
                    help="tensor prefixes to drop (vision tower, projector)")
    ap.add_argument("--head", default="lm_head.weight", help="the untied head tensor to keep as is")
    ap.add_argument("--text-config", default="text_config", help="config.json key that holds the text config")
    ap.add_argument("--copy", nargs="*", default=["tokenizer.json", "tokenizer_config.json",
                                                   "chat_template.jinja", "generation_config.json"])
    ap.add_argument("--report", default=None, help="write the JSON report here (also printed)")
    ap.add_argument("--prove", action="store_true", help="compare full-model and derived logits in fp32 eager")
    ap.add_argument("--prove-ids", default=None, help="JSON list of token ids for --prove")
    ap.add_argument("--prove-text", default="The capital of France is Paris. Water boils at 100 degrees Celsius at sea level.")
    ap.add_argument("--prove-max-tokens", type=int, default=48)
    ap.add_argument("--prove-bar", type=float, default=1e-3)
    ap.add_argument("--full-class", default=None,
                    help="transformers class of the full VLM (default: AutoModelForImageTextToText)")
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()
    if os.path.exists(args.out):
        sys.exit(f"refusing: {args.out} exists")

    report = derive(args)
    text = json.dumps(report, indent=2)
    print(text)
    if args.report:
        with open(args.report, "w") as fh:
            fh.write(text + "\n")
    if not args.prove:
        return 0

    report["prove"] = prove(args)
    text = json.dumps(report, indent=2)
    print(json.dumps(report["prove"], indent=2))
    if args.report:
        with open(args.report, "w") as fh:
            fh.write(text + "\n")
    return 0 if report["prove"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
