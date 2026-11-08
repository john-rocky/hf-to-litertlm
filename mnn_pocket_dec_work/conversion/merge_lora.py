"""Fold the LoRA adapter of MNN-Pocket-0.3B-Dec into its backbone in float32 (the weights every export reads).

    python conversion/merge_lora.py --src <MNN-Pocket-0.3B-Dec download> --out merged_fp32.safetensors

Run it in the reference environment (requirements-reference.txt, the original repository's versions). The base is
base/model.safetensors (bf16, widened to fp32 exactly). For every (lora_A, lora_B) pair of
adapter/adapter_model.safetensors: W' = W + (B @ A) * (lora_alpha / r) in fp32, scale 2, never rounded; adapter key
`base_model.model.<x>.lora_{A,B}.weight` -> base key `<x>.weight`. The output holds the backbone only (embed_tokens,
layers, norm; no lm_head), keys sorted, no metadata. Four torch threads, as in the shipped run.
Expected: 1,211,358,392 bytes, SHA-256 a2f58515fc25ceb1974cf95452f8cd85c72efbbbb550354db1a4118fc2ed38d3."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

EXPECTED_SHA256 = "a2f58515fc25ceb1974cf95452f8cd85c72efbbbb550354db1a4118fc2ed38d3"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="the downloaded taobao-mnn/MNN-Pocket-0.3B-Dec directory")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    torch.set_num_threads(4)
    cfg = json.loads((a.src / "adapter/adapter_config.json").read_text())
    assert cfg["r"] == 16 and cfg["lora_alpha"] == 32 and not cfg["use_rslora"] and not cfg["use_dora"]
    assert not cfg.get("modules_to_save") and cfg["bias"] == "none"
    scale = cfg["lora_alpha"] / cfg["r"]
    with safe_open(a.src / "base/model.safetensors", "pt") as f:
        base = {k: f.get_tensor(k).float() for k in f.keys()}
    with safe_open(a.src / "adapter/adapter_model.safetensors", "pt") as f:
        adapter = {k: f.get_tensor(k) for k in f.keys()}
    stems = sorted({k[: -len(".lora_A.weight")] for k in adapter if k.endswith(".lora_A.weight")})
    assert len(adapter) == 2 * len(stems) == 392
    own = dict(base)
    for stem in stems:
        A, B = adapter[stem + ".lora_A.weight"], adapter[stem + ".lora_B.weight"]
        assert A.dtype == B.dtype == torch.float32 and A.shape[0] == 16 and B.shape[1] == 16
        name = stem.removeprefix("base_model.model.") + ".weight"
        assert name in own, name
        own[name] = own[name] + (B @ A) * scale
    keep = {k: own[k].contiguous() for k in sorted(own) if k.startswith("model.")}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(keep, str(a.out))
    sha = sha256_file(a.out)
    print(f"{a.out}: {a.out.stat().st_size:,} bytes, {len(keep)} tensors, {len(stems)} folded pairs, SHA-256 {sha}")
    print("same as the shipped run" if sha == EXPECTED_SHA256 else f"DIFFERS from the shipped run ({EXPECTED_SHA256})")


if __name__ == "__main__":
    main()
