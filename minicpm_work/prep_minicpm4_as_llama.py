#!/usr/bin/env python3
"""Port MiniCPM4/4.1 (custom `MiniCPMForCausalLM`, muP scalings) to a plain
LlamaForCausalLM checkpoint that litert-torch export_hf converts natively.

The remote modeling code (written for transformers 4.46) does three things a
stock llama does not; all are static weight transforms, so we fold them:
  1. inputs_embeds = embed_tokens(ids) * scale_emb          -> embed *= scale_emb
  2. hidden = residual + out * (scale_depth / sqrt(L))      -> o_proj, down_proj *= that
  3. logits = lm_head(h / (hidden_size / dim_model_base))   -> lm_head /= that
Tied-embedding checkpoints (0.5B) must be untied first (1 and 3 conflict).
longrope with long_factor == short_factor and scale == 1 (mscale == 1) survives
as a stock transformers `rope_scaling` entry in the emitted llama config.

Usage:
  prep_minicpm4_as_llama.py openbmb/MiniCPM4-0.5B out_dir/
"""
import json
import math
import os
import shutil
import sys

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file


def main():
    src, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    src_dir = src if os.path.isdir(src) else snapshot_download(
        src, allow_patterns=[
            "*.json", "*.model", "model.safetensors", "*.safetensors"])

    cfg = json.load(open(os.path.join(src_dir, "config.json")))
    L = cfg["num_hidden_layers"]
    scale_emb = cfg["scale_emb"]
    depth_mult = cfg["scale_depth"] / math.sqrt(L)
    logit_div = cfg["hidden_size"] / cfg["dim_model_base"]
    print(f"scale_emb={scale_emb} depth_mult={depth_mult:.6f} logit_div={logit_div}")

    idx_path = os.path.join(src_dir, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        sd = {}
        idx = json.load(open(idx_path))
        for shard in sorted(set(idx["weight_map"].values())):
            sd.update(load_file(os.path.join(src_dir, shard)))
    else:
        sd = load_file(os.path.join(src_dir, "model.safetensors"))
    out = {}
    emb = sd["model.embed_tokens.weight"].to(torch.float32)
    if "lm_head.weight" in sd:
        head = sd["lm_head.weight"].to(torch.float32)
    else:
        print("tied embeddings -> untying")
        head = emb.clone()
    out["model.embed_tokens.weight"] = (emb * scale_emb).to(torch.bfloat16)
    out["lm_head.weight"] = (head / logit_div).to(torch.bfloat16)

    for k, v in sd.items():
        if k in ("model.embed_tokens.weight", "lm_head.weight"):
            continue
        if k.endswith("self_attn.o_proj.weight") or k.endswith("mlp.down_proj.weight"):
            out[k] = (v.to(torch.float32) * depth_mult).to(torch.bfloat16)
        else:
            out[k] = v
    save_file(out, os.path.join(out_dir, "model.safetensors"),
              metadata={"format": "pt"})

    rope_scaling = cfg.get("rope_scaling")
    if rope_scaling:
        rs_type = rope_scaling.get("rope_type") or rope_scaling.get("type")
        assert rs_type == "longrope", rs_type
        lf, sf = rope_scaling["long_factor"], rope_scaling["short_factor"]
        orig = rope_scaling["original_max_position_embeddings"]
        assert lf == sf, "long!=short needs the static-rope fold, not this script"
        assert cfg["max_position_embeddings"] == orig, "scale!=1 -> mscale!=1"
        rope_scaling = {
            "rope_type": "longrope",
            "long_factor": lf,
            "short_factor": sf,
            "original_max_position_embeddings": orig,
        }

    new_cfg = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "attention_bias": False,
        "bos_token_id": cfg.get("bos_token_id", 1),
        "eos_token_id": cfg.get("eos_token_id", 2),
        "hidden_act": cfg["hidden_act"],
        "hidden_size": cfg["hidden_size"],
        "initializer_range": cfg.get("initializer_range", 0.02),
        "intermediate_size": cfg["intermediate_size"],
        "max_position_embeddings": cfg["max_position_embeddings"],
        "num_attention_heads": cfg["num_attention_heads"],
        "num_hidden_layers": L,
        "num_key_value_heads": cfg["num_key_value_heads"],
        "rms_norm_eps": cfg["rms_norm_eps"],
        "rope_theta": cfg.get("rope_theta", 10000.0),
        "rope_scaling": rope_scaling,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "vocab_size": cfg["vocab_size"],
    }
    json.dump(new_cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=1)

    for f in ("generation_config.json", "tokenizer.json", "tokenizer.model",
              "tokenizer_config.json", "special_tokens_map.json",
              "added_tokens.json", "chat_template.jinja"):
        p = os.path.join(src_dir, f)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(out_dir, f))
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
