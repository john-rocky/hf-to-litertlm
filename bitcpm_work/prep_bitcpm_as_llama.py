#!/usr/bin/env python3
"""Normalize openbmb/BitCPM-CANN-1B for litert-torch export_hf.

The 1B is already a stock LlamaForCausalLM (no muP scalings, no remote code) —
unlike MiniCPM4 there is nothing to fold. This only:
  1. rewrites pytorch_model.bin -> model.safetensors
  2. emits a minimal llama config (drops the no-op rope_scaling "factor" key;
     keeps longrope with long_factor == short_factor, mscale == 1, which the
     export_static_longrope.py wrapper makes static)
  3. copies tokenizer files and synthesizes added_tokens.json from
     tokenizer_config.json's added_tokens_decoder (the 1B repo does not ship
     added_tokens.json, but fix_sp_added_tokens.py needs one to append
     <|im_end|>=73440 etc. to the SP model)

Usage: prep_bitcpm_as_llama.py <hf_id_or_dir> out_dir/
"""
import json
import os
import shutil
import sys

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file


def main():
    src, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    src_dir = src if os.path.isdir(src) else snapshot_download(src)

    sd = torch.load(os.path.join(src_dir, "pytorch_model.bin"),
                    map_location="cpu", weights_only=True)
    sd = {k: v.contiguous() for k, v in sd.items()}
    assert "lm_head.weight" in sd, "expected untied lm_head"
    save_file(sd, os.path.join(out_dir, "model.safetensors"),
              metadata={"format": "pt"})

    cfg = json.load(open(os.path.join(src_dir, "config.json")))
    rs = cfg["rope_scaling"]
    assert rs["long_factor"] == rs["short_factor"]
    assert cfg["max_position_embeddings"] == rs["original_max_position_embeddings"]
    new_cfg = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "attention_bias": False,
        "bos_token_id": cfg.get("bos_token_id", 1),
        "eos_token_id": cfg.get("eos_token_id", 2),
        "head_dim": cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"]),
        "hidden_act": cfg["hidden_act"],
        "hidden_size": cfg["hidden_size"],
        "initializer_range": cfg.get("initializer_range", 0.02),
        "intermediate_size": cfg["intermediate_size"],
        "max_position_embeddings": cfg["max_position_embeddings"],
        "num_attention_heads": cfg["num_attention_heads"],
        "num_hidden_layers": cfg["num_hidden_layers"],
        "num_key_value_heads": cfg["num_key_value_heads"],
        "rms_norm_eps": cfg["rms_norm_eps"],
        "rope_theta": cfg.get("rope_theta", 10000.0),
        "rope_scaling": {
            "rope_type": "longrope",
            "long_factor": rs["long_factor"],
            "short_factor": rs["short_factor"],
            "original_max_position_embeddings": rs["original_max_position_embeddings"],
        },
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "vocab_size": cfg["vocab_size"],
    }
    json.dump(new_cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=1)

    for f in ("generation_config.json", "tokenizer.json", "tokenizer.model",
              "tokenizer_config.json", "special_tokens_map.json"):
        p = os.path.join(src_dir, f)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(out_dir, f))

    tok_cfg = json.load(open(os.path.join(src_dir, "tokenizer_config.json")))
    added = {v["content"]: int(k)
             for k, v in tok_cfg["added_tokens_decoder"].items()}
    json.dump(added, open(os.path.join(out_dir, "added_tokens.json"), "w"),
              indent=1)
    print(f"wrote {out_dir} (added_tokens: {sorted(added.values())[-8:]})")


if __name__ == "__main__":
    main()
