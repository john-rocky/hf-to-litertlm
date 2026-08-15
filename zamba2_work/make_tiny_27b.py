#!/usr/bin/env python3
"""Build a tiny random-init Zamba2 checkpoint exercising the 2.7B config branches.

The 2.7B differs from the shipped 1.2B in exactly the branches this tiny pins:
  - num_mem_blocks = 2 (two shared transformer blocks, tied in a cycle over
    hybrid OCCURRENCE order) — including the upstream get_layers block_id bug
    (transformers assigns block_id by GLOBAL layer index; the checkpoint layout
    and the tie cycle follow hybrid order — see model_ext/zamba2/patch.py).
    The hybrid placement [2, 4, 7] is deliberately NOT parity-alternating
    (layer ids even/even/odd vs orders 0/1/2), reproducing the 2.7B shape
    where the two conventions disagree.
  - use_mem_rope = False (no rotary on the shared attention)
  - use_shared_attention_adapter = False (no q/k/v LoRA)
  - use_shared_mlp_adapter = True (per-position gate_up LoRA)

Weights are saved through save_pretrained's tie-dedup, so the checkpoint's
key layout follows the same convention as Zyphra's published 2.7B (unique
block A carries even-order adapter slots, block B odd ones).

  PYTHONPATH=~/code/litert-torch .venv-092/bin/python zamba2_work/make_tiny_27b.py
"""
import json
import os
import shutil

import torch

from litert_torch.generative.export_hf.model_ext.zamba2 import patch as zpatch
from transformers.models.zamba2 import modeling_zamba2
from transformers.models.zamba2.configuration_zamba2 import Zamba2Config

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "tiny_ckpt_27b")

# Structure-only fix on the plain class (the export path gets it via the
# PatchedZamba2Model class swap; here we need the same structure for save).
modeling_zamba2.Zamba2Model.get_layers = zpatch._get_layers_hybrid_order

cfg = Zamba2Config(
    hidden_size=128,
    attention_hidden_size=256,
    attention_head_dim=64,
    num_attention_heads=4,
    num_key_value_heads=4,
    num_query_groups=4,
    kv_channels=32,
    intermediate_size=256,
    ffn_hidden_size=256,
    hidden_act="gelu",
    num_hidden_layers=9,
    layers_block_type=[
        "mamba", "mamba", "hybrid", "mamba", "hybrid",
        "mamba", "mamba", "hybrid", "mamba",
    ],
    hybrid_layer_ids=[2, 4, 7],
    num_mem_blocks=2,
    use_mem_rope=False,
    use_shared_attention_adapter=False,
    use_shared_mlp_adapter=True,
    adapter_rank=16,
    n_mamba_heads=4,
    mamba_headdim=64,
    mamba_expand=2,
    mamba_d_state=32,
    mamba_d_conv=4,
    mamba_ngroups=1,
    chunk_size=16,
    use_mamba_kernels=False,
    vocab_size=32000,
    tie_word_embeddings=True,
    max_position_embeddings=4096,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
)

torch.manual_seed(0)
model = modeling_zamba2.Zamba2ForCausalLM(cfg).eval()

# Sanity of the fixed structure before saving.
for lid, want in ((2, [0, 2]), (4, [1])):
    ff = model.model.layers[lid].shared_transformer.feed_forward
    real = [i for i, m in enumerate(ff.gate_up_proj_adapter_list)
            if not isinstance(m, torch.nn.Identity)]
    assert real == want, f"layer {lid}: adapter slots {real}, want {want}"
assert model.model._tied_weights_keys == {
    "layers.7.shared_transformer": "layers.2.shared_transformer",
}, model.model._tied_weights_keys
assert not hasattr(model.model, "rotary_emb") or model.model.rotary_emb is None

os.makedirs(OUT, exist_ok=True)
model.save_pretrained(OUT)

# Real tokenizer files from the 1.2B tiny (same Zamba2 metaspace tokenizer).
for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
          "generation_config.json"):
    src = os.path.join(HERE, "tiny_ckpt", f)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(OUT, f))

saved = json.load(open(os.path.join(OUT, "model.safetensors.index.json")))\
    if os.path.exists(os.path.join(OUT, "model.safetensors.index.json")) else None
if saved is None:
    from safetensors import safe_open
    with safe_open(os.path.join(OUT, "model.safetensors"), framework="pt") as f:
        keys = sorted(f.keys())
else:
    keys = sorted(saved["weight_map"])
tied_dupes = [k for k in keys if k.startswith("model.layers.7.shared_transformer")]
adapters = [k for k in keys if "adapter" in k]
print("saved keys:", len(keys))
print("tied-dup keys saved (want 0):", len(tied_dupes))
print("adapter keys:")
for k in adapters:
    print("  ", k)
print("TINY-27B checkpoint at", OUT)
