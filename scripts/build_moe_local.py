"""Build a TINY in-library MoE model for a given arch to vote the C9 general fix
(force experts_implementation='batched_mm') across MoE FAMILIES beyond DeepSeek.

    ~/clipconv/bin/python scripts/build_moe_local.py <model_type> <out_dir>
      model_type: mixtral | qwen3_moe | qwen2_moe | olmoe | ...

Builds from the in-library Config (small dims, real MoE routing), no remote code,
no download. Tokenizer falls back to Qwen3-0.6B.
"""
import sys, types
class _D:
    def __getattr__(self, n): return lambda *a, **k: None
    def __call__(self, *a, **k): return None
_pp = types.ModuleType("scipy.sparse.linalg._propack"); _pp.__file__ = "<stub>"; _pp.__spec__ = None
for _nm in ("_spropack", "_dpropack", "_cpropack", "_zpropack"): setattr(_pp, _nm, _D())
sys.modules["scipy.sparse.linalg._propack"] = _pp
_opt = types.ModuleType("scipy.optimize"); _opt.__file__ = "<stub>"; _opt.__spec__ = None
_opt.linear_sum_assignment = lambda *a, **k: None
sys.modules["scipy.optimize"] = _opt

import torch
import transformers
from transformers import AutoTokenizer

model_type = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else f"out/{model_type}-moe-local"

cfg_cls = transformers.AutoConfig.for_model(model_type).__class__
# Small, valid MoE geometry shared by the standard (non-MLA) MoE arches.
common = dict(
    hidden_size=128, intermediate_size=256, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
    vocab_size=1024, torch_dtype="float32",
)
moe = dict(
    num_local_experts=4, num_experts_per_tok=2,        # mixtral / olmoe naming
    num_experts=4,                                      # qwen*_moe naming
    moe_intermediate_size=128, shared_expert_intermediate_size=128,
    decoder_sparse_step=1, norm_topk_prob=True,
)
raw = {**common, **moe}
# keep only keys the config accepts
accepted = {k: v for k, v in raw.items() if k in cfg_cls().to_dict() or hasattr(cfg_cls(), k)}
config = cfg_cls(**accepted)
print(f"{model_type}: experts-relevant cfg:",
      {k: getattr(config, k, None) for k in
       ("num_local_experts","num_experts","num_experts_per_tok","moe_intermediate_size",
        "decoder_sparse_step","num_hidden_layers")})

torch.manual_seed(0)
model = transformers.AutoModelForCausalLM.from_config(config).eval()
model.save_pretrained(OUT)
AutoTokenizer.from_pretrained("litert-community/Qwen3-0.6B").save_pretrained(OUT)
print("SAVED", OUT)
