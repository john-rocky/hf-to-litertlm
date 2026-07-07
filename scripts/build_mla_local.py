"""Build a TINY in-library MLA model (DeepSeek-V2/V3/V3.2) for converter probing.

Builds from the in-library transformers config class so the real modeling code is
traced (no stale remote `auto_map`). 2 layers, both DENSE (first_k_dense_replace=2)
so MLA is isolated from MoE (C9). Realistic MLA geometry.

    ~/clipconv/bin/python scripts/build_mla_local.py <model_type> <out_dir> [q_lora]
      model_type: deepseek_v2 | deepseek_v3 | deepseek_v32
      q_lora:     "none" -> q_lora_rank=None (DeepSeek-V2-Lite branch: q_proj),
                  else uses the config default (q_a_proj/q_b_proj branch).
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

import torch, transformers

model_type = sys.argv[1]
OUT = sys.argv[2]
q_lora = (len(sys.argv) > 3 and sys.argv[3].lower() == "none")

cfg_cls = transformers.AutoConfig.for_model(model_type).__class__
arch = {"deepseek_v2": "DeepseekV2ForCausalLM", "deepseek_v3": "DeepseekV3ForCausalLM",
        "deepseek_v32": "DeepseekV32ForCausalLM"}[model_type]
model_cls = getattr(transformers, arch)

config = cfg_cls(
    vocab_size=1024,
    hidden_size=128,
    intermediate_size=256,
    moe_intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
    kv_lora_rank=512,
    q_lora_rank=(None if q_lora else 384),
    qk_rope_head_dim=64,
    qk_nope_head_dim=128,
    v_head_dim=128,
    first_k_dense_replace=2,   # both layers dense -> isolate MLA from MoE
    n_routed_experts=4,
    max_position_embeddings=512,
    architectures=[arch],
    torch_dtype="float32",
)
print("MLA cfg:", {k: getattr(config, k, None) for k in
      ("hidden_size", "num_attention_heads", "head_dim", "kv_lora_rank",
       "q_lora_rank", "qk_rope_head_dim", "qk_nope_head_dim", "v_head_dim",
       "first_k_dense_replace", "num_hidden_layers")})

torch.manual_seed(0)
model = model_cls(config).eval()
model.save_pretrained(OUT)
tok = transformers.AutoTokenizer.from_pretrained(
    "hf-internal-testing/tiny-random-DeepseekV3ForCausalLM", trust_remote_code=True)
tok.save_pretrained(OUT)
print("SAVED", OUT, "q_lora_rank=None" if q_lora else "q_lora_rank=384")
