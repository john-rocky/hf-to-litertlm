"""Build a TINY in-library `deepseek_v3` with MoE ENABLED (layer 1 routed) to probe
the C9 MoE wall on the MLA frontier family. Layer 0 stays dense (first_k_dense_replace=1)
so the MLA path (C17, proven) is exercised first, then the MoE layer is reached.

Grouping is made valid for a tiny model: n_group=1, topk_group=1 (stock tiny-random
has topk_group=4 > n_group=2 which is invalid). n_routed_experts=4, top-2 routing.
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
from transformers import AutoConfig, AutoTokenizer
from transformers import DeepseekV3Config, DeepseekV3ForCausalLM

SRC = "hf-internal-testing/tiny-random-DeepseekV3ForCausalLM"
OUT = sys.argv[1] if len(sys.argv) > 1 else "out/dsv3-moe-local"

raw = AutoConfig.from_pretrained(SRC, trust_remote_code=True).to_dict()
for k in ("auto_map", "_name_or_path", "transformers_version"):
    raw.pop(k, None)
raw["num_hidden_layers"] = 2
raw["first_k_dense_replace"] = 1     # layer 0 dense (MLA), layer 1 = routed MoE
raw["n_group"] = 1                    # valid grouping for the tiny model
raw["topk_group"] = 1
raw["n_routed_experts"] = 4
raw["num_experts_per_tok"] = 2
raw["n_shared_experts"] = 1
raw["moe_intermediate_size"] = 256
raw["architectures"] = ["DeepseekV3ForCausalLM"]
raw["torch_dtype"] = "float32"

config = DeepseekV3Config.from_dict(raw)
print("MoE cfg:", {k: getattr(config, k, None) for k in
      ("first_k_dense_replace", "num_hidden_layers", "n_routed_experts",
       "num_experts_per_tok", "n_shared_experts", "n_group", "topk_group",
       "moe_intermediate_size", "scoring_func")})

torch.manual_seed(0)
model = DeepseekV3ForCausalLM(config).eval()
model.save_pretrained(OUT)
try:
    tok = AutoTokenizer.from_pretrained(SRC, trust_remote_code=True)
except Exception as e:
    print("tok fallback:", e)
    tok = AutoTokenizer.from_pretrained("litert-community/Qwen3-0.6B")
tok.save_pretrained(OUT)
print("SAVED", OUT)
