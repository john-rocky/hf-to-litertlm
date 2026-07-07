"""Build a TINY in-library transformers `deepseek_v3` (MLA) model for converter probing.

The HF tiny-random DeepSeek repos carry a stale remote `modeling_deepseek.py`
(`auto_map`) that imports the removed `is_torch_fx_available` → ImportError at load.
We rebuild from the SAME config dict but drop `auto_map` so transformers uses its
in-library `deepseek_v3` modeling (5.12.1), with realistic MLA geometry
(kv_lora_rank=512, q_lora_rank=1536, qk_rope=64, qk_nope=128, v_head_dim=128) but
only 2 layers, both DENSE (first_k_dense_replace=2) so MLA is isolated from MoE (C9).
"""
import sys, types
# scipy stubs (macOS 27 / py3.10 _propack dlopen fails) — same as scripts/probe_convert.py
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
OUT = sys.argv[1] if len(sys.argv) > 1 else "out/dsv3-tiny-local"

raw = AutoConfig.from_pretrained(SRC, trust_remote_code=True).to_dict()
for k in ("auto_map", "_name_or_path", "transformers_version"):
    raw.pop(k, None)
raw["num_hidden_layers"] = 2
raw["first_k_dense_replace"] = 2   # both layers dense MLP -> no MoE routing (isolate MLA)
raw["architectures"] = ["DeepseekV3ForCausalLM"]
raw["torch_dtype"] = "float32"

config = DeepseekV3Config.from_dict(raw)
print("MLA cfg:", {k: getattr(config, k, None) for k in
      ("hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim",
       "kv_lora_rank", "q_lora_rank", "qk_rope_head_dim", "qk_nope_head_dim",
       "v_head_dim", "first_k_dense_replace", "num_hidden_layers")})

torch.manual_seed(0)
model = DeepseekV3ForCausalLM(config).eval()
model.save_pretrained(OUT)

try:
    tok = AutoTokenizer.from_pretrained(SRC, trust_remote_code=True)
except Exception as e:
    print("tok from SRC failed, falling back to Qwen3-0.6B:", e)
    tok = AutoTokenizer.from_pretrained("litert-community/Qwen3-0.6B")
tok.save_pretrained(OUT)
print("SAVED", OUT)
