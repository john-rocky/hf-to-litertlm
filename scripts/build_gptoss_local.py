"""Build a TINY in-library gpt_oss model (bf16) for converter probing.

gpt-oss is MoE (sparse experts) + attention sinks + sliding/full alternating
attention + YaRN rope. The real openai/gpt-oss-20b ships MXFP4-quantized experts;
this tiny build is FLOAT (bf16) and exercises the same modeling code to find the
ARCH conversion walls (MoE / sinks / sliding-window) WITHOUT the 42 GB dequant load.

    ~/clipconv/bin/python scripts/build_gptoss_local.py <out_dir>
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

OUT = sys.argv[1] if len(sys.argv) > 1 else "out/gptoss-tiny-local"

config = transformers.GptOssConfig(
    vocab_size=1024,
    hidden_size=128,
    intermediate_size=128,
    num_hidden_layers=2,
    layer_types=["sliding_attention", "full_attention"],
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    num_local_experts=4,
    num_experts_per_tok=2,
    sliding_window=8,
    max_position_embeddings=64,
    architectures=["GptOssForCausalLM"],
    torch_dtype="float32",
)
print("gpt_oss cfg:", {k: getattr(config, k, None) for k in
      ("hidden_size", "num_hidden_layers", "layer_types", "num_attention_heads",
       "num_key_value_heads", "head_dim", "num_local_experts", "num_experts_per_tok",
       "sliding_window", "vocab_size")})

torch.manual_seed(0)
model = transformers.GptOssForCausalLM(config).eval()
model.save_pretrained(OUT)
# Reuse the gpt-oss tokenizer if cached; else fall back to a generic fast tokenizer.
try:
    tok = transformers.AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
except Exception:
    tok = transformers.AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
tok.save_pretrained(OUT)
print("SAVED", OUT)
