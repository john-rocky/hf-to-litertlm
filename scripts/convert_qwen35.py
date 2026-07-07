"""Convert Qwen3.5 (GatedDeltaNet hybrid) to .litertlm via litert_torch's qwen3_5 builder.

Purpose: the RUNTIME-GAP REPRO. Qwen3.5 is the newest (Feb 2026) on-target small model,
but every size is a linear/full-attention HYBRID (18-24 `linear_attention` +
6-8 `full_attention` layers, GatedDeltaNet). litert_torch HAS a dedicated `qwen3_5`
export builder (cache_implementation=LiteRTQwen35HybridCache) — so it CONVERTS — but the
current LiteRT-LM runtime can't execute the linear-attention cache states (same class as
LFM2 / Granite-4-h, which convert + GPU-delegate but fail at invoke). This script produces
the converted artifact + logs documenting that runtime limitation.

    python scripts/convert_qwen35.py Qwen/Qwen3.5-0.8B out/qwen35-0.8b-int8 [quant_recipe]
"""
import os
import sys

from litert_torch.generative.export_hf.export import export

model_id = sys.argv[1]
out_dir = sys.argv[2]
quant = sys.argv[3] if len(sys.argv) > 3 else "dynamic_wi8_afp32"

export(
    model=model_id,
    output_dir=out_dir,
    prefill_lengths=[int(os.environ.get("PREFILL", "128"))],
    cache_length=int(os.environ.get("CACHE", "1024")),
    quantization_recipe=quant,
    trust_remote_code=True,
)
print("EXPORT_DONE")
