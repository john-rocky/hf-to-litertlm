"""Pure OFFICIAL litert-torch conversion of Qwen3-1.7B — no monkeypatches.
int4 = blockwise-128 + OCTAV via recipe.json (the only non-default choice)."""
from litert_torch.generative.export_hf.export import export
export(
    model="src_models/qwen3-1.7b",
    output_dir="out/qwen3-1.7b-stock",
    quantization_recipe="qwen3_int4_block128_octav.json",
    prefill_lengths=[128],
    cache_length=4096,
    trust_remote_code=True,
)
print("STOCK_EXPORT_DONE")
