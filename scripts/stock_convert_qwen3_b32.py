from litert_torch.generative.export_hf.export import export
export(model="src_models/qwen3-1.7b", output_dir="out/qwen3-1.7b-stock-b32",
       quantization_recipe="qwen3_int4_block32_octav.json",
       prefill_lengths=[128], cache_length=4096, trust_remote_code=True)
print("STOCK_EXPORT_DONE")
