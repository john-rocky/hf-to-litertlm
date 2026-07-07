"""Pure OFFICIAL litert-torch conversion — NO monkeypatches, NO custom code.

Proves Falcon3-3B converts to a working .litertlm with the stock tool alone:
  - graph: stock from_pretrained -> export
  - tokenizer: stock (HF tokenizer)
  - chat template: stock default (use_jinja_template=True -> raw jinja embedded)
  - int4: blockwise via a recipe.json passed to the stock export (the ONLY
    non-default choice; the tool's default named int4 is channelwise)

  python scripts/stock_convert_test.py
"""
from litert_torch.generative.export_hf.export import export

export(
    model="src_models/falcon3-3b",
    output_dir="out/falcon3-stock",
    quantization_recipe="falcon_int4_block128.json",
    prefill_lengths=[128],
    cache_length=2048,
    trust_remote_code=True,
)
print("STOCK_EXPORT_DONE")
