"""`ModelExtension` for LFM3 decoders (`lfm3`), out of tree.

Load with `--extension_modules=lfm3_ext.extension` (put `lfm3_work/` on
PYTHONPATH) or pass `model_extension=Lfm3Extension(...)`; `export_lfm3.py`
does the latter for checkpoints whose decoder sits under another key prefix
(VL, Omni, Audio).

- `patch_model`: `patch.py` (LFM2's LiteRT short conv at width 7, pad guard
  from position monotonicity, LFM2's composite RMSNorm).
- exportables: `exportables.py` (token or embeddings input, optional `hidden`
  output).
- `ExecutorMetadata`: the stock builder (`model_ext/metadata_builder.py`)
  already maps `layer_types`: `kv_cache_c_{i}` = TYPE_LINEAR_ATTENTION for
  conv layers, `kv_cache_{k,v}_{i}` = global KV for full and (without a ring
  buffer) sliding layers, `max_history_size` 0.
- `LlmMetadata`: stop tokens = the generation config's EOS ids only (the
  generic builder also adds the template's end string plus punctuation-
  prefixed variants as `token_str` stops, which strip a reply's final mark on
  BPE vocabularies); `llm_model_type` stays `generic_model` (no LiteRT-LM
  parser reads LFM3's XML tool calls, and a typed bundle changes three GPU
  executor defaults).
"""

from litert_torch.generative.export_hf.model_ext import extension as extension_lib

from . import exportables
from . import patch


@extension_lib.register_model_extension()
class Lfm3Extension(extension_lib.ModelExtension):
  """LFM3 hybrid (short conv + full / sliding attention) decoders."""

  model_types = ("lfm3",)

  def __init__(self, emit_hidden: bool = False):
    """Args: emit_hidden: add the `hidden` output to prefill and decode."""
    self.emit_hidden = emit_hidden

  def patch_model(self, model, export_config):
    return patch.patch_lfm3_model(model, export_config)

  def get_prefill_decode_exportables(self, model_config, export_config):
    del model_config
    return exportables.prefill_decode_classes(
        export_config, emit_hidden=self.emit_hidden
    )

  def build_llm_metadata(
      self,
      source_model_artifacts,
      export_config,
      exported_model_artifacts,
      llm_metadata,
  ):
    del export_config, exported_model_artifacts
    model = source_model_artifacts.model
    gen = getattr(model, "generation_config", None)
    eos = getattr(gen, "eos_token_id", None)
    if eos is None:
      eos = source_model_artifacts.tokenizer.eos_token_id
    eos_ids = [eos] if isinstance(eos, int) else list(eos)
    del llm_metadata.stop_tokens[:]
    for token_id in eos_ids:
      llm_metadata.stop_tokens.add().token_ids.ids.append(int(token_id))
    return llm_metadata
