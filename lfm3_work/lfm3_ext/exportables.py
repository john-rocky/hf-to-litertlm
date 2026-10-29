"""Prefill / decode exportables for LFM3 decoders.

The generic decoder-only exportables (token input) and their externalized-
embedder twins (`embeddings [1, T, hidden]` input), plus one option: a second
output `hidden` = the final-norm output (`embedding_norm`, the lm_head input),
`[1, T, hidden]` from prefill and `[1, 1, hidden]` from decode. The Omni and
Audio models run their heads on it; the text model leaves it off.

Inputs, outputs and state names are the generic ones: `tokens` or
`embeddings`, `input_pos [T]`, `mask [1, 1, T, cache_length]`, state
`kv_cache_k_{i}` / `kv_cache_v_{i}` (attention layers) and `kv_cache_c_{i}`
(conv layers), outputs `kv_cache_*` and `logits` (decode; prefill only with
`prefill_logits`).
"""

from litert_torch.generative.export_hf.core import exportable_module
from litert_torch.generative.export_hf.core.external_emb import exportable_module as external_emb_module
import torch


def _prefill(module, tokens, embeddings, input_pos, kv_cache, mask, local_mask, kwargs):
  """Prefill as the generic exportables run it, plus the optional `hidden`."""
  kwargs.update(module.composite_flags())
  inputs = module.adapt_inputs(
      tokens,
      embeddings,
      input_pos,
      kv_cache,
      mask,
      local_mask=local_mask,
      use_bool_mask=module.export_config.extra_kwargs.get("use_bool_mask", False),
      **kwargs,
  )
  inputs |= module.attention_kwargs()
  emit_logits = bool(module.export_config.prefill_logits)
  if emit_logits:
    inputs["logits_to_keep"] = 1
  inputs["prefill_logits"] = emit_logits or module.EMIT_HIDDEN
  output = module.model(**inputs)
  outputs = {"kv_cache": output.past_key_values}
  if emit_logits:
    outputs["logits"] = output.logits.to(torch.float32)
  if module.EMIT_HIDDEN:
    outputs["hidden"] = output.hidden_states[-1].to(torch.float32)
  return outputs


def _decode(module, tokens, embeddings, input_pos, kv_cache, mask, local_mask, kwargs):
  """Decode as the generic exportables run it, plus the optional `hidden`."""
  kwargs.update(module.composite_flags())
  inputs = module.adapt_inputs(
      tokens,
      embeddings,
      input_pos,
      kv_cache,
      mask,
      local_mask=local_mask,
      use_bool_mask=module.export_config.extra_kwargs.get("use_bool_mask", False),
      **kwargs,
  )
  inputs |= module.attention_kwargs()
  output = module.model(**inputs)
  outputs = {
      "kv_cache": output.past_key_values,
      "logits": output.logits.to(torch.float32),
  }
  if module.EMIT_HIDDEN:
    outputs["hidden"] = output.hidden_states[-1].to(torch.float32)
  return outputs


class Lfm3Prefill(exportable_module.LiteRTExportableModuleForDecoderOnlyLMPrefill):
  """Token-input prefill."""

  EMIT_HIDDEN = False

  def forward(self, tokens, input_pos, kv_cache, mask, local_mask=None, **kwargs):
    return _prefill(self, tokens, None, input_pos, kv_cache, mask, local_mask, kwargs)


class Lfm3Decode(exportable_module.LiteRTExportableModuleForDecoderOnlyLMGenerate):
  """Token-input decode."""

  EMIT_HIDDEN = False

  def forward(self, tokens, input_pos, kv_cache, mask, local_mask=None, **kwargs):
    if self.export_verifier:
      raise NotImplementedError("lfm3: MTP verifier signatures are not wired.")
    return _decode(self, tokens, None, input_pos, kv_cache, mask, local_mask, kwargs)


class Lfm3PrefillEmbeddings(
    external_emb_module.LiteRTExportableModuleForDecoderOnlyLMPrefillExternalEmbedder
):
  """Embeddings-input prefill (`externalize_embedder=True`)."""

  EMIT_HIDDEN = False

  def forward(self, embeddings, input_pos, kv_cache, mask, local_mask=None, **kwargs):  # pylint: disable=arguments-renamed
    return _prefill(self, None, embeddings, input_pos, kv_cache, mask, local_mask, kwargs)


class Lfm3DecodeEmbeddings(
    external_emb_module.LiteRTExportableModuleForDecoderOnlyLMGenerateExternalEmbedder
):
  """Embeddings-input decode (`externalize_embedder=True`)."""

  EMIT_HIDDEN = False

  def forward(self, embeddings, input_pos, kv_cache, mask, local_mask=None, **kwargs):  # pylint: disable=arguments-renamed
    if self.export_verifier:
      raise NotImplementedError("lfm3: MTP verifier signatures are not wired.")
    return _decode(self, None, embeddings, input_pos, kv_cache, mask, local_mask, kwargs)


def _with_hidden(cls):
  return type(cls.__name__ + "Hidden", (cls,), {"EMIT_HIDDEN": True})


_CLASSES = {
    (False, False): (Lfm3Prefill, Lfm3Decode),
    (True, False): (Lfm3PrefillEmbeddings, Lfm3DecodeEmbeddings),
}
_CLASSES.update({
    (emb, True): tuple(_with_hidden(c) for c in pair)
    for (emb, _), pair in list(_CLASSES.items())
})


def prefill_decode_classes(export_config, emit_hidden: bool = False):
  """Returns `(prefill_cls, decode_cls)` for `export_config`."""
  if getattr(export_config, "split_cache", False):
    raise NotImplementedError("lfm3: split_cache is not wired.")
  return _CLASSES[(bool(export_config.externalize_embedder), bool(emit_hidden))]
