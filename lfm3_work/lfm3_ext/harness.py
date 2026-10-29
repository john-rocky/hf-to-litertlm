"""Runs the exporter's PyTorch module the way LiteRT-LM drives the exported graph.

Eager twin of the exported signatures, for oracle checks against HF: the same
exportable modules (`exportables.py`) and LiteRT-LM cache objects the export
traces, fed what the runtime writes into the signature inputs (LiteRT-LM
v0.18.0 / v0.18.1, `runtime/executor/llm_litert_compiled_model_executor.cc` and
`litert_compiled_model_executor_utils.cc`):

- `mask` float32 `[1, 1, T, cache]`: -0.7 * FLT_MAX everywhere, row i of the
  first `steps` rows 0 on `[0, start + i]` (`InitializeAttentionMask`,
  `FillAttentionMask`, no sliding window on the global mask).
- a prefill call shorter than its signature: positions 0 and the pad token in
  the unused tail (memset of the position buffer, `GetPadTokenId`).
"""

import dataclasses

import numpy as np
import torch

from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.core import export_lib

from . import exportables
from . import patch

MASK_FILL = -0.7 * float(np.finfo(np.float32).max)


def runtime_mask(start, steps, rows, cache_length):
  """The runtime's float attention mask for one call."""
  mask = torch.full((1, 1, rows, cache_length), MASK_FILL, dtype=torch.float32)
  for i in range(steps):
    mask[0, 0, i, : start + i + 1] = 0.0
  return mask


def make_export_config(cache_length, externalize_embedder=False, **kwargs):
  return exportable_module_config.ExportableModuleConfig(
      model="lfm3-harness",
      cache_length=cache_length,
      externalize_embedder=externalize_embedder,
      quantization_recipe="none",
      **kwargs,
  )


@dataclasses.dataclass
class Runner:
  """Prefill / decode on the exporter's module with a persistent cache."""

  model: torch.nn.Module
  export_config: exportable_module_config.ExportableModuleConfig
  pad_token_id: int

  def __post_init__(self):
    prefill_cls, decode_cls = exportables.prefill_decode_classes(self.export_config)
    artifacts = export_lib.SourceModelArtifacts(
        model=self.model,
        model_config=self.model.config,
        text_model_config=self.model.config,
        tokenizer=None,
    )
    self.prefill_module = prefill_cls(self.model, self.export_config, artifacts)
    self.decode_module = decode_cls(self.model, self.export_config, artifacts)
    self.embeddings = self.export_config.externalize_embedder
    self.cache = None
    self.reset()

  @property
  def cache_length(self):
    return self.export_config.cache_length

  def reset(self):
    inputs, _ = self.prefill_module.get_sample_kv_cache(self.model.config)
    self.cache = inputs["kv_cache"]

  def _embed(self, tokens):
    return self.model.get_input_embeddings()(tokens)

  @torch.no_grad()
  def prefill(self, token_ids, start, pad_to=None, pad_embedding=None):
    """Prefills `token_ids` at `start`; returns logits `[n, vocab]` (float32).

    Args:
      token_ids: Real tokens of the call.
      start: Position of the first token.
      pad_to: Signature length; the tail is padded as the runtime pads it.
      pad_embedding: Embeddings input only: what the padded slots hold
        (default random, to prove the guard ignores them).
    """
    n = len(token_ids)
    rows = pad_to or n
    tokens = torch.full((1, rows), self.pad_token_id, dtype=torch.int32)
    tokens[0, :n] = torch.tensor(token_ids, dtype=torch.int32)
    pos = torch.zeros(rows, dtype=torch.int32)
    pos[:n] = torch.arange(start, start + n, dtype=torch.int32)
    mask = runtime_mask(start, n, rows, self.cache_length)
    module = self.prefill_module
    if self.embeddings:
      emb = self._embed(tokens)
      if rows > n:
        emb[0, n:] = (
            pad_embedding
            if pad_embedding is not None
            else torch.randn_like(emb[0, n:]) * 3.0
        )
      tok_arg, emb_arg = None, emb
    else:
      tok_arg, emb_arg = tokens, None
    inputs = module.adapt_inputs(
        tok_arg, emb_arg, pos, self.cache, mask, use_bool_mask=False,
        **module.composite_flags(),
    )
    inputs |= module.attention_kwargs()
    inputs["prefill_logits"] = True
    out = module.model(**inputs)
    return out.logits[0, :n].float()

  @torch.no_grad()
  def decode(self, token_id, pos):
    """Decodes one token at `pos` through the decode exportable's forward."""
    tokens = torch.tensor([[token_id]], dtype=torch.int32)
    mask = runtime_mask(pos, 1, 1, self.cache_length)
    input_pos = torch.tensor([pos], dtype=torch.int32)
    if self.embeddings:
      out = self.decode_module(self._embed(tokens), input_pos, self.cache, mask)
    else:
      out = self.decode_module(tokens, input_pos, self.cache, mask)
    return out["logits"][0, 0].float()


def run_paths(runner, ids, chunk=128, split=None):
  """Logits `[len(ids), vocab]` for each path on one sequence.

  (i) one prefill call; (ii) decode token by token; (iii) prefill in `chunk`
  calls, the last one padded to `chunk`; (iv) the first `split` tokens as
  (iii), then decode (the padded call's state is read by the decode steps).
  """
  out = {}
  runner.reset()
  out["i_prefill"] = runner.prefill(ids, 0)
  runner.reset()
  out["ii_decode"] = torch.stack([runner.decode(t, p) for p, t in enumerate(ids)])
  runner.reset()
  parts = []
  for s in range(0, len(ids), chunk):
    parts.append(runner.prefill(ids[s : s + chunk], s, pad_to=chunk))
  out["iii_chunked"] = torch.cat(parts)
  if split:
    runner.reset()
    parts = []
    for s in range(0, split, chunk):
      parts.append(runner.prefill(ids[s : min(s + chunk, split)], s, pad_to=chunk))
    parts.append(torch.stack([runner.decode(ids[p], p) for p in range(split, len(ids))]))
    out["iv_chunked_then_decode"] = torch.cat(parts)
  return out


def compare(ref, got, tie_gap=1e-3):
  """max |dlogit| and top-1 agreement over positions whose ref top-2 gap >= tie_gap."""
  ref = ref.float()
  got = got.float()
  top2 = torch.topk(ref, 2, dim=-1).values
  gap = top2[:, 0] - top2[:, 1]
  keep = gap >= tie_gap
  delta = (ref - got).abs().amax(dim=-1)
  agree = ref.argmax(-1) == got.argmax(-1)
  finite = bool(torch.isfinite(got).all())
  return {
      "positions": int(ref.shape[0]),
      "excluded_near_tie": int((~keep).sum()),
      "max_abs_dlogit": float(delta[keep].max()) if keep.any() else 0.0,
      "max_abs_dlogit_all": float(delta.max()),
      "top1_agree": int(agree[keep].sum()),
      "top1_of": int(keep.sum()),
      "finite": finite,
  }


def patched(model, export_config):
  """Context: the exporter's patches plus the attention it traces with."""
  model.config._attn_implementation = "lrt_transposed_attention"  # pylint: disable=protected-access
  return patch.patch_lfm3_model(model, export_config)
