"""Convert Nanbeige/Nanbeige4.2-3B (looped transformer, num_loops=2) to .litertlm.

Nanbeige 4.2 is a "looped" decoder: its 22 llama-style layers run TWICE per token
(config.num_loops=2), the final RMSNorm is also applied between the loops
(skip_loop_final_norm=False), and each (loop, layer) pair has its OWN KV cache slot
(cache_layer_idx = layer_idx + loop_idx * num_hidden_layers → 44 slots). Weights are
shared between the two passes, so the exported flatbuffer stores each weight once —
only compute (and KV) doubles.

The HF modeling is old-style (transformers-4.42-era attention classes), so the generic
export_hf pipeline can't drive it directly:
  - NanbeigeDecoderLayer.__init__ does NANBEIGE_ATTENTION_CLASSES[_attn_implementation]
    → KeyError on the pipeline's 'lrt_transposed_attention'
  - the LiteRT cache would build num_hidden_layers (22) slots, not 44
  - forward() doesn't accept the pipeline's k_ts_idx / v_ts_idx kwargs

Fix, entirely from this script (no site-packages edits):
  1. Pre-load the trust_remote_code dynamic module (sys.modules-cached, so patches
     persist into from_pretrained) and replace NanbeigeForCausalLM.forward with an
     export-friendly port of the loop (verified by scripts/parity_nanbeige42.py):
     embed → [22 layers → final-norm] × num_loops → lm_head, attention routed through
     the pipeline's transposed-attention helper, cache slot = loop_idx*22 + layer_idx.
  2. Register a 'NanbeigeLoopCache' (num_hidden_layers × num_loops slots) in
     CACHE_REGISTRY and pass cache_implementation='NanbeigeLoopCache'.

Everything else (FORCE_SPM + added-tokens fix — 4.2 has the same 166100–166106 added
special tokens as 4.1 —, OCTAV/blockwise recipes, simple-ChatML template forcing,
externalized embedder, packaging) mirrors scripts/export_simple_template.py.

Usage (mirrors export_simple_template.py):
    FORCE_SPM=1 EXTERNALIZE_EMBEDDER=1 CACHE=4096 \
    python convert_nanbeige42.py Nanbeige/Nanbeige4.2-3B out/nanbeige4.2-3b \
        templates/chatml_simple.jinja BOCTAV4
"""

import sys

# --- scipy stub prelude (same rationale as export_simple_template.py) ---
import types as _types  # noqa: E402


class _StubLeaf:
  def __getattr__(self, n):
    return lambda *a, **k: None

  def __call__(self, *a, **k):
    return None


def _scipy_healthy():
  try:
    import scipy.sparse.linalg._propack  # noqa: F401
    import scipy.optimize  # noqa: F401
    return True
  except Exception:
    return False


if not _scipy_healthy():
  _pp = _types.ModuleType("scipy.sparse.linalg._propack")
  _pp.__file__ = "<stub:scipy._propack>"
  _pp.__spec__ = None
  for _nm in ("_spropack", "_dpropack", "_cpropack", "_zpropack"):
    setattr(_pp, _nm, _StubLeaf())
  sys.modules["scipy.sparse.linalg._propack"] = _pp

  _opt = _types.ModuleType("scipy.optimize")
  _opt.__file__ = "<stub:scipy.optimize>"
  _opt.__spec__ = None
  _opt.linear_sum_assignment = lambda *a, **k: None
  sys.modules["scipy.optimize"] = _opt

import transformers  # noqa: E402

model_id = sys.argv[1] if len(sys.argv) > 1 else "Nanbeige/Nanbeige4.2-3B"
out_dir = sys.argv[2] if len(sys.argv) > 2 else "out/nanbeige4.2-3b"
template_path = sys.argv[3] if len(sys.argv) > 3 else "templates/chatml_simple.jinja"
quant = sys.argv[4] if len(sys.argv) > 4 else "BOCTAV4"

SIMPLE_TEMPLATE = open(template_path).read()

# Force every tokenizer loaded during export to use the simple ChatML template
# (Nanbeige 4.2's real template has tool-calling + media branches minja can't render).
_orig_from_pretrained = transformers.AutoTokenizer.from_pretrained


def _patched_from_pretrained(*args, **kwargs):
  tok = _orig_from_pretrained(*args, **kwargs)
  try:
    tok.chat_template = SIMPLE_TEMPLATE
  except Exception as e:  # pylint: disable=broad-except
    print(f"WARN could not set chat_template: {e}")
  return tok


transformers.AutoTokenizer.from_pretrained = _patched_from_pretrained

import os  # noqa: E402
import dataclasses  # noqa: E402
from litert_torch.generative.export_hf.core import export_lib  # noqa: E402
from litert_torch.generative.tools import (  # noqa: E402
    tokenizer_to_sentencepiece_lib as _tok_spm,
)


def _force_spm_export_tokenizer(source_model_artifacts, export_config, exported):
  tok = source_model_artifacts.tokenizer
  vf = getattr(tok, "vocab_file", None)
  if vf and not str(vf).endswith((".model", ".spiece", ".spm")):
    tok.vocab_file = None
  spm = _tok_spm.convert(tok)
  if os.environ.get("FIX_ADDED_TOKENS", "1") != "0":
    spm = _append_added_tokens_to_spm(spm, tok, source_model_artifacts)
  path = os.path.join(export_config.work_dir, "tokenizer.spiece")
  with open(path, "wb") as f:
    f.write(spm)
  print("FORCED sentencepiece tokenizer")
  return dataclasses.replace(exported, tokenizer_model_path=path)


def _append_added_tokens_to_spm(spm_bytes, tok, source_model_artifacts):
  """Append added special tokens (dropped by tok_spm.convert) as USER_DEFINED SP pieces
  at their exact ids, padding to the model vocab so generated ids can't be out of range.
  Nanbeige 4.2: <|im_start|>=166100 … </tool_call>=166106, embedding vocab 166144."""
  from sentencepiece import sentencepiece_model_pb2 as _spb
  mp = _spb.ModelProto()
  mp.ParseFromString(spm_bytes)
  base_n = len(mp.pieces)
  by_id = {}
  for i, t in getattr(tok, "added_tokens_decoder", {}).items():
    by_id[int(i)] = getattr(t, "content", str(t))
  if not by_id or max(by_id) < base_n:
    return spm_bytes
  target = max(by_id) + 1
  for attr in ("model", "pytorch_model"):
    m = getattr(source_model_artifacts, attr, None)
    vs = getattr(getattr(m, "config", None), "vocab_size", None)
    if isinstance(vs, int) and vs > target:
      target = vs
      break
  while len(mp.pieces) < target:
    idx = len(mp.pieces)
    p = mp.pieces.add()
    if idx in by_id:
      p.piece = by_id[idx]; p.score = 0.0
      p.type = _spb.ModelProto.SentencePiece.USER_DEFINED
    else:
      p.piece = f"<unused_{idx}>"; p.score = 0.0
      p.type = _spb.ModelProto.SentencePiece.UNUSED
  print(f"FIX_ADDED_TOKENS: SP {base_n} -> {len(mp.pieces)} pieces "
        f"(appended {sum(1 for i in by_id if i >= base_n)} added tokens incl. "
        f"{by_id.get(max(by_id))!r})")
  return mp.SerializeToString()


if os.environ.get("FORCE_SPM"):
  export_lib.export_tokenizer = _force_spm_export_tokenizer

# Custom quant recipes by NAME (same set as export_simple_template.py).
import copy  # noqa: E402
import ai_edge_quantizer.recipe as _aqr  # noqa: E402

_I4 = _aqr.dynamic_wi4_afp32()[0]
_I8 = copy.deepcopy(_I4)
_I8["op_config"]["weight_tensor_config"]["num_bits"] = 8


def _mk_alg(int4_rule, ops_int8):
  rules = [int4_rule]
  for op in ops_int8:
    rr = copy.deepcopy(_I8)
    rr["operation"] = op
    rules.append(rr)
  return rules


_O4 = copy.deepcopy(_I4)
_O4["algorithm_key"] = _aqr.AlgorithmName.OCTAV
_BO4 = copy.deepcopy(_O4)
_BO4["op_config"]["weight_tensor_config"]["granularity"] = "BLOCKWISE_32"
_BO4_128 = copy.deepcopy(_O4)
_BO4_128["op_config"]["weight_tensor_config"]["granularity"] = "BLOCKWISE_128"
_aqr.OCTAV4 = lambda: _mk_alg(_O4, ["EMBEDDING_LOOKUP"])
_aqr.BOCTAV4 = lambda: _mk_alg(_BO4, ["EMBEDDING_LOOKUP"])
_aqr.BOCTAV4_128 = lambda: _mk_alg(_BO4_128, ["EMBEDDING_LOOKUP"])

# ---------------------------------------------------------------------------
# Nanbeige 4.2 loop support
# ---------------------------------------------------------------------------
import torch  # noqa: E402
from transformers.dynamic_module_utils import get_class_from_dynamic_module  # noqa: E402
from transformers.modeling_outputs import CausalLMOutputWithPast  # noqa: E402
from litert_torch.generative.export_hf.core import attention as _lrt_attn  # noqa: E402
from litert_torch.generative.export_hf.core import cache as _lrt_cache  # noqa: E402
from litert_torch.generative.export_hf.core import cache_base as _lrt_cache_base  # noqa: E402


def _nb_num_loops(cfg) -> int:
  """Mirror NanbeigeModel._get_num_loops for the released (simple) variant."""
  loop_weights = getattr(cfg, "loop_loss_weights", None)
  if loop_weights:
    return len(loop_weights) + 1
  return getattr(cfg, "num_loops", 1)


def _nb_assert_simple_variant(cfg):
  """The port below only implements the released 4.2-3B config. Refuse the exotic
  variants (they change the forward graph) instead of silently mis-converting."""
  for flag in (
      "enable_double_loop_split",
      "loop_share_kv",
      "enable_hyper_connection",
      "enable_mhc",
      "enable_depth_attention",
      "ngram_insert_all_layers",
      "qk_layernorm",
  ):
    if getattr(cfg, flag, False):
      raise NotImplementedError(f"Nanbeige variant with {flag}=True is not supported")
  if getattr(cfg, "insert_ngram_layer_idx", None):
    raise NotImplementedError("Nanbeige n-gram layer fusion is not supported")
  if getattr(cfg, "emb_neighbor_num", None) is not None:
    raise NotImplementedError("Nanbeige n-gram embeddings are not supported")
  if getattr(cfg, "pretraining_tp", 1) > 1:
    raise NotImplementedError("pretraining_tp > 1 is not supported")
  rs = getattr(cfg, "rope_scaling", None)
  # transformers 5.x normalizes rope_scaling=None into {'rope_type': 'default', ...}.
  if rs is not None and not (
      isinstance(rs, dict) and rs.get("rope_type", rs.get("type")) == "default"
  ):
    raise NotImplementedError(f"rope_scaling {rs!r} is not supported")


def install_nanbeige_loop_patches(model_ref: str):
  """Load the Nanbeige dynamic module and make it exportable. Returns the module."""
  causal_lm_cls = get_class_from_dynamic_module(
      "modeling_nanbeige.NanbeigeForCausalLM", model_ref
  )
  nb = sys.modules[causal_lm_cls.__module__]

  # 1) Any _attn_implementation string (the pipeline sets 'lrt_transposed_attention')
  #    resolves to the base attention class; its forward is bypassed below anyway.
  class _AnyAttn(dict):

    def __missing__(self, key):
      return nb.NanbeigeAttention

  nb.NANBEIGE_ATTENTION_CLASSES = _AnyAttn(dict(nb.NANBEIGE_ATTENTION_CLASSES))

  # transformers 5.x rewrites rope_scaling=None into {'rope_type': 'default', ...};
  # the 4.42-era _init_rope then KeyErrors on rope_scaling['type']. Unwrap the
  # normalized no-op dict back to None before the original branch logic runs.
  _orig_init_rope = nb.NanbeigeAttention._init_rope

  def _patched_init_rope(self):
    rs = self.config.rope_scaling
    if isinstance(rs, dict) and rs.get("rope_type", rs.get("type")) == "default":
      self.config.rope_scaling = None
    _orig_init_rope(self)

  nb.NanbeigeAttention._init_rope = _patched_init_rope

  # ⚠ transformers 5.x META-DEVICE LOAD ZEROES INIT-COMPUTED BUFFERS: the old-style
  # NanbeigeRotaryEmbedding computes its non-persistent inv_freq buffer ONLY in
  # __init__, which under 5.x from_pretrained runs on the meta device — the buffer
  # then materializes as ZEROS. cos=1/sin=0 for every position → the model sees a
  # bag of tokens (rope-ZERO trap, same family as the PaddleOCR tf5.x referee bug).
  # Recompute inv_freq on the fly from the python-scalar attrs (dim/base survive).
  def _fixed_rope_forward(self, x, position_ids):
    inv_freq = 1.0 / (
        self.base ** (
            torch.arange(0, self.dim, 2, dtype=torch.float32, device=x.device)
            / self.dim
        )
    )
    pos = position_ids[:, None, :].float()
    freqs = (inv_freq[None, :, None].expand(pos.shape[0], -1, 1) @ pos).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(x.dtype), emb.sin().to(x.dtype)

  nb.NanbeigeRotaryEmbedding.forward = _fixed_rope_forward

  rotate_half = nb.rotate_half

  def _layer_forward(layer, h, cos, sin, kv, cache_idx, mask, k_ts_idx, v_ts_idx):
    """One decoder layer against LiteRT static cache slot `cache_idx`.
    Same math as NanbeigeAttention eager forward (scale 1/sqrt(head_dim), llama
    rotate-half RoPE), with attention over the full static cache buffer."""
    attn = layer.self_attn
    residual = h
    x = layer.input_layernorm(h)
    b, t, _ = x.shape
    q = attn.q_proj(x).view(b, t, attn.num_heads, attn.head_dim).transpose(1, 2)
    k = attn.k_proj(x).view(b, t, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
    v = attn.v_proj(x).view(b, t, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    k_full, v_full = kv.layers[cache_idx].update(k, v)
    attn_out, _ = _lrt_attn.transposed_attention(
        attn, q, k_full, v_full, mask,
        scaling=None, k_ts_idx=k_ts_idx, v_ts_idx=v_ts_idx,
    )  # [b, t, num_heads, head_dim]
    attn_out = attn_out.reshape(b, t, attn.num_heads * attn.head_dim)
    h = residual + attn.o_proj(attn_out)
    residual = h
    y = layer.post_attention_layernorm(h)
    y = layer.mlp(y)
    return residual + y

  def _export_forward(
      self,
      input_ids=None,
      inputs_embeds=None,
      position_ids=None,
      past_key_values=None,
      cache_position=None,
      attention_mask=None,
      use_cache=True,
      k_ts_idx=2,
      v_ts_idx=3,
      **kwargs,
  ):
    """Export-friendly NanbeigeForCausalLM.forward: unrolled num_loops passes over
    the shared 22 layers, per-(loop,layer) cache slots, final norm after each loop
    (skip_loop_final_norm=False layout)."""
    del cache_position, use_cache, kwargs
    cfg = self.config
    model = self.model
    assert past_key_values is not None, "export forward requires the LiteRT cache"
    assert attention_mask is not None, "export forward requires the runtime mask"

    h = model.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds

    # RoPE cos/sin once per forward (identical across layers/loops): llama convention.
    # inv_freq is computed from config, NOT read from the module buffer — the buffer
    # is zeroed by the transformers-5.x meta-device load (see _fixed_rope_forward).
    head_dim = cfg.head_dim
    inv_freq = 1.0 / (
        float(cfg.rope_theta)
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )  # [head_dim/2] fp32
    pos = position_ids.float()  # [1, T]
    freqs = pos[:, :, None] * inv_freq[None, None, :]  # [1, T, head_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [1, T, head_dim]
    cos = emb.cos()[:, None, :, :]  # [1, 1, T, head_dim]
    sin = emb.sin()[:, None, :, :]

    num_loops = _nb_num_loops(cfg)
    n_layers = cfg.num_hidden_layers
    skip_loop_final_norm = getattr(cfg, "skip_loop_final_norm", False)
    for loop_idx in range(num_loops):
      for i, layer in enumerate(model.layers):
        h = _layer_forward(
            layer, h, cos, sin, past_key_values,
            loop_idx * n_layers + i, attention_mask, k_ts_idx, v_ts_idx,
        )
      if not skip_loop_final_norm:
        h = model.norm(h)
    if skip_loop_final_norm:
      h = model.norm(h)

    logits = self.lm_head(h)
    return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)

  causal_lm_cls._nb42_original_forward = causal_lm_cls.forward  # for parity reference
  causal_lm_cls.forward = _export_forward
  # The patched forward is fully static/exportable; silence the pipeline's warnings.
  causal_lm_cls._can_compile_fullgraph = True  # pylint: disable=protected-access
  causal_lm_cls._supports_attention_backend = True  # pylint: disable=protected-access
  nb.NanbeigeModel._can_compile_fullgraph = True  # pylint: disable=protected-access
  nb.NanbeigeModel._supports_attention_backend = True  # pylint: disable=protected-access
  print("PATCHED NanbeigeForCausalLM.forward -> looped export port "
        "(44-slot cache, transposed attention)")
  return nb


class NanbeigeLoopCache(_lrt_cache.LiteRTLMCache):
  """Registry entry building num_hidden_layers x num_loops cache slots.

  create_from_config returns a plain LiteRTLMCache instance (NOT this subclass):
  torch.export flattens the cache by exact pytree-registered type, and only
  LiteRTLMCache is registered."""

  @classmethod
  def create_from_config(cls, model_config, export_config, **kwargs):
    num_layers = model_config.num_hidden_layers * _nb_num_loops(model_config)
    layers = [
        _lrt_cache.LiteRTLMCacheLayer.create_from_config(
            model_config, i, export_config, **kwargs
        )
        for i in range(num_layers)
    ]
    print(f"NanbeigeLoopCache: {num_layers} KV slots "
          f"({model_config.num_hidden_layers} layers x {_nb_num_loops(model_config)} loops)")
    return _lrt_cache.LiteRTLMCache(layers)


_lrt_cache_base.CACHE_REGISTRY["NanbeigeLoopCache"] = NanbeigeLoopCache


def main():
  cfg = transformers.AutoConfig.from_pretrained(model_id, trust_remote_code=True)
  _nb_assert_simple_variant(cfg)
  install_nanbeige_loop_patches(model_id)

  from litert_torch.generative.export_hf.export import export  # noqa: E402

  quant_recipe = None if quant.upper() in ("NONE", "FP32") else quant
  export(
      model=model_id,
      output_dir=out_dir,
      prefill_lengths=[int(os.environ.get("PREFILL", "128"))],
      cache_length=int(os.environ.get("CACHE", "1024")),
      quantization_recipe=quant_recipe,
      use_jinja_template=False,
      cache_implementation="NanbeigeLoopCache",
      externalize_embedder=bool(os.environ.get("EXTERNALIZE_EMBEDDER")),
      trust_remote_code=True,
  )
  # The bundle embeds NO Jinja: use_jinja_template=False packs only prefix/suffix
  # markers parsed from `template_path`. (A default litert-torch export embeds the
  # vendor's Jinja template verbatim; Nanbeige's calls Python-style .get()/.startswith(),
  # which the runtime's minijinja renderer rejects on the first message.)
  print(f"Template mode: prefix/suffix prompt_templates from {template_path} "
        "(use_jinja_template=False -- no Jinja embedded)")
  print("EXPORT_DONE")


if __name__ == "__main__":
  main()
