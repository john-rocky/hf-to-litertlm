"""Export an HF model to .litertlm but FORCE a simple ChatML chat template.

The runtime applies the LlmMetadata's *structured* prompt_templates (the official
litert-community models use this — simple `<|im_start|>role\n` prefixes). Without
--use_jinja_template, litert-torch's `parse_chat_template` tries to extract that
structured form by applying the tokenizer's chat_template to sample messages — but
for COMPLEX templates (LFM2 namespaces, Qwen/MiniCPM tool+thinking logic) the
extraction fails and it falls back to embedding the raw jinja, which the runtime's
minimal jinja engine (minja) can't render → empty / garbage output on device.

Fix: monkeypatch the tokenizer's chat_template to a minimal ChatML template before
export, so parse_chat_template cleanly extracts the structured prefixes — matching
the official models. Usage:

    python export_simple_template.py <hf_model> <out_dir> <template.jinja> [quant_recipe]
"""

import sys

# --- scipy stub prelude (macOS / clipconv main-HEAD env): scipy 1.15.3's compiled
#     _propack fails to dlopen here, and transformers' D-FINE detection loss (pulled in
#     transitively when litert_torch calls AttentionInterface.register) imports
#     scipy.optimize. Neither is used by LLM conversion — stub the broken leaves so the
#     import chain succeeds; the REAL csgraph / sparse.linalg then load. No-op on a clean
#     scipy (e.g. .venv 1.17). Mirrors scripts/probe_convert.py. ---
import types as _types  # noqa: E402


class _StubLeaf:
  def __getattr__(self, n):
    return lambda *a, **k: None

  def __call__(self, *a, **k):
    return None


def _scipy_healthy():
  # Only stub when scipy is actually broken (clipconv env: scipy 1.15.3's _propack
  # fails to dlopen). On a clean scipy (.venv 1.17) the stub would ITSELF break the
  # scipy.sparse.csgraph / _svdp import chain that externalize_embedder pulls in
  # (the stub _propack lacks slansvd) — so skip stubbing when the real one works.
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

model_id = sys.argv[1]
out_dir = sys.argv[2]
template_path = sys.argv[3]
quant = sys.argv[4] if len(sys.argv) > 4 else "dynamic_wi4_afp32"

SIMPLE_TEMPLATE = open(template_path).read()

# Force every tokenizer loaded during export to use the simple ChatML template.
_orig_from_pretrained = transformers.AutoTokenizer.from_pretrained


def _patched_from_pretrained(*args, **kwargs):
  tok = _orig_from_pretrained(*args, **kwargs)
  try:
    tok.chat_template = SIMPLE_TEMPLATE
  except Exception as e:  # pylint: disable=broad-except
    print(f"WARN could not set chat_template: {e}")
  return tok


transformers.AutoTokenizer.from_pretrained = _patched_from_pretrained

# Force a SentencePiece tokenizer in the .litertlm. export_tokenizer defaults to
# saving the HF tokenizer.json (→ HF_Tokenizer_Zlib), which the runtime
# MIS-TOKENIZES (the prompt comes out as garbage → degenerate output). The working
# official litert-community models embed SP_Tokenizer (HF→sentencepiece converted).
import os  # noqa: E402
import dataclasses  # noqa: E402
from litert_torch.generative.export_hf.core import export_lib  # noqa: E402
from litert_torch.generative.tools import (  # noqa: E402
    tokenizer_to_sentencepiece_lib as _tok_spm,
)


def _force_spm_export_tokenizer(source_model_artifacts, export_config, exported):
  tok = source_model_artifacts.tokenizer
  # BPE tokenizers (Qwen/Llama-BPE) expose a `vocab_file` pointing at vocab.json.
  # tokenizer_to_sentencepiece.convert() then tries to parse that JSON as a
  # sentencepiece ModelProto → "Wire format was corrupt". Only SP-native tokenizers
  # (Gemma/Llama-SP, vocab_file = *.model/*.spiece) belong on that fast path; for the
  # rest, clear vocab_file so convert() builds a real SP model from vocab+merges (the
  # path the working int8 export took when no vocab.json happened to be cached).
  vf = getattr(tok, "vocab_file", None)
  if vf and not str(vf).endswith((".model", ".spiece", ".spm")):
    tok.vocab_file = None
  spm = _tok_spm.convert(tok)
  # ADDED-TOKENS FIX: convert() builds the SP model from the base vocab+merges and DROPS
  # the tokenizer's added special tokens (e.g. `<think>`=166103 on Nanbeige). A thinking
  # model then GENERATES `<think>` and the runtime crashes "Token id out of range".
  # Append each added_token as a USER_DEFINED piece at its exact id, and pad up to the
  # model's embedding vocab_size so no generated id can be out of range. Gated by env
  # FIX_ADDED_TOKENS=1 (default on when FORCE_SPM is set).
  if os.environ.get("FIX_ADDED_TOKENS", "1") != "0":
    spm = _append_added_tokens_to_spm(spm, tok, source_model_artifacts)
  path = os.path.join(export_config.work_dir, "tokenizer.spiece")
  with open(path, "wb") as f:
    f.write(spm)
  print("FORCED sentencepiece tokenizer")
  return dataclasses.replace(exported, tokenizer_model_path=path)


def _append_added_tokens_to_spm(spm_bytes, tok, source_model_artifacts):
  """Append added special tokens (dropped by tok_spm.convert) as USER_DEFINED SP pieces
  at their exact ids, padding to the model vocab so generated ids can't be out of range."""
  from sentencepiece import sentencepiece_model_pb2 as _spb
  mp = _spb.ModelProto()
  mp.ParseFromString(spm_bytes)
  base_n = len(mp.pieces)
  by_id = {}
  for i, t in getattr(tok, "added_tokens_decoder", {}).items():
    by_id[int(i)] = getattr(t, "content", str(t))
  if not by_id or max(by_id) < base_n:
    return spm_bytes  # nothing beyond the base vocab -> no fix needed
  # target = model embedding vocab if we can find it, else just past the last added id
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

# Register custom mixed-int4 recipes by NAME (export_lib does
# recipe_lib.__dict__[name]()). int4 default + keep the vocab embedding/lm_head
# (EMBEDDING_LOOKUP — the int4-killer for small models) at int8.
import copy  # noqa: E402
import ai_edge_quantizer.recipe as _aqr  # noqa: E402

_I4 = _aqr.dynamic_wi4_afp32()[0]
_I8 = copy.deepcopy(_I4)
_I8["op_config"]["weight_tensor_config"]["num_bits"] = 8


def _mk(ops_int8):
  rules = [_I4]
  for op in ops_int8:
    rr = copy.deepcopy(_I8)
    rr["operation"] = op
    rules.append(rr)
  return rules


_aqr.MIXED4 = lambda: _mk(["EMBEDDING_LOOKUP"])
_aqr.MIXED4B = lambda: _mk(["EMBEDDING_LOOKUP", "FULLY_CONNECTED"])

# Better int4: replace naive min-max with OCTAV (optimal-clipping, data-free) /
# MSE for the int4 weights, keeping the int8 embedding. Reduces PTQ degradation
# (e.g. GSM8K) without calibration data — the fix for "int4 is measurably worse than bf16".
_O4 = copy.deepcopy(_I4)
_O4["algorithm_key"] = _aqr.AlgorithmName.OCTAV
_M4 = copy.deepcopy(_I4)
_M4["algorithm_key"] = _aqr.AlgorithmName.MSE


def _mk_alg(int4_rule, ops_int8):
  rules = [int4_rule]
  for op in ops_int8:
    rr = copy.deepcopy(_I8)
    rr["operation"] = op
    rules.append(rr)
  return rules


_aqr.OCTAV4 = lambda: _mk_alg(_O4, ["EMBEDDING_LOOKUP"])
_aqr.MSE4 = lambda: _mk_alg(_M4, ["EMBEDDING_LOOKUP"])

# BLOCKWISE int4 (block size 32) — the granularity the official litert-community
# models use. `dynamic_wi4_afp32` defaults to CHANNELWISE int4, which catastrophically
# collapses small models (Qwen3-0.6B: 0% GSM8K, degenerate looping). Blockwise int4
# matches the official conversion (≈42% on 0.6B). Keep the vocab embedding at int8.
_B4 = copy.deepcopy(_I4)
_B4["op_config"]["weight_tensor_config"]["granularity"] = "BLOCKWISE_32"
# Blockwise int4 + OCTAV optimal-clipping = best data-free int4.
_BO4 = copy.deepcopy(_O4)
_BO4["op_config"]["weight_tensor_config"]["granularity"] = "BLOCKWISE_32"
_aqr.BMIX4 = lambda: _mk_alg(_B4, ["EMBEDDING_LOOKUP"])
_aqr.BMIX4B = lambda: _mk_alg(_B4, ["EMBEDDING_LOOKUP", "FULLY_CONNECTED"])
_aqr.BOCTAV4 = lambda: _mk_alg(_BO4, ["EMBEDDING_LOOKUP"])

# Blockwise-128 int4 — coarser blocks = 1/4 the scales = lighter dequant =
# faster GPU decode (the granularity the official Gemma block128 bundles use),
# at a small quality cost vs block32. The on-device speed/quality knob.
_B4_128 = copy.deepcopy(_I4)
_B4_128["op_config"]["weight_tensor_config"]["granularity"] = "BLOCKWISE_128"
_aqr.BMIX4_128 = lambda: _mk_alg(_B4_128, ["EMBEDDING_LOOKUP"])

# Blockwise-128 + OCTAV: the iPhone knob. Block128's 1/4 scales shrink the main
# TFLiteModel section under the iOS ~2 GiB single-section mmap limit (block32 leaves
# a 4B model's section at ~2.11 GiB → iPhone load fails), while OCTAV keeps int4 at
# parity. block32 BOCTAV4 stays the Mac/Android best-quality build.
_BO4_128 = copy.deepcopy(_O4)
_BO4_128["op_config"]["weight_tensor_config"]["granularity"] = "BLOCKWISE_128"
_aqr.BOCTAV4_128 = lambda: _mk_alg(_BO4_128, ["EMBEDDING_LOOKUP"])

from litert_torch.generative.export_hf.export import export  # noqa: E402

# GPTQREC_GCD_FIX=1: dequantized_weight_recovery derives each block's scale as the
# min positive diff of the levels PRESENT in that block — underdetermined when all
# present levels share a common divisor (e.g. {0,±2,±4} → 2s, or a 2-level block
# {0,6s} → 6s): the roundtrip then fails validation even though the tensor IS on a
# grid (first hit: granite gs64 down_proj — smaller blocks miss adjacent levels more
# often). Refine only the failing blocks: divide the block scale by the smallest
# integer k that makes the roundtrip bit-exact with levels in [-8,7]. Any exactly
# roundtripping scale is a valid transport — recovering the "true" GPTQ scale is not
# required. (fp32-exactness holds: diffs of on-grid fp32 values and their integer
# quotients are exactly representable.)
if os.environ.get("GPTQREC_GCD_FIX"):
  import numpy as _np  # noqa: E402
  from ai_edge_quantizer.algorithms.uniform_quantize import (  # noqa: E402
      dequantized_weight_recovery as _dwr,
  )
  from ai_edge_quantizer.algorithms.utils import common_utils as _dwr_cu  # noqa: E402

  _orig_get_zp_scale = _dwr.get_zp_scale_from_dequantized_symmetric_weights

  def _gcd_fixed_get_zp_scale(dequant_vals, quantized_dimension=None, block_size=0,
                              min_scale=1e-9):
    zp, scales = _orig_get_zp_scale(dequant_vals, quantized_dimension, block_size,
                                    min_scale)
    if block_size and quantized_dimension is not None:
      blocks = _dwr_cu.reshape_to_blocks(dequant_vals, quantized_dimension,
                                         block_size).astype(_np.float64)
      flat = scales.reshape(-1, 1).astype(_np.float64)
      q = blocks / flat
      bad = _np.nonzero(_np.abs(q - _np.round(q)).max(axis=1) > 1e-6)[0]
      fixed = 0
      for i in bad:
        for k in range(2, 17):
          s2 = flat[i, 0] / k
          q2 = blocks[i] / s2
          r2 = _np.round(q2)
          if _np.abs(q2 - r2).max() < 1e-6 and r2.min() >= -8 and r2.max() <= 7:
            flat[i, 0] = s2
            fixed += 1
            break
      if fixed:
        print(f"GCD_FIX: refined {fixed}/{len(bad)} degenerate blocks"
              f" (of {flat.shape[0]})")
      scales = flat.reshape(scales.shape).astype(scales.dtype)
    return zp, scales

  _dwr.get_zp_scale_from_dequantized_symmetric_weights = _gcd_fixed_get_zp_scale
  print("PATCHED dequantized_weight_recovery scale fn (GCD-degenerate block fix)")

# LongRoPE fix for Phi-3/Phi-4 (rope_scaling=longrope), gated by env PHI3_STATIC_ROPE=1.
# The `@dynamic_rope_update` decorator on Phi3RotaryEmbedding.forward swaps short/long factor on
# `seq_len > original_max(4096)` → data-dependent guard under torch.export. We export with
# cache ≤ original_max so the SHORT factor (rotary init) is always correct → replace forward with a
# static version. CRITICAL: applied HERE, AFTER `import litert_torch` — importing
# transformers.models.phi3 BEFORE litert_torch corrupts litert_converter's MLIR dialect loading
# (bogus `litert_converter.mlir.dialects.tfl ModuleNotFoundError`). Gated so non-Phi runs never import it.
if os.environ.get("PHI3_STATIC_ROPE"):
  import torch as _torch  # noqa: E402
  import transformers.models.phi3.modeling_phi3 as _mp  # noqa: E402

  @_torch.no_grad()
  def _phi3_static_rope_forward(self, x, position_ids):
    inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
    position_ids_expanded = position_ids[:, None, :].float()
    device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
    with _torch.autocast(device_type=device_type, enabled=False):
      freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
      emb = _torch.cat((freqs, freqs), dim=-1)
      cos = emb.cos() * self.attention_scaling
      sin = emb.sin() * self.attention_scaling
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

  _mp.Phi3RotaryEmbedding.forward = _phi3_static_rope_forward
  print("PATCHED Phi3RotaryEmbedding.forward -> static (longrope, post-litert-import)")

# use_jinja_template defaults to True (→ embeds raw jinja, which the runtime's
# minja can't render → broken prompt). Force False so parse_chat_template extracts
# the STRUCTURED prompt_templates (simple ChatML prefixes) the runtime applies —
# matching the official litert-community models.
# "NONE" → no quantization (fp32 reference, for logit-parity isolation of the converter).
quant_recipe = None if quant.upper() in ("NONE", "FP32") else quant

export(
    model=model_id,
    output_dir=out_dir,
    prefill_lengths=[int(os.environ.get("PREFILL", "128"))],
    cache_length=int(os.environ.get("CACHE", "1024")),
    quantization_recipe=quant_recipe,
    use_jinja_template=False,
    experimental_use_mixed_precision=bool(os.environ.get("MIXED")),
    # EXTERNALIZE_EMBEDDER=1 splits the (tied) embedding into its own .litertlm
    # section — the generic equivalent of Gemma's PLE embedding-mmap. Keeps the main
    # TFLiteModel weights section under the iOS ~2GiB single-section mmap limit so
    # big (28-layer 3B) models load on iPhone. No effect on weights/parity.
    externalize_embedder=bool(os.environ.get("EXTERNALIZE_EMBEDDER")),
    trust_remote_code=True,
)
print("EXPORT_DONE")
