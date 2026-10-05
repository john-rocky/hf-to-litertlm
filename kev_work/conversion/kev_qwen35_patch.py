# Adapted from a patch for the Qwen3.5 export in litert-torch (changed: used without the patch registry, and the
# text model reads the valid mask from `self._kev_valid`, set by kev_graph.py, instead of `input_ids != 0`).
# Copyright 2026 The LiteRT Torch Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Patch for Qwen3.5 (GatedDeltaNet hybrid linear-attention), transformers >= 5.14.

The cache side needs no model patch: `layer_types[i] == "linear_attention"` routes to
LiteRTLMLinearAttentionCacheLayer, whose always-truthy `has_previous_state` makes
`create_recurrent_attention_mask` return None (no data-dependent guard is reached) and
statically selects the state-continuation branches — seq_len==1 signatures trace the
fused single-step path (`causal_conv1d_update` rolls the conv state in-place),
seq_len>1 signatures trace the chunked path with the cached conv/recurrent state
consumed, so multi-chunk prefill composes.

What DOES need a patch is the prefill-pad guard: the LiteRT-LM engine runs prefill
chunks PARTIALLY FILLED (the chunk planner zero-pads the remainder chunk), and
`apply_mask_to_padding_states` is a no-op at batch size 1 — so pad tokens would
poison the conv window and the gated-delta recurrent state (the gated-delta analog
of the LFM2 ShortConv / Granite-4-h mamba prefill-pad bug). The guard:

  1. valid mask = `input_ids != 0` (the engine zero-fills pad tokens), stashed
     directly on every GatedDeltaNet instance by the patched text model (there is
     no mask route to the mixer: `create_recurrent_attention_mask` yields None);
  2. pad positions' hidden states zeroed before the projections (in_proj_* have
     no bias, so q/k/v/z/b contribute nothing for pads);
  3. pad positions' `a` forced to -30 pre-softplus -> g = -A*softplus(a+dt_bias)
     ~ 0 -> exp(g) ~ 1: the recurrent-state decay is an IDENTITY step on pads
     (zeroed input alone would still decay the state by softplus(dt_bias));
  4. pad positions re-zeroed after the conv (the causal conv smears valid tokens
     into pad positions, giving them nonzero k/v; k must be 0 so pads inject
     nothing into the recurrent state);
  5. the stored conv window is gathered ending at the last VALID column via an
     in-graph one-hot matmul (the upstream LFM2 0.9.2 technique), instead of the
     chunk's last K columns (which are pads).

The guarded `forward` is generated from the installed transformers source with
anchored string replacements (exec'd); anchors are asserted so a modeling change
fails loudly instead of silently skipping the guard.
"""

import inspect
import textwrap

import torch
from transformers.models.qwen3_5 import modeling_qwen3_5


class PatchedQwen3_5TextRotaryEmbedding(modeling_qwen3_5.Qwen3_5TextRotaryEmbedding):
  """Text-only 1D RoPE. For text, all M-RoPE channels carry identical position
  ids, and the interleaved-section overwrite is then the identity — but the
  stock forward's `expand(3, ...)` and strided slice-assignment lower to
  BROADCAST_TO / STABLEHLO_PAD (plus int64 index math), all rejected by GPU
  delegates. Compute the mathematically identical plain-RoPE form instead."""

  def forward(self, x, position_ids):
    if position_ids.ndim == 3:
      position_ids = position_ids[0]
    freqs = position_ids[:, :, None].float() * self.inv_freq[None, None, :].float()
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos() * self.attention_scaling
    sin = emb.sin() * self.attention_scaling
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class PatchedQwen3_5TextModel(modeling_qwen3_5.Qwen3_5TextModel):
  """Export-friendly Qwen3.5 text model (class swap; persists on the instance)."""

  def forward(self, input_ids=None, position_ids=None, **kwargs):
    # Engine pads prefill chunks with token id 0; derive the per-position valid
    # mask here and stash it directly on every GatedDeltaNet instance — the
    # linear-attention layers receive attention_mask=None during export (see
    # module docstring), so the guarded forward reads `self._litert_valid`.
    # Kev change (a): the graph wrapper sets `self._kev_valid` ([1, L] float32,
    # 1 = real token, 0 = pad; None = no guard). Kev pads with 248044 and token
    # id 0 ("!") is a real token, so `input_ids != 0` cannot mark pads here.
    valid = getattr(self, "_kev_valid", None)
    for module in self.modules():
      if isinstance(module, PatchedQwen3_5GatedDeltaNet):
        module._litert_valid = valid
    # The stock forward turns 2-D position ids into the 4-channel M-RoPE form
    # via `expand(4, ...)` — a BROADCAST_TO the GPU delegates reject. Build the
    # same shape with a PACK instead (channels are identical for text; the
    # patched rotary only reads channel 0 anyway).
    if position_ids is not None and position_ids.ndim == 2:
      position_ids = torch.stack(
          [position_ids, position_ids, position_ids, position_ids], dim=0)
    return super().forward(input_ids=input_ids, position_ids=position_ids,
                           **kwargs)


def _litert_interleave_heads(x, r):
  """`repeat_interleave(r, dim=2)` for [b, s, h, d] without rank-5 tensors or
  BROADCAST_TO: fold the batch dim (b == 1 in this export) so the copy axis
  fits in rank 4, materialize the copies with concat, and reinterleave with
  rank-4 reshapes. Flattening (h, r) h-major reproduces repeat_interleave's
  [h0, h0, h1, h1, ...] order exactly (bitwise, checked vs the stock op)."""
  b, s, h, d = x.shape
  y = x.reshape(s, h, 1, d)
  y = torch.cat([y] * r, dim=2)
  return y.reshape(b, s, h * r, d)


def _unwrap_forward():
  """`Qwen3_5GatedDeltaNet.forward` is wrapped by `@force_accelerate_hooks`
  (no functools.wraps, so `inspect.unwrap` can't see through it); the real
  function rides in the wrapper's closure."""
  wrapped = modeling_qwen3_5.Qwen3_5GatedDeltaNet.forward
  for cell in wrapped.__closure__ or ():
    value = cell.cell_contents
    if callable(value) and getattr(value, "__name__", "") == "forward":
      return value
  raise AssertionError("could not unwrap Qwen3_5GatedDeltaNet.forward")


def _build_guarded_forward():
  """Vendored `Qwen3_5GatedDeltaNet.forward` with the pad guard, generated from
  the installed source via anchored replacements."""
  src = inspect.getsource(_unwrap_forward())
  src = textwrap.dedent(src)
  # Drop decorator lines (accelerate-hook plumbing; inert for export and its
  # factory name need not resolve in the exec namespace).
  lines = src.split("\n")
  first_def = next(i for i, l in enumerate(lines) if l.startswith("def "))
  src = "\n".join(lines[first_def:])

  # NOTE: all anchors/replacements are in POST-DEDENT coordinates (the file's
  # indentation minus 4: method body at 4, else-branch bodies at 8/12).

  # (1)+(2) guard activation + input zeroing. The valid mask only applies on the
  # prefill signatures (seq_len > 1); decode must NOT mask (a legitimately
  # generated token id 0 would be zeroed, and the single-step branch never sees
  # engine pads).
  anchor = ("    hidden_states = apply_mask_to_padding_states"
            "(hidden_states, attention_mask)\n")
  assert src.count(anchor) == 1, "pad-guard anchor 1 not found"
  src = src.replace(
      anchor,
      anchor
      + "    valid = getattr(self, '_litert_valid', None)\n"
      "    if valid is not None and hidden_states.shape[1] > 1:\n"
      "        valid = valid.to(hidden_states.dtype)\n"
      "        hidden_states = hidden_states * valid[..., None]\n"
      "    else:\n"
      "        valid = None\n",
  )

  # (3) decay identity on pads (pre-softplus -30 -> softplus ~ 0 -> exp(g) ~ 1).
  # `a` from zeroed input is 0, and softplus(0 + dt_bias) != 0 would still decay
  # the recurrent state at every pad position.
  anchor = "    a = self.in_proj_a(hidden_states)\n"
  assert src.count(anchor) == 1, "pad-guard anchor 2 not found"
  src = src.replace(
      anchor,
      anchor
      + "    if valid is not None:\n"
      "        a = a * valid[..., None] + (1.0 - valid[..., None]) * (-30.0)\n",
  )

  # (4) post-conv zeroing (q/k/v for the delta rule). The causal conv smears
  # valid tokens into pad positions; k must be 0 there so pads inject nothing.
  anchor = ("    mixed_qkv = mixed_qkv.transpose(1, 2)\n"
            "    query, key, value = torch.split(\n")
  assert src.count(anchor) == 1, "pad-guard anchor 3 not found"
  src = src.replace(
      anchor,
      "    mixed_qkv = mixed_qkv.transpose(1, 2)\n"
      "    if valid is not None:\n"
      "        mixed_qkv = mixed_qkv * valid[..., None]\n"
      "    query, key, value = torch.split(\n",
  )

  # (5) conv window gathered at the last VALID column (one-hot matmul). The
  # stock store keeps the chunk's last K columns, which are pads. On the
  # continuation path mixed_qkv is [cached K-wide context | chunk], which
  # `_w - valid.shape[-1]` accounts for.
  anchor = (
      "        if cache_params is not None:\n"
      "            new_conv_state = F.pad(mixed_qkv, "
      "(self.conv_kernel_size - mixed_qkv.shape[-1], 0))\n"
      "            cache_params.update_conv_state(new_conv_state, "
      "self.layer_idx)\n"
  )
  assert src.count(anchor) == 1, "pad-guard anchor 4 not found"
  src = src.replace(
      anchor,
      "        if cache_params is not None:\n"
      "            if valid is not None:\n"
      "                _w = mixed_qkv.shape[-1]\n"
      "                _K = self.conv_kernel_size\n"
      "                _padded = F.pad(mixed_qkv, (_K, 0))\n"
      # ⚠ KEEP THE BATCH DIMENSION, and stay off int64.
      # int64 index math is rejected outright by every LiteRT GPU delegate.
      # A RANK-0 scalar is worse than rejected — it is silently wrong:
      # `rank0 + arange` disagrees with CPU on the Metal/CL delegates, the
      # selector built from it comes back ALL ZERO, and the stored conv
      # state is then 0.0 in every layer while CPU has real values
      # (measured on the granite twin of this guard, 2026-08-13, with
      # tools/backend_diff; `sum(-1, keepdim=True)` makes it exact).
      "                _valid_len = valid.sum(-1, keepdim=True).to(_padded.dtype)\n"
      "                _base = float(_w - valid.shape[-1]) + _valid_len\n"
      "                _idx = _base + torch.arange(_K, dtype=_padded.dtype, "
      "device=_padded.device)\n"
      "                _cols = torch.arange(_w + _K, dtype=_padded.dtype, "
      "device=_padded.device)\n"
      "                _onehot = torch.relu(1.0 - torch.abs(\n"
      "                    _cols[None, :, None] - _idx[:, None, :]))\n"
      "                new_conv_state = torch.matmul(_padded, _onehot)\n"
      "            else:\n"
      "                new_conv_state = F.pad(mixed_qkv, "
      "(self.conv_kernel_size - mixed_qkv.shape[-1], 0))\n"
      "            cache_params.update_conv_state(new_conv_state, "
      "self.layer_idx)\n",
  )

  # (6) rank-4 head interleave (4B+: num_v_heads > num_k_heads; ratio 1 on the
  # 0.8B never traces this branch). The stock `repeat_interleave(r, dim=2)` on
  # rank-4 q/k lowers through rank-5 unsqueeze->expand->reshape and the GPU
  # delegate rejects it ("RESHAPE: Tensor dimensions must be less than 5").
  anchor = (
      "    if self.num_v_heads // self.num_k_heads > 1:\n"
      "        query = query.repeat_interleave("
      "self.num_v_heads // self.num_k_heads, dim=2)\n"
      "        key = key.repeat_interleave("
      "self.num_v_heads // self.num_k_heads, dim=2)\n")
  assert src.count(anchor) == 1, "head-interleave anchor not found"
  src = src.replace(
      anchor,
      "    if self.num_v_heads // self.num_k_heads > 1:\n"
      "        query = _litert_interleave_heads("
      "query, self.num_v_heads // self.num_k_heads)\n"
      "        key = _litert_interleave_heads("
      "key, self.num_v_heads // self.num_k_heads)\n",
  )

  namespace = dict(vars(modeling_qwen3_5))
  namespace["torch"] = torch
  namespace["_litert_interleave_heads"] = _litert_interleave_heads
  exec(compile(src, "<qwen3_5_pad_guard>", "exec"), namespace)  # pylint: disable=exec-used
  return namespace["forward"]


def _chunk_rule_constants(chunk_size, dtype, device):
  """Graph constants built from python data: `torch.eye` / `torch.triu(ones)`
  created inside a traced function lower to STABLEHLO_IOTA patterns that no
  released TFLite kernel set registers; data-built tensors lift as constants.
  Returns (eye, tril-including-diagonal float mask, strictly-lower float mask)."""
  eye = torch.tensor(
      [[1.0 if i == j else 0.0 for j in range(chunk_size)]
       for i in range(chunk_size)], dtype=dtype, device=device)
  tril0_f = torch.tensor(
      [[1.0 if j <= i else 0.0 for j in range(chunk_size)]
       for i in range(chunk_size)], dtype=dtype, device=device)
  tril1_f = torch.tensor(
      [[1.0 if j < i else 0.0 for j in range(chunk_size)]
       for i in range(chunk_size)], dtype=dtype, device=device)
  return eye, tril0_f, tril1_f


def _rank4_chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=64,
                                  initial_state=None, output_final_state=False,
                                  use_qk_l2norm_in_kernel=False, **kwargs):
  """Rank-<=4 re-expression of `torch_chunk_gated_delta_rule` for batch 1.

  The reference kernel chunks tensors to [B, H, n_chunks, 64, dim] — rank 5 —
  and every LiteRT GPU delegate caps tensors at rank 4 (the graph then fails
  engine creation on GPU, Metal and OpenCL alike). Our static export is always
  batch 1, so the batch dim is folded away and every intermediate stays rank 4
  ([H, n, c, d]). Numerical equivalence to the reference is asserted at patch
  build time (see below), so a modeling change fails loudly."""
  initial_dtype = query.dtype
  if use_qk_l2norm_in_kernel:
    query = modeling_qwen3_5.l2norm(query, dim=-1, eps=1e-6)
    key = modeling_qwen3_5.l2norm(key, dim=-1, eps=1e-6)
  # [1, L, H, D] -> [H, L, D]
  q, k, v = [x.squeeze(0).transpose(0, 1).contiguous().to(torch.float32)
             for x in (query, key, value)]
  b_t, g_t = [x.squeeze(0).transpose(0, 1).contiguous().to(torch.float32)
              for x in (beta, g)]
  num_heads, seq_len, k_dim = k.shape
  v_dim = v.shape[-1]
  pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
  if pad_size:
    # PAD here is mis-EXECUTED by the ML Drift GPU delegate (numerics, not
    # rejection): the rank-3 [H, T, 1] pad lands shifted one row on the head
    # axis (head h reads head h-1, head 0 reads zeros) and the [H, T, D] pad
    # scrambles rows in another pattern, fp16 and fp32 alike. CONCATENATION
    # with a zeros constant is exact on the same shapes, so express the tail
    # padding as concat (rank-2 PAD is additionally rejected outright).
    zero_qk = torch.zeros(num_heads, pad_size, k_dim,
                          dtype=q.dtype, device=q.device)
    zero_v = torch.zeros(num_heads, pad_size, v_dim,
                         dtype=v.dtype, device=v.device)
    zero_t = torch.zeros(num_heads, pad_size, dtype=g_t.dtype,
                         device=g_t.device)
    q = torch.cat([q, zero_qk], dim=1)
    k = torch.cat([k, zero_qk], dim=1)
    v = torch.cat([v, zero_v], dim=1)
    b_t = torch.cat([b_t, zero_t], dim=-1)
    g_t = torch.cat([g_t, zero_t], dim=-1)
  total = seq_len + pad_size
  n = total // chunk_size
  q = q * (q.shape[-1] ** -0.5)
  v_beta = v * b_t.unsqueeze(-1)
  k_beta = k * b_t.unsqueeze(-1)
  # chunk: [H, n, c, d]
  q, k, v, k_beta, v_beta = [
      x.reshape(num_heads, n, chunk_size, x.shape[-1])
      for x in (q, k, v, k_beta, v_beta)]
  g_t = g_t.reshape(num_heads, n, chunk_size)
  eye, tril0_f, tril1_f = _chunk_rule_constants(chunk_size, torch.float32,
                                                q.device)
  g_c = g_t.cumsum(dim=-1)
  # tril()/masked_fill lower to SELECT_V2 with a broadcast condition, which GPU
  # delegates reject — express both as multiplies with float mask constants.
  diff = (g_c.unsqueeze(-1) - g_c.unsqueeze(-2)) * tril0_f
  decay_mask = diff.exp().float() * tril0_f
  attn = -((k_beta @ k.transpose(-1, -2)) * decay_mask) * tril1_f
  # The reference's in-place row update (attn[..., i, :i] = ...) lowers to
  # index_put = BROADCAST_TO + SELECT storms; build the inverted matrix by
  # stacking rows instead (PACK/CONCAT are delegate-friendly).
  rows = [attn[..., 0, :]]
  for i in range(1, chunk_size):
    prev = torch.stack(rows, dim=-2)                    # [H, n, i, c]
    row_full = attn[..., i, :]                          # [H, n, c]
    row_lead = row_full[..., :i]                        # [H, n, i]
    sub = prev[..., :i]                                 # [H, n, i, i]
    upd = row_lead + (row_lead.unsqueeze(-1) * sub).sum(-2)
    rows.append(torch.cat([upd, row_full[..., i:]], dim=-1))
  attn = torch.stack(rows, dim=-2) + eye                # [H, n, c, c]
  v2 = attn @ v_beta
  k_cumdecay = attn @ (k_beta * g_c.exp().unsqueeze(-1))
  if initial_state is None:
    state = torch.zeros(num_heads, k_dim, v_dim, dtype=v2.dtype, device=v2.device)
  else:
    state = initial_state.reshape(-1, initial_state.shape[-2],
                                  initial_state.shape[-1]).to(v2)
  outs = []
  for i in range(n):
    q_i, k_i, v_i = q[:, i], k[:, i], v2[:, i]                       # [H, c, d]
    attn_i = q_i @ k_i.transpose(-1, -2) * decay_mask[:, i]
    v_prime = k_cumdecay[:, i] @ state
    v_new = v_i - v_prime
    attn_inter = (q_i * g_c[:, i, :, None].exp()) @ state
    outs.append(attn_inter + attn_i @ v_new)
    state = (state * g_c[:, i, -1, None, None].exp()
             + (k_i * (g_c[:, i, -1, None] - g_c[:, i]).exp()[..., None])
             .transpose(-1, -2) @ v_new)
  last_state = state.unsqueeze(0) if output_final_state else None
  out = torch.stack(outs, dim=1)                        # [H, n, c, dv]
  out = out.reshape(num_heads, total, v_dim)[:, :seq_len]
  out = out.unsqueeze(0).transpose(1, 2).contiguous().to(initial_dtype)
  return out, last_state


def _assert_rank4_kernel_equivalence():
  """Build-time gate: the rank-4 kernel must match the installed reference
  bit-for-tolerance on random inputs, so transformers drift fails loudly."""
  torch.manual_seed(0)
  H, L, DK, DV, C = 3, 10, 8, 8, 4
  q = torch.randn(1, L, H, DK)
  k = torch.randn(1, L, H, DK)
  v = torch.randn(1, L, H, DV)
  g = -torch.rand(1, L, H)
  b = torch.rand(1, L, H)
  s0 = torch.randn(1, H, DK, DV)
  ref_out, ref_state = modeling_qwen3_5.torch_chunk_gated_delta_rule(
      q, k, v, g, b, chunk_size=C, initial_state=s0, output_final_state=True,
      use_qk_l2norm_in_kernel=True)
  got_out, got_state = _rank4_chunk_gated_delta_rule(
      q, k, v, g, b, chunk_size=C, initial_state=s0, output_final_state=True,
      use_qk_l2norm_in_kernel=True)
  assert torch.allclose(ref_out, got_out, atol=1e-5), (
      "rank-4 chunk kernel diverged from reference (out)")
  assert torch.allclose(ref_state, got_state, atol=1e-5), (
      "rank-4 chunk kernel diverged from reference (state)")


_assert_rank4_kernel_equivalence()


class PatchedQwen3_5GatedDeltaNet(modeling_qwen3_5.Qwen3_5GatedDeltaNet):
  forward = _build_guarded_forward()

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # __init__ binds the module-level torch fallback per instance; rebind to the
    # rank-4 rewrite (only when the torch fallback was selected — a real FLA
    # kernel install means no tracing concern and no swap).
    if self.chunk_gated_delta_rule is modeling_qwen3_5.torch_chunk_gated_delta_rule:
      self.chunk_gated_delta_rule = _rank4_chunk_gated_delta_rule
