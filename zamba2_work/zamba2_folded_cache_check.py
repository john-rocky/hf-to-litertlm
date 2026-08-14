#!/usr/bin/env python3
"""Layer-level gate for the Zamba2 folded-scan CACHE paths.

`patch._selfcheck_folded_zamba2` only covers the cache-free chunked scan. The
exporter also traces (a) chunked prefill that CONTINUES from a stored conv/SSM
state, (b) the single-token decode step, and (c) partially-filled prefill
chunks where the pad guard must make pad positions exact no-ops — on Zamba2
that last one is the min-only dt clamp floor trap (pads decay the state unless
dt is forced to 0 AFTER the clamp; NemotronH lesson).

The reference is the installed transformers `torch_forward` run once over the
whole sequence with no cache. The rewrite must reproduce it while walking the
same sequence in chunks through a stub cache implementing the export contract
(`cat(old, new)[..., -K:]` for conv, plain store for the SSM state).

  PYTHONPATH=~/code/litert-torch <venv>/bin/python zamba2_folded_cache_check.py
"""
import argparse

import torch

from litert_torch.generative.export_hf.model_ext.nemotron_h import (
    patch as np_,
)
from litert_torch.generative.export_hf.model_ext.zamba2 import patch as zp  # noqa: F401 (asserts run on import context entry)
from transformers.models.zamba2 import modeling_zamba2 as mod
from transformers.models.zamba2.configuration_zamba2 import Zamba2Config


class _StubCacheLayer:

  def __init__(self, conv_dim, kernel, heads, head_dim, state, batch=1):
    self._conv = torch.zeros(batch, conv_dim, kernel)
    self._rec = torch.zeros(batch, heads, head_dim, state)
    self.conv_kernel_size = kernel

  @property
  def conv_states(self):
    return [self._conv]

  @property
  def recurrent_states(self):
    return [self._rec]


class _StubCache:
  """The export cache contract, minus the torch.export machinery."""

  def __init__(self, layer_cfg):
    self.layers = [_StubCacheLayer(**layer_cfg)]

  def has_previous_state(self, layer_idx):
    del layer_idx
    return True  # always truthy, as the export cache traces it

  def update_conv_state(self, conv_states, layer_idx=0, *args, **kwargs):
    del args, kwargs
    layer = self.layers[layer_idx]
    merged = torch.cat([layer._conv, conv_states], dim=-1)[
        ..., -layer.conv_kernel_size :
    ]
    layer._conv = merged
    return merged

  def update_recurrent_state(self, recurrent_states, layer_idx=0, *args, **kwargs):
    del args, kwargs
    self.layers[layer_idx]._rec = recurrent_states
    return recurrent_states


def _build(chunk=8, heads=8, state=16, groups=2, hidden=64):
  cfg = Zamba2Config(
      hidden_size=hidden,
      num_hidden_layers=2,
      layers_block_type=["mamba", "hybrid"],
      n_mamba_heads=heads,
      mamba_expand=2,
      mamba_d_state=state,
      mamba_d_conv=4,
      mamba_ngroups=groups,
      chunk_size=chunk,
      use_mamba_kernels=False,
  )
  layer = mod.Zamba2MambaMixer(cfg, layer_idx=0).eval().float()
  cache_cfg = dict(
      conv_dim=layer.conv_dim,
      kernel=layer.conv_kernel_size,
      heads=layer.num_heads,
      head_dim=layer.head_dim,
      state=layer.ssm_state_size,
  )
  return cfg, layer, cache_cfg


def _reference(layer, x):
  """Installed transformers scan, one shot, no cache."""
  with torch.no_grad():
    return mod.Zamba2MambaMixer.torch_forward(layer, x)


def _walk(layer, cache_cfg, x, plan, valid_per_step=None):
  """Run the rewrite over `x` in `plan`-sized steps through the stub cache."""
  cache = _StubCache(cache_cfg)
  out, pos = [], 0
  for i, step in enumerate(plan):
    piece = x[:, pos : pos + step]
    layer._litert_valid = None if valid_per_step is None else valid_per_step[i]
    with torch.no_grad():
      out.append(
          np_._folded_torch_forward_nemotron(layer, piece, cache_params=cache)
      )
    pos += step
  assert pos == x.shape[1], f"plan covers {pos} of {x.shape[1]} tokens"
  return torch.cat(out, dim=1)


def _rel(a, b):
  return ((a - b).abs().max() / b.abs().max()).item()


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--tol", type=float, default=5e-4)
  args = ap.parse_args()
  torch.manual_seed(0)

  cfg, layer, cache_cfg = _build()
  chunk = cfg.chunk_size
  results = []

  # 1. Chunked prefill continuation: the state handed between chunks must
  #    reproduce a single one-shot pass.
  x = torch.randn(1, 4 * chunk, cfg.hidden_size)
  ref = _reference(layer, x)
  results.append(
      ("prefill x4 chunks (continuation)", _rel(_walk(layer, cache_cfg, x, [chunk] * 4), ref))
  )

  # 2. Ragged chunk plan — the engine's planner does not hand out equal chunks,
  #    and a mid-sequence chunk shorter than chunk_size exercises the pad-to-
  #    boundary path WITH a live incoming state.
  results.append(
      ("prefill ragged plan 12/9/6/5", _rel(_walk(layer, cache_cfg, x[:, :32], [12, 9, 6, 5]), ref[:, :32]))
  )

  # 3. Prefill then decode: single-token steps after a chunked prefill.
  x2 = torch.randn(1, chunk * 2 + 3, cfg.hidden_size)
  ref2 = _reference(layer, x2)
  results.append(
      ("prefill 2 chunks + 3 decode steps", _rel(_walk(layer, cache_cfg, x2, [chunk, chunk, 1, 1, 1]), ref2))
  )

  # 4. Decode from the very first token (the length-1 prefill signature also
  #    lands here, since has_previous_state is always truthy).
  results.append(
      ("all-decode walk (16 steps)", _rel(_walk(layer, cache_cfg, x2[:, :16], [1] * 16), ref2[:, :16]))
  )

  # 5. Pad guard: a partially filled chunk must leave the same state behind as
  #    the same real tokens with no pads — on Zamba2 this is the min-only
  #    clamp floor trap. Feed [7 real + 5 pad] then 9 real, and compare the 7
  #    real outputs and the following 9 against the one-shot run.
  real, pad = 7, 5
  xg = torch.zeros(1, real + pad + 9, cfg.hidden_size)
  xg[:, :real] = x2[:, :real]
  xg[:, real + pad :] = x2[:, real : real + 9]
  valid_chunk = torch.zeros(1, real + pad)
  valid_chunk[:, :real] = 1.0
  got = _walk(
      layer,
      cache_cfg,
      xg,
      [real + pad, 9],
      valid_per_step=[valid_chunk, torch.ones(1, 9)],
  )
  guarded = torch.cat([got[:, :real], got[:, real + pad :]], dim=1)
  results.append(
      ("underfilled chunk (pad guard)", _rel(guarded, ref2[:, : real + 9]))
  )

  worst = 0.0
  for name, rel in results:
    ok = "OK  " if rel <= args.tol else "FAIL"
    worst = max(worst, rel)
    print(f"{ok} {name:36s} relative max|diff| = {rel:.2e}")
  print(f"\nworst = {worst:.2e} (tol {args.tol:.0e})")
  return 0 if worst <= args.tol else 1


if __name__ == "__main__":
  raise SystemExit(main())
