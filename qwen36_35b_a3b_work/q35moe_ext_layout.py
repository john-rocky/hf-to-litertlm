"""The `moe` op's weight layout from the Qwen3.5 MoE checkpoint (shared by q35moe_ext.py and graft_moe.py).

HF transformers 5.22.2 Qwen3_5MoeExperts: gate, up = linear(x, gate_up_proj[e]).chunk(2, dim=-1) -> gate = rows [:I] of
gate_up_proj[e] ([2I, D]), up = rows [I:]; out = linear(act(gate) * up, down_proj[e]) with down_proj[e] [D, I].
The op (as LiteRT's moe kernels read it): gate / ff1 [I, E, 1, D], linear [D, E, 1, I], row = out x E + expert.
"""
import torch

OHWI_SUFFIXES = (".mlp.experts.w_gate", ".mlp.experts.w_ff1", ".mlp.experts.w_lin")


def ohwi_from_checkpoint(suffix, gate_up_proj, down_proj):
  """gate_up_proj [E, 2I, D], down_proj [E, D, I] (any float dtype) -> the op's fp32 tensor for `suffix`."""
  if suffix.endswith("w_lin"):
    w = down_proj
  else:
    i_dim = gate_up_proj.shape[1] // 2
    w = gate_up_proj[:, :i_dim, :] if suffix.endswith("w_gate") else gate_up_proj[:, i_dim:, :]
  e, o, n = w.shape
  return w.to(torch.float32).permute(1, 0, 2).contiguous().reshape(o, e, 1, n)
