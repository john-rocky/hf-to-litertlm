"""Patch transformers' Cohere2 rope to CohereCompass's Llama-style layout.

Why: North-Micro-Vision's text decoder is Cohere2 in every respect except the rope
pair layout -- Compass rotates half-split pairs (i, i+D/2) with cos/sin =
cat(freqs, freqs) (Llama style), stock Cohere2 rotates interleaved pairs (2i, 2i+1)
via torch.stack(...).flatten with cos/sin = repeat_interleave(freqs, 2). Besides
being a different layout, the stock formulation lowers to a 5-D CONCATENATION and a
BROADCAST_TO, both rejected by the LiteRT GPU delegate. With this patch the exported
graph is the standard Llama rope path and the Compass weights load verbatim.

Import and call patch_cohere2_rope() BEFORE the model is instantiated (prep parity
check and export both do).
"""
import torch
from transformers.models.cohere2 import modeling_cohere2 as _mc2


def _rotate_half(x):
  x1 = x[..., : x.shape[-1] // 2]
  x2 = x[..., x.shape[-1] // 2:]
  return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
  cos = cos.unsqueeze(unsqueeze_dim)
  sin = sin.unsqueeze(unsqueeze_dim)
  q_embed = (q * cos) + (_rotate_half(q) * sin)
  k_embed = (k * cos) + (_rotate_half(k) * sin)
  return q_embed, k_embed


@torch.no_grad()
def _rotary_forward(self, x, position_ids):
  inv_freq_expanded = self.inv_freq[None, :, None].float()          # [1, D/2, 1]
  position_ids_expanded = position_ids[:, None, :].float()          # [bs, 1, seq]
  freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)  # [bs, seq, D/2]
  emb = torch.cat((freqs, freqs), dim=-1)                           # Llama layout
  cos = emb.cos() * self.attention_scaling
  sin = emb.sin() * self.attention_scaling
  return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def patch_cohere2_rope():
  _mc2.rotate_half = _rotate_half
  _mc2.apply_rotary_pos_emb = _apply_rotary_pos_emb
  _mc2.Cohere2RotaryEmbedding.forward = _rotary_forward
  return _mc2
