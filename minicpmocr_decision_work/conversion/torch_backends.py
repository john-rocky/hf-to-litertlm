"""The separation test's backends: the author's float32 torch modules behind the host's tower and decide interfaces. The host's own
tables go in (the patches, the bilinear position rows, the rotary tables, the embeddings), so everything the host computes is
under test and no graph is. Runs in the reference environment (transformers 5.10.4, torch 2.8.0), with the same depthwise patch as the reference."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class TorchTower:
    max_n = None

    def __init__(self, visual):
        self.v = visual

    def __call__(self, patches, pos, cos, sin) -> np.ndarray:
        v, n = self.v, patches.shape[0]
        h = v.patch_embed(torch.from_numpy(patches)) + torch.from_numpy(pos)
        cu = torch.tensor([0, n], dtype=torch.long)
        pe = (torch.from_numpy(cos), torch.from_numpy(sin))
        for blk in v.blocks:
            h = blk(h, cu_seqlens=cu, position_embeddings=pe)
        return v.merger(h).float().numpy()


class _HostRotary(nn.Module):
    """Stands where Qwen3_5TextRotaryEmbedding stands and returns the host's tables."""

    def __init__(self):
        super().__init__()
        self.tables = None

    def forward(self, x, position_ids=None):
        return self.tables


class TorchDecide:
    max_len = None

    def __init__(self, language_model):
        self.lm = language_model
        self.rot = _HostRotary()
        self.lm.rotary_emb = self.rot

    def __call__(self, embeds, cos, sin, planes) -> np.ndarray:
        T = embeds.shape[0]
        self.rot.tables = (torch.from_numpy(cos)[None], torch.from_numpy(sin)[None])
        out = self.lm(inputs_embeds=torch.from_numpy(embeds)[None], attention_mask=torch.ones(1, T, dtype=torch.long),
                      position_ids=torch.from_numpy(planes[:, None, :]).long(), use_cache=False)
        return out.last_hidden_state[0].float().numpy()
