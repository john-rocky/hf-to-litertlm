"""Shaw-gather rewrite of _SeparateQKVAttention.forward for the LiteRT export.

Why: the remote code computes the relative-position bias as
    rel  = rel_pos_emb(attention_dists[:blk, :blk])        # [blk, blk, d]
    bias = einsum("g h c d, c r d -> g h c r", qf, rel)
Under torch.export the (static) index slice constant-folds, so the lowered
graph embeds a [blk, blk, 128] fp32 constant PER LAYER PER DISTINCT BLOCK
LENGTH PER SIGNATURE — ~8.4 MB each at blk=128, ~0.5 GB total across the
3-signature export (fp32 2.37 GB vs the 1.89 GB the weights account for), and
FC-only dynamic int8 cannot reach them (they feed BATCH_MATMUL).

Fix (Shaw et al.'s standard re-association): dot q with the WHOLE embedding
table first, then gather columns by the index matrix:
    qW   = qf @ rel_pos_emb.weight.T                       # [g, h, c, 1025]
    bias[g,h,c,r] = qW[g,h,c, idx[c,r]]                    # gather, static idx
Same sums re-associated — float noise only; the eager A/B and the tflite
smokes in convert_granite_speech_ctc.py quantify it (argmax-id match + logit
max|diff|). The [1025, 128] table stays a single shared weight.

Everything else in the forward is copied verbatim from the repo's
granite_encoder.py (true-length tail block, pad-to-8 for SDPA, key-pad mask).
apply() swaps the forward on the dynamically-loaded class.
"""

import torch
from torch import nn
from torch.nn.attention import sdpa_kernel


def _shaw_forward(self, hidden_states, attention_dists, mask=None):
    from granite_encoder_dyn import _SDPA_BACKENDS  # resolved by apply()

    hidden_states = self.pre_norm(hidden_states)
    n = hidden_states.shape[1]
    c = self.context_size
    nb_full = n // c
    nr = n % c

    q = self.to_q(hidden_states)
    k = self.to_k(hidden_states)
    v = self.to_v(hidden_states)

    def block_attn(qb, kb, vb, blk, mb):
        h, d = self.num_heads, self.dim_head
        b = qb.shape[0]
        g = b * (qb.shape[1] // blk)
        qf, kf, vf = (t.reshape(g, blk, h, d).transpose(1, 2) for t in (qb, kb, vb))
        idx = attention_dists[:blk, :blk]                       # [blk, blk] i64, static
        qW = torch.matmul(qf, self.rel_pos_emb.weight.transpose(0, 1))  # [g,h,blk,V]
        bias = torch.gather(qW, -1, idx.expand(g, h, blk, blk)) * self.scale

        if mb is not None:
            key_mask = mb.reshape(g, blk)[:, None, None, :].to(bias.dtype)
            bias = bias + key_mask * (-torch.finfo(bias.dtype).max)

        pad = ((blk + 7) & ~7) - blk
        if pad:
            qf = nn.functional.pad(qf, (0, 0, 0, pad))
            kf = nn.functional.pad(kf, (0, 0, 0, pad))
            vf = nn.functional.pad(vf, (0, 0, 0, pad))
            neg = -torch.finfo(bias.dtype).max
            bias = nn.functional.pad(bias, (0, pad, 0, 0), value=neg)
            bias = nn.functional.pad(bias, (0, 0, 0, pad), value=0.0)
        bias = bias.contiguous()
        with sdpa_kernel(_SDPA_BACKENDS):
            of = nn.functional.scaled_dot_product_attention(
                qf, kf, vf, attn_mask=bias, scale=self.scale)
        if pad:
            of = of[:, :, :blk, :]
        return of.transpose(1, 2).reshape(b, -1, h * d)

    outs = []
    L = nb_full * c
    if nb_full > 0:
        mb = mask[:, :L] if mask is not None else None
        outs.append(block_attn(q[:, :L], k[:, :L], v[:, :L], c, mb))
    if nr > 0:
        mb = mask[:, L:n] if mask is not None else None
        outs.append(block_attn(q[:, L:], k[:, L:], v[:, L:], nr, mb))
    out = torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]
    return self.dropout(self.to_out(out[:, :n, :]))


def apply(model):
    """Swap the Shaw forward onto the loaded model's attention class."""
    import sys

    attn_cls = type(model.encoder.layers[0].attn)
    enc_mod = sys.modules[attn_cls.__module__]
    # _shaw_forward imports _SDPA_BACKENDS through this alias, keeping the
    # backend list single-sourced in the repo's own granite_encoder.py.
    sys.modules["granite_encoder_dyn"] = enc_mod
    attn_cls.forward = _shaw_forward
    return model
