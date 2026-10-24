"""Derived M-RoPE for WeVisOne-VL-2B (Qwen3-VL text rotary) at a fixed image grid: the checkpoint's 3-channel
(t, h, w) positions computed from the runtime's 1-D position p, for one image after a fixed P-token prefix.

Copied from decider2bv_work/scripts/mrope_derived.py (which fixes P = 1 and an 8x8 grid) and generalized to a prefix
of P tokens and a GH x GW merged grid. Layout of transformers 5.12.1 `Qwen3VLModel.get_rope_index` +
`get_vision_position_ids` for `<prefix of P tokens> <|image_pad|> x N <|vision_end|> text...` (N = GH * GW):

    p < P              -> (p, p, p)                                         system turn + `<|im_start|>user\\n<|vision_start|>`
    P <= p < P + N     -> (P, P + (p - P) // GW, P + (p - P) % GW)          image tokens, raster order
    p >= P + N         -> (p - D, p - D, p - D),  D = N - max(GH, GW)       `<|vision_end|>`, the question, the answer

The three channels are recombined per rotary frequency exactly as `Qwen3VLTextRotaryEmbedding.apply_interleaved_mrope`
does (mrope_section [24, 20, 20], interleaved: h on frequencies 1, 4, ..., 58, w on 2, 5, ..., 59, t on the rest) and
the halves are concatenated. Everything is element-wise: interval tests are step functions of the float position,
step(x) = relu(x) - relu(x - 1) (1 for x >= 1, 0 for x <= 0 on integers; never clamp, which lowers to RELU_0_TO_1),
the floor division by GW is a sum of GH - 1 steps, and the channel selection is a sum of three products with constant
0/1-masked inv_freq vectors (each product is the exact float32 pos_c * inv_freq[j] or an exact 0).
"""
import torch


def _step(x):
    """1 where the (integer-valued) x >= 1, 0 where x <= 0."""
    return torch.relu(x) - torch.relu(x - 1.0)


class Grid:
    def __init__(self, prefix, gh, gw):
        self.P = int(prefix)
        self.GH = int(gh)
        self.GW = int(gw)
        self.N = self.GH * self.GW
        self.D = self.N - max(self.GH, self.GW)

    def __repr__(self):
        return f"Grid(P={self.P}, GH={self.GH}, GW={self.GW}, N={self.N}, D={self.D})"


def derived_thw(p, grid):
    """p: float tensor of 1-D positions (any shape). Returns (t, h, w) float tensors of the same shape."""
    P, N, GW, GH = float(grid.P), float(grid.N), grid.GW, grid.GH
    start = _step(p - (P - 1.0))                    # p >= P
    after = _step(p - (P + N - 1.0))                # p >= P + N
    img = start - after                             # P <= p < P + N
    before = 1.0 - start                            # p < P
    q = p - P                                       # image slot 0 .. N - 1 (only read where img = 1)
    row = _step(q - float(GW - 1))
    for k in range(2, GH):
        row = row + _step(q - float(k * GW - 1))    # floor(q / GW) for 0 <= q <= N - 1
    col = q - float(GW) * row
    shifted = after * (p - float(grid.D))
    t = before * p + img * P + shifted
    h = before * p + img * (P + row) + shifted
    w = before * p + img * (P + col) + shifted
    return t, h, w


def reference_thw(length, grid):
    """Integer (t, h, w) by the plain piecewise formula (no steps), for the checks."""
    out = []
    for p in range(length):
        if p < grid.P:
            out.append((p, p, p))
        elif p < grid.P + grid.N:
            q = p - grid.P
            out.append((grid.P, grid.P + q // grid.GW, grid.P + q % grid.GW))
        else:
            out.append((p - grid.D,) * 3)
    return out


def channel_masks(n_freq, mrope_section):
    """0/1 masks over the n_freq rotary frequencies selecting the t / h / w channel, in the order of
    Qwen3VLTextRotaryEmbedding.apply_interleaved_mrope (h and w overwrite the t base at interleaved indices)."""
    owner = [0] * n_freq
    for dim, offset in enumerate((1, 2), start=1):
        for j in range(offset, mrope_section[dim] * 3, 3):
            owner[j] = dim
    return [[1.0 if owner[j] == c else 0.0 for j in range(n_freq)] for c in range(3)], owner


def derived_cos_sin(position_ids, inv_freq, mrope_section, grid, attention_scaling=1.0, dtype=torch.float32):
    """position_ids: [B, S] or [C, B, S] 1-D positions (channel 0 is read). Returns cos, sin [B, S, 2 * n_freq]."""
    if position_ids.ndim == 3:
        position_ids = position_ids[0]
    p = position_ids.float()
    t, h, w = derived_thw(p, grid)
    masks, _ = channel_masks(inv_freq.shape[0], mrope_section)
    inv = inv_freq.float()
    inv_t = inv * torch.tensor(masks[0], dtype=torch.float32, device=inv.device)
    inv_h = inv * torch.tensor(masks[1], dtype=torch.float32, device=inv.device)
    inv_w = inv * torch.tensor(masks[2], dtype=torch.float32, device=inv.device)
    freqs = (t[:, :, None] * inv_t[None, None, :] + h[:, :, None] * inv_h[None, None, :]
             + w[:, :, None] * inv_w[None, None, :])
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos() * attention_scaling
    sin = emb.sin() * attention_scaling
    return cos.to(dtype=dtype), sin.to(dtype=dtype)


def install(rotary, grid):
    """Replace one Qwen3VLTextRotaryEmbedding instance's forward with the derived one. Returns an undo function."""
    def forward(x, position_ids):
        return derived_cos_sin(position_ids, rotary.inv_freq, rotary.mrope_section, grid,
                               rotary.attention_scaling, dtype=x.dtype)

    had = "forward" in rotary.__dict__
    previous = rotary.__dict__.get("forward")
    rotary.forward = forward

    def undo():
        if had:
            rotary.forward = previous
        else:
            del rotary.forward
    return undo
