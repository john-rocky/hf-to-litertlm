"""Derived M-RoPE for decider-2b-vision at a fixed image grid: the author's 3-channel (t, h, w) positions are computed
inside the graph from the runtime's 1-D position p (image first, one GxG image).

Layout of the author's positions on this path (transformers 5.17.0 Qwen3_5Model.get_rope_index +
get_vision_position_ids, verified per row by scripts/test_mrope_derived.py): the row is
`<|vision_start|> <|image_pad|> x N <|vision_end|> text...`, so with a merged grid of GH x GW tokens (N = GH * GW)

    p = 0                 -> (0, 0, 0)                                        <|vision_start|>
    1 <= p <= N           -> (1, 1 + (p - 1) // GW, 1 + (p - 1) % GW)         image tokens, raster order
    p >= N + 1            -> (p - D, p - D, p - D),  D = N - max(GH, GW)      <|vision_end|> and all text after it

G = 256: GH = GW = 8, N = 64, D = 56. The three channels are then recombined per frequency index exactly as
Qwen3_5TextRotaryEmbedding.recomposition_frequencies does (interleaved sections: h on indices 1, 4, ..., 3*sh - 2,
w on 2, 5, ..., 3*sw - 1, t on the rest), and the halves are concatenated.

Everything is element-wise so the graph stays in the op set the qwen35 patch was built for: the interval tests are
step functions of the float position (step(x) = 1 for x >= 1, 0 for x <= 0 on integers), the floor division by GW is
a sum of GH - 1 steps, and the channel selection is a sum of three products with constant 0/1-masked inv_freq vectors
(each product is either the exact float32 product pos_c * inv_freq[j] or an exact 0, so the selected frequency is
bit-identical to the HF path). No GATHER, BROADCAST_TO, strided slice assignment or int64 index math.

`install(patch_module)` replaces `PatchedQwen3_5TextRotaryEmbedding.forward` of the qwen35 litert-torch patch for the
lifetime of the calling process only (the export driver calls it before `export()`); nothing on disk changes.

Step form (STEP_FORM): round 4 writes step(x) = relu(x) - relu(x - 1) ('relu_diff'). Rounds 2 and 3 exported
clamp(x, 0, 1) ('clamp'), which the converter folds into RELU_0_TO_1, an op the WebGPU delegate does not support
(round 3 engine-creation refusal). The two forms are equal, bit for bit, on integer-valued float32 x with |x| < 2**24
(every position here); set_step_form('clamp') reproduces the round-2/3 graphs.
"""
import torch

G_IMG = 256
PATCH = 16
MERGE = 2
GH = GW = G_IMG // PATCH // MERGE          # 8 x 8 merged grid
N_IMG = GH * GW                            # 64 image tokens at positions 1..64
OFFSET = N_IMG - max(GH, GW)               # 56: text after the image resumes at p - 56 (65 -> 9)


def channel_masks(n_freq, mrope_section):
    """0/1 masks over the n_freq rotary frequencies selecting the t / h / w channel, in the order of
    Qwen3_5TextRotaryEmbedding.recomposition_frequencies (h and w overwrite the t base at interleaved indices)."""
    owner = [0] * n_freq
    for dim, offset in enumerate((1, 2), start=1):
        for j in range(offset, mrope_section[dim] * 3, 3):
            owner[j] = dim
    return [[1.0 if owner[j] == c else 0.0 for j in range(n_freq)] for c in range(3)], owner


def _step_clamp(x):
    """Rounds 2-3: min(max(x, 0), 1); the converter folds it into RELU_0_TO_1."""
    return torch.clamp(x, min=0.0, max=1.0)


def _step_relu_diff(x):
    """Round 4: relu(x) - relu(x - 1); RELU and SUB only, no RELU_0_TO_1 / MINIMUM / MAXIMUM pattern to fold."""
    return torch.relu(x) - torch.relu(x - 1.0)


STEP_FORMS = {'clamp': _step_clamp, 'relu_diff': _step_relu_diff}
STEP_FORM = 'relu_diff'


def set_step_form(name):
    """Select the step form for this process (module global, read at trace time)."""
    global STEP_FORM
    assert name in STEP_FORMS, name
    STEP_FORM = name


def _step(x):
    """1 where the (integer-valued) x >= 1, 0 where x <= 0, in the selected STEP_FORM."""
    return STEP_FORMS[STEP_FORM](x)


def derived_thw(p):
    """p: float tensor of 1-D positions (any shape). Returns (t, h, w) float tensors of the same shape."""
    after = _step(p - float(N_IMG))                 # p >= N + 1
    img = _step(p) - after                          # 1 <= p <= N
    q = p - 1.0                                     # image slot 0 .. N - 1
    row = _step(q - float(GW - 1))
    for k in range(2, GH):
        row = row + _step(q - float(k * GW - 1))    # floor(q / GW) for 0 <= q <= N - 1
    col = q - float(GW) * row
    shifted = after * (p - float(OFFSET))
    t = img + shifted
    h = img * (1.0 + row) + shifted
    w = img * (1.0 + col) + shifted
    return t, h, w


def derived_cos_sin(position_ids, inv_freq, mrope_section, attention_scaling=1.0, dtype=torch.float32):
    """position_ids: [B, S] or [C, B, S] 1-D positions (channel 0 is read, all channels carry the same value).
    Returns cos, sin of shape [B, S, 2 * len(inv_freq)] in `dtype`."""
    if position_ids.ndim == 3:
        position_ids = position_ids[0]
    p = position_ids.float()                        # [B, S]
    t, h, w = derived_thw(p)
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


def derived_forward(self, x, position_ids):
    """Drop-in forward for the qwen35 patch's PatchedQwen3_5TextRotaryEmbedding (same signature and outputs)."""
    return derived_cos_sin(position_ids, self.inv_freq, self.mrope_section, self.attention_scaling, dtype=x.dtype)


def install(patch_module):
    """Swap the patched rotary's forward for the derived one (process-local). Returns the replaced function."""
    cls = patch_module.PatchedQwen3_5TextRotaryEmbedding
    previous = cls.forward
    cls.forward = derived_forward
    return previous
