"""Row lengths and positions: the bucket of a record, its three M-RoPE position planes and the cos / sin tables the row
graph takes as inputs (the graph holds no COS / SIN / position table of its own).

Buckets: one row graph per length L in BUCKETS; a record of T tokens runs on the smallest L >= T, right-padded. A
record longer than the largest graph is refused (`RequestTooLong`), never truncated.

Position planes [3, T] (t, h, w) of the author's model (transformers Qwen3_5Model.get_rope_index; closed form recorded
with the reference, P = token_offset = the <|vision_start|> index, one picture of H x W merged tokens, N = H * W):
    text-only record: (i, i, i)
    i <= P -> (i, i, i);  picture token k at P+1+k -> (P+1, P+1 + k // W, P+1 + k % W);
    i >= P+N+1 -> i - (N - max(H, W)) on all three planes
Pad positions (right of the record) continue the planes' maximum + 1, + 2, ... on all three planes; their outputs are
never read and the graph's `valid` mask hides them as keys.

cos / sin (the author's Qwen3_5TextRotaryEmbedding: default rope, theta 1e7, rotary dim 64 of the 256-wide heads,
attention scaling 1, interleaved mrope_section [11, 11, 10]):
    inv_freq[j] = 1 / theta^(2j / 64)                    j = 0..31, every step in float32 as the author's code
    angle[p][i, j] = float32(inv_freq[j] * pos[p][i])    p = t, h, w
    interleave: j % 3 == 1 and j < 33 -> h plane, j % 3 == 2 and j < 30 -> w plane, every other j -> t plane
    cos = cos(cat(angle, angle)), sin = sin(cat(angle, angle))      float32 [1, L, 64]
The cosine and sine are taken in float64 on the float32 angle and rounded to float32 (torch's float32 cos / sin sit
within 1 ulp of these)."""
from __future__ import annotations

import numpy as np

from .errors import RequestTooLong

BUCKETS = (256, 512, 1024, 2048, 4096)
ROTARY_DIM = 64
ROPE_THETA = 10000000.0
MROPE_SECTION = (11, 11, 10)


def pick_length(tokens: int, buckets=BUCKETS) -> int:
    """The smallest row graph length that holds `tokens`; RequestTooLong past the largest."""
    for length in sorted(buckets):
        if tokens <= length:
            return length
    raise RequestTooLong(f"the record has {tokens} tokens; the largest row graph holds {max(buckets)}")


def text_positions(tokens: int) -> np.ndarray:
    """[3, T] int64 planes of a record without a picture: (i, i, i)."""
    return np.broadcast_to(np.arange(tokens, dtype=np.int64), (3, tokens)).copy()


def image_positions(tokens: int, token_offset: int, h: int, w: int) -> np.ndarray:
    """[3, T] int64 planes of a record with one picture of h x w merged tokens right after the prefix."""
    p, n = int(token_offset), int(h) * int(w)
    if not p + 1 + n < tokens:
        raise ValueError(f"a picture of {n} tokens at {p} does not fit a record of {tokens} tokens")
    pos = text_positions(tokens)
    k = np.arange(n, dtype=np.int64)
    pos[0, p + 1:p + 1 + n] = p + 1
    pos[1, p + 1:p + 1 + n] = p + 1 + k // w
    pos[2, p + 1:p + 1 + n] = p + 1 + k % w
    pos[:, p + n + 1:] = np.arange(p + n + 1, tokens, dtype=np.int64) - (n - max(h, w))
    return pos


def record_positions(tokens: int, token_offset: int | None = None, image_grid=None) -> np.ndarray:
    """[3, T] planes of a record (text-only when `image_grid` is None)."""
    if image_grid is None:
        return text_positions(tokens)
    return image_positions(tokens, token_offset, *image_grid)


def padded_positions(pos3: np.ndarray, length: int) -> np.ndarray:
    """[3, L]: the record's planes, then maximum + 1, + 2, ... on all three planes for the pad positions."""
    n = pos3.shape[1]
    if n > length:
        raise ValueError(f"{n} positions do not fit length {length}")
    out = np.empty((3, length), dtype=np.int64)
    out[:, :n] = pos3
    out[:, n:] = int(pos3.max()) + 1 + np.arange(length - n, dtype=np.int64)[None, :]
    return out


def inv_freq(theta: float = ROPE_THETA, rotary_dim: int = ROTARY_DIM) -> np.ndarray:
    """float32 [rotary_dim / 2] = 1 / theta^(arange(0, dim, 2) / dim), every step in float32."""
    expo = np.arange(0, rotary_dim, 2, dtype=np.float32) / np.float32(rotary_dim)
    return (np.float32(1.0) / np.power(np.float32(theta), expo, dtype=np.float32)).astype(np.float32)


def mrope_tables(pos3: np.ndarray, freq: np.ndarray | None = None,
                 section: tuple[int, int, int] = MROPE_SECTION) -> tuple[np.ndarray, np.ndarray]:
    """[3, L] position planes -> cos, sin float32 [1, L, 2 * len(freq)]."""
    freq = inv_freq() if freq is None else freq
    ang = (pos3.astype(np.float32)[:, :, None] * freq[None, None, :]).astype(np.float32)        # [3, L, d/2]
    a = ang[0].copy()
    for plane, offset in ((1, 1), (2, 2)):
        stop = section[plane] * 3
        a[:, offset:stop:3] = ang[plane][:, offset:stop:3]
    emb = np.concatenate([a, a], axis=-1).astype(np.float64)
    return np.cos(emb).astype(np.float32)[None], np.sin(emb).astype(np.float32)[None]
