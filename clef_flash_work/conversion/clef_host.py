"""Host side of the clef-flash row graph, NumPy only (no torch): the embedding rows and the M-RoPE tables the graph
takes as inputs.

    table = EmbedTable(snapshot)                       # the checkpoint's bf16 embed_tokens, memory-mapped
    embeds, valid, cos, sin = row_inputs(ids, 1024, table)          # a text record: positions (i, i, i)

EmbedTable: `model.language_model.embed_tokens.weight` of the pinned checkpoint (BF16 [248320, 4096]) read in place
from its safetensors shard (header offsets; nothing copied), rows converted to float32 exactly (a bf16 value is the
top 16 bits of the float32 with the same value).

M-RoPE (the author's Qwen3_5TextRotaryEmbedding, transformers 5.14.1 / 5.17.0; default rope, attention_scaling 1):
    inv_freq[j] = 1 / theta^(2j / rotary_dim)            j = 0..rotary_dim/2 - 1 (float32; theta 1e7, rotary_dim 64)
    angle[p][i, j] = float32(inv_freq[j] * pos[p][i])    p = t, h, w planes
    interleave (mrope_section [11, 11, 10]): j % 3 == 1 and j < 33 -> h plane, j % 3 == 2 and j < 30 -> w plane,
                                             every other j -> t plane
    cos = cos(cat(angle, angle)), sin = sin(cat(angle, angle))            [1, L, rotary_dim] float32
cos / sin are evaluated in float64 on the float32 angle and rounded to float32 (the correctly rounded value; torch's
float32 cos may sit an ulp away, recorded by the torch check). Pad positions (right of the record) continue the last
position + 1, + 2, ... on all three planes (for a text record: positions 0..L-1, the constant positions of Kev's form);
their outputs are never read and the mask hides them as keys."""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

HIDDEN = 4096
ROTARY_DIM = 64
ROPE_THETA = 10000000.0
MROPE_SECTION = (11, 11, 10)
EMBED_KEY = "model.language_model.embed_tokens.weight"


class EmbedTable:
    def __init__(self, snapshot: str | Path, key: str = EMBED_KEY):
        snapshot = Path(snapshot)
        index = json.loads((snapshot / "model.safetensors.index.json").read_text())
        shard = snapshot / index["weight_map"][key]
        with open(shard, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        e = header[key]
        assert e["dtype"] == "BF16" and len(e["shape"]) == 2, e
        start, end = e["data_offsets"]
        rows, dim = e["shape"]
        assert end - start == rows * dim * 2
        self.u16 = np.memmap(shard, dtype=np.uint16, mode="r", offset=8 + n + start, shape=(rows, dim))
        self.info = {"key": key, "shard": shard.name, "shape": [rows, dim], "dtype": "BF16",
                     "file_offset": 8 + n + start, "bytes": end - start}

    def rows(self, ids) -> np.ndarray:
        """float32 [len(ids), dim]: the bf16 rows widened exactly."""
        u = np.asarray(self.u16[np.asarray(ids, dtype=np.int64)], dtype=np.uint32) << 16
        return u.view(np.float32)


def inv_freq(theta: float = ROPE_THETA, rotary_dim: int = ROTARY_DIM) -> np.ndarray:
    """float32 [rotary_dim / 2] = 1 / theta^(arange(0, dim, 2) / dim), every step in float32 as the author's code."""
    expo = np.arange(0, rotary_dim, 2, dtype=np.float32) / np.float32(rotary_dim)
    return (np.float32(1.0) / np.power(np.float32(theta), expo, dtype=np.float32)).astype(np.float32)


def text_positions(n: int) -> np.ndarray:
    """[3, n] int64 position planes of a text-only record: (i, i, i)."""
    return np.broadcast_to(np.arange(n, dtype=np.int64), (3, n)).copy()


def padded_positions(pos3: np.ndarray, L: int) -> np.ndarray:
    """[3, L]: the record's planes, then last + 1, + 2, ... on all three planes for the pad positions."""
    n = pos3.shape[1]
    assert n <= L, (n, L)
    out = np.empty((3, L), dtype=np.int64)
    out[:, :n] = pos3
    out[:, n:] = int(pos3.max()) + 1 + np.arange(L - n, dtype=np.int64)[None, :]
    return out


def mrope_tables(pos3: np.ndarray, freq: np.ndarray | None = None,
                 section: tuple[int, int, int] = MROPE_SECTION) -> tuple[np.ndarray, np.ndarray]:
    """[3, L] position planes -> cos, sin float32 [1, L, 2 * len(freq)]."""
    freq = inv_freq() if freq is None else freq
    ang = (pos3.astype(np.float32)[:, :, None] * freq[None, None, :]).astype(np.float32)      # [3, L, d/2]
    a = ang[0].copy()
    for plane, offset in ((1, 1), (2, 2)):
        length = section[plane] * 3
        a[:, offset:length:3] = ang[plane][:, offset:length:3]
    emb = np.concatenate([a, a], axis=-1).astype(np.float64)
    return (np.cos(emb).astype(np.float32)[None], np.sin(emb).astype(np.float32)[None])


def row_inputs(ids, L: int, table: EmbedTable, pos3: np.ndarray | None = None):
    """One record -> (embeds [1, L, d], valid [1, L], cos [1, L, r], sin [1, L, r]), all float32; pads after the
    record: embeds 0, valid 0."""
    n = len(ids)
    assert 0 < n <= L, (n, L)
    embeds = np.zeros((1, L, table.u16.shape[1]), dtype=np.float32)
    embeds[0, :n] = table.rows(ids)
    valid = np.zeros((1, L), dtype=np.float32)
    valid[0, :n] = 1.0
    pos3 = text_positions(n) if pos3 is None else pos3
    cos, sin = mrope_tables(padded_positions(pos3, L))
    return embeds, valid, cos, sin
