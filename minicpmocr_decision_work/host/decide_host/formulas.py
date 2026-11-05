"""Host-side formulas of the decision path, NumPy only (no torch): token ids, M-RoPE planes and tables, vision positions,
vision rotary tables, the bilinear position rows, and the bf16 table reader. Each one is the author's expression
(transformers 5.10.4, models/qwen3_5/modeling_qwen3_5.py + vision_utils.py).
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

IMAGE_PAD, VISION_START, VISION_END, PAD, POOLED = 248056, 248053, 248054, 248044, 271
TEXT_ROTARY_DIM, TEXT_THETA = 64, 10000000.0
MROPE_SECTION = (11, 11, 10)
VISION_HEAD_DIM, VISION_THETA = 64, 10000.0
MERGE, SIDE = 2, 48
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
INSTRUCTION = ("第一张图是原图，第二张图是上述 OCR 文本渲染得到的图片。\n"
               "请判断该 OCR 文本及其对应渲染图是否与原图内容一致。\n"
               "这是一个二分类任务：一致标为 1，不一致标为 0。")
ASSISTANT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def build_prompt(text: str, m1: int, m2: int) -> str:
    """score.py build_prompt with the two placeholders expanded as the processor does (m = merged tokens of the picture)."""
    a = "<|vision_start|>" + "<|image_pad|>" * m1 + "<|vision_end|>"
    b = "<|vision_start|>" + "<|image_pad|>" * m2 + "<|vision_end|>"
    return f"<|im_start|>user\n{a}\nOCR文本：\n{text}\n{b}\n{INSTRUCTION}{ASSISTANT_SUFFIX}"


def mrope_planes(ids: np.ndarray, grids: list[tuple[int, int]]) -> np.ndarray:
    """[3, T] int64 planes (Qwen3_5Model.get_rope_index, modeling L1311-1402, one record, images only). grids = patch grids
    (gh, gw) per picture in prompt order. Text: arange + current_pos. A run of <|image_pad|>: t = current_pos,
    h = arange(H) + current_pos, w = arange(W) + current_pos on the merged grid; then current_pos += max(H, W)."""
    ids = np.asarray(ids, dtype=np.int64)
    T = len(ids)
    pos = np.zeros((3, T), dtype=np.int64)
    is_img = ids == IMAGE_PAD
    i, cur, g = 0, 0, 0
    while i < T:
        j = i
        while j < T and is_img[j] == is_img[i]:
            j += 1
        n = j - i
        if is_img[i]:
            H, W = grids[g][0] // MERGE, grids[g][1] // MERGE
            g += 1
            assert H * W == n, (H, W, n)
            k = np.arange(n, dtype=np.int64)
            pos[0, i:j] = cur
            pos[1, i:j] = cur + k // W
            pos[2, i:j] = cur + k % W
            cur += max(H, W)
        else:
            pos[:, i:j] = cur + np.arange(n, dtype=np.int64)
            cur += n
        i = j
    return pos


def padded_planes(pos3: np.ndarray, L: int) -> np.ndarray:
    """[3, L]: the record's planes, then last + 1, + 2, ... on all planes (never read; the mask hides pads as keys)."""
    n = pos3.shape[1]
    out = np.empty((3, L), dtype=np.int64)
    out[:, :n] = pos3
    out[:, n:] = int(pos3.max()) + 1 + np.arange(L - n, dtype=np.int64)[None, :]
    return out


def text_inv_freq() -> np.ndarray:
    expo = np.arange(0, TEXT_ROTARY_DIM, 2, dtype=np.float32) / np.float32(TEXT_ROTARY_DIM)
    return (np.float32(1.0) / np.power(np.float32(TEXT_THETA), expo, dtype=np.float32)).astype(np.float32)


def mrope_tables(pos3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[3, L] planes -> cos, sin float32 [1, L, 64] (Qwen3_5TextRotaryEmbedding.forward + apply_interleaved_mrope). The
    float32 angle's cosine in float64, rounded to float32: torch's float32 cos may sit an ulp away (checked and recorded)."""
    ang = (pos3.astype(np.float32)[:, :, None] * text_inv_freq()[None, None, :]).astype(np.float32)
    a = ang[0].copy()
    for plane, offset in ((1, 1), (2, 2)):
        a[:, offset:MROPE_SECTION[plane] * 3:3] = ang[plane][:, offset:MROPE_SECTION[plane] * 3:3]
    emb = np.concatenate([a, a], axis=-1).astype(np.float64)
    return np.cos(emb).astype(np.float32)[None], np.sin(emb).astype(np.float32)[None]


def vision_position_ids(gh: int, gw: int) -> np.ndarray:
    """[N, 2] int64 (row, col) in merge-block-major order (vision_utils.get_vision_position_ids)."""
    r = np.arange(gh, dtype=np.int64)[:, None].repeat(gw, axis=1).reshape(gh // MERGE, MERGE, gw // MERGE, MERGE)
    c = np.arange(gw, dtype=np.int64)[None, :].repeat(gh, axis=0).reshape(gh // MERGE, MERGE, gw // MERGE, MERGE)
    return np.stack([r.transpose(0, 2, 1, 3).ravel(), c.transpose(0, 2, 1, 3).ravel()], axis=-1)


def vision_inv_freq() -> np.ndarray:
    dim = VISION_HEAD_DIM // 2
    expo = np.arange(0, dim, 2, dtype=np.float32) / np.float32(dim)
    return (np.float32(1.0) / np.power(np.float32(VISION_THETA), expo, dtype=np.float32)).astype(np.float32)


def vision_rotary(gh: int, gw: int) -> tuple[np.ndarray, np.ndarray]:
    """cos, sin float32 [N, 64] (modeling L1102-1108: emb = cat(rot, rot), rot = (pos[..., None] * inv_freq).flatten(1))."""
    pid = vision_position_ids(gh, gw)
    rot = (pid.astype(np.float32)[:, :, None] * vision_inv_freq()[None, None, :]).astype(np.float32).reshape(len(pid), -1)
    emb = np.concatenate([rot, rot], axis=-1).astype(np.float64)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def _linspace(end: float, steps: int) -> np.ndarray:
    """torch.linspace(0, end, steps), float32: i * step on the first half, end - (steps - 1 - i) * step on the second. The
    product is not rounded before the subtraction (torch 2.8.0 on arm64 fuses it; measured 244,649 / 244,649 equal for
    steps 2..699); an x86 or GPU build of the same line may land an ulp away, which is below every position row's float noise."""
    if steps == 1:
        return np.zeros(1, dtype=np.float32)
    step = float(np.float32(end / (steps - 1)))
    i = np.arange(steps, dtype=np.int64)
    half = steps // 2
    return np.where(i < half, i.astype(np.float64) * step, end - (steps - 1 - i).astype(np.float64) * step).astype(np.float32)


def bilinear_taps(gh: int, gw: int) -> tuple[np.ndarray, np.ndarray]:
    """vision_utils.get_vision_bilinear_indices_and_weights for one picture: idx int64 [4, N], w float32 [4, N]."""
    hg, wg = _linspace(float(SIDE - 1), gh), _linspace(float(SIDE - 1), gw)
    hf, wf = hg.astype(np.int64), wg.astype(np.int64)
    hc, wc = np.minimum(hf + 1, SIDE - 1), np.minimum(wf + 1, SIDE - 1)
    hr, wr = (hg - hf.astype(np.float32)).astype(np.float32), (wg - wf.astype(np.float32)).astype(np.float32)
    corners = [(hf[:, None] * SIDE + wf[None, :]), (hf[:, None] * SIDE + wc[None, :]),
               (hc[:, None] * SIDE + wf[None, :]), (hc[:, None] * SIDE + wc[None, :])]
    one = np.float32(1.0)
    weights = [((one - hr)[:, None] * (one - wr)[None, :]), ((one - hr)[:, None] * wr[None, :]),
               (hr[:, None] * (one - wr)[None, :]), (hr[:, None] * wr[None, :])]
    h_idx = np.arange(gh, dtype=np.int64).reshape(gh // MERGE, MERGE)
    w_idx = np.arange(gw, dtype=np.int64).reshape(gw // MERGE, MERGE)
    reorder = (h_idx[:, :, None, None] * gw + w_idx[None, None, :, :]).transpose(0, 2, 1, 3).ravel()
    idx = np.stack([c.ravel()[reorder] for c in corners])
    w = np.stack([np.asarray(c, dtype=np.float32).ravel()[reorder] for c in weights])
    return idx, w


def pos_rows(table: np.ndarray, gh: int, gw: int) -> np.ndarray:
    """float32 [N, 1024] = (pos_embed(idx) * w[:, :, None]).sum(0) (modeling L1100), table = the bf16 [2304, 1024] as float32."""
    idx, w = bilinear_taps(gh, gw)
    out = table[idx[0]] * w[0][:, None]
    for i in (1, 2, 3):
        out = (out + table[idx[i]] * w[i][:, None]).astype(np.float32)
    return out


class Bf16Table:
    """One bf16 tensor of the checkpoint, memory-mapped, rows widened exactly (a bf16 value is the top 16 bits of the float32)."""

    def __init__(self, safetensors: str | Path, key: str):
        with open(safetensors, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            e = json.loads(f.read(n))[key]
        assert e["dtype"] == "BF16" and len(e["shape"]) == 2, e
        start, end = e["data_offsets"]
        self.shape = tuple(e["shape"])
        assert end - start == self.shape[0] * self.shape[1] * 2
        self.u16 = np.memmap(safetensors, dtype=np.uint16, mode="r", offset=8 + n + start, shape=self.shape)

    def rows(self, ids=None) -> np.ndarray:
        u = self.u16 if ids is None else self.u16[np.asarray(ids, dtype=np.int64)]
        return (np.asarray(u, dtype=np.uint32) << 16).view(np.float32)


def fill_embeds(ids: np.ndarray, L: int, table: Bf16Table, rows: list[np.ndarray]):
    """(embeds [1, L, 2048], valid [1, L]) float32: the table's rows, each picture's merged rows at its <|image_pad|> run in
    prompt order (the author's masked_scatter), zeros and valid 0 on the right pads."""
    ids = np.asarray(ids, dtype=np.int64)
    T = len(ids)
    embeds = np.zeros((1, L, table.shape[1]), dtype=np.float32)
    embeds[0, :T] = table.rows(ids)
    where = np.flatnonzero(ids == IMAGE_PAD)
    img = np.concatenate(rows, axis=0)
    assert len(where) == len(img), (len(where), len(img))
    embeds[0, where] = img
    valid = np.zeros((1, L), dtype=np.float32)
    valid[0, :T] = 1.0
    return embeds, valid


def softmax_prob1(hidden_row: np.ndarray, score_w: np.ndarray) -> tuple[np.ndarray, float]:
    """score = hidden @ W^T (no bias), prob = softmax in float32. -> (logits [2], prob_1)."""
    logits = (score_w @ hidden_row.astype(np.float32)).astype(np.float32)
    e = np.exp(logits - logits.max()).astype(np.float32)
    return logits, float((e / e.sum()).astype(np.float32)[1])
