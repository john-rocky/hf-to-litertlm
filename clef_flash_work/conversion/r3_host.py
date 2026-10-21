"""Round 3 host side for pictures (NumPy + Pillow + tokenizers, no torch): the tile's patches, the record's ids with the
image block, the M-RoPE planes of an image record, and the row graph's inputs with the tower's rows in place.

    from tokenizers import Tokenizer
    patches = tile_patches("fixtures/images/shapes_224.png", 256)        # [256, 1536] float32 = the processor's pixel_values
    enc = encode_image_record(tok, request, grid=8)                       # ids with <|vision_start|> + 64 pads + ... after the prefix
    embeds, valid, cos, sin = image_row_inputs(enc["ids"], 1024, table, rows, P, 8, 8)

Pixels (one fixed tile per tower graph; the reference's g256 / g448 arms, scripts/clef_ref.py `load_image`):
    PIL open -> convert("RGB") -> resize((tile, tile), BICUBIC)   (aspect NOT kept; Pillow's own integer resampler)
    -> /255 -> (x - 0.5) / 0.5 in float64 -> float32 -> merge-block-major patches [4 G^2, 1536], vector (C, T=2, 16, 16)
Copied from the Core AI zoo's conversion/clef_flash/host.py (`preprocess` resize="pil" and `patchify`, whose output
was bit-equal to the processor's pixel_values on 56/56 runs there); the processor's own resize is not run (it is the
identity at 256 / 448 = multiples of 32 inside [65,536, 16,777,216] pixels).

ids: the author's encode_record with media: prefix + [<|vision_start|>, N x <|image_pad|>, <|vision_end|>, "\\n"] + state +
schema + suffix; token_offset P = len(prefix) (the <|vision_start|> index); the spans shift by the N + 3 media ids
(scripts/clef_tok.py does the text pieces).

M-RoPE planes (the reference's closed form, ref/header.json `mrope_closed_form`; merged grid H x W, N = H * W):
    i <= P -> (i, i, i); image token k at P+1+k -> (P+1, P+1 + k // W, P+1 + k % W); i >= P+N+1 -> i - (N - max(H, W))
cos / sin = scripts/clef_host.py `mrope_tables` of those planes (pads continue last + 1, + 2, ... on all planes).

Row graph inputs: embeds = the checkpoint's bf16 embed_tokens rows of the ids (float32), the image-pad rows replaced by
the tower's rows (the author's masked_scatter of image_embeds); valid = 1 on the record, 0 on the right pads."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clef_host as CH  # noqa: E402
import clef_tok as CT  # noqa: E402

PATCH, MERGE, TEMPORAL = 16, 2, 2
IMAGE_MEAN, IMAGE_STD = 0.5, 0.5
TILES = {"g256": 256, "g448": 448}
VISION_START, VISION_END, IMAGE_PAD = 248053, 248054, 248056
BUCKETS = (256, 512, 1024, 2048, 4096)


def tile_grid(tile: int) -> int:
    """Merged rows per side of a square tile (256 -> 8, 448 -> 14)."""
    assert tile % (PATCH * MERGE) == 0, tile
    return tile // (PATCH * MERGE)


def patchify(x: np.ndarray) -> np.ndarray:
    """Normalized [H, W, C] -> [gh*gw, C*T*P*P] in Qwen's merge-block-major order: patches iterate
    (block_row, block_col, y-in-block, x-in-block), channel outermost inside the vector, the still frame repeated at both
    temporal slots (the Qwen2VLImageProcessor reshape / permute; zoo host.py, unchanged)."""
    h, w, c = x.shape
    assert h % (PATCH * MERGE) == 0 and w % (PATCH * MERGE) == 0, (h, w)
    gh, gw = h // PATCH, w // PATCH
    t = x.transpose(2, 0, 1).reshape(c, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH)
    t = t.transpose(1, 4, 2, 5, 0, 3, 6).reshape(gh * gw, 1, c, PATCH, PATCH)
    t = np.broadcast_to(t[:, :, :, None], (gh * gw, 1, c, TEMPORAL, PATCH, PATCH))
    return t.reshape(gh * gw, c * TEMPORAL * PATCH * PATCH)


def tile_pixels(image, tile: int) -> np.ndarray:
    """Image (path or PIL image) -> uint8 [tile, tile, 3]: RGB, Pillow BICUBIC to the tile."""
    from PIL import Image

    im = Image.open(image) if isinstance(image, (str, Path)) else image
    return np.asarray(im.convert("RGB").resize((tile, tile), Image.Resampling.BICUBIC))


def tile_patches(image, tile: int) -> np.ndarray:
    """Image -> patches [4 G^2, 1536] float32, G = tile // 32."""
    tile_grid(tile)
    x = (tile_pixels(image, tile).astype(np.float64) / 255.0 - IMAGE_MEAN) / IMAGE_STD
    return np.ascontiguousarray(patchify(x).astype(np.float32))


def image_block(tok, n: int) -> list[int]:
    """The processor's media ids of one image of n merged tokens: <|vision_start|>, n pads, <|vision_end|>, \\n."""
    ids = CT.tokens(tok, "<|vision_start|>" + "<|image_pad|>" * n + "<|vision_end|>\n")
    ok = (ids[:1] == [VISION_START] and ids[1:n + 1] == [IMAGE_PAD] * n and ids[n + 1:n + 2] == [VISION_END]
          and len(ids) == n + 3)
    if not ok:
        raise ValueError("the tokenizer does not keep the image placeholder tokens whole")
    return ids


def encode_image_record(tok, request: dict, n_image: int) -> dict:
    """The author's encode_record for a request with one image of n_image merged tokens -> {"ids", "questions",
    "token_offset"} (spans [start, end) in the full ids)."""
    enc = CT.encode_record(tok, request)
    P = enc["prefix_len"]
    block = image_block(tok, n_image)
    fixed = len(enc["ids"]) - enc["prefix_len"]      # the text path's length after the prefix
    assert P + len(block) + fixed <= CT.MAX_LENGTH, "state cut with media is not handled (records here are short)"
    shift = len(block)
    for q in enc["questions"]:
        q["question_span"] = [q["question_span"][0] + shift, q["question_span"][1] + shift]
        q["option_spans"] = [[a + shift, b + shift] for a, b in q["option_spans"]]
    ids = enc["ids"][:P] + block + enc["ids"][P:]
    return {"ids": ids, "questions": enc["questions"], "token_offset": P}


def image_positions(T: int, P: int, H: int, W: int) -> np.ndarray:
    """[3, T] int64 M-RoPE planes of a record with one image of H x W merged tokens right after the prefix (P ids)."""
    N = H * W
    assert P + 1 + N < T, (T, P, N)
    pos = np.broadcast_to(np.arange(T, dtype=np.int64), (3, T)).copy()
    k = np.arange(N, dtype=np.int64)
    pos[0, P + 1:P + 1 + N] = P + 1
    pos[1, P + 1:P + 1 + N] = P + 1 + k // W
    pos[2, P + 1:P + 1 + N] = P + 1 + k % W
    pos[:, P + N + 1:] = np.arange(P + N + 1, T, dtype=np.int64) - (N - max(H, W))
    return pos


def bucket(T: int, buckets=BUCKETS) -> int:
    """The smallest row graph length that holds T tokens (None = refused)."""
    return next((L for L in buckets if T <= L), None)


def image_row_inputs(ids, L: int, table, rows: np.ndarray | None, P: int, H: int, W: int):
    """One image record -> (embeds [1, L, d], valid [1, L], cos [1, L, r], sin [1, L, r]) float32. rows = the tower's
    image rows [H * W, d] (None = keep the table rows of the pads); the pad rows after the record: embeds 0, valid 0."""
    ids = np.asarray(ids, dtype=np.int64)
    T, N = len(ids), H * W
    assert T <= L, (T, L)
    assert ids[P] == VISION_START and (ids[P + 1:P + 1 + N] == IMAGE_PAD).all() and ids[P + 1 + N] == VISION_END
    embeds = np.zeros((1, L, table.u16.shape[1]), dtype=np.float32)
    embeds[0, :T] = table.rows(ids)
    if rows is not None:
        rows = np.asarray(rows, dtype=np.float32)
        assert rows.shape == (N, table.u16.shape[1]), rows.shape
        embeds[0, P + 1:P + 1 + N] = rows
    valid = np.zeros((1, L), dtype=np.float32)
    valid[0, :T] = 1.0
    cos, sin = CH.mrope_tables(CH.padded_positions(image_positions(T, P, H, W), L))
    return embeds, valid, cos, sin
