"""d1-3B on LiteRT: the picture path of the reference host (numpy + Pillow only; no PyTorch, no transformers).

Every step below is a copy of what the provider's code (LiquidAI/d1-3B at
da1fe36a, `runner.SystemOne`) and transformers 5.14.1 (`Lfm2VlProcessor`, `Lfm2VlImageProcessor`,
`Siglip2VisionEmbeddings`, `Lfm2VlMultiModalProjector`, `Lfm2VlModel.get_placeholder_mask`) do for a request with
pictures, written so that a host in any language can follow it. The numbers are the d1-3B ones
(`processor_config.json`, `config.json`); `VisionSettings` holds them.

1. `cap_pixels`: a picture larger than 1024 * 1024 pixels is shrunk with Pillow's BICUBIC to
   (int(w * s), int(h * s)), s = sqrt(1024 * 1024 / (w * h)) (each side at least 1); the picture is converted to RGB.
2. `preprocess` (the image processor, torchvision backend, CPU):
   - `smart_resize`: the single-tile / thumbnail size, both sides multiples of 32 (patch 16 x downsample 2), area
     between 64 and 256 tokens' worth of pixels (64 * 32 * 32 .. 256 * 32 * 32), Python `round` (half to even).
   - `is_too_large`: round_by_factor(h, 32) * round_by_factor(w, 32) > 256 * 32 * 32 * 2.0 (sides at least 16).
   - Large pictures are split: the grid (cols, rows) is the closest aspect ratio among 2..10 tiles
     (`grid_layout`, ties broken toward the larger grid when the picture covers more than half of its area), the
     picture is resized to (rows * 512, cols * 512) and cut into 512 x 512 tiles in row-major order, and a thumbnail
     at the smart_resize size follows the tiles. Small pictures are one tile at the smart_resize size.
   - Resizing = torchvision `resize(uint8, BICUBIC, antialias=True)`, which on the CPU is PyTorch's uint8 separable
     antialias kernel (`resize_uint8_bicubic_aa`): Keys cubic a = -0.5, support 2 * scale when shrinking, weights
     computed in float64, normalised, scaled to int16 at the largest precision that keeps them below 2^15, applied
     with integer arithmetic and rounding (+ 2^(p-1), >> p, clamp 0..255); width before height; a side that
     keeps its size is not touched.
   - Each tile: float32, (x - 127.5) / 127.5 (the processor's fused 1/255 rescale and mean = std = 0.5), cut into
     16 x 16 patches in raster order, each patch flattened as (row, column, channel) = 768 values, padded with zero
     patches to 1024; `pixel_attention_mask` int32 [1024] (1 = real patch), `spatial_shapes` [patch rows, patch cols].
3. `image_token_ids`: per picture `<|image_start|>`, then for a split picture `<|img_row_r_col_c|>` + 256 x `<image>`
   per tile (row-major) and `<|img_thumbnail|>` + n x `<image>` for the thumbnail, for a single tile n x `<image>`;
   then `<|image_end|>`. n = ceil(h / 16 / 2) * ceil(w / 16 / 2) for the thumbnail / single-tile size (h, w).
   `row_ids` encodes a rendered request (whose text holds one `<image>` per picture, as the chat template writes it)
   as the processor does: the text between the markers through the tokenizer, each marker replaced by its picture's
   ids.
4. `resize_positions`: the tower's 16 x 16 position table resized to a tile's patch grid (h, w) with bilinear
   antialias (align_corners False), as `Siglip2VisionEmbeddings.resize_positional_embeddings` does through
   `F.interpolate(mode="bilinear", antialias=True)`; float32 arithmetic in the kernel's order (weights in float32,
   width before height, fused multiply-add as PyTorch's arm64 build computes it). Rows = the grid in raster
   order.
5. The tower graph maps `pixels` [1,1024,768] + `pos` [1,1024,H] + `mask` [1,1024] to `features` [1,1024,H]; the real
   patches are the leading h * w rows.
6. `pixel_unshuffle`: the real features as an (h, w, H) grid -> (h/2, w/2, 4H): output channel j * 2H + k * H + c of
   cell (r, q) is input (2r + j, 2q + k, c); rows in raster order. The projector graph maps `soft` [1,256,4H] (the
   cells, zero rows after them) to `mm` [1,256,d]; the leading (h/2)(w/2) rows are the picture tokens.
7. `insert_embeddings`: the row's embeddings are the text table's rows, and the k-th `<image>` position takes the k-th
   picture token (tiles in order, pictures in order) = `get_placeholder_mask` + `masked_scatter`; the counts must
   agree.
8. `LiteRTGraph` runs one graph through the CompiledModel API (CPU or GPU, shapes from the signature);
   `picture_tokens` chains steps 5-6 for every tile of a picture. The row then goes to the embeds variant of the row
   graph (host/d1_litert.py), right-padded, and is read out at its last real token as for text.
9. `VisionPath` holds the tower and projector graphs, the position table
   (`load_position_table`, vision_position_table.safetensors) and the token ids (`token_ids`, contract.json);
   `D1Host` (host/d1_litert.py item 9) calls its `pictures` (step 1-2 per picture: a PIL image, a path or an image
   file's bytes, `open_picture`) and `tokens` (steps 4-6 for every tile of every picture, in order).
contract.json `vision` holds the same rules as data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np

VISION_MAX_PIXELS = 1024 * 1024
IMAGE_MARKER = "<image>"


@dataclass(frozen=True)
class VisionSettings:
    """d1-3B's processor_config.json / config.json values."""

    patch: int = 16
    downsample: int = 2
    tile: int = 512
    min_tiles: int = 2
    max_tiles: int = 10
    min_image_tokens: int = 64
    max_image_tokens: int = 256
    max_pixels_tolerance: float = 2.0
    max_num_patches: int = 1024
    use_thumbnail: bool = True
    mean: float = 0.5
    std: float = 0.5
    rescale: float = 0.00392156862745098      # 1 / 255 as processor_config.json writes it

    @property
    def tokens_per_tile(self) -> int:
        side = math.ceil((self.tile // self.patch) / self.downsample)
        return side * side


SETTINGS = VisionSettings()


# --------------------------------------------------------------------------- #
# 1. cap_pixels (runner.cap_pixels)
# --------------------------------------------------------------------------- #


def cap_pixels(image, max_pixels: int = VISION_MAX_PIXELS):
    """The provider's `cap_pixels`: RGB, and shrunk with BICUBIC to at most `max_pixels`."""
    from PIL import Image

    image = image.convert("RGB")
    w, h = image.size
    if w * h <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / (w * h))
    return image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BICUBIC)


# --------------------------------------------------------------------------- #
# 2a. PyTorch's uint8 separable antialias resize (bicubic), ported
# --------------------------------------------------------------------------- #


def _cubic_aa(x: float) -> float:
    """HelperInterpCubic::aa_filter<double, true>: Keys cubic with a = -0.5 (cubic_convolution1 / 2)."""
    a = -0.5
    x = abs(x)
    if x < 1.0:
        return ((a + 2) * x - (a + 3)) * x * x + 1
    if x < 2.0:
        return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a
    return 0.0


def aa_weights_int16(in_size: int, out_size: int, interp_size: int = 4,
                     filt: Callable[[float], float] = _cubic_aa) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """`_compute_index_ranges_int16_weights` (antialias, align_corners False, no scale argument): per output index the
    starting source index, the number of taps, the int16 weights [out, max_taps], and the precision p."""
    scale = in_size / out_size
    support = (interp_size * 0.5) * scale if scale >= 1.0 else interp_size * 0.5
    max_taps = int(math.ceil(support)) * 2 + 1
    xmins, sizes = np.zeros(out_size, np.int64), np.zeros(out_size, np.int64)
    weights = np.zeros((out_size, max_taps), np.float64)
    wt_max = 0.0
    for i in range(out_size):
        center = scale * (i + 0.5)
        invscale = 1.0 / scale if scale >= 1.0 else 1.0
        xmin = max(int(center - support + 0.5), 0)
        xsize = min(int(center + support + 0.5), in_size) - xmin
        xsize = min(max(xsize, 0), max_taps)
        total = 0.0
        ws = []
        for j in range(xsize):
            w = filt((j + xmin - center + 0.5) * invscale)
            ws.append(w)
            total += w
        wmax_i = 0.0
        if total != 0.0:
            for j in range(xsize):
                ws[j] = ws[j] / total
                wmax_i = max(wmax_i, ws[j])
        wt_max = max(wt_max, wmax_i)
        xmins[i], sizes[i] = xmin, xsize
        weights[i, :xsize] = ws
    precision = 0
    while precision < 22:
        if int(0.5 + wt_max * (1 << (precision + 1))) >= (1 << 15):
            break
        precision += 1
    scaled = weights * (1 << precision)
    w16 = np.where(scaled < 0, np.trunc(-0.5 + scaled), np.trunc(0.5 + scaled)).astype(np.int64)
    return xmins, sizes, w16, precision


def _resample_axis_uint8(arr: np.ndarray, axis: int, out_size: int) -> np.ndarray:
    """One separable pass (basic_loop_separable_1d_{horizontal,vertical}<uint8_t>) along `axis` of a uint8 array."""
    in_size = arr.shape[axis]
    xmins, sizes, w16, p = aa_weights_int16(in_size, out_size)
    taps = w16.shape[1]
    idx = xmins[:, None] + np.arange(taps)[None, :]
    used = np.arange(taps)[None, :] < sizes[:, None]
    idx = np.where(used, idx, 0)
    w = np.where(used, w16, 0)
    src = np.moveaxis(arr, axis, 0).astype(np.int64)            # [in, ...]
    gathered = src[idx]                                          # [out, taps, ...]
    acc = (gathered * w.reshape(w.shape + (1,) * (src.ndim - 1))).sum(axis=1) + (1 << (p - 1))
    out = np.clip(acc >> p, 0, 255).astype(np.uint8)
    return np.moveaxis(out, 0, axis)


def resize_uint8_bicubic_aa(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """torchvision `resize(uint8 CHW, [out_h, out_w], BICUBIC, antialias=True)` on the CPU, for an [H, W, C] array:
    width before height; an unchanged side is skipped and an unchanged picture is returned as it is."""
    h, w = arr.shape[:2]
    if (out_h, out_w) == (h, w):
        return arr
    if out_w != w:
        arr = _resample_axis_uint8(arr, 1, out_w)
    if out_h != h:
        arr = _resample_axis_uint8(arr, 0, out_h)
    return arr


# --------------------------------------------------------------------------- #
# 2b. the image processor (Lfm2VlImageProcessor, transformers 5.14.1)
# --------------------------------------------------------------------------- #


def round_by_factor(number: float, factor: int) -> int:
    return round(number / factor) * factor


def smart_resize(height: int, width: int, s: VisionSettings = SETTINGS) -> Tuple[int, int]:
    """(width, height) of a single tile or the thumbnail."""
    total = s.patch * s.downsample
    min_pixels = s.min_image_tokens * s.patch ** 2 * s.downsample ** 2
    max_pixels = s.max_image_tokens * s.patch ** 2 * s.downsample ** 2
    h_bar = max(total, round_by_factor(height, total))
    w_bar = max(total, round_by_factor(width, total))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(total, math.floor(height / beta / total) * total)
        w_bar = max(total, math.floor(width / beta / total) * total)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / total) * total
        w_bar = math.ceil(width * beta / total) * total
    return w_bar, h_bar


def is_too_large(height: int, width: int, s: VisionSettings = SETTINGS) -> bool:
    total = s.patch * s.downsample
    h_bar = max(s.patch, round_by_factor(height, total))
    w_bar = max(s.patch, round_by_factor(width, total))
    return h_bar * w_bar > s.max_image_tokens * s.patch ** 2 * s.downsample ** 2 * s.max_pixels_tolerance


def target_ratios(min_tiles: int, max_tiles: int) -> List[Tuple[int, int]]:
    ratios = [(w, h) for n in range(min_tiles, max_tiles + 1) for w in range(1, n + 1) for h in range(1, n + 1)
              if min_tiles <= w * h <= max_tiles]
    return sorted(set(ratios), key=lambda x: x[0] * x[1])


def closest_aspect_ratio(aspect_ratio: float, ratios: Sequence[Tuple[int, int]], width: int, height: int,
                         image_size: int) -> Tuple[int, int]:
    best_diff, best = float("inf"), (1, 1)
    area = width * height
    for ratio in ratios:
        diff = abs(aspect_ratio - ratio[0] / ratio[1])
        if diff < best_diff:
            best_diff, best = diff, ratio
        elif diff == best_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best = ratio
    return best


def grid_layout(height: int, width: int, s: VisionSettings = SETTINGS) -> Tuple[int, int]:
    """(grid columns, grid rows) of a picture that is split."""
    return closest_aspect_ratio(width / height, target_ratios(s.min_tiles, s.max_tiles), width, height, s.tile)


@dataclass
class Tile:
    pixels: np.ndarray          # float32 [max_num_patches, patch * patch * 3], zero rows after the real patches
    mask: np.ndarray            # int32 [max_num_patches]
    grid: Tuple[int, int]       # (patch rows h, patch cols w) = spatial_shapes


@dataclass
class Picture:
    tiles: List[Tile]           # row-major tiles, then the thumbnail (or one tile)
    rows: int                   # grid rows (1 for a single tile)
    cols: int
    size: Tuple[int, int]       # (height, width) of the thumbnail / single tile = the processor's image_sizes
    input_size: Tuple[int, int]  # (height, width) after cap_pixels


def to_patches(tile_u8: np.ndarray, s: VisionSettings = SETTINGS) -> Tile:
    """A resized uint8 [h, w, 3] tile -> normalised float32 patches, padded; the mask; the patch grid."""
    mean = np.float32(s.mean * (1.0 / s.rescale))
    std = np.float32(s.std * (1.0 / s.rescale))
    x = (tile_u8.astype(np.float32) - mean) / std                # [h, w, 3]
    h, w = x.shape[0] // s.patch, x.shape[1] // s.patch
    p = x.reshape(h, s.patch, w, s.patch, 3).transpose(0, 2, 1, 3, 4).reshape(h * w, s.patch * s.patch * 3)
    n = h * w
    if n > s.max_num_patches:
        raise ValueError(f"a tile of {n} patches exceeds {s.max_num_patches}")
    pixels = np.zeros((s.max_num_patches, p.shape[1]), np.float32)
    pixels[:n] = p
    mask = np.zeros(s.max_num_patches, np.int32)
    mask[:n] = 1
    return Tile(pixels, mask, (h, w))


def preprocess(image, s: VisionSettings = SETTINGS) -> Picture:
    """One picture (a PIL image already through `cap_pixels`) -> its tiles, as Lfm2VlImageProcessor makes them."""
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = arr.shape[:2]
    new_w, new_h = smart_resize(height, width, s)
    splitting = not (s.min_tiles == s.max_tiles == 1)
    if is_too_large(height, width, s) and splitting:
        cols, rows = grid_layout(height, width, s)
        big = resize_uint8_bicubic_aa(arr, s.tile * rows, s.tile * cols)
        tiles = [big[r * s.tile:(r + 1) * s.tile, c * s.tile:(c + 1) * s.tile] for r in range(rows) for c in range(cols)]
        if s.use_thumbnail and rows * cols != 1:
            tiles.append(resize_uint8_bicubic_aa(arr, new_h, new_w))
    else:
        rows = cols = 1
        tiles = [resize_uint8_bicubic_aa(arr, new_h, new_w)]
    return Picture([to_patches(t, s) for t in tiles], rows, cols, (new_h, new_w), (height, width))


# --------------------------------------------------------------------------- #
# 3. token ids (processing_lfm2_vl._build_image_tokens)
# --------------------------------------------------------------------------- #


def tokens_for_size(size: Tuple[int, int], s: VisionSettings = SETTINGS) -> int:
    h, w = size
    return math.ceil((h // s.patch) / s.downsample) * math.ceil((w // s.patch) / s.downsample)


def image_markup(n: int) -> str:
    """What the chat template writes at the head of the user turn for n pictures (`SystemOne._image_markup`)."""
    return IMAGE_MARKER * n


def image_token_ids(pic: Picture, ids: Mapping[str, int], s: VisionSettings = SETTINGS) -> List[int]:
    """`ids`: the contract's token_ids (image, image_start, image_end, img_thumbnail, img_row_col: {"r,c": id})."""
    out = [ids["image_start"]]
    n = tokens_for_size(pic.size, s)
    if pic.rows > 1 or pic.cols > 1:
        for r in range(pic.rows):
            for c in range(pic.cols):
                out.append(ids["img_row_col"][f"{r + 1},{c + 1}"])
                out += [ids["image"]] * s.tokens_per_tile
        if s.use_thumbnail:
            out.append(ids["img_thumbnail"])
            out += [ids["image"]] * n
    else:
        out += [ids["image"]] * n
    out.append(ids["image_end"])
    return out


def row_ids(encode: Callable[[str], List[int]], text: str, pictures: Sequence[Picture], ids: Mapping[str, int],
            s: VisionSettings = SETTINGS) -> List[int]:
    """A rendered request with one `<image>` per picture -> token ids (the processor's expansion + tokenizer)."""
    parts = text.split(IMAGE_MARKER)
    if len(parts) - 1 != len(pictures):
        raise ValueError(f"{len(parts) - 1} picture markers for {len(pictures)} pictures")
    out: List[int] = []
    for k, part in enumerate(parts):
        out += encode(part) if part else []
        if k < len(pictures):
            out += image_token_ids(pictures[k], ids, s)
    return out


# --------------------------------------------------------------------------- #
# 4. position table (F.interpolate bilinear antialias, float32), ported
# --------------------------------------------------------------------------- #


def _linear_weights_f32(in_size: int, out_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HelperInterpLinear::compute_index_ranges_weights<float> (antialias): float32 weights, as the kernel rounds
    them (float * double products in double, stored as float)."""
    f32, f64 = np.float32, np.float64
    scale = f32(in_size) / f32(out_size)
    support = f32(f64(2 * 0.5) * f64(scale)) if scale >= 1.0 else f32(2 * 0.5)
    max_taps = int(math.ceil(float(support))) * 2 + 1
    xmins, sizes = np.zeros(out_size, np.int64), np.zeros(out_size, np.int64)
    weights = np.zeros((out_size, max_taps), np.float32)
    for i in range(out_size):
        center = f32(f64(scale) * f64(i + 0.5))
        invscale = f32(f64(1.0) / f64(scale)) if scale >= 1.0 else f32(1.0)
        xmin = max(int(f64(f32(center - support)) + 0.5), 0)
        xsize = min(int(f64(f32(center + support)) + 0.5), in_size) - xmin
        xsize = min(max(xsize, 0), max_taps)
        total = f32(0.0)
        ws = []
        for j in range(xsize):
            arg = f32((f64(f32(f32(j + xmin) - center)) + 0.5) * f64(invscale))
            x = abs(arg)
            w = f32(f64(1.0) - f64(x)) if x < 1.0 else f32(0.0)
            ws.append(w)
            total = f32(total + w)
        if total != 0.0:
            ws = [f32(w / total) for w in ws]
        xmins[i], sizes[i] = xmin, xsize
        weights[i, :xsize] = ws
    return xmins, sizes, weights


def _resample_axis_f32(arr: np.ndarray, axis: int, out_size: int) -> np.ndarray:
    """interpolate_separable_1d<float>: out = t0 * w0, then out += t_j * w_j. PyTorch's arm64 build fuses the
    multiply-add (one rounding), so each step is computed as float64(out) + float64(t_j) * float64(w_j) (the product
    is exact in float64) rounded to float32; an unfused float32 port differs from torch by up to ~1e-6."""
    xmins, sizes, w = _linear_weights_f32(arr.shape[axis], out_size)
    src = np.moveaxis(arr, axis, 0)
    out = np.empty((out_size,) + src.shape[1:], np.float32)
    for i in range(out_size):
        acc = src[xmins[i]] * w[i, 0]
        for j in range(1, int(sizes[i])):
            acc = (acc.astype(np.float64) + src[xmins[i] + j].astype(np.float64) * np.float64(w[i, j])).astype(np.float32)
        out[i] = acc
    return np.moveaxis(out, 0, axis)


def resize_positions(table: np.ndarray, h: int, w: int) -> np.ndarray:
    """table float32 [S, S, C] (the tower's position embedding as [num_patches] reshaped to S x S) -> [h * w, C]."""
    t = np.asarray(table, np.float32)
    if t.shape[1] != w:
        t = _resample_axis_f32(t, 1, w)
    if t.shape[0] != h:
        t = _resample_axis_f32(t, 0, h)
    return np.ascontiguousarray(t.reshape(h * w, t.shape[2]))


def tower_inputs(tile: Tile, table: np.ndarray, s: VisionSettings = SETTINGS) -> Dict[str, np.ndarray]:
    """One tile -> the tower graph's inputs: pixels [1, N, 768], pos [1, N, C] (rows past the grid repeat row 0, as
    the transformers code fills them; the mask hides them), mask float32 [1, N]."""
    h, w = tile.grid
    pos = resize_positions(table, h, w)
    full = np.repeat(pos[:1], s.max_num_patches, axis=0)
    full[: h * w] = pos
    return {"pixels": tile.pixels[None], "pos": full[None].astype(np.float32),
            "mask": tile.mask.astype(np.float32)[None]}


# --------------------------------------------------------------------------- #
# 6. pixel unshuffle and the projector input
# --------------------------------------------------------------------------- #


def pixel_unshuffle(features: np.ndarray, grid: Tuple[int, int], factor: int = 2) -> np.ndarray:
    """The tower's real-patch features [h * w, C] (raster order) -> [(h/f)(w/f), f * f * C], as
    `Lfm2VlMultiModalProjector.pixel_unshuffle` orders them."""
    h, w = grid
    if h % factor or w % factor:
        raise ValueError(f"grid {grid} is not divisible by {factor}")
    c = features.shape[-1]
    x = np.asarray(features[: h * w]).reshape(h // factor, factor, w // factor, factor, c)
    return np.ascontiguousarray(x.transpose(0, 2, 1, 3, 4).reshape((h // factor) * (w // factor), factor * factor * c))


def projector_input(cells: np.ndarray, rows: int = 256) -> np.ndarray:
    """[n, 4C] -> soft [1, rows, 4C] with zero rows after the n cells."""
    n = cells.shape[0]
    if n > rows:
        raise ValueError(f"{n} cells exceed the projector graph's {rows} rows")
    soft = np.zeros((1, rows, cells.shape[1]), np.float32)
    soft[0, :n] = cells
    return soft


# --------------------------------------------------------------------------- #
# 7. the row's embeddings
# --------------------------------------------------------------------------- #


def insert_embeddings(ids: Sequence[int], text_rows: Callable[[Sequence[int]], np.ndarray], picture_tokens: np.ndarray,
                      image_id: int) -> np.ndarray:
    """[n, d]: text rows everywhere, the picture tokens (in order) at the `<image>` positions."""
    ids = np.asarray(ids, np.int64)
    slots = ids == image_id
    if int(slots.sum()) != picture_tokens.shape[0]:
        raise ValueError(f"{int(slots.sum())} <image> positions, {picture_tokens.shape[0]} picture tokens")
    out = np.empty((len(ids), picture_tokens.shape[1]), np.float32)
    if (~slots).any():
        out[~slots] = text_rows(ids[~slots].tolist())
    out[slots] = picture_tokens
    return out


# --------------------------------------------------------------------------- #
# graphs on the CompiledModel API, and a picture's tokens
# --------------------------------------------------------------------------- #


class LiteRTGraph:
    """One graph through `ai_edge_litert.compiled_model` (signature `serving_default`): named float32 inputs, one
    output, shapes read from the signature. accelerator "cpu" (XNNPACK, `threads`) or "gpu" (precision "fp32" =
    GpuOptions(enforce_f32=True), "default" = the delegate's default precision)."""

    SIGNATURE = "serving_default"

    def __init__(self, path, accelerator: str = "cpu", precision: str = "fp32", threads: int = 4):
        from pathlib import Path

        from ai_edge_litert.compiled_model import CompiledModel, CpuOptions, GpuOptions, HardwareAccelerator, Options

        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"no graph at {self.path}")
        if accelerator == "gpu":
            if precision not in ("fp32", "default"):
                raise ValueError("precision must be 'fp32' or 'default'")
            options = Options(hardware_accelerators=HardwareAccelerator.GPU,
                              gpu_options=GpuOptions(enforce_f32=precision == "fp32"))
        elif accelerator == "cpu":
            options = Options(hardware_accelerators=HardwareAccelerator.CPU, cpu_options=CpuOptions(num_threads=threads))
        else:
            raise ValueError("accelerator must be 'cpu' or 'gpu'")
        self.model = CompiledModel.from_file(str(self.path), options=options)
        ins = self.model.get_input_tensor_details(self.SIGNATURE)
        outs = self.model.get_output_tensor_details(self.SIGNATURE)
        if len(outs) != 1:
            self.model.close()
            raise ValueError(f"{self.path.name} has {len(outs)} outputs, expected 1")
        self.input_shapes = {n: tuple(int(x) for x in d["shape"]) for n, d in ins.items()}
        self.output_name = next(iter(outs))
        self.output_shape = tuple(int(x) for x in outs[self.output_name]["shape"])
        self.inputs = {n: self.model.create_input_buffer_by_name(self.SIGNATURE, n) for n in ins}
        self.outputs = {self.output_name: self.model.create_output_buffer_by_name(self.SIGNATURE, self.output_name)}
        try:
            self.fully_accelerated = bool(self.model.is_fully_accelerated())
        except Exception:   # informational only
            self.fully_accelerated = None

    def __call__(self, **feeds: np.ndarray) -> np.ndarray:
        if sorted(feeds) != sorted(self.inputs):
            raise ValueError(f"{self.path.name} takes {sorted(self.inputs)}, got {sorted(feeds)}")
        for n, v in feeds.items():
            if tuple(v.shape) != self.input_shapes[n]:
                raise ValueError(f"{self.path.name}: {n} must be {self.input_shapes[n]}, got {tuple(v.shape)}")
            self.inputs[n].write(np.ascontiguousarray(v, dtype=np.float32))
        self.model.run_by_name(self.SIGNATURE, self.inputs, self.outputs)
        size = int(np.prod(self.output_shape))
        return np.asarray(self.outputs[self.output_name].read(size, np.float32), np.float32).reshape(self.output_shape)

    def close(self) -> None:
        for buffer in list(self.inputs.values()) + list(self.outputs.values()):
            try:
                buffer.destroy()
            except Exception:
                pass
        self.inputs, self.outputs = {}, {}
        self.model.close()


def picture_tokens(pic: Picture, tower: Callable[..., np.ndarray], projector: Callable[..., np.ndarray],
                   table: np.ndarray, s: VisionSettings = SETTINGS) -> np.ndarray:
    """A picture's tokens [n, d] in the order the row's `<image>` positions take them: per tile (row-major tiles, then
    the thumbnail) the tower, the real patches, the unshuffle, the projector, its leading (h/2)(w/2) rows.
    `tower(pixels=, pos=, mask=)` -> [1, N, H]; `projector(soft=)` -> [1, R, d]; `table` = the position table [S, S, H]."""
    out = []
    for tile in pic.tiles:
        h, w = tile.grid
        features = tower(**tower_inputs(tile, table, s))[0]
        cells = pixel_unshuffle(features[: h * w], tile.grid, s.downsample)
        mm = projector(soft=projector_input(cells))[0]
        out.append(mm[: cells.shape[0]])
    return np.concatenate(out).astype(np.float32)


# --------------------------------------------------------------------------- #
# the picture path as one object (host/d1_litert.py D1Host, item 9)
# --------------------------------------------------------------------------- #


def load_position_table(path) -> np.ndarray:
    """vision_position_table.safetensors (`table`, float32 [S, S, H]) -> the table."""
    from safetensors.numpy import load_file

    t = load_file(str(path))["table"]
    if t.dtype != np.float32 or t.ndim != 3 or t.shape[0] != t.shape[1]:
        raise ValueError(f"position table {t.dtype} {t.shape}, expected float32 [S, S, H]")
    return t


def token_ids(contract: Mapping) -> Dict:
    """contract.json (or its `vision.token_ids`) -> the ids `image_token_ids` takes."""
    t = contract["vision"]["token_ids"] if "vision" in contract else contract
    out: Dict = {k: int(t[k]) for k in ("image", "image_start", "image_end", "img_thumbnail")}
    out["img_row_col"] = {k: int(v) for k, v in t["img_row_col"].items()}
    return out


def open_picture(x):
    """A picture as a request carries it: a PIL image, a path, or the bytes of an image file."""
    from PIL import Image

    if hasattr(x, "convert"):
        return x
    if isinstance(x, (bytes, bytearray)):
        import io

        return Image.open(io.BytesIO(x))
    return Image.open(x)


class VisionPath:
    """The picture path of a host: `pictures` = cap_pixels + preprocess per picture, `tokens` = every picture's tokens
    through the tower and projector graphs (`picture_tokens`), pictures in order. `tower` / `projector` are callables
    with the graphs' signatures (`LiteRTGraph`, or a torch form in a test); `ids` = `token_ids(contract)`."""

    def __init__(self, tower: Callable[..., np.ndarray], projector: Callable[..., np.ndarray], position_table: np.ndarray,
                 ids: Mapping, s: VisionSettings = SETTINGS):
        self.tower, self.projector, self.ids, self.s = tower, projector, ids, s
        self.table = np.asarray(position_table, np.float32)

    @classmethod
    def from_files(cls, tower_file, projector_file, table_file, ids: Mapping, accelerator: str = "cpu",
                   precision: str = "fp32", threads: int = 4) -> "VisionPath":
        tower = LiteRTGraph(tower_file, accelerator, precision, threads)
        projector = LiteRTGraph(projector_file, accelerator, precision, threads)
        return cls(tower, projector, load_position_table(table_file), ids)

    def pictures(self, images: Sequence) -> List[Picture]:
        return [preprocess(cap_pixels(open_picture(x)), self.s) for x in images]

    def tokens(self, pics: Sequence[Picture]) -> np.ndarray:
        if not pics:
            raise ValueError("no pictures")
        return np.concatenate([picture_tokens(p, self.tower, self.projector, self.table, self.s) for p in pics])

    def close(self) -> None:
        for g in (self.tower, self.projector):
            if hasattr(g, "close"):
                g.close()
