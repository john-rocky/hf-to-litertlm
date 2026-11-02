"""The picture path of the d1-2B host, in numpy: what LFM3-VL's image processor does to one picture.

The provider (`media.Reader._encode`) caps a picture at the checkpoint's pixel budget with Pillow (`media.cap_pixels`,
imported from the snapshot, not ported) and hands it to `Lfm3VlProcessor` (the snapshot's vendored copy, which
transformers 5.19.0 runs because it ships no LFM3: `Lfm3VlImageProcessor` on transformers' `TorchvisionBackend`). For a
picture that processor does, on the CPU:

1. `pil_to_tensor`: the RGB picture as uint8 [3, H, W].
2. `smart_resize(H, W, factor=patch_size * merge_size = 32, min_pixels=size.shortest_edge = 65,536,
   max_pixels=size.longest_edge = 16,777,216)`: both sides multiples of 32 (Python `round`, half to even), the area
   brought inside the bounds by a common factor (floor when shrinking, ceil when growing); a ratio above 200 is refused.
3. `tvF.resize(uint8, [h, w], BICUBIC, antialias=True)`: torchvision 0.29.1 keeps uint8 for bicubic on the CPU, so
   this is PyTorch's uint8 separable antialias kernel: Keys cubic (a = -0.5), support 2 * scale when shrinking (2
   when growing), weights computed in float64 and normalised, scaled to int16 at the largest precision p that keeps
   them below 2^15, applied in integer arithmetic (+ 2^(p-1), >> p, clamp 0..255); width first, then height; a side
   that keeps its size is not resampled (`resize_uint8_bicubic_aa`, ported from the d1-3B host, where the same kernel
   served LFM2-VL; test_host.py checks it bit for bit against the processor).
4. rescale and normalise fused: mean and std times 1 / rescale_factor (float32: 127.5), then (x - 127.5) / 127.5 in
   float32 (subtract, then divide, as `tvF.normalize` does).
5. `patchify`: patches of 16 x 16 in merge-block order (block (i, j) of 2 x 2 patches row-major, its patches (a, b)
   row-major), each patch vector (channel, temporal 2, 16, 16) with both temporal copies equal: [gh * gw, 1536],
   grid (gh, gw) = (h / 16, w / 16).

The processor then writes `<|image|>` gh * gw / 4 times between `<|vision_start|>` and `<|vision_end|>` (D1Host makes
that markup and checks it reads back through the tokenizer, as `Reader._encode` does).
"""

from __future__ import annotations

import json
import math
import os
from typing import Callable

import numpy as np


# --------------------------------------------------------------------------- #
# smart_resize (image_processing_lfm3_vl.py, copied: the module imports torch)
# --------------------------------------------------------------------------- #


def smart_resize(height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56,
                 max_pixels: int = 14 * 14 * 4 * 1280) -> tuple[int, int]:
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


# --------------------------------------------------------------------------- #
# PyTorch's uint8 separable antialias bicubic resize (UpSampleKernel.cpp), ported
# --------------------------------------------------------------------------- #


def _cubic_aa(x: float) -> float:
    """HelperInterpCubic::aa_filter<double, true>: Keys cubic with a = -0.5."""
    a = -0.5
    x = abs(x)
    if x < 1.0:
        return ((a + 2) * x - (a + 3)) * x * x + 1
    if x < 2.0:
        return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a
    return 0.0


def aa_weights_int16(in_size: int, out_size: int, interp_size: int = 4,
                     filt: Callable[[float], float] = _cubic_aa) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """`_compute_index_ranges_int16_weights` (antialias, align_corners False, no scale given): per output index its
    first source index, its number of taps, the int16 weights [out, max_taps] and the precision p."""
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
    """One separable pass (basic_loop_aa_{horizontal,vertical}<uint8_t>) along `axis` of a uint8 array.

    Exact integer arithmetic: int32 when the weights keep every partial sum inside it (255 * the positive weights +
    the rounding term, and 255 * the negative ones, below 2^31: always so for these kernels), else int64; the results
    are the same either way."""
    in_size = arr.shape[axis]
    xmins, sizes, w16, p = aa_weights_int16(in_size, out_size)
    taps = w16.shape[1]
    used = np.arange(taps)[None, :] < sizes[:, None]
    idx = np.where(used, xmins[:, None] + np.arange(taps)[None, :], 0)
    w = np.where(used, w16, 0)
    high = 255 * int(np.where(w > 0, w, 0).sum(1).max()) + (1 << (p - 1))
    low = -255 * int(np.where(w < 0, w, 0).sum(1).min())
    acc_type = np.int32 if max(high, low) < 2 ** 31 else np.int64
    src = np.ascontiguousarray(np.moveaxis(arr, axis, 0))        # uint8 [in, ...], each source line contiguous
    shape = (out_size,) + src.shape[1:]
    lanes = (out_size,) + (1,) * (src.ndim - 1)
    acc = np.full(shape, 1 << (p - 1), dtype=acc_type)
    term = np.empty(shape, dtype=acc_type)
    for t in range(taps):                                        # tap by tap: [out, ...] each, no [out, taps, ...]
        if not used[:, t].any():
            continue
        np.multiply(src[idx[:, t]], w[:, t].astype(acc_type).reshape(lanes), out=term)
        acc += term
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
# the processor's picture settings and steps
# --------------------------------------------------------------------------- #


class PictureProcessor:
    """`Lfm3VlImageProcessor` with the snapshot's processor_config.json, for one capped RGB picture."""

    def __init__(self, snapshot_dir: str):
        with open(os.path.join(snapshot_dir, "processor_config.json")) as f:
            doc = json.load(f)
        self.processor_class = doc.get("processor_class", "Lfm3VlProcessor")
        c = doc["image_processor"]
        if c.get("image_processor_type") != "Lfm3VlImageProcessor":
            raise ValueError(f"the host ports Lfm3VlImageProcessor, not {c.get('image_processor_type')}")
        if not (c.get("do_resize", True) and c.get("do_rescale", True) and c.get("do_normalize", True)):
            raise ValueError("the host ports the processor with resize, rescale and normalize all on")
        if c.get("resample", 3) != 3:  # PILImageResampling.BICUBIC
            raise ValueError(f"the host ports bicubic resampling, not {c.get('resample')}")
        self.patch = int(c.get("patch_size", 16))
        self.merge = int(c.get("merge_size", 2))
        self.temporal = int(c.get("temporal_patch_size", 2))
        size = c.get("size") or {}
        self.min_pixels = int(size.get("shortest_edge", 65_536))
        self.max_pixels = int(size.get("longest_edge", 16_777_216))
        rescale = float(c.get("rescale_factor", 1 / 255))
        # `_fuse_mean_std_and_rescale_factor`: torch.tensor(mean) * (1.0 / rescale_factor), float32
        self.mean = (np.asarray(c.get("image_mean", [0.5] * 3), np.float32) * np.float32(1.0 / rescale)).astype(np.float32)
        self.std = (np.asarray(c.get("image_std", [0.5] * 3), np.float32) * np.float32(1.0 / rescale)).astype(np.float32)

    @property
    def factor(self) -> int:
        return self.patch * self.merge

    def target_size(self, height: int, width: int) -> tuple[int, int]:
        return smart_resize(height, width, factor=self.factor, min_pixels=self.min_pixels, max_pixels=self.max_pixels)

    def __call__(self, image) -> tuple[np.ndarray, tuple[int, int]]:
        """A capped RGB picture (PIL) -> (pixel_values float32 [gh * gw, 3 * temporal * patch * patch], (gh, gw))."""
        if getattr(image, "mode", "RGB") != "RGB":
            raise ValueError("the picture must be RGB (media.cap_pixels converts it)")
        arr = np.asarray(image, dtype=np.uint8)  # [H, W, 3]
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"an RGB picture is [H, W, 3], not {arr.shape}")
        h, w = self.target_size(arr.shape[0], arr.shape[1])
        arr = resize_uint8_bicubic_aa(arr, h, w)
        x = arr.transpose(2, 0, 1).astype(np.float32)  # [3, h, w]
        x = (x - self.mean[:, None, None]) / self.std[:, None, None]
        return self.patchify(x)

    def patchify(self, x: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        """[C, h, w] float32 -> [gh * gw, C * temporal * patch * patch] in merge-block order, and (gh, gw)."""
        c, h, w = x.shape
        p, m, t = self.patch, self.merge, self.temporal
        gh, gw = h // p, w // p
        blocks = x.reshape(c, gh // m, m, p, gw // m, m, p).transpose(1, 4, 2, 5, 0, 3, 6)  # [gh/m, gw/m, a, b, C, py, px]
        out = np.broadcast_to(blocks[:, :, :, :, :, None], blocks.shape[:5] + (t,) + blocks.shape[5:])
        return np.ascontiguousarray(out.reshape(gh * gw, c * t * p * p), dtype=np.float32), (gh, gw)
