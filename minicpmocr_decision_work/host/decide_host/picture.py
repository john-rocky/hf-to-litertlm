"""Host side of the picture path, NumPy only (no torch): what score.py's processor call does to one picture.

score.py passes file paths, so AutoProcessor (transformers 5.10.4) loads Qwen2VLImageProcessor (TorchvisionBackend), reads
the file with torchvision.io.decode_image(mode=RGB) and resizes the uint8 tensor with tvF.resize(BICUBIC, antialias=True).
_do_native_uint8_resize_on_cpu(BICUBIC) is True on every CPU, so the uint8 tensor goes straight into
F.interpolate(mode="bicubic", align_corners=False, antialias=True): torch's fixed-point resampler
(aten/src/ATen/native/cpu/UpsampleKernel.cpp, a port of Pillow's Resample.c with int16 weights):

  per axis, width first, then height (an axis whose size does not change is skipped):
    scale = in / out; support = 2 * max(scale, 1); invscale = 1 / max(scale, 1)
    for each output index i: center = scale * (i + 0.5)
       xmin = max(int(center - support + 0.5), 0); xsize = min(int(center + support + 0.5), in) - xmin
       w[j] = keys_cubic((j + xmin - center + 0.5) * invscale)  (a = -0.5), normalised by their sum (float64)
    precision p = the largest p < 22 with int(0.5 + max_w * 2^(p+1)) < 2^15 (max_w over the whole axis)
    w16[j] = int16(w[j] * 2^p +- 0.5, away from zero)
    out[i] = clip((2^(p-1) + sum_j src[xmin + j] * w16[j]) >> p, 0, 255)      (int32, arithmetic shift, uint8 between the axes)

Then pixel_values = (u8 - 127.5) / 127.5 in float32 (the fused rescale + normalize), merge-block-major patches, each
patch twice along the temporal axis: [N, 3 * 2 * 16 * 16 = 1536].
"""
from __future__ import annotations

import math

import numpy as np

FACTOR, MIN_PIXELS, MAX_PIXELS = 32, 65536, 16777216
PATCH, MERGE, TEMPORAL = 16, 2, 2


def smart_resize(height: int, width: int) -> tuple[int, int]:
    """qwen2_vl smart_resize with factor 32 and the checkpoint's budget. round() is Python's (half to even)."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError("aspect ratio over 200")
    h_bar, w_bar = round(height / FACTOR) * FACTOR, round(width / FACTOR) * FACTOR
    if h_bar * w_bar > MAX_PIXELS:
        beta = math.sqrt((height * width) / MAX_PIXELS)
        h_bar = max(FACTOR, math.floor(height / beta / FACTOR) * FACTOR)
        w_bar = max(FACTOR, math.floor(width / beta / FACTOR) * FACTOR)
    elif h_bar * w_bar < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (height * width))
        h_bar = math.ceil(height * beta / FACTOR) * FACTOR
        w_bar = math.ceil(width * beta / FACTOR) * FACTOR
    return h_bar, w_bar


def _keys_cubic(x: np.ndarray) -> np.ndarray:
    a = -0.5
    x = np.abs(x)
    near = ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    far = (((x - 5.0) * x + 8.0) * x - 4.0) * a
    return np.where(x < 1.0, near, np.where(x < 2.0, far, 0.0))


def _axis_weights(in_size: int, out_size: int):
    """-> xmin [out], w16 [out, k] int32, precision."""
    scale = in_size / out_size
    fscale = max(scale, 1.0)
    support = 2.0 * fscale
    invscale = 1.0 / fscale
    k = int(math.ceil(support)) * 2 + 1
    center = scale * (np.arange(out_size, dtype=np.float64) + 0.5)
    xmin = np.maximum((center - support + 0.5).astype(np.int64), 0)
    xsize = np.minimum((center + support + 0.5).astype(np.int64), in_size) - xmin
    xsize = np.clip(xsize, 0, k)
    j = np.arange(k, dtype=np.float64)[None, :]
    w = _keys_cubic((j + xmin[:, None] - center[:, None] + 0.5) * invscale)
    w = np.where(j < xsize[:, None], w, 0.0)
    total = w.sum(axis=1, keepdims=True)
    # float64 running sum in index order, as the kernel: np.sum of a short axis adds in order for k < 8 only, so redo it
    total = np.cumsum(w, axis=1)[:, -1:]
    w = np.where(total != 0.0, w / np.where(total != 0.0, total, 1.0), w)
    max_w = float(w.max())
    p = 0
    while p < 22:
        if int(0.5 + max_w * (1 << (p + 1))) >= (1 << 15):
            break
        p += 1
    v = w * (1 << p)
    w16 = np.where(v < 0, v - 0.5, v + 0.5).astype(np.int64).astype(np.int16).astype(np.int32)
    return xmin, xsize, w16, p


def _resize_axis(src: np.ndarray, out_size: int, axis: int) -> np.ndarray:
    """src uint8 [..]; resample along `axis` (0 = height, 1 = width of an HWC array)."""
    in_size = src.shape[axis]
    xmin, xsize, w16, p = _axis_weights(in_size, out_size)
    a = np.moveaxis(src, axis, 0).astype(np.int32)                         # [in, other, C]
    out = np.full((out_size,) + a.shape[1:], 1 << (p - 1), dtype=np.int32)
    for j in range(w16.shape[1]):
        idx = np.minimum(xmin + j, in_size - 1)
        mask = j < xsize
        if not mask.any():
            break
        wj = w16[:, j].reshape((out_size,) + (1,) * (a.ndim - 1))
        out += a[idx] * wj
    out = np.clip(out >> p, 0, 255).astype(np.uint8)
    return np.moveaxis(out, 0, axis)


def resize_uint8(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """uint8 HWC -> uint8 HWC, torch's uint8 bicubic antialias (width first, then height)."""
    out = image
    if out.shape[1] != width:
        out = _resize_axis(out, width, 1)
    if out.shape[0] != height:
        out = _resize_axis(out, height, 0)
    return out


def to_pixel_values(image: np.ndarray) -> np.ndarray:
    """uint8 [H, W, 3] with H, W multiples of 32 -> float32 [N, 1536], the processor's order."""
    h, w = image.shape[:2]
    gh, gw = h // PATCH, w // PATCH
    # the processor fuses rescale and normalize: mean = std = 0.5 * 255 = 127.5, float32 (image_processing_backends.py L320-345)
    x = (image.astype(np.float32) - np.float32(127.5)) / np.float32(127.5)
    x = np.transpose(x, (2, 0, 1))                         # [C, H, W]
    x = x.reshape(3, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH)
    x = np.transpose(x, (1, 4, 2, 5, 0, 3, 6))             # [gh/2, gw/2, merge, merge, C, patch, patch]
    x = np.repeat(x[:, :, :, :, :, None, :, :], TEMPORAL, axis=5)
    return x.reshape(gh * gw, 3 * TEMPORAL * PATCH * PATCH).astype(np.float32)
