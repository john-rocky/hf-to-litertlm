#!/usr/bin/env python3
"""App-side tiling of one picture for a fixed-size (GxG) fast_vlm bundle of Xiaomi-OCR-0 (a form measured during the
conversion and not shipped: the 768-pixel tiles of a page did not fit a 4,096-token KV cache).

Qwen3.5's processor has no slice marks: every tile is one more image of the same user message
(<|vision_start|><|image_pad|><|vision_end|> per image, no separator, no "Picture N:"), and the runtime fills the
rendered image tokens in order. The rule:
  1. (H', W') = the processor's own target for the picture (smart_resize, multiples of 32) = what native sees.
  2. R = ceil(f H' / G), C = ceil(f W' / G) with f = 1 if R C <= N_MAX (12); else the (R, C) with R C <= N_MAX that
     keeps the largest f = min(R G / H', C G / W') (one uniform down-scale, never an up-scale).
  3. The EXIF-applied picture is resized (PIL BICUBIC) to (round32(f W'), round32(f H')) and cut at 32-px-aligned
     boundaries b_k = round(k L / n / 32) * 32 into R x C tiles (each <= G x G), each pasted top-left on a white G x G.
  4. Images of the message: [thumbnail = the picture padded to a white square, then G x G (the p{G} picture)] and
     the tiles row-major; t{G}n = the tiles alone.
"""
import math

from PIL import Image

N_MAX = 12


def smart_resize(h, w, factor=32, min_pixels=65536, max_pixels=16777216):
    """transformers.models.qwen2_vl.image_processing_qwen2_vl.smart_resize with this repo's processor_config."""
    h_bar = round(h / factor) * factor
    w_bar = round(w / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        h_bar = max(factor, math.floor(h / beta / factor) * factor)
        w_bar = max(factor, math.floor(w / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h_bar = math.ceil(h * beta / factor) * factor
        w_bar = math.ceil(w * beta / factor) * factor
    return h_bar, w_bar


def plan(h_nat, w_nat, g, n_max=N_MAX):
    R, C = math.ceil(h_nat / g), math.ceil(w_nat / g)
    if R * C <= n_max:
        return R, C, 1.0
    best = None
    for r in range(1, n_max + 1):
        for c in range(1, n_max // r + 1):
            f = min(r * g / h_nat, c * g / w_nat, 1.0)
            if best is None or f > best[2] + 1e-12:
                best = (r, c, f)
    return best


def cuts(length, n):
    b = [int(round(k * length / n / 32.0)) * 32 for k in range(n + 1)]
    b[-1] = length
    return b


def pad_square(img, g):
    s = max(img.size)
    sq = Image.new("RGB", (s, s), "white")
    sq.paste(img, ((s - img.width) // 2, (s - img.height) // 2))
    return sq.resize((g, g), Image.BICUBIC)


def tile_images(img, g, thumbnail=True):
    """img: RGB, EXIF already applied. -> (list of GxG images, meta)."""
    h_nat, w_nat = smart_resize(img.height, img.width)
    R, C, f = plan(h_nat, w_nat, g)
    H = max(32, int(round(f * h_nat / 32.0)) * 32)
    W = max(32, int(round(f * w_nat / 32.0)) * 32)
    big = img.resize((W, H), Image.BICUBIC)
    ys, xs = cuts(H, R), cuts(W, C)
    tiles, boxes = [], []
    for r in range(R):
        for c in range(C):
            box = (xs[c], ys[r], xs[c + 1], ys[r + 1])
            assert box[2] - box[0] <= g and box[3] - box[1] <= g, (box, g)
            canvas = Image.new("RGB", (g, g), "white")
            canvas.paste(big.crop(box), (0, 0))
            tiles.append(canvas)
            boxes.append(list(box))
    imgs = ([pad_square(img, g)] if thumbnail else []) + tiles
    meta = {"tiling": {"native_hw": [h_nat, w_nat], "grid_rc": [R, C], "scale": round(f, 6), "resized_hw": [H, W],
                       "boxes_xyxy": boxes, "thumbnail": thumbnail, "n_images": len(imgs), "g": g}}
    return imgs, meta
