"""Picture -> the tower graph's patches, at one fixed square tile per tower graph (256 -> 8 x 8 = 64 picture tokens,
448 -> 14 x 14 = 196).

    open -> convert("RGB") -> resize((tile, tile), BICUBIC)        Pillow's own integer resampler; the aspect is NOT kept
    -> /255 -> (x - 0.5) / 0.5 in float64 -> float32
    -> merge-block-major patches [4 G^2, 1536], G = tile // 32, each vector (C, T=2, 16, 16)

This is what the reference's g256 / g448 arms fed the author's processor (the picture resized with Pillow before the
processor, whose own resize is the identity at multiples of 32), and the patches are the processor's `pixel_values`
bit for bit. The author's native arm (the processor's dynamic resolution) is not a tower graph here. EXIF orientation
is not applied (the reference passed opened PIL images, which the processor takes as they are)."""
from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np

from .errors import UnsupportedMedia

PATCH, MERGE, TEMPORAL = 16, 2, 2
IMAGE_MEAN, IMAGE_STD = 0.5, 0.5
TILES = (256, 448)


def tile_grid(tile: int) -> int:
    """Merged rows per side of a square tile (256 -> 8, 448 -> 14)."""
    if tile not in TILES:
        raise UnsupportedMedia(f"tile {tile}: the tower graphs read {' or '.join(map(str, TILES))} px tiles")
    return tile // (PATCH * MERGE)


def open_image(image):
    """A PIL image, a uint8 [H, W, 3] array, the bytes of an image file, a local path, or a base64 data URI -> PIL."""
    from PIL import Image

    if isinstance(image, Image.Image):
        return image
    if isinstance(image, np.ndarray):
        return Image.fromarray(np.asarray(image, dtype=np.uint8))
    try:
        if isinstance(image, (bytes, bytearray)):
            return Image.open(io.BytesIO(bytes(image)))
        if isinstance(image, str) and image.startswith("data:image/"):
            return Image.open(io.BytesIO(base64.b64decode(image.split(",", 1)[1])))
        if isinstance(image, (str, Path)) and Path(image).is_file():
            return Image.open(image)
    except Exception as e:                          # Pillow's UnidentifiedImageError, a bad base64 string, ...
        raise UnsupportedMedia(f"the picture could not be opened: {e}") from e
    raise UnsupportedMedia("a picture is a PIL image, a uint8 array, image file bytes, a local path or a data URI "
                           "(the host does not fetch URLs)")


def patchify(x: np.ndarray) -> np.ndarray:
    """Normalized [H, W, C] -> [gh*gw, C*T*P*P] in Qwen's merge-block-major order: patches iterate
    (block_row, block_col, y-in-block, x-in-block), channel outermost inside the vector, the still frame repeated at
    both temporal slots (the Qwen2VLImageProcessor reshape / permute)."""
    h, w, c = x.shape
    if h % (PATCH * MERGE) or w % (PATCH * MERGE):
        raise ValueError(f"{h}x{w} is not divisible by patch * merge = {PATCH * MERGE}")
    gh, gw = h // PATCH, w // PATCH
    t = x.transpose(2, 0, 1).reshape(c, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH)
    t = t.transpose(1, 4, 2, 5, 0, 3, 6).reshape(gh * gw, 1, c, PATCH, PATCH)
    t = np.broadcast_to(t[:, :, :, None], (gh * gw, 1, c, TEMPORAL, PATCH, PATCH))
    return t.reshape(gh * gw, c * TEMPORAL * PATCH * PATCH)


def tile_pixels(image, tile: int) -> np.ndarray:
    """Picture -> uint8 [tile, tile, 3]: RGB, Pillow BICUBIC to the tile."""
    from PIL import Image

    tile_grid(tile)
    return np.asarray(open_image(image).convert("RGB").resize((tile, tile), Image.Resampling.BICUBIC))


def tile_patches(image, tile: int) -> np.ndarray:
    """Picture -> patches [4 G^2, 1536] float32, G = tile // 32 (the tower graph's input)."""
    x = (tile_pixels(image, tile).astype(np.float64) / 255.0 - IMAGE_MEAN) / IMAGE_STD
    return np.ascontiguousarray(patchify(x).astype(np.float32))
