"""d1-omni-600M on LiteRT: the picture path of the reference host (numpy + Pillow only; no PyTorch, no transformers).

Every step is a copy of what the provider's code (LiquidAI/d1-omni-600M at
414f8d64, `vision.py` `layout` / `preprocess` / `Vision.forward` / `Projector`) and transformers 5.19.0
(`load_image`, `Siglip2VisionEmbeddings.resize_positional_embeddings`) do for a request with pictures, written so that
a host in any language can follow it.

1. `load_image`: Pillow open, EXIF orientation applied (`ImageOps.exif_transpose`), RGB (= transformers' load_image,
   the reader the model card uses).
2. `layout(width, height)` (the provider's, verbatim): factor 32; the thumbnail / single-crop size (h, w) =
   max(32, round(side / 32) * 32) with Python's round (half to even); above 256 * 1024 pixels both sides become
   max(32, floor(side / beta / 32) * 32), beta = sqrt(h * w / 262144); below 64 * 1024 pixels
   ceil(side * beta / 32) * 32, beta = sqrt(65536 / (h * w)). The picture is split ("tiled") when
   max(16, round(h / 32) * 32) * max(16, round(w / 32) * 32) > 524,288; the grid (cols, rows) is the closest ratio
   cols / rows to w / h among the grids of 2..10 tiles sorted by tile count (a tie goes to the later grid when
   w * h > 0.5 * 512 * 512 * cols * rows).
3. `preprocess`: a tiled picture is resized to (rows * 512, cols * 512) and cut into 512 x 512 tiles in row-major
   order; the thumbnail (the whole picture at the layout size) follows; a small picture is the thumbnail alone.
   Resizing = the FLOAT path (the oracle's, ref/records_ref.json `load.resize_path`: torchvision 0.24's resize_image
   for a uint8 CPU tensor): uint8 -> float32 -> bilinear, antialias, align_corners False (PyTorch's separable kernel:
   width first, then height; a side that keeps its size is skipped; a picture that keeps its size is returned as it
   is) -> round half to even -> clamp 0..255 -> uint8. Each crop: float32 (x - 127.5) / 127.5, 16 x 16 patches in
   raster order, each flattened as (row, column, channel) = 768 values, zero patches after the real ones up to 1024;
   `mask` 1 for a real patch; the patch grid (ph, pw) = the crop size / 16.
4. `resize_positions`: the tower's 16 x 16 position table (checkpoint `vision.tower.vision_model.embeddings.
   position_embedding.weight` [256, 768] reshaped row-major) resized to (ph, pw) with bilinear antialias
   (align_corners False), float32 arithmetic in the kernel's order; rows in raster order; the rows past the grid
   repeat row 0 (as transformers fills them; the mask hides them).
5. The tower graph maps pixels [1,1024,768] + pos [1,1024,768] + mask [1,1024] to features [1,1024,768]; the real
   patches are the first ph * pw rows.
6. `pixel_unshuffle`: the real features as a (ph, pw, 768) grid -> (ph/2, pw/2, 3072): output channel
   j * 1536 + k * 768 + c of cell (r, q) is input (2r + j, 2q + k, c); cells in raster order (the provider's
   Projector reshape / permute). The projector graph maps soft [1,256,3072] (the cells, zero rows after them) to
   prefix [1,256,1024]; the first (ph/2)(pw/2) rows are the crop's prefix embeddings.
7. `image_prefix`: crops in order (tiles row-major, then the thumbnail), each crop's rows concatenated -> [P, 1024],
   P = sum over crops of (ph/2)(pw/2). This is the `prefix` input of the decision graph (media = 1 at 0..P-1).
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

TILE, PATCH, MAX_PATCHES = 512, 16, 1024
FACTOR, MAX_PIXELS, MIN_PIXELS = 32, 256 * 1024, 64 * 1024
PROJ_ROWS = 256
HIDDEN = 768
POSITION_KEY = "vision.tower.vision_model.embeddings.position_embedding.weight"


# --------------------------------------------------------------------------- 1-2. reading and layout


def load_image(path):
    """transformers.image_utils.load_image for a local file: open, EXIF transpose, RGB."""
    from PIL import Image, ImageOps

    image = Image.open(str(path))
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def layout(width: int, height: int) -> dict:
    """The provider's vision.layout (LFM2-VL's smart resize, tile grid and thumbnail), verbatim."""
    if min(width, height) < 1:
        raise ValueError("empty image")
    factor, maximum, minimum = FACTOR, MAX_PIXELS, MIN_PIXELS
    h, w = max(factor, round(height / factor) * factor), max(factor, round(width / factor) * factor)
    if h * w > maximum:
        beta = math.sqrt(height * width / maximum)
        h = max(factor, math.floor(height / beta / factor) * factor)
        w = max(factor, math.floor(width / beta / factor) * factor)
    elif h * w < minimum:
        beta = math.sqrt(minimum / (height * width))
        h = math.ceil(height * beta / factor) * factor
        w = math.ceil(width * beta / factor) * factor
    large = max(16, round(height / factor) * factor) * max(16, round(width / factor) * factor) > maximum * 2
    grid = (1, 1)
    if large:
        ratios = sorted({(x, y) for n in range(2, 11) for x in range(1, n + 1) for y in range(1, n + 1)
                         if 2 <= x * y <= 10}, key=lambda r: r[0] * r[1])
        best = float("inf")
        for ratio in ratios:
            diff = abs(width / height - ratio[0] / ratio[1])
            if diff < best or (diff == best and width * height > 0.5 * TILE * TILE * ratio[0] * ratio[1]):
                grid, best = ratio, diff
    return {"grid": grid, "thumbnail": (h, w), "tiled": large}


# --------------------------------------------------------------------------- 3-4. PyTorch's float32 bilinear
# antialias interpolation (interpolate_separable_1d<float>), ported


def _linear_weights_f32(in_size: int, out_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HelperInterpLinear::compute_index_ranges_weights<float> (antialias, align_corners False): float32 weights, as
    the kernel rounds them (float * double products in double, stored as float)."""
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
    """One separable pass along `axis` of a float32 array: out = t0 * w0, then out += t_j * w_j. PyTorch's arm64 build
    fuses the multiply-add (one rounding), so each step is float64(out) + float64(t_j) * float64(w_j) (the product is
    exact in float64) rounded to float32."""
    xmins, sizes, w = _linear_weights_f32(arr.shape[axis], out_size)
    src = np.moveaxis(np.asarray(arr, np.float32), axis, 0)
    out = np.empty((out_size,) + src.shape[1:], np.float32)
    for i in range(out_size):
        acc = src[xmins[i]] * w[i, 0]
        for j in range(1, int(sizes[i])):
            acc = (acc.astype(np.float64) + src[xmins[i] + j].astype(np.float64) * np.float64(w[i, j])).astype(np.float32)
        out[i] = acc
    return np.moveaxis(out, 0, axis)


def resize_float(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """uint8 [H, W, 3] -> uint8 [out_h, out_w, 3] on the float path: float32, bilinear antialias (width, then height;
    an unchanged side skipped), round half to even, clamp 0..255, uint8. An unchanged size returns the input."""
    h, w = arr.shape[:2]
    if (out_h, out_w) == (h, w):
        return arr
    x = arr.astype(np.float32)
    if out_w != w:
        x = _resample_axis_f32(x, 1, out_w)
    if out_h != h:
        x = _resample_axis_f32(x, 0, out_h)
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- 3. crops and patches


def to_patches(crop_u8: np.ndarray) -> Dict[str, np.ndarray]:
    """A resized uint8 [h, w, 3] crop -> pixels float32 [1024, 768] (zero rows after the real patches), mask int32
    [1024], grid (ph, pw)."""
    x = (crop_u8.astype(np.float32) - np.float32(127.5)) / np.float32(127.5)
    ph, pw = x.shape[0] // PATCH, x.shape[1] // PATCH
    n = ph * pw
    if n > MAX_PATCHES:
        raise ValueError(f"a crop of {n} patches exceeds {MAX_PATCHES}")
    p = x.reshape(ph, PATCH, pw, PATCH, 3).transpose(0, 2, 1, 3, 4).reshape(n, PATCH * PATCH * 3)
    pixels = np.zeros((MAX_PATCHES, PATCH * PATCH * 3), np.float32)
    pixels[:n] = p
    mask = np.zeros(MAX_PATCHES, np.int32)
    mask[:n] = 1
    return {"pixels": pixels, "mask": mask, "grid": (ph, pw)}


def crops_of(image) -> Tuple[List[np.ndarray], dict]:
    """A PIL image -> the uint8 crops in the provider's order (tiles row-major, then the thumbnail) and the plan."""
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = arr.shape[:2]
    plan = layout(width, height)
    crops = []
    if plan["tiled"]:
        gw, gh = plan["grid"]
        big = resize_float(arr, gh * TILE, gw * TILE)
        crops = [big[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE] for r in range(gh) for c in range(gw)]
    th, tw = plan["thumbnail"]
    crops.append(resize_float(arr, th, tw))
    return crops, {**plan, "input_hw": (height, width)}


def preprocess(image) -> Dict[str, np.ndarray]:
    """The provider's preprocess(): pixel_values float32 [C, 1024, 768], spatial_shapes int64 [C, 2],
    pixel_attention_mask int32 [C, 1024] (C = crops)."""
    crops, plan = crops_of(image)
    cs = [to_patches(c) for c in crops]
    return {"pixel_values": np.stack([c["pixels"] for c in cs]),
            "spatial_shapes": np.asarray([c["grid"] for c in cs], np.int64),
            "pixel_attention_mask": np.stack([c["mask"] for c in cs]), "plan": plan}


def prefix_length(spatial_shapes) -> int:
    return int(sum((int(h) // 2) * (int(w) // 2) for h, w in np.asarray(spatial_shapes).tolist()))


# --------------------------------------------------------------------------- 4. position table


def read_position_table(weights_path) -> np.ndarray:
    """The checkpoint's position table as float32 [16, 16, 768], read with a plain safetensors parse (8-byte header
    length, JSON header, raw little-endian data); only this tensor's bytes are read."""
    with open(weights_path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
        meta = header[POSITION_KEY]
        if meta["dtype"] != "F32" or meta["shape"] != [256, HIDDEN]:
            raise ValueError(f"unexpected position table {meta['dtype']} {meta['shape']}")
        a, b = meta["data_offsets"]
        f.seek(8 + n + a)
        raw = f.read(b - a)
    return np.frombuffer(raw, dtype="<f4").astype(np.float32).reshape(16, 16, HIDDEN).copy()


def load_position_table(npy_path) -> np.ndarray:
    """The position table as shipped (host/vision_position_table.npy = read_position_table's output, written once from
    the checkpoint): float32 [16, 16, 768]."""
    t = np.load(str(npy_path), allow_pickle=False)
    if t.dtype != np.float32 or t.shape != (16, 16, HIDDEN):
        raise ValueError(f"{npy_path}: expected float32 (16, 16, {HIDDEN}), got {t.dtype} {t.shape}")
    return t


def resize_positions(table: np.ndarray, h: int, w: int) -> np.ndarray:
    """table float32 [S, S, C] -> [h * w, C] (bilinear antialias, width then height, unchanged sides skipped)."""
    t = np.asarray(table, np.float32)
    if t.shape[1] != w:
        t = _resample_axis_f32(t, 1, w)
    if t.shape[0] != h:
        t = _resample_axis_f32(t, 0, h)
    return np.ascontiguousarray(t.reshape(h * w, t.shape[2]))


def positions_padded(table: np.ndarray, grid: Tuple[int, int]) -> np.ndarray:
    """[1024, C]: the resized rows, then row 0 repeated (transformers' padding of the resized table)."""
    h, w = grid
    pos = resize_positions(table, h, w)
    full = np.repeat(pos[:1], MAX_PATCHES, axis=0)
    full[: h * w] = pos
    return full


def tower_inputs(crop: Dict[str, np.ndarray], table: np.ndarray) -> Dict[str, np.ndarray]:
    """One crop -> the tower graph's inputs: pixels [1, 1024, 768], pos [1, 1024, 768], mask float32 [1, 1024]."""
    return {"pixels": crop["pixels"][None], "pos": positions_padded(table, crop["grid"])[None].astype(np.float32),
            "mask": crop["mask"].astype(np.float32)[None]}


# --------------------------------------------------------------------------- 6. unshuffle and the projector input


def pixel_unshuffle(features: np.ndarray, grid: Tuple[int, int], factor: int = 2) -> np.ndarray:
    """The real-patch features [h * w, C] (raster order) -> [(h/f)(w/f), f * f * C] in the provider's Projector order:
    channel j * f * C + k * C + c of cell (r, q) = input (f r + j, f q + k, c)."""
    h, w = grid
    if h % factor or w % factor:
        raise ValueError(f"grid {grid} is not divisible by {factor}")
    c = features.shape[-1]
    x = np.asarray(features[: h * w]).reshape(h // factor, factor, w // factor, factor, c)
    return np.ascontiguousarray(x.transpose(0, 2, 1, 3, 4).reshape((h // factor) * (w // factor), factor * factor * c))


def projector_input(cells: np.ndarray, rows: int = PROJ_ROWS) -> np.ndarray:
    """[n, 4C] -> soft [1, rows, 4C] with zero rows after the n cells."""
    n = cells.shape[0]
    if n > rows:
        raise ValueError(f"{n} cells exceed the projector graph's {rows} rows")
    soft = np.zeros((1, rows, cells.shape[1]), np.float32)
    soft[0, :n] = cells
    return soft


# --------------------------------------------------------------------------- 7. a picture's prefix


def image_prefix(image, tower, projector, table: np.ndarray) -> np.ndarray:
    """A PIL image -> prefix [P, 1024]: per crop (tiles row-major, then the thumbnail) the tower, its real patches,
    the unshuffle, the projector, the first (ph/2)(pw/2) rows. `tower(pixels=, pos=, mask=)` -> [1, 1024, 768];
    `projector(soft=)` -> [1, 256, 1024]."""
    crops, _ = crops_of(image)
    out = []
    for c in crops:
        crop = to_patches(c)
        h, w = crop["grid"]
        features = tower(**tower_inputs(crop, table))[0]
        cells = pixel_unshuffle(features[: h * w], (h, w))
        out.append(projector(soft=projector_input(cells))[0][: cells.shape[0]])
    return np.concatenate(out).astype(np.float32)


class LiteRTGraph:
    """One graph through ai_edge_litert's CompiledModel (its only signature): named float32 inputs, one output, shapes
    read from the signature. accelerator "cpu" (XNNPACK, `threads`) or "gpu" (precision "fp32" =
    GpuOptions(enforce_f32=True), "default" = the delegate's default precision)."""

    def __init__(self, path, accelerator: str = "cpu", precision: str = "fp32", threads: int = 4):
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
        self.signature = next(iter(self.model.get_signature_list()))
        ins = self.model.get_input_tensor_details(self.signature)
        outs = self.model.get_output_tensor_details(self.signature)
        if len(outs) != 1:
            self.model.close()
            raise ValueError(f"{self.path.name} has {len(outs)} outputs, expected 1")
        self.input_shapes = {n: tuple(int(x) for x in d["shape"]) for n, d in ins.items()}
        self.output_name = next(iter(outs))
        self.output_shape = tuple(int(x) for x in outs[self.output_name]["shape"])
        self.inputs = {n: self.model.create_input_buffer_by_name(self.signature, n) for n in ins}
        self.outputs = {self.output_name: self.model.create_output_buffer_by_name(self.signature, self.output_name)}
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
        self.model.run_by_name(self.signature, self.inputs, self.outputs)
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
