#!/usr/bin/env python3
"""Read a picture with MiniCPM-OCR-1.0's LiteRT-LM bundle: one padded 448x448 picture for a text region, or the
model's own page tiling (an overview plus up to 9 slices of 448x448 in one message) for a page.

  python ocr_page.py MiniCPM-OCR-1.0-VL_int8.litertlm page.jpg [--prompt "Text Recognition:"]
      [--backend cpu|gpu] [--vision-backend cpu|gpu] [--tile auto|always|never] [--cache-dir DIR]

Needs litert-lm (pip install litert-lm) and Pillow. The bundle resizes every picture to 448x448 without keeping the
aspect ratio, so this script prepares each picture at 448x448 itself:
  - a region (area up to 2 x 448^2 by default): pasted centred on a white square, then resized to 448x448;
  - a page (larger): the grid of MiniCPMOcrImageProcessor.get_sliced_grid (an overview + at most 9 slices, scale 448) - the whole
    picture padded to a white square as the overview, then every grid cell resized to 448x448 - sent as the images of
    one message, with the extra-context variable `slice_cols` (the grid's columns) that tells the bundle's chat
    template to mark the 2nd and later images as slices, row by row, the way the model's processor lays them out.
EXIF orientation is applied first (the runtime does not read it). One Engine per call: the text decoder carries a
recurrent state, so a new conversation needs a new Engine (LiteRT-LM issue #3165).
"""
import argparse
import base64
import io
import math
import sys

from PIL import Image, ImageOps

G = 448            # the bundle's image size (fast_vlm 448x448 = 256 image tokens)
MAX_SLICES = 9     # MiniCPMOcrImageProcessor.max_slice_nums
SCALE = 448        # MiniCPMOcrImageProcessor.scale_resolution


def sliced_grid(width, height, max_slice_nums=MAX_SLICES, scale_resolution=SCALE):
  """[rows, cols] of the processor's slicing, or None when it does not slice (transformers
  MiniCPMOcrImageProcessor.get_sliced_grid, Apache-2.0, same arithmetic)."""
  log_ratio = math.log(width / height)
  ratio = width * height / (scale_resolution * scale_resolution)
  multiple = min(math.ceil(ratio), max_slice_nums)
  if multiple <= 1:
    return None
  best, min_error = [1, 1], float("inf")
  for num_slices in (multiple - 1, multiple, multiple + 1):
    if num_slices == 1 or num_slices > max_slice_nums:
      continue
    for rows in range(1, num_slices + 1):
      if num_slices % rows == 0:
        cols = num_slices // rows
        error = abs(log_ratio - math.log(cols / rows))
        if error < min_error or (error == min_error and rows > best[0]):
          best, min_error = [rows, cols], error
  return best


def pad_square(img, g=G):
  side = max(img.size)
  square = Image.new("RGB", (side, side), "white")
  square.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
  return square.resize((g, g), Image.BICUBIC)


def cells(img, grid, g=G):
  rows, cols = grid
  w, h = img.size
  out = []
  for r in range(rows):
    for c in range(cols):
      box = (round(w * c / cols), round(h * r / rows), round(w * (c + 1) / cols), round(h * (r + 1) / rows))
      out.append(img.crop(box).resize((g, g), Image.BICUBIC))
  return out


def pictures(img, tile="auto", tile_above=2.0):
  """-> (list of 448x448 pictures, slice_cols or None)."""
  grid = sliced_grid(img.width, img.height)
  big = img.width * img.height > tile_above * G * G
  if grid is None or tile == "never" or (tile == "auto" and not big):
    return [pad_square(img)], None
  return [pad_square(img)] + cells(img, grid), grid[1]


def png_item(picture):
  buf = io.BytesIO()
  picture.save(buf, format="PNG")
  return {"type": "image", "blob": base64.b64encode(buf.getvalue()).decode("ascii")}


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("bundle")
  ap.add_argument("image")
  ap.add_argument("--prompt", default="Text Recognition:")
  ap.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
  ap.add_argument("--vision-backend", choices=("cpu", "gpu"), default=None, help="default: the --backend value")
  ap.add_argument("--tile", choices=("auto", "always", "never"), default="auto")
  ap.add_argument("--max-output-tokens", type=int, default=1536)
  ap.add_argument("--cache-dir", default=None, help="where the runtime writes its caches (default: beside the bundle)")
  a = ap.parse_args()
  import litert_lm

  try:
    img = ImageOps.exif_transpose(Image.open(a.image)).convert("RGB")
  except (OSError, ValueError) as e:
    sys.exit(f"cannot read the picture {a.image!r}: {e}")
  pics, slice_cols = pictures(img, a.tile)
  backend = {"cpu": litert_lm.interfaces.Backend.CPU, "gpu": litert_lm.interfaces.Backend.GPU}
  engine = litert_lm.Engine(a.bundle, backend=backend[a.backend](),
                            vision_backend=backend[a.vision_backend or a.backend](),
                            max_num_images=len(pics), cache_dir=a.cache_dir)
  conv = engine.create_conversation(**({"extra_context": {"slice_cols": slice_cols}} if slice_cols else {}))
  content = [png_item(p) for p in pics] + [{"type": "text", "text": a.prompt}]
  reply = conv.send_message({"role": "user", "content": content}, max_output_tokens=a.max_output_tokens)
  parts = reply.get("content", [])
  text = "".join(p.get("text", "") for p in parts if isinstance(p, dict)) if isinstance(parts, list) else str(parts)
  print(f"[{img.width}x{img.height}: {len(pics)} picture(s), slice_cols={slice_cols}]", file=sys.stderr)
  print(text)


if __name__ == "__main__":
  main()
