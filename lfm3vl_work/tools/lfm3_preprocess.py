"""Host-side image preparation for the LFM3-VL fixed-grid tower (fast_vlm / generic image path), and how it differs
from the Hugging Face processor.

  to_square(path_or_image, g) -> PIL.Image      ImageOps.exif_transpose -> RGB -> resize((g, g), BICUBIC)
  save_square(src, g, dst)                      the same, written as PNG (what the gate feeds the runtime)
  as_input(path_or_image, g) -> np.float32 [1, g, g, 3] in [0, 1]   (the encoder's input tensor, NHWC)

Why a host step: LiteRT-LM v0.18.1 decodes the picture with stb and STRETCHES it to the bundle's image_tensor_height /
width with Catmull-Rom (support/preprocessor/stb_image_preprocessor.cc L171-193, no aspect ratio, no padding, no EXIF
orientation; the fixed-size resize returns u8,
STBIR_TYPE_UINT8_SRGB, L185-191), then hands the encoder float(u8) * (1/255) (L203-210). A g x g PNG asks for a
same-size resize, read as the identity (stbir at scale 1; not measured on the runtime), so what the model sees is
this module's output. The normalisation (the exact u8 back, then (u8 - 127.5) / 127.5 = the HF processor's own
values) and the 16 x 16 patchify are inside the encoder graph (tools/lfm3_vision.py).

Difference from HF: Lfm3VlImageProcessor (image_processing_lfm3_vl.py L50-76, L138-162) keeps the aspect ratio with
smart_resize (both sides rounded to a multiple of 32, 256^2 <= pixels <= 4096^2) and then runs the tower on that
(H/16) x (W/16) grid: (H/32) * (W/32) image tokens, a different count per picture. The fixed grid squashes every
picture to g x g ((g/32)^2 tokens: 256 at 512, 576 at 768); for a g x g input (or one smart_resize maps to g x g)
the two are the same pixels. The measured cost of the squash on the fixture set is on the model card (litert-community/LFM3-VL-2B-LiteRT):
with the source model in float32 every scored answer stayed right (18 of 18) and the first 32 tokens stayed the same
on 16 of 23 questions.
"""
import numpy as np
from PIL import Image, ImageOps


def to_square(src, g):
  im = src if isinstance(src, Image.Image) else Image.open(src)
  return ImageOps.exif_transpose(im).convert("RGB").resize((g, g), Image.BICUBIC)


def save_square(src, g, dst):
  to_square(src, g).save(dst, optimize=False)
  return dst


def as_input(src, g):
  return (np.asarray(to_square(src, g), dtype=np.float32) / 255.0)[None]
