"""Export the Shieldstral vision graphs for a hand-assembled .litertlm.

  VISION_ENCODER : [1, S, S, 3] NHWC in [0,1]  ->  [1, G*G, D_vision]
  VISION_ADAPTER : [1, G*G, D_vision]          ->  [1, G*(G/m+... ), D_text]

The runtime hands the encoder an NHWC image already scaled to [0,1] and does no
normalisation, so the CLIP mean/std and the NCHW transpose are baked into the
encoder graph. The adapter carries the 2x2 patch merge, the projector, and the
constant `[IMG_BREAK]` / `[IMG_END]` rows that pixtral's token expansion requires
(see verify_adapter.py — the block is corr 1.0 against transformers' own embeddings).

    python convert_shieldstral_vision.py <checkpoint> <out_dir> [--size 560]
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import transformers

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from static_pixtral import StaticPatchMerger, StaticPixtralTower  # noqa: E402
from verify_adapter import VisionAdapter  # noqa: E402

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


class VisionEncoder(nn.Module):
  """NHWC [0,1] image -> pixtral tower features, normalisation baked in."""

  def __init__(self, tower, size):
    super().__init__()
    self.tower = StaticPixtralTower(tower, size)
    self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1), persistent=False)
    self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1), persistent=False)

  def forward(self, image_nhwc):                 # [1, S, S, 3] in [0,1]
    x = image_nhwc.permute(0, 3, 1, 2)           # -> NCHW
    x = (x - self.mean) / self.std
    return self.tower(x)


class ProjTail(nn.Module):
  """projector norm -> static 2x2 merge -> linear_1 -> act -> linear_2."""

  def __init__(self, proj, merger):
    super().__init__()
    self.p, self.merger = proj, merger

  def forward(self, tower_out):                  # [1, G*G, D_v]
    x = self.p.norm(tower_out[0])
    x = self.merger(x.unsqueeze(0))
    x = self.p.linear_1(x)
    x = self.p.act(x)
    return self.p.linear_2(x)


class Adapter(nn.Module):
  def __init__(self, inner):
    super().__init__()
    self.inner = inner

  def forward(self, tower_out):                  # [1, G*G, D_v] -> [1, N, D_text]
    return self.inner((tower_out,)).unsqueeze(0)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("checkpoint")
  ap.add_argument("out_dir")
  ap.add_argument("--size", type=int, default=560)
  args = ap.parse_args()
  os.makedirs(args.out_dir, exist_ok=True)

  import litert_torch  # noqa: F401  — import before model submodules (dialect loading)

  full = transformers.AutoModelForImageTextToText.from_pretrained(
      args.checkpoint, dtype=torch.float32, low_cpu_mem_usage=True).eval()
  tok = transformers.AutoTokenizer.from_pretrained(args.checkpoint)

  tower, proj = full.model.vision_tower, full.model.multi_modal_projector
  encoder = VisionEncoder(tower, args.size).eval()
  grid = encoder.tower.grid
  merged = grid // proj.patch_merger.spatial_merge_size

  emb = full.get_input_embeddings()
  brk = emb(torch.tensor([tok.convert_tokens_to_ids("[IMG_BREAK]")]))[0]
  end = emb(torch.tensor([tok.convert_tokens_to_ids("[IMG_END]")]))[0]
  tail = ProjTail(proj, StaticPatchMerger(proj.patch_merger, grid).eval()).eval()
  adapter = Adapter(VisionAdapter(tail, merged, brk, end).eval()).eval()

  img = torch.rand(1, args.size, args.size, 3)
  with torch.no_grad():
    feats = encoder(img)
    block = adapter(feats)
  n_tokens = block.shape[1]
  print(f"encoder {tuple(img.shape)} -> {tuple(feats.shape)}")
  print(f"adapter -> {tuple(block.shape)}  ({merged}x{merged} image rows + {merged} markers "
        f"= {n_tokens} embeddings per image)")

  from litert_torch import convert

  enc_tfl = convert(encoder, (img.clone(),))
  enc_path = os.path.join(args.out_dir, "vision_encoder.tflite")
  enc_tfl.export(enc_path)
  print("wrote", enc_path, os.path.getsize(enc_path) // 1024 // 1024, "MB")

  ad_tfl = convert(adapter, (feats.clone(),))
  ad_path = os.path.join(args.out_dir, "vision_adapter.tflite")
  ad_tfl.export(ad_path)
  print("wrote", ad_path, os.path.getsize(ad_path) // 1024 // 1024, "MB")

  with open(os.path.join(args.out_dir, "vision_meta.txt"), "w") as f:
    f.write(f"image_size={args.size}\ngrid={grid}\nmerged={merged}\n"
            f"tokens_per_image={n_tokens}\nvision_dim={feats.shape[-1]}\n"
            f"text_dim={block.shape[-1]}\n")
  print("VISION_EXPORT_DONE")


if __name__ == "__main__":
  sys.exit(main())
