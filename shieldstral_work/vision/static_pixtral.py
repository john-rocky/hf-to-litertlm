"""Static single-image rewrite of the pixtral tower + Mistral3 patch merger.

Pixtral is a dynamic-resolution tower, but its dynamism is unusually shallow, and
at a fixed square resolution every dynamic piece collapses to a constant:

  * the per-image crop `embed[..., :h//p, :w//p]` is the identity,
  * `position_ids_in_meshgrid` becomes a compile-time constant table — but the
    ids must keep indexing the TRAINED grid, so `max_width` stays
    `config.image_size // patch_size` (110) and NOT the new grid width,
  * `generate_block_attention_mask` for one image is all-zeros = full attention,
    so the mask is dropped entirely,
  * the patch merger's `split(tokens_per_image)` / `view(h, w, d)` specialise,
    and `unfold` becomes a fixed strided gather-free reshape.

Patchify stays a single Conv2d over the raster image and the sequence keeps
raster order, so no GATHER_ND is introduced (the mobile-GPU vision blocker).

    python static_pixtral.py <checkpoint> [--size 560]
"""

import argparse
import sys

import torch
import torch.nn as nn


class StaticPixtralTower(nn.Module):
  """PixtralVisionModel specialised to one square image of `size` pixels."""

  def __init__(self, tower, size):
    super().__init__()
    cfg = tower.config
    self.patch_size = cfg.patch_size
    assert size % cfg.patch_size == 0, f"{size} not a multiple of patch {cfg.patch_size}"
    self.grid = size // cfg.patch_size

    self.patch_conv = tower.patch_conv
    self.ln_pre = tower.ln_pre
    self.transformer = tower.transformer
    self.patch_positional_embedding = tower.patch_positional_embedding

    # ids = h * max_width + w, with max_width from the TRAINED grid (110 here).
    max_width = cfg.image_size // cfg.patch_size
    h = torch.arange(self.grid).view(-1, 1)
    w = torch.arange(self.grid).view(1, -1)
    self.register_buffer("position_ids", (h * max_width + w).reshape(1, -1), persistent=False)

  def forward(self, pixel_values):            # [1, 3, S, S]
    x = self.patch_conv(pixel_values)          # [1, D, G, G]
    x = x.flatten(2).transpose(1, 2)           # [1, G*G, D] — raster order, as eager
    x = self.ln_pre(x)
    pos = self.patch_positional_embedding(x, self.position_ids)
    # Single image => the block-diagonal mask is all-zeros => plain full attention.
    out = self.transformer(x, attention_mask=None, position_embeddings=pos)
    return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]


class StaticPatchMerger(nn.Module):
  """Mistral3PatchMerger specialised to one square image (no split, no image_sizes)."""

  def __init__(self, merger, grid):
    super().__init__()
    self.m = merger.spatial_merge_size
    self.grid = grid
    assert grid % self.m == 0, f"grid {grid} not divisible by merge {self.m}"
    self.merging_layer = merger.merging_layer

  def forward(self, feats):                    # [1, G*G, D]
    d = feats.shape[-1]
    g, m = self.grid, self.m
    x = feats.view(g, g, d).permute(2, 0, 1).unsqueeze(0)          # [1, D, G, G]
    # unfold with kernel == stride == m is a pure strided reshape; express it that
    # way so the graph stays free of im2col-style ops.
    x = x.view(1, d, g // m, m, g // m, m)                          # [1,D,G/m,m,G/m,m]
    x = x.permute(0, 1, 3, 5, 2, 4).reshape(1, d * m * m, (g // m) * (g // m))
    x = x.squeeze(0).t()                                            # [(G/m)^2, D*m*m]
    return self.merging_layer(x)


def build(checkpoint, size, dtype=torch.float32):
  import transformers
  full = transformers.AutoModelForImageTextToText.from_pretrained(
      checkpoint, dtype=dtype, low_cpu_mem_usage=True).eval()
  tower = full.model.vision_tower
  proj = full.model.multi_modal_projector
  static_tower = StaticPixtralTower(tower, size).eval()
  static_merger = StaticPatchMerger(proj.patch_merger, static_tower.grid).eval()
  return full, tower, proj, static_tower, static_merger


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("checkpoint")
  ap.add_argument("--size", type=int, default=560)
  args = ap.parse_args()

  full, tower, proj, static_tower, static_merger = build(args.checkpoint, args.size)
  print(f"grid = {static_tower.grid}x{static_tower.grid} patches "
        f"-> {(static_tower.grid // static_merger.m) ** 2} tokens after {static_merger.m}x{static_merger.m} merge")

  torch.manual_seed(0)
  px = torch.randn(1, 3, args.size, args.size)
  sizes = torch.tensor([[args.size, args.size]])

  with torch.no_grad():
    ref = tower(px, image_sizes=sizes).last_hidden_state
    got = static_tower(px)
  print(f"tower   : ref {tuple(ref.shape)} vs static {tuple(got.shape)}")
  report("tower", ref, got)

  with torch.no_grad():
    ref_m = proj.patch_merger(ref[0], sizes)
    got_m = static_merger(got)
  print(f"merger  : ref {tuple(ref_m.shape)} vs static {tuple(got_m.shape)}")
  report("merger", ref_m, got_m)

  # Full projector path (norm + merger + linear stack), which is what the adapter ships.
  with torch.no_grad():
    ref_p = proj(ref[0], sizes)
  print(f"projector out: {tuple(ref_p.shape)}")


def report(label, a, b):
  a, b = a.reshape(-1).float(), b.reshape(-1).float()
  if a.shape != b.shape:
    print(f"  {label}: SHAPE MISMATCH {a.shape} vs {b.shape}")
    return
  corr = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
  print(f"  {label}: corr={corr:.8f}  max|diff|={(a - b).abs().max().item():.3e}")


if __name__ == "__main__":
  sys.exit(main())
