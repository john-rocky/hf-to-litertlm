"""Prove the vision adapter reproduces HF's image embedding block exactly.

The runtime's single-image contract injects ONE contiguous block of embeddings at
the soft-token position. Pixtral does not fit that shape as-is: its processor
expands one `[IMG]` into a grid of `[IMG]` tokens with `[IMG_BREAK]` at the end of
every row and one `[IMG_END]` at the end, and those two markers keep their ordinary
*text* embeddings — they are not replaced by vision features.

  560x560 -> 40x40 patches -> 20x20 merged tokens
          -> 400 [IMG] + 19 [IMG_BREAK] + 1 [IMG_END] = 420 positions

Escape: fold the marker embeddings into the adapter. They are constants (rows of the
decoder's embedding table), so the adapter can emit the full 420-row block and the
runtime injects it unchanged. This script checks that block against the embeddings
transformers itself builds for the same image.

    python verify_adapter.py <checkpoint> [--size 560]
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import transformers
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from static_pixtral import StaticPatchMerger, StaticPixtralTower  # noqa: E402


class VisionAdapter(nn.Module):
  """Projected patch features -> the exact 420-row block the decoder expects.

  Row layout: [20 image rows] each followed by a marker; the marker is
  `[IMG_BREAK]` for rows 0..18 and `[IMG_END]` for row 19.
  Implemented as a concat with a constant column — no gather, no dynamic shapes.
  """

  def __init__(self, projector, merged_grid, break_emb, end_emb):
    super().__init__()
    self.g = merged_grid
    self.projector = projector
    markers = break_emb.repeat(merged_grid, 1).clone()   # [G, D]
    markers[-1] = end_emb
    self.register_buffer("markers", markers.unsqueeze(1), persistent=False)  # [G,1,D]

  def forward(self, merged_feats):          # [G*G, D_vision] -> [G*G+G, D_text]
    x = self.projector(merged_feats)        # [400, 3072]
    d = x.shape[-1]
    x = x.view(self.g, self.g, d)           # [20, 20, 3072]
    x = torch.cat([x, self.markers], dim=1)  # [20, 21, 3072]
    return x.reshape(self.g * (self.g + 1), d)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("checkpoint")
  ap.add_argument("--size", type=int, default=560)
  args = ap.parse_args()

  full = transformers.AutoModelForImageTextToText.from_pretrained(
      args.checkpoint, dtype=torch.float32, low_cpu_mem_usage=True).eval()
  proc = transformers.AutoProcessor.from_pretrained(args.checkpoint)
  tok = proc.tokenizer

  tower = full.model.vision_tower
  proj = full.model.multi_modal_projector
  static_tower = StaticPixtralTower(tower, args.size).eval()
  merged_grid = static_tower.grid // proj.patch_merger.spatial_merge_size

  emb = full.get_input_embeddings()
  brk = emb(torch.tensor([tok.convert_tokens_to_ids("[IMG_BREAK]")]))[0]
  end = emb(torch.tensor([tok.convert_tokens_to_ids("[IMG_END]")]))[0]

  static_merger = StaticPatchMerger(proj.patch_merger, static_tower.grid).eval()

  class ProjTail(nn.Module):
    """The projector minus its merger: norm -> merger -> linear stack."""

    def __init__(self, p, merger):
      super().__init__()
      self.p, self.merger = p, merger

    def forward(self, tower_out):
      x = self.p.norm(tower_out[0])
      x = self.merger(x.unsqueeze(0))
      x = self.p.linear_1(x)
      x = self.p.act(x)
      return self.p.linear_2(x)

  tail = ProjTail(proj, static_merger).eval()
  adapter = VisionAdapter(tail, merged_grid, brk, end).eval()

  # A real image, not noise.
  img = Image.open(os.path.join(os.path.dirname(__file__), "probe.png")).convert("RGB") \
      if os.path.exists(os.path.join(os.path.dirname(__file__), "probe.png")) else \
      Image.effect_mandelbrot((args.size, args.size), (-2, -1.5, 1, 1.5), 40).convert("RGB")
  img = img.resize((args.size, args.size))

  msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "probe"}]}]
  txt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
  batch = proc(text=txt, images=[img], return_tensors="pt")
  px, ids = batch["pixel_values"], batch["input_ids"]

  img_id = tok.convert_tokens_to_ids("[IMG]")
  brk_id = tok.convert_tokens_to_ids("[IMG_BREAK]")
  end_id = tok.convert_tokens_to_ids("[IMG_END]")
  mask = torch.isin(ids[0], torch.tensor([img_id, brk_id, end_id]))
  start = int(mask.nonzero()[0]); stop = int(mask.nonzero()[-1]) + 1
  print(f"image block occupies positions {start}..{stop} ({stop-start} rows)")

  with torch.no_grad():
    ref_embeds = full.model.get_input_embeddings()(ids)
    # get_image_features returns the raw tower output here (1600x1024); the
    # projected features the decoder actually receives come from the projector.
    sizes = batch.get("image_sizes")
    if sizes is None:
      sizes = torch.tensor([[args.size, args.size]])
    tower_ref = tower(px, image_sizes=sizes).last_hidden_state
    feats = proj(tower_ref[0], sizes)
    ref_block = ref_embeds[0, start:stop].clone()
    # HF scatters the vision features onto the [IMG] rows only.
    img_rows = (ids[0, start:stop] == img_id)
    ref_block[img_rows] = feats.reshape(-1, ref_block.shape[-1]).to(ref_block.dtype)

    tower_out = static_tower(px)
    got_block = adapter((tower_out,))

  print(f"ref block {tuple(ref_block.shape)}  vs adapter {tuple(got_block.shape)}")
  a, b = ref_block.reshape(-1).float(), got_block.reshape(-1).float()
  if a.shape != b.shape:
    print("SHAPE MISMATCH — adapter layout does not match the processor's expansion")
    return 1
  corr = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
  print(f"corr = {corr:.8f}   max|diff| = {(a-b).abs().max().item():.3e}")
  print("PASS" if corr > 0.9999 else "FAIL")
  return 0


if __name__ == "__main__":
  sys.exit(main())
