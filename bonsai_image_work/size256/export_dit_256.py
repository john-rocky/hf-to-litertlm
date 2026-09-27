"""Export the Bonsai Image DiT at a smaller output size (default 256x256).

Same weights and the same fp64-RoPE -> float32 patch as ../export_dit.py; the
only thing that changes is the traced image-token count: SIZE/8 latent, 2x2
patches -> (SIZE/16)^2 tokens (256x256 -> 16x16 grid -> 256 tokens, vs 1024 at
512x512). The text side stays at 256 prompt tokens.

The fidelity print uses REAL position ids (the pipeline's [0, h, w, 0] /
[0, 0, 0, i] builders): the original export checked the rope patch with all-zero
ids, which proves nothing (cos=1, sin=0 at pos 0 for any dtype).

    SIZE=256 WORK=~/models/bonsai-image-256 python export_dit_256.py
Then, from $WORK:  python ../quantize_dit.py   (SRC=dit_fp32.tflite)
                   python fix_scales.py dit_int4b32.tflite dit_256_int4b32.tflite
"""
import glob
import os
import time

import torch
import diffusers.models.transformers.transformer_flux2 as flux2mod
from diffusers import Flux2Transformer2DModel

SNAP = os.environ.get("SNAP") or sorted(glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--prism-ml--bonsai-image-ternary-4B-unpacked/snapshots/*")))[-1]
WORK = os.path.expanduser(os.environ.get("WORK", "~/models/bonsai-image-256"))
SIZE = int(os.environ.get("SIZE", "256"))
GRID = SIZE // 16
B, IMG, TXT = 1, GRID * GRID, 256
OUT = os.path.join(WORK, "dit_fp32.tflite")
os.makedirs(WORK, exist_ok=True)
print(f"snapshot {SNAP}\nsize {SIZE} -> grid {GRID}x{GRID} = {IMG} image tokens", flush=True)


class DiT(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids):
        return self.m(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states,
                      timestep=timestep, img_ids=img_ids, txt_ids=txt_ids, return_dict=False)[0]


m = Flux2Transformer2DModel.from_pretrained(f"{SNAP}/transformer", torch_dtype=torch.float32).eval()
C, JD, AX = m.config.in_channels, m.config.joint_attention_dim, len(m.config.axes_dims_rope)
print(f"params {sum(p.numel() for p in m.parameters())/1e9:.3f} B", flush=True)

# real position ids, as generate.py builds them
one = torch.arange(1)
img_ids = torch.cartesian_prod(one, torch.arange(GRID), torch.arange(GRID), one).float()
txt_ids = torch.cartesian_prod(one, one, one, torch.arange(TXT)).float()
g = torch.Generator().manual_seed(0)
args = (torch.randn(B, IMG, C, generator=g), torch.randn(B, TXT, JD, generator=g),
        torch.tensor([1.0]), img_ids, txt_ids)
wrapped = DiT(m).eval()

with torch.no_grad():
    ref64 = wrapped(*args)                       # untouched fp64-rope reference
print(f"ref(fp64 rope) mean={ref64.mean():.6f} std={ref64.std():.6f}", flush=True)

flux2mod.maybe_adjust_dtype_for_device = lambda dtype, device: (
    torch.float32 if dtype == torch.float64 else dtype)
with torch.no_grad():
    ref32 = wrapped(*args)                       # after the patch
d = (ref32 - ref64).abs()
print(f"ref(fp32 rope) mean={ref32.mean():.6f} std={ref32.std():.6f} | "
      f"max|d|={d.max():.3e} rel={d.max()/ref64.abs().max():.3e}", flush=True)
torch.save({"ref64": ref64, "ref32": ref32, "args": args, "size": SIZE},
           os.path.join(WORK, f"dit_ref_{SIZE}.pt"))

import litert_torch
t0 = time.time()
lm = litert_torch.convert(wrapped, args)
print(f"convert OK in {time.time()-t0:.0f}s", flush=True)
t0 = time.time()
lm.export(OUT)
print(f"export OK in {time.time()-t0:.0f}s -> {OUT} {os.path.getsize(OUT)/2**30:.2f} GiB", flush=True)
