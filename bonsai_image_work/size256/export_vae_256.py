"""Export the Bonsai Image VAE decoder at a smaller output size (default 256x256).

Identical to ../export_vae.py except the fixed latent side: SIZE/8 (256x256 ->
32x32 latent, vs 64x64 at 512x512). Same weights, fp32.

    SIZE=256 WORK=~/models/bonsai-image-256 python export_vae_256.py
"""
import glob
import os
import time

import torch
from diffusers import AutoencoderKLFlux2

SNAP = os.environ.get("SNAP") or sorted(glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--prism-ml--bonsai-image-ternary-4B-unpacked/snapshots/*")))[-1]
WORK = os.path.expanduser(os.environ.get("WORK", "~/models/bonsai-image-256"))
SIZE = int(os.environ.get("SIZE", "256"))
LAT = SIZE // 8
OUT = os.path.join(WORK, f"vae_dec_{SIZE}_fp32.tflite")
os.makedirs(WORK, exist_ok=True)


class Decoder(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        return self.vae.decode(latents, return_dict=False)[0]


vae = AutoencoderKLFlux2.from_pretrained(f"{SNAP}/vae", torch_dtype=torch.float32).eval()
lc = vae.config.latent_channels
print(f"VAE latent_channels={lc} params={sum(p.numel() for p in vae.parameters())/1e6:.1f} M "
      f"| size {SIZE} -> latent {LAT}x{LAT}", flush=True)

dec = Decoder(vae).eval()
z = torch.randn(1, lc, LAT, LAT, generator=torch.Generator().manual_seed(0))
with torch.no_grad():
    ref = dec(z)
print(f"decode {tuple(z.shape)} -> {tuple(ref.shape)} min={ref.min():.3f} max={ref.max():.3f}",
      flush=True)
torch.save({"z": z, "ref": ref, "size": SIZE}, os.path.join(WORK, f"vae_ref_{SIZE}.pt"))

import litert_torch
t0 = time.time()
m = litert_torch.convert(dec, (z,))
print(f"convert OK in {time.time()-t0:.0f}s", flush=True)
m.export(OUT)
print(f"export OK -> {OUT} {os.path.getsize(OUT)/2**20:.1f} MiB", flush=True)
