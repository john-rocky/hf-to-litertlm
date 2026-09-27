"""PSNR and sharpness of PNGs: compare.py <reference.png> <test.png> [<test2.png> ...]

PSNR is against the reference (same size required). Sharpness = variance of the
3x3 Laplacian of the grayscale image (0-255 scale), reported for every file
including the reference; it is reference-free, so it separates real blur from
sampling drift (a different draw scores low PSNR but keeps its sharpness).
"""
import sys

import numpy as np
from PIL import Image


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float64)


def psnr(a, b):
    mse = ((a - b) ** 2).mean()
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def lap_var(rgb):
    g = rgb @ np.array([0.299, 0.587, 0.114])
    lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    return float(lap.var())


def hf_share(rgb, cutoff=0.25):
    """Share of spectral energy beyond `cutoff` of Nyquist (DC removed)."""
    g = rgb @ np.array([0.299, 0.587, 0.114])
    f = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean()))) ** 2
    h, w = g.shape
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt(((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2)
    return float(f[r > cutoff].sum() / f.sum())


ref = load(sys.argv[1])
print(f"{'file':40s} {'PSNR dB':>8s} {'lap var':>9s} {'hf share':>9s}")
print(f"{sys.argv[1]:40s} {'ref':>8s} {lap_var(ref):9.1f} {hf_share(ref):9.4f}")
for p in sys.argv[2:]:
    t = load(p)
    ps = f"{psnr(ref, t):8.2f}" if t.shape == ref.shape else "  (size)"
    print(f"{p:40s} {ps} {lap_var(t):9.1f} {hf_share(t):9.4f}")
