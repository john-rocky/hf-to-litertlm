"""Save the provider processor's pixel values of pictures (no model weights needed): out/pixels/<stem>.npz with
pixel_values [gh*gw, 1536] float32 and grid [gh, gw]. The export checks of the tower read them without torch models.

  $REF conversion/save_pixels.py out/media/coco_val2017_000000039769.jpg [more pictures]
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import seqs_only  # noqa: E402


def main():
    import numpy as np
    from PIL import Image

    eng = seqs_only.provider_engine()
    media_mod = sys.modules[type(eng).__module__.rsplit(".", 1)[0] + ".media"]
    out_dir = os.path.join(WORK, "out/pixels")
    os.makedirs(out_dir, exist_ok=True)
    for p in sys.argv[1:]:
        _, out = eng.media._encode(media_mod.Media("image", Image.open(p)))
        pv, grid = out["pixel_values"].float().numpy(), out["image_grid_thw"]
        gh, gw = int(grid[0, 1]), int(grid[0, 2])
        dst = os.path.join(out_dir, os.path.splitext(os.path.basename(p))[0] + ".npz")
        np.savez(dst, pixel_values=pv.astype(np.float32), grid=np.array([gh, gw]))
        print(dst, pv.shape, (gh, gw))


if __name__ == "__main__":
    main()
