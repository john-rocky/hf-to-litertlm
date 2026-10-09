"""Round 6d: the HF processor's tensors for pictures outside d1_vision_host_check.CASES (the model card's COCO cats
photo, measurement only: never committed, never shipped), and the host's preprocessing against them.

    $REF scripts/d1v_extra_pictures.py coco_cats:cache/realv/coco_cats.jpg \
        [--out-dir cache/realv] [--rtag realv] [--root <dir>]

The same calls as d1_vision_host_check.py makes for its synthetic pictures: the provider's `runner.cap_pixels` and
the host's `cap_pixels` (same size, same pixels), then `AutoProcessor.from_pretrained(hf_small)(text=["<image>"],
images=[[picture]], return_tensors="pt", add_special_tokens=False)` (as `SystemOne._image_inputs` calls it, before the
provider's cut) and `Lfm2VlImageProcessor(..., return_row_col_info=True)` against `host/d1_vision.py preprocess`:
pixel_values (float32 bits), pixel_attention_mask, spatial_shapes, image_rows / image_cols / image_sizes.
Outputs (never overwritten): <out-dir>/proc_<name>.npz (the torch check's input) and
results/{rtag}_extra_pictures.json (file sha256, decoded RGB sha256, sizes, tiles, the equalities).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "scripts"))
sys.path.insert(0, str(K / "host"))
import d1_vision as V  # noqa: E402
from d1_common import HF_SMALL, provider  # noqa: E402
from d1_vision_host_check import bits_equal, plain  # noqa: E402


def main() -> int:
    import torch
    from transformers import AutoProcessor

    ap = argparse.ArgumentParser()
    ap.add_argument("pictures", nargs="+", help="name:image (K-relative)")
    ap.add_argument("--out-dir", default="cache/realv")
    ap.add_argument("--rtag", default="realv")
    ap.add_argument("--root", default=None, help="output root for the results json (default K)")
    a = ap.parse_args()
    root = K if not a.root else (Path(a.root) if Path(a.root).is_absolute() else K / a.root)
    out = root / f"results/{a.rtag}_extra_pictures.json"
    out_dir = K / a.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    assert not out.exists(), f"refusing to overwrite {out}"
    t0 = time.time()
    runner = provider("runner")
    proc = AutoProcessor.from_pretrained(str(HF_SMALL))
    ip = proc.image_processor
    rows = []
    for spec in a.pictures:
        name, img = spec.split(":")
        path = K / img
        npz_out = out_dir / f"proc_{name}.npz"
        assert not npz_out.exists(), f"refusing to overwrite {npz_out}"
        back = Image.open(path)
        cap_p, cap_h = runner.cap_pixels(back), V.cap_pixels(back)
        cap_equal = cap_p.size == cap_h.size and np.array_equal(np.asarray(cap_p), np.asarray(cap_h))
        hf = proc(text=[V.IMAGE_MARKER], images=[[cap_p]], return_tensors="pt", add_special_tokens=False)
        info = ip([cap_p], return_row_col_info=True, return_tensors="pt")
        pic = V.preprocess(cap_h)
        pv = np.stack([t.pixels for t in pic.tiles])
        mask = np.stack([t.mask for t in pic.tiles])
        spatial = np.array([list(t.grid) for t in pic.tiles], dtype=np.int64)
        np.savez_compressed(npz_out, pixel_values=hf["pixel_values"].numpy(),
                            pixel_attention_mask=hf["pixel_attention_mask"].numpy(),
                            spatial_shapes=hf["spatial_shapes"].numpy())
        row = {"name": name, "image": img, "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "file_bytes": path.stat().st_size, "format": back.format, "mode": back.mode,
               "input_wh": list(back.size), "capped_wh": list(cap_h.size),
               "rgb_sha256": hashlib.sha256(np.asarray(back.convert("RGB")).tobytes()).hexdigest(),
               "cap_pixels_equal": cap_equal, "rows_cols": [pic.rows, pic.cols], "tiles": len(pic.tiles),
               "tile_grids": [list(t.grid) for t in pic.tiles], "thumbnail_or_single_hw": list(pic.size),
               "image_tokens": V.tokens_for_size(pic.size) + (pic.rows * pic.cols * 256 if len(pic.tiles) > 1 else 0),
               "pixel_values_bits_equal": bits_equal(hf["pixel_values"].numpy(), pv),
               "pixel_values_shape": list(hf["pixel_values"].shape),
               "pixel_attention_mask_equal": bits_equal(hf["pixel_attention_mask"].numpy(), mask),
               "spatial_shapes_equal": bits_equal(hf["spatial_shapes"].numpy(), spatial),
               "image_rows_cols_sizes_equal": (plain(info["image_rows"]) == [pic.rows] and plain(info["image_cols"]) == [pic.cols]
                                               and plain(info["image_sizes"]) == [list(pic.size)]),
               "hf_rows_cols_sizes": [plain(info["image_rows"]), plain(info["image_cols"]), plain(info["image_sizes"])],
               "npz": str(npz_out.relative_to(K))}
        row["pass"] = all(row[k] for k in ("cap_pixels_equal", "pixel_values_bits_equal", "pixel_attention_mask_equal",
                                           "spatial_shapes_equal", "image_rows_cols_sizes_equal"))
        rows.append(row)
    doc = {"what": "round 6d: processor tensors and host preprocessing for extra pictures (measurement only)",
           "processor": type(proc).__name__, "image_processor": type(ip).__name__,
           "backend": getattr(ip, "backend", None), "torch": torch.__version__,
           "cpu_capability": torch.backends.cpu.get_cpu_capability(), "pictures": rows,
           "pass": all(r["pass"] for r in rows), "seconds": round(time.time() - t0, 1)}
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(doc, indent=1))
    return 0 if doc["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
