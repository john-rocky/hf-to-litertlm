"""Round 3 acceptance 4-5 / round 6d acceptance 2 (torch side): VisionTower / Projector vs transformers'
Siglip2VisionModel and Lfm2VlModel.get_image_features, on every tile of the synthetic pictures (and extra pictures).

    $EXPORT scripts/d1_vision_torch_check.py [--source tiny --tag tiny]
    $EXPORT scripts/d1_vision_torch_check.py \
        --source <snapshot> --tag real --rtag realv --log-prefix r6d_ --rel-bar 1e-4 \
        --table cache/real/tables/vision_position_table.safetensors \
        --extra coco_cats:cache/realv/coco_cats.jpg:cache/realv/proc_coco_cats.npz          (round 6d, the real weights)

Inputs: cache/vision/img_<case>.png and cache/vision/proc_<case>.npz (the HF processor's tensors, written by
d1_vision_host_check.py in venv-ref; this venv has no torchvision); --extra name:image:npz adds a picture whose
processor tensors d1v_extra_pictures.py wrote (round 6d: the model card's COCO cats photo, measurement only). The
host's preprocessing is run again here and must give the processor's pixel_values bit for bit (acceptance 1 in this
venv's numpy). --table: the position table file the host ships (vision_position_table.safetensors, key `table`); its
bits are compared with the weights' table and the host inputs are built from the file (default: from the weights).
Per picture (all tiles in one batch, as the provider runs them):
  HF uncut = Siglip2VisionModel(pixel_values [T, 1024, 768], spatial_shapes, pixel_attention_mask) (transformers'
             own forward, SDPA); HF cut + mm = Lfm2VlModel.get_image_features on the provider's cut to max(mask.sum(1))
             patches (`SystemOne._image_inputs`): its last_hidden_state (the cut tower output) and pooler_output
             (pixel_unshuffle + projector inside); the real rows of uncut and cut are compared (the provider's "same
             answer" claim).
  ours     = VisionTower(pixels, pos, mask) per tile with the host's inputs (`d1_vision.tower_inputs`); real rows vs HF
             uncut and HF cut: max |diff| and relative = max |diff| / max |HF value| of the tile.
  padding  = the tile's pad rows of pixels and pos replaced by N(0, 3^2) noise: the real rows must be bit-equal.
  projector: ours = the host's pixel_unshuffle of OUR features + Projector vs HF mm; unshuffle alone = the host's vs
             Lfm2VlMultiModalProjector.pixel_unshuffle on the same features (bit-equal); projector alone = Projector on
             the host-unshuffled HF cut features vs HF mm.
Bar: round 3 = max |diff| <= 1e-5 (tiny) on the tower and the projector chain; --rel-bar R (round 6d) = relative <= R
on the tower (uncut and cut: the planned bar) and on the projector alone (the projector graph's own form); the chain
is recorded, not gated: on the real weights its error is the tower's float32 rounding carried through the projector
(results/realv_tower_fp64_noise.json: on the worst tile transformers' own float32 chain is 1.44e-4 relative from
float64, the export form's 2.6e-5). Pad rows bit-equal and unshuffle bit-equal on every tile, host inputs equal on
every picture, no non-finite value, in both.
Outputs (Out names, d1_vision_graph.Out): results/{rtag}_vision_torch_check.json; {cache}/{tag}_torch_rows.npz, per
tile key <case>__<tile>: features (ours, real rows [n, H]), hf (HF uncut, real rows), soft (the unshuffled cells
[k, 4H]), mm (ours [k, d]), hfmm (HF [k, d]) — the LiteRT checks read the HF rows from here instead of running
transformers again.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "scripts"))
sys.path.insert(0, str(K / "host"))
import d1_vision as V  # noqa: E402
import d1_vision_graph as VG  # noqa: E402
from d1_vision_host_check import CASES, bits_equal  # noqa: E402

BAR = 1e-5


class _Half(torch.nn.Module):
    """What Lfm2VlModel.get_image_features reads from `self`: the tower and the projector."""

    def __init__(self, tower, proj):
        super().__init__()
        self.vision_tower, self.multi_modal_projector = tower, proj


def pictures(extra: list[str]) -> list[tuple[str, Path, Path]]:
    """(name, image file, processor npz): the 18 synthetic pictures, then each --extra name:image:npz (K-relative)."""
    out = [(name, K / f"cache/vision/img_{name}.png", K / f"cache/vision/proc_{name}.npz") for name, *_ in CASES]
    for spec in extra:
        name, img, npz = spec.split(":")
        out.append((name, K / img, K / npz))
    return out


def position_table(tower, table_file: str | None) -> tuple[np.ndarray, dict]:
    """The [S, S, H] table the host uses: --table's file (compared bit for bit with the weights') or the weights'."""
    side = tower.embeddings.position_embedding_size
    weights = tower.embeddings.position_embedding.weight.detach().numpy().reshape(side, side, -1).astype(np.float32)
    info = {"from": "weights", "shape": list(weights.shape)}
    if not table_file:
        return weights, info
    from safetensors.numpy import load_file

    p = K / table_file
    t = load_file(str(p))["table"]
    info = {"from": table_file, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "shape": list(t.shape),
            "dtype": str(t.dtype), "bits_equal_weights": bits_equal(t, weights)}
    assert info["bits_equal_weights"], info
    return t, info


def rel(d: float, ref: np.ndarray) -> float:
    m = float(np.abs(ref).max())
    return d / m if m else float("inf")


def main() -> int:
    from transformers.models.lfm2_vl.modeling_lfm2_vl import Lfm2VlModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="tiny")
    VG.Out.add_args(ap)
    ap.add_argument("--table", default=None, help="the position table file (K-relative safetensors, key `table`)")
    ap.add_argument("--extra", action="append", default=[], help="name:image:processor_npz (K-relative), repeatable")
    ap.add_argument("--rel-bar", type=float, default=None, help="relative bar (round 6d); default = round 3's 1e-5 abs")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--only", default="", help="comma-separated picture names (a timing probe; not a pass record)")
    a = ap.parse_args()
    o = VG.Out.from_args(a)
    out = o.result(f"{o.rtag}_vision_torch_check{'_only' if a.only else ''}.json")
    rows_out = o.cache(f"{o.tag}_torch_rows{'_only' if a.only else ''}.npz")
    for p in (out, rows_out):
        assert not p.exists(), f"refusing to overwrite {p}"
    torch.set_num_threads(a.threads)
    t0 = time.time()
    hf_tower, hf_proj, cfg, info = VG.load_vision(a.source)            # transformers' own forward (reference)
    tower_m, proj_m, _ = VG.graphs(a.source)                            # the export forms (separate copies)
    half = _Half(hf_tower, hf_proj).eval()
    table, table_info = position_table(hf_tower, a.table)
    load_s = round(time.time() - t0, 1)
    rng = np.random.default_rng(3)
    store, rows = {}, []
    pics = pictures(a.extra)
    if a.only:
        keep = set(a.only.split(","))
        pics = [p for p in pics if p[0] in keep]
    for name, img_path, npz_path in pics:
        tp = time.time()
        npz = np.load(npz_path)
        pv, pm, ss = npz["pixel_values"], npz["pixel_attention_mask"], npz["spatial_shapes"]
        pic = V.preprocess(V.cap_pixels(Image.open(img_path)))
        host_pv = np.stack([t.pixels for t in pic.tiles])
        host_equal = bits_equal(pv, host_pv) and bits_equal(pm, np.stack([t.mask for t in pic.tiles]))
        n_cut = int(pm.sum(1).max())
        with torch.no_grad():
            hf_un = hf_tower(pixel_values=torch.from_numpy(pv), spatial_shapes=torch.from_numpy(ss),
                             pixel_attention_mask=torch.from_numpy(pm)).last_hidden_state.numpy()
            got = Lfm2VlModel.get_image_features(half, pixel_values=torch.from_numpy(pv[:, :n_cut]),
                                                 spatial_shapes=torch.from_numpy(ss),
                                                 pixel_attention_mask=torch.from_numpy(pm[:, :n_cut]), return_dict=True)
            hf_cut, hf_mm = got.last_hidden_state.numpy(), [t.numpy() for t in got.pooler_output]
        t_hf = time.time() - tp
        tiles = []
        for ti, tile in enumerate(pic.tiles):
            hh, ww = tile.grid
            n = hh * ww
            ins = V.tower_inputs(tile, table)
            with torch.no_grad():
                feat = tower_m(**{k: torch.from_numpy(v) for k, v in ins.items()})["features"][0].numpy()
            real = feat[:n]
            hf_real = hf_un[ti, :n]
            d_un = float(np.abs(real.astype(np.float64) - hf_real).max())
            d_cut = float(np.abs(real.astype(np.float64) - hf_cut[ti, :n]).max())
            d_cut_un = float(np.abs(hf_cut[ti, :n].astype(np.float64) - hf_real).max())
            pad_bits = None
            if n < V.SETTINGS.max_num_patches:
                noisy = {k: v.copy() for k, v in ins.items()}
                noisy["pixels"][0, n:] = (rng.standard_normal(noisy["pixels"][0, n:].shape) * 3).astype(np.float32)
                noisy["pos"][0, n:] = (rng.standard_normal(noisy["pos"][0, n:].shape) * 3).astype(np.float32)
                with torch.no_grad():
                    feat2 = tower_m(**{k: torch.from_numpy(v) for k, v in noisy.items()})["features"][0].numpy()
                pad_bits = bits_equal(feat2[:n], real)
            cells = V.pixel_unshuffle(real, tile.grid)
            with torch.no_grad():
                hf_cells = hf_proj.pixel_unshuffle(torch.from_numpy(np.ascontiguousarray(real)).reshape(1, hh, ww, -1))
            unshuffle_bits = bits_equal(hf_cells.reshape(-1, hf_cells.shape[-1]).numpy(), cells)
            soft = V.projector_input(cells)
            with torch.no_grad():
                mm = proj_m(torch.from_numpy(soft))["mm"][0].numpy()
                hf_only = proj_m(torch.from_numpy(V.projector_input(V.pixel_unshuffle(hf_cut[ti, :n], tile.grid))))["mm"][0].numpy()
            k = cells.shape[0]
            ref_mm = hf_mm[ti]
            assert ref_mm.shape[0] == k == V.tokens_for_size(pic.size if ti == len(pic.tiles) - 1 else (512, 512)), (name, ti)
            d_mm = float(np.abs(mm[:k].astype(np.float64) - ref_mm).max())
            d_proj_only = float(np.abs(hf_only[:k].astype(np.float64) - ref_mm).max())
            key = f"{name}__{ti}"
            store[f"features__{key}"] = np.ascontiguousarray(real)
            store[f"hf__{key}"] = np.ascontiguousarray(hf_real)
            store[f"soft__{key}"] = np.ascontiguousarray(cells)
            store[f"mm__{key}"] = np.ascontiguousarray(mm[:k])
            store[f"hfmm__{key}"] = np.ascontiguousarray(ref_mm)
            tiles.append({"tile": ti, "grid": [hh, ww], "real": n, "tokens": k, "tower_vs_hf_uncut": d_un,
                          "tower_vs_hf_uncut_rel": rel(d_un, hf_real), "tower_vs_hf_cut": d_cut,
                          "tower_vs_hf_cut_rel": rel(d_cut, hf_cut[ti, :n]), "hf_cut_vs_uncut": d_cut_un,
                          "pad_noise_bits_equal": pad_bits, "unshuffle_bits_equal": unshuffle_bits,
                          "projector_chain_vs_hf": d_mm, "projector_chain_vs_hf_rel": rel(d_mm, ref_mm),
                          "projector_alone_vs_hf": d_proj_only, "projector_alone_vs_hf_rel": rel(d_proj_only, ref_mm),
                          "max_abs_feature": float(np.abs(real).max()), "max_abs_hf_feature": float(np.abs(hf_real).max()),
                          "max_abs_mm": float(np.abs(mm[:k]).max()), "max_abs_hf_mm": float(np.abs(ref_mm).max()),
                          "nonfinite": int((~np.isfinite(feat)).sum() + (~np.isfinite(mm)).sum())})
        rows.append({"case": name, "image": str(img_path.relative_to(K)), "tiles": len(tiles),
                     "host_inputs_equal_processor": host_equal, "cut": n_cut, "hf_seconds": round(t_hf, 1),
                     "seconds": round(time.time() - tp, 1),
                     **{f: max(t[f] for t in tiles) for f in ("tower_vs_hf_uncut", "tower_vs_hf_uncut_rel",
                                                               "tower_vs_hf_cut", "tower_vs_hf_cut_rel", "hf_cut_vs_uncut",
                                                               "projector_chain_vs_hf", "projector_chain_vs_hf_rel",
                                                               "projector_alone_vs_hf", "projector_alone_vs_hf_rel")},
                     "pad_noise_bits_equal": [t["pad_noise_bits_equal"] for t in tiles],
                     "unshuffle_bits_equal": all(t["unshuffle_bits_equal"] for t in tiles), "per_tile": tiles})
        print(f"{name}: {len(tiles)} tiles, tower rel {rows[-1]['tower_vs_hf_uncut_rel']:.3e} "
              f"(abs {rows[-1]['tower_vs_hf_uncut']:.3e}), mm rel {rows[-1]['projector_chain_vs_hf_rel']:.3e}, "
              f"{rows[-1]['seconds']} s", flush=True)
    np.savez(rows_out, **store)
    flat = [t for r in rows for t in r["per_tile"]]
    pads = [t["pad_noise_bits_equal"] for t in flat if t["pad_noise_bits_equal"] is not None]
    mx = lambda f: max(t[f] for t in flat)  # noqa: E731
    summary = {"cases": len(rows), "tiles": len(flat),
               "host_inputs_equal_processor": sum(r["host_inputs_equal_processor"] for r in rows),
               **{f"{f}_max": mx(f) for f in ("tower_vs_hf_uncut", "tower_vs_hf_uncut_rel", "tower_vs_hf_cut",
                                               "tower_vs_hf_cut_rel", "hf_cut_vs_uncut", "projector_chain_vs_hf",
                                               "projector_chain_vs_hf_rel", "projector_alone_vs_hf",
                                               "projector_alone_vs_hf_rel")},
               "pad_tiles": len(pads), "pad_bits_equal": sum(pads),
               "unshuffle_bits_equal": sum(t["unshuffle_bits_equal"] for t in flat),
               "max_abs_feature": mx("max_abs_feature"), "max_abs_hf_feature": mx("max_abs_hf_feature"),
               "max_abs_mm": mx("max_abs_mm"), "max_abs_hf_mm": mx("max_abs_hf_mm"),
               "nonfinite": sum(t["nonfinite"] for t in flat)}
    if a.rel_bar is None:
        bar = {"kind": "absolute", "value": BAR}
        within = (summary["tower_vs_hf_uncut_max"] <= BAR and summary["tower_vs_hf_cut_max"] <= BAR
                  and summary["projector_chain_vs_hf_max"] <= BAR)
    else:
        bar = {"kind": "relative (max |diff| / max |HF value| per tile) on the tower and the projector alone; the "
                       "projector chain is recorded, not gated", "value": a.rel_bar}
        within = (summary["tower_vs_hf_uncut_rel_max"] <= a.rel_bar and summary["tower_vs_hf_cut_rel_max"] <= a.rel_bar
                  and summary["projector_alone_vs_hf_rel_max"] <= a.rel_bar)
    summary["pass"] = (summary["host_inputs_equal_processor"] == len(rows) and within and summary["nonfinite"] == 0
                       and summary["pad_bits_equal"] == len(pads) and summary["unshuffle_bits_equal"] == len(flat))
    doc = {"what": "VisionTower / Projector (export forms) vs transformers, every tile (round 3 acceptance 4-5; round 6d "
                   "acceptance 2)", "source": info, "out": o.as_dict(), "position_table": table_info, "bar": bar,
           "only": a.only or None, "summary": summary, "cases": rows, "rows_npz": o.rel(rows_out),
           "load_seconds": load_s, "seconds": round(time.time() - t0, 1), "threads": a.threads,
           "peak_rss_bytes_getrusage": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "torch": torch.__version__,
           "attn_implementation": info.get("attn_implementation")}
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
