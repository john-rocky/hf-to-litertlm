#!/usr/bin/env python3
"""Round 3 acceptance 3: the rewritten tower (scripts/r3_tower.py ClefTower, float32) against the author's model.visual
(transformers 5.17.0 Qwen3_5VisionModel, sdpa, torch 2.9.0 = venv-ref = the reference's versions) on the 28 fixed-grid
reference runs, plus the constants the export uses.

    HF_HUB_OFFLINE=1 venv-ref/bin/python scripts/r3_tower_torch.py            -> results/r3_tower_torch.json,
        cache/r3/consts_g256.npz, cache/r3/consts_g448.npz, cache/r3/torch/<id>__<arm>.npy (the rewrite's rows, fp32)
    venv-ref/bin/python scripts/r3_tower_torch.py --tiny                      -> results/r3_tiny_torch.json (a random
        tiny config: depth 2, hidden 64, 4 heads, the same patch / grid shapes; no checkpoint read)

1. The author's module: Qwen3_5VisionModel(vision_config) with the checkpoint's model.visual.* tensors (bf16 -> float32,
   strict load), attn sdpa, torch threads 10 (the reference's). Control: its pooler_output on the reference's
   pixel_values vs the reference npz image_embeds (bit-equal expected: same versions, same input).
2. Constants per grid from the author's own code path (the forward's expressions, unchanged): 5.17.0's bilinear taps
   (get_vision_interpolation_indices_and_weights, align_corners, merge-block-major) -> pos_embeds = (pos_embed(idx) * w)
   .sum(1) [N, 1152]; get_vision_position_ids -> rotary_pos_emb -> cos / sin [N, 72] -> cache/r3/consts_<grid>.npz.
3. The rewrite (float32, the same weights, those constants) on the same pixel_values: per run cos, rel_rms
   (||a - b|| / ||b||), max |diff|, min row cos against the author's rows and against the npz rows. No element bar
   (an element bar below the float32 noise floor decides nothing, and a few massive activations dominate a norm).
4. float64 (2 runs per grid): the author's module in float64 with apply_rotary_pos_emb_vision's float32 casts removed
   (cos / sin stay the float32 values the author computes) vs the rewrite in float64 with the same constants (pos_embeds
   recomputed by the same expression in float64): the identity of the two functions; and both float32 forms against
   that float64 truth (the float32 noise floor of each).
5. Negative control: the same pixels in raster patch order through the rewrite must land far from the author's rows."""
import argparse
import contextlib
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "scripts"))
import r3_tower as RT  # noqa: E402

REV = "f0b7d3cd65296abfe459e6c228b618f5d8b26d2f"
SNAP = K / "hf/hub/models--Cloudflare--clef-flash/snapshots" / REV
FP64_RUNS = (("img_01", "g256"), ("photo_01", "g256"), ("img_04", "g448"), ("photo_04", "g448"))


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def stats(got, want) -> dict:
    g = np.asarray(got, np.float64).reshape(-1, np.shape(want)[-1])
    w = np.asarray(want, np.float64).reshape(g.shape)
    d = g - w
    rows = (g * w).sum(-1) / np.maximum(np.linalg.norm(g, axis=-1) * np.linalg.norm(w, axis=-1), 1e-300)
    return {"cos": float(g.ravel() @ w.ravel() / (np.linalg.norm(g) * np.linalg.norm(w))),
            "rel_rms": float(np.linalg.norm(d) / np.linalg.norm(w)), "max_abs": float(np.abs(d).max()),
            "min_row_cos": float(rows.min()), "ref_absmax": float(np.abs(w).max()),
            "bit_equal": bool(np.array_equal(np.asarray(got), np.asarray(want))), "finite": bool(np.isfinite(g).all())}


def raster_perm(g: int) -> np.ndarray:
    """raster-ordered patches = block_major_patches[perm] (zoo gate_tower.py's negative control)."""
    r, c = np.divmod(np.arange(g * g), g)
    m = 2
    return (((r // m) * (g // m) + c // m) * m + r % m) * m + c % m


@contextlib.contextmanager
def author_fp64_rope(M):
    """apply_rotary_pos_emb_vision without its float32 casts (float64 q / k stay float64; cos / sin promoted)."""
    orig = M.apply_rotary_pos_emb_vision

    def keep(q, k, cos, sin):
        cos, sin = cos.unsqueeze(-2).to(q.dtype), sin.unsqueeze(-2).to(q.dtype)
        return (q * cos) + (M.rotate_half(q) * sin), (k * cos) + (M.rotate_half(k) * sin)

    M.apply_rotary_pos_emb_vision = keep
    try:
        yield
    finally:
        M.apply_rotary_pos_emb_vision = orig


def author_consts(visual, M, grid_side: int) -> dict:
    """The forward's own expressions for one grid (transformers 5.17.0 Qwen3_5VisionModel.forward)."""
    from transformers.vision_utils import get_vision_interpolation_indices_and_weights, get_vision_position_ids
    grid_thw = torch.tensor([[1, grid_side, grid_side]])
    idx, w = get_vision_interpolation_indices_and_weights(
        grid_thw, num_grid_per_side=visual.num_grid_per_side, mode=visual.interpolation_mode,
        align_corners=visual.interpolation_align_corners, spatial_merge_size=visual.config.spatial_merge_size)
    pid = get_vision_position_ids(grid_thw, visual.spatial_merge_size)
    with torch.no_grad():
        pos_embeds = (visual.pos_embed(idx) * w[:, :, None]).sum(1)
        cos, sin = visual.rotary_pos_emb(torch.zeros(1, dtype=visual.pos_embed.weight.dtype), pid)
    return {"grid_thw": grid_thw, "interp_indices": idx, "interp_weights": w, "position_ids": pid,
            "pos_embeds": pos_embeds, "cos": cos, "sin": sin}


def build_author(cfg, sd: dict | None, M):
    cfg._attn_implementation = "sdpa"
    visual = M.Qwen3_5VisionModel(cfg)
    if sd is not None:
        visual.load_state_dict(sd, strict=True)
    return visual.eval().requires_grad_(False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--threads", type=int, default=10)
    a = ap.parse_args()
    out = K / ("results/r3_tiny_torch.json" if a.tiny else "results/r3_tower_torch.json")
    assert not out.exists(), f"refusing to overwrite {out}"
    import transformers
    from transformers import AutoConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M

    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    t_all = time.time()
    doc = {"what": "round 3: the rewritten tower (r3_tower.ClefTower) vs the author's model.visual" + (" (tiny random config)" if a.tiny else ""),
           "versions": {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__,
                        "numpy": np.__version__}, "threads": torch.get_num_threads(),
           "r3_tower_py_sha256": sha256_file(K / "scripts/r3_tower.py")}
    vcfg = AutoConfig.from_pretrained(str(SNAP)).vision_config
    if a.tiny:
        vcfg.hidden_size, vcfg.num_heads, vcfg.intermediate_size, vcfg.depth, vcfg.out_hidden_size = 64, 4, 96, 2, 80
        torch.manual_seed(0)
        visual = build_author(vcfg, None, M)
        with torch.no_grad():                    # non-trivial biases / norms (post_init leaves some zero / one)
            for n_, p_ in visual.named_parameters():
                p_.copy_(torch.randn_like(p_) * (0.02 if p_.dim() > 1 else 0.1) + (1.0 if n_.endswith("norm1.weight")
                         or n_.endswith("norm2.weight") or n_.endswith("norm.weight") else 0.0))
        sd32 = {k: v.detach().clone() for k, v in visual.state_dict().items()}
        heads = vcfg.num_heads
        doc["tiny_config"] = {k: getattr(vcfg, k) for k in ("hidden_size", "num_heads", "intermediate_size", "depth",
                                                            "out_hidden_size", "patch_size", "num_position_embeddings")}
    else:
        t0 = time.time()
        sd32 = RT.load_visual_state(SNAP)
        visual = build_author(vcfg, sd32, M)
        heads = RT.HEADS
        doc["load_seconds"] = round(time.time() - t0, 1)
        doc["visual_tensors"] = len(sd32)
        doc["visual_params"] = int(sum(v.numel() for v in sd32.values()))
    doc["author"] = {"class": type(visual).__name__, "attn_implementation": visual.config._attn_implementation,
                     "num_grid_per_side": visual.num_grid_per_side, "interpolation": [visual.interpolation_mode,
                                                                                     visual.interpolation_align_corners],
                     "rope_type": visual.rotary_pos_emb.rope_type, "inv_freq": visual.rotary_pos_emb.inv_freq.tolist()}
    # constants
    consts, towers = {}, {}
    for grid, side in RT.GRIDS.items():
        c = author_consts(visual, M, side)
        consts[grid] = c
        if not a.tiny:
            p = K / f"cache/r3/consts_{grid}.npz"
            p.parent.mkdir(parents=True, exist_ok=True)
            if p.exists():                       # an earlier invocation: it must hold the same constants
                old = np.load(p)
                assert all(np.array_equal(old[n], c[n].numpy()) for n in ("pos_embeds", "cos", "sin")), f"{p} differs"
                p = p.with_name(p.stem + ".again.npz")
            np.savez(p, pos_embeds=c["pos_embeds"].numpy().astype(np.float32), cos=c["cos"].numpy().astype(np.float32),
                     sin=c["sin"].numpy().astype(np.float32), interp_indices=c["interp_indices"].numpy(),
                     interp_weights=c["interp_weights"].numpy(), position_ids=c["position_ids"].numpy(),
                     grid_thw=c["grid_thw"].numpy(), transformers=np.array(transformers.__version__),
                     torch=np.array(torch.__version__))
            doc.setdefault("consts", {})[grid] = {"file": str(p.relative_to(K)), "sha256": sha256_file(p),
                                                  "pos_embeds": list(c["pos_embeds"].shape), "cos": list(c["cos"].shape)}
        towers[grid] = RT.ClefTower(sd32, c["pos_embeds"], c["cos"], c["sin"], heads=heads).eval()
    # runs
    if a.tiny:
        g = torch.Generator().manual_seed(1)
        runs = [(f"rand{i}", grid, torch.randn(side * side, RT.PATCH_DIM, generator=g).numpy(), None)
                for grid, side in RT.GRIDS.items() for i in range(2)]
    else:
        ref = json.loads((K / "ref/records_ref.json").read_text())
        runs = []
        for r in ref["rows"]:
            if r["arm"] in RT.GRIDS:
                z = np.load(K / f"ref/npz/{r['id']}__{r['arm']}.npz")
                runs.append((r["id"], r["arm"], z["pixel_values"], z["image_embeds"]))
        assert len(runs) == 28, len(runs)
    cache = K / "cache/r3/torch"
    if not a.tiny:
        cache.mkdir(parents=True, exist_ok=True)
    rows, walls = [], {"author": [], "rewrite": []}
    for rid, grid, pv, ref_rows in runs:
        side = RT.GRIDS[grid]
        t0 = time.perf_counter()
        au = visual(torch.from_numpy(pv), grid_thw=consts[grid]["grid_thw"]).pooler_output.numpy()
        walls["author"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        mine = towers[grid](torch.from_numpy(pv)[None])["rows"][0].numpy()
        walls["rewrite"].append(time.perf_counter() - t0)
        item = {"id": rid, "arm": grid, "rows": int(mine.shape[0]), "rewrite_vs_author": stats(mine, au)}
        if ref_rows is not None:
            item["author_vs_reference_npz"] = stats(au, ref_rows)
            item["rewrite_vs_reference_npz"] = stats(mine, ref_rows)
            np.save(cache / f"{rid}__{grid}.npy", mine.astype(np.float32))
        perm = raster_perm(side)
        neg = towers[grid](torch.from_numpy(np.ascontiguousarray(pv[perm]))[None])["rows"][0].numpy()
        item["negative_raster_order_vs_author"] = {k: v for k, v in stats(neg, au).items() if k in ("cos", "rel_rms", "min_row_cos")}
        rows.append(item)
        print(f"{rid:10s} {grid} rewrite vs author cos {item['rewrite_vs_author']['cos']:.10f} rel_rms "
              f"{item['rewrite_vs_author']['rel_rms']:.3e}" + (f" | author vs npz bit {item['author_vs_reference_npz']['bit_equal']}"
                                                               if ref_rows is not None else ""), flush=True)
    # float64
    fp64 = []
    pick = [(rid, grid, pv) for rid, grid, pv, _ in runs if a.tiny or (rid, grid) in FP64_RUNS]
    if pick:
        sd64 = {k: v.to(torch.float64) for k, v in sd32.items()}
        vis64 = build_author(vcfg, sd64, M).double() if not a.tiny else build_author(vcfg, sd64, M).double()
        with author_fp64_rope(M):
            for rid, grid, pv in pick:
                c = consts[grid]
                with torch.no_grad():
                    pe64 = (vis64.pos_embed(c["interp_indices"]) * c["interp_weights"][:, :, None]).sum(1)
                t64 = RT.ClefTower(sd64, pe64, c["cos"].double(), c["sin"].double(), heads=heads).eval()
                x64 = torch.from_numpy(pv).double()
                au64 = vis64(x64, grid_thw=c["grid_thw"]).pooler_output.numpy()
                mi64 = t64(x64[None])["rows"][0].numpy()
                au32 = visual(torch.from_numpy(pv), grid_thw=c["grid_thw"]).pooler_output.numpy()
                mi32 = towers[grid](torch.from_numpy(pv)[None])["rows"][0].numpy()
                fp64.append({"id": rid, "arm": grid, "rewrite64_vs_author64": stats(mi64, au64),
                             "author32_vs_author64": stats(au32, au64), "rewrite32_vs_author64": stats(mi32, au64)})
                print(f"fp64 {rid} {grid}: identity rel_rms {fp64[-1]['rewrite64_vs_author64']['rel_rms']:.3e}, author32 "
                      f"{fp64[-1]['author32_vs_author64']['rel_rms']:.3e}, rewrite32 {fp64[-1]['rewrite32_vs_author64']['rel_rms']:.3e}",
                      flush=True)
                del t64
    def worst(key, sub, fn=max, src=rows):
        r = fn(src, key=lambda x: x[sub][key])
        return {"value": r[sub][key], "run": f"{r['id']}/{r['arm']}"}
    summary = {"runs": len(rows),
               "rewrite_vs_author": {"min_cos": worst("cos", "rewrite_vs_author", min), "max_rel_rms": worst("rel_rms", "rewrite_vs_author"),
                                     "max_abs": worst("max_abs", "rewrite_vs_author"),
                                     "min_row_cos": worst("min_row_cos", "rewrite_vs_author", min)},
               "negative_max_cos": max(r["negative_raster_order_vs_author"]["cos"] for r in rows),
               "seconds_per_run_median": {k: round(float(np.median(v)), 3) for k, v in walls.items()}}
    if not a.tiny:
        summary["author_vs_reference_npz_bit_equal"] = f"{sum(r['author_vs_reference_npz']['bit_equal'] for r in rows)}/{len(rows)}"
        summary["author_vs_reference_npz_max_abs"] = max(r["author_vs_reference_npz"]["max_abs"] for r in rows)
        summary["rewrite_vs_reference_npz"] = {"min_cos": worst("cos", "rewrite_vs_reference_npz", min),
                                               "max_rel_rms": worst("rel_rms", "rewrite_vs_reference_npz")}
        for grid in RT.GRIDS:
            gr = [r for r in rows if r["arm"] == grid]
            summary[f"by_grid_{grid}"] = {"runs": len(gr), "min_cos": min(r["rewrite_vs_author"]["cos"] for r in gr),
                                          "max_rel_rms": max(r["rewrite_vs_author"]["rel_rms"] for r in gr)}
    if fp64:
        summary["fp64"] = {"runs": len(fp64), "identity_max_rel_rms": max(x["rewrite64_vs_author64"]["rel_rms"] for x in fp64),
                           "identity_max_abs": max(x["rewrite64_vs_author64"]["max_abs"] for x in fp64),
                           "author32_max_rel_rms": max(x["author32_vs_author64"]["rel_rms"] for x in fp64),
                           "rewrite32_max_rel_rms": max(x["rewrite32_vs_author64"]["rel_rms"] for x in fp64)}
    doc.update(summary=summary, runs=rows, fp64=fp64, seconds=round(time.time() - t_all, 1))
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    ok = summary["negative_max_cos"] < 0.99 and (not fp64 or summary["fp64"]["identity_max_rel_rms"] <= 1e-10)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
