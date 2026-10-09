"""Do the row graph's norm reductions fit fp16 storage, and which per-site pre-scale k keeps them inside it?
(round 2 acceptance 9 = the probe; round 5 acceptance 2 = the lower side, --norm-scale, --recommend, --vision)

    $EXPORT scripts/d1_norm_range.py --tag tiny5 --recommend      # tiny, every row
    $EXPORT scripts/d1_norm_range.py --vision --tag tiny5 --recommend   # tiny VL tower
    $EXPORT scripts/d1_norm_range.py \
        --source <snapshot> --tag real --shards 8 --recommend --headroom 4 --floor 6.1e-5   (the real weights)

Why (cf. Kev-0.8B LiteRT, r14_norm_range_probe.py): the converter lowers RMSNorm's mean(x^2) to SUM (+ a
constant MUL); a GPU path that stores in fp16 (Metal default precision, Android FP16_WITH_FP32_ACCUM) keeps that SUM in
fp16, so a sum of squares above 65,504 becomes inf, rsqrt(inf) = 0 and the row comes back exactly 0 with no non-finite
value. The other side: a mean(x^2) below the fp16 normal range (6.1e-5) loses its bits (subnormal) or flushes to 0, and
rsqrt(0 + eps) of a flushed eps is inf (an earlier conversion: one k for every norm pushed a small
norm's variance into the subnormals). The sites of LFM2: per layer `operator_norm` and `ffn_norm` (d values), per
attention layer `self_attn.q_layernorm` and `self_attn.k_layernorm` (head_dim values per head), and the final
`embedding_norm` (d values).
Per fixture row (fixtures/rows.json; ids folded into the tiny vocabulary with i % 255 for the tiny model), through
D1Prefill at the smallest bucket of --buckets that holds the row (right pad, valid), float32: for every module, over the
real positions and over all positions (pads included: an inf or NaN there reaches real rows through 0 * inf in the
masked attention), the maximum of sum(x^2) along the normalised axis (per head for q/k), the largest |x|, and the
minimum of mean(x^2) (the variance the rsqrt reads). With --norm-scale the module's input is taken as its reduction sees
it (x * 2^-k). Output: results/{tag}_norm_range.json (never overwritten): per site (aggregated over layers and per
layer) the maxima / minima over all rows, the margin 65,504 / max, the rows over 65,504, the top 10 rows.
--recommend [--headroom H (4)] [--floor F (6.1e-5)] (reads results/{tag}_norm_range.json when it exists, else probes
first; the probe must be unscaled): per site, from the all-position values (s = max sum of squares, v = min variance),
    k_low  = the smallest k in -15..15 with s * 4^-k <= 65,504 / H         (the overflow side)
    k_high = the largest k in -15..15 with v * 4^-k >= F * H               (the subnormal side; v = 0 -> none fits)
    k      = the k of [k_low, k_high] nearest 0 (the least change: 0 when it fits, else k_low > 0 to shrink or
             k_high < 0 to grow), or no k when k_low > k_high (status "conflict": no single power of two keeps both
             sides; another rewrite is needed)
-> results/norm_scale_{tag}.json (never overwritten): `k` (the dict d1_prefill_graph --norm-scale reads), per site the
bounds, the margins at k on both sides, and the inputs.
--recommend --per-module [--out <path>] (round 6b): the same bounds per RMSNorm module (`per_module` of the probe), `k`
keyed by module name (`layers.4.operator_norm`, `layers.5.self_attn.q_layernorm`, `embedding_norm`, ...): on d1-3B one k
per site had no solution (layers 0-4 carry sums of squares <= 1.8 and variances ~2e-5, layers 5-12 sums near 920), one
k per layer has. --out names the recommendation file (default results/norm_scale_{tag}[_vision].json).
--vision: the picture tower instead (the tiny VL tower of d1_vision_graph, or <snapshot>'s): the 18 pictures of
d1_vision_host_check.CASES through the host's preprocessing (host/d1_vision.py) and the export form VisionTower, per
tile; per LayerNorm module (sites layer_norm1, layer_norm2 per encoder layer, post_layernorm; the converter lowers
LayerNorm to MEAN, SQUARED_DIFFERENCE, MEAN, RSQRT), over the real patches and over all 1,024 patch rows: max of
sum((x - mean)^2), max |sum x| (the first MEAN), max |x|, min of mean((x - mean)^2). -> results/{tag}_vision_norm_range.json;
with --recommend also the same k bounds (information: no LayerNorm pre-scale is implemented; LayerNorm is
scale-equivariant the same way, eps x 4^-k).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d1_common import K, ROWS, sha256_file  # noqa: E402

FP16_MAX = 65504.0
FP16_MIN_NORMAL = 6.1e-5
BUCKETS = (64, 128, 256, 512, 1024, 2048, 4096)
SITES = ("operator_norm", "ffn_norm", "q_layernorm", "k_layernorm", "embedding_norm")
VISION_SITES = ("layer_norm1", "layer_norm2", "post_layernorm")


def site_of(name: str, mod) -> str | None:
    """The site of an RMSNorm module by its own name (`post_operator_norm` / `post_ffn_norm` are nn.Identity here)."""
    last = name.rsplit(".", 1)[-1]
    return last if last in SITES and type(mod).__name__ == "RMSNorm" else None


def vision_site_of(name: str, mod) -> str | None:
    last = name.rsplit(".", 1)[-1]
    return last if last in VISION_SITES and type(mod).__name__ == "LayerNorm" else None


def _rec(rec: dict, name: str, site: str, n_real: int, ss, x_abs, var, extra=None):
    """Fold one position-set into the module's record [site, ss_real, absmax_real, ss_all, var_real_min, var_all_min,
    (vision) abs_sum_real, abs_sum_all]; ss / var / x_abs are torch tensors [positions, heads or 1(, D)]."""
    r = rec.setdefault(name, [site, 0.0, 0.0, 0.0, math.inf, math.inf] + ([0.0, 0.0] if extra is not None else []))
    r[1] = max(r[1], float(ss[:n_real].max()))
    r[2] = max(r[2], float(x_abs[:n_real].max()))
    r[3] = max(r[3], float(ss.max()))
    r[4] = min(r[4], float(var[:n_real].min()))
    r[5] = min(r[5], float(var.min()))
    if extra is not None:
        r[6] = max(r[6], float(extra[:n_real].max()))
        r[7] = max(r[7], float(extra.max()))


def worker(source: str, tag: str, i: int, n: int, buckets: tuple[int, ...], norm_scale: str) -> None:
    import torch

    import d1_prefill_graph as G

    torch.set_num_threads(1)
    lm, _ = G.load(source)
    ns = G.apply_norm_scale(lm, norm_scale or None)
    tiny = source == "tiny"
    pad_id = G.TINY_PAD_ID if tiny else 124893
    rows = [r for r in json.loads(ROWS.read_text())["rows"] if not r.get("image_expansion_pending")][i::n]
    cur = {"rec": None}

    def note(name, site, x):
        rec = cur["rec"]
        if rec is None:
            return
        L, nreal = cur["L"], cur["n"]
        x = x.detach().float()
        x = x.reshape(L, -1, x.shape[-1])            # [L, heads or 1, D]
        ss = (x * x).sum(-1)                         # [L, heads or 1]
        _rec(rec, name, site, nreal, ss, x.abs().amax(-1), ss / x.shape[-1])

    hooks = []
    for name, mod in lm.named_modules():
        site = site_of(name, mod)
        if site is not None:
            # the input as the reduction sees it: x * 2^-k when the module is pre-scaled
            hooks.append(mod.register_forward_pre_hook(
                lambda m, args, name=name, site=site: note(name, site, args[0] * getattr(m, "_d1_norm_scale", 1.0))))
    graphs, out = {}, []
    t0 = time.perf_counter()
    for r in rows:
        ids = [x % 255 for x in r["ids"]] if tiny else r["ids"]
        L = next(b for b in buckets if b >= len(ids))
        if L not in graphs:
            graphs[L] = G.graph(lm, L)
        cur.update(rec={}, L=L, n=len(ids))
        with torch.no_grad():
            h = graphs[L](*G.row_inputs(ids, L, pad_id))["hidden"]
        out.append({"key": f"{r['id']}/{r['qid']}", "row_len": len(ids), "L": L, "sites": cur["rec"],
                    "finite": bool(torch.isfinite(h).all())})
        cur["rec"] = None
    for hk in hooks:
        hk.remove()
    (K / f"cache/{tag}/norm_range_{i}of{n}.json").write_text(
        json.dumps({"rows": out, "norm_scale": ns["k"], "seconds": round(time.perf_counter() - t0, 1)}))


def vision_worker(source: str, tag: str, i: int, n: int) -> None:
    import numpy as np
    import torch
    from PIL import Image

    sys.path.insert(0, str(K / "host"))
    import d1_vision as V
    import d1_vision_graph as VG
    from d1_vision_host_check import CASES

    torch.set_num_threads(1)
    tower, _, info = VG.graphs(source)
    emb = tower.tower.embeddings.position_embedding
    side = emb.weight.shape[0]
    table = emb.weight.detach().numpy().reshape(int(round(side ** 0.5)), int(round(side ** 0.5)), -1).astype(np.float32)
    cur = {"rec": None}

    def note(name, site, x):
        rec = cur["rec"]
        if rec is None:
            return
        x = x.detach().float()[0]                    # [N, H]
        mean = x.mean(-1, keepdim=True)
        d = x - mean
        ss = (d * d).sum(-1)                         # sum of squared differences per patch row
        _rec(rec, name, site, cur["n"], ss[:, None], x.abs().amax(-1)[:, None], (ss / x.shape[-1])[:, None],
             extra=x.sum(-1).abs()[:, None])

    hooks = []
    for name, mod in tower.named_modules():
        site = vision_site_of(name, mod)
        if site is not None:
            hooks.append(mod.register_forward_pre_hook(lambda m, args, name=name, site=site: note(name, site, args[0])))
    out = []
    t0 = time.perf_counter()
    for case, *_ in CASES[i::n]:
        pic = V.preprocess(V.cap_pixels(Image.open(K / f"cache/vision/img_{case}.png")))
        for ti, tile in enumerate(pic.tiles):
            hh, ww = tile.grid
            ins = V.tower_inputs(tile, table)
            cur.update(rec={}, n=hh * ww)
            with torch.no_grad():
                f = tower(**{k: torch.from_numpy(v) for k, v in ins.items()})["features"]
            out.append({"key": f"{case}/tile{ti}", "patches": hh * ww, "sites": cur["rec"],
                        "finite": bool(torch.isfinite(f).all())})
            cur["rec"] = None
    for hk in hooks:
        hk.remove()
    (K / f"cache/{tag}/vision_norm_range_{i}of{n}.json").write_text(
        json.dumps({"rows": out, "info": info, "seconds": round(time.perf_counter() - t0, 1)}))


def summary(vals, vision: bool = False) -> dict:
    """vals = [(ss_real, absmax_real, ss_all, var_real_min, var_all_min[, abs_sum_real, abs_sum_all]), ...]"""
    mx, mxa = max(v[0] for v in vals), max(v[2] for v in vals)
    out = {"max_sum_of_squares_real": mx, "max_abs_value_real": max(v[1] for v in vals),
           "max_sum_of_squares_all_positions": mxa,
           "min_variance_real": min(v[3] for v in vals), "min_variance_all_positions": min(v[4] for v in vals),
           "margin_to_fp16_max_real": FP16_MAX / mx if mx else None,
           "margin_to_fp16_max_all_positions": FP16_MAX / mxa if mxa else None,
           "rows_over_fp16_max_real": sum(v[0] > FP16_MAX for v in vals),
           "rows_over_fp16_max_all_positions": sum(v[2] > FP16_MAX for v in vals),
           "rows_variance_below_fp16_min_normal_all_positions": sum(v[4] < FP16_MIN_NORMAL for v in vals)}
    if vision:
        out.update(max_abs_sum_real=max(v[5] for v in vals), max_abs_sum_all_positions=max(v[6] for v in vals))
    return out


def recommend_one(s: float, v: float, headroom: float, floor: float, k_max: int = 15) -> dict:
    """k bounds for one site from s = max sum of squares, v = min variance (all positions); k may be negative (grow)."""
    hi_cap, lo_cap = FP16_MAX / headroom, floor * headroom
    ks = range(-k_max, k_max + 1)
    fit_hi = [k for k in ks if s * 4.0 ** -k <= hi_cap]          # overflow side: every k from k_low up
    fit_lo = [k for k in ks if v * 4.0 ** -k >= lo_cap]          # subnormal side: every k up to k_high
    k_low = min(fit_hi) if fit_hi else None
    k_high = max(fit_lo) if fit_lo else None
    ok = k_low is not None and k_high is not None and k_low <= k_high
    k = min(max(0, k_low), k_high) if ok else None               # the feasible k nearest 0 (the least change)
    at = k if ok else (k_low if k_low is not None else 0)
    return {"k_low_overflow_side": k_low, "k_high_subnormal_side": k_high, "k": k,
            "status": "ok" if ok else "conflict",
            "upper_margin_at_k": hi_cap * headroom / (s * 4.0 ** -at) if s else None,
            "lower_margin_at_k": (v * 4.0 ** -at) / floor if v > 0 else 0.0,
            "inputs": {"max_sum_of_squares_all_positions": s, "min_variance_all_positions": v}}


def recommend_per_module(doc: dict, headroom: float, floor: float) -> dict:
    """--recommend --per-module (round 6b): the same bounds per RMSNorm module (doc["per_module"]: every layer's
    operator_norm / ffn_norm / q / k layernorm and embedding_norm) instead of one k per site: a site whose layers span
    a sum of squares near 1,000 and a variance near 2e-5 has no single power-of-two k, while each layer alone may.
    The `k` dict is keyed by module name (d1_prefill_graph.parse_norm_scale takes module names and sites)."""
    assert not any(doc.get("norm_scale", {}).values()), "recommend from an unscaled probe (norm_scale all 0)"
    mods = {}
    for name, d in doc["per_module"].items():
        r = recommend_one(d["max_sum_of_squares_all_positions"], d["min_variance_all_positions"], headroom, floor)
        r["site"] = name.rsplit(".", 1)[-1]
        mods[name] = r
    k = {m: r["k"] for m, r in mods.items()}
    by_site = {}
    for m, r in mods.items():
        by_site.setdefault(r["site"], {}).setdefault(str(r["k"]), []).append(m)
    return {"what": "per-module pre-scale k for fp16 storage (row graph RMSNorm, one k per module)",
            "source_json": doc["_path"], "source": doc["source"], "tag": doc["tag"], "rows": doc["rows"],
            "headroom": headroom, "floor": floor, "fp16_max": FP16_MAX,
            "rule": "per module: k_low = least k with s 4^-k <= 65504 / headroom; k_high = largest k with v 4^-k >= "
                    "floor x headroom; k = the k of [k_low, k_high] nearest 0, none if k_low > k_high (conflict); s, v "
                    "over all positions of every fixture row; k in -15..15",
            "k": k if all(v is not None for v in k.values()) else None, "k_partial": k,
            "conflicts": sorted(m for m, r in mods.items() if r["status"] != "ok"),
            "modules_by_site_and_k": {s: {kk: len(v) for kk, v in d.items()} for s, d in by_site.items()},
            "modules": mods}


def recommend(doc: dict, headroom: float, floor: float, vision: bool) -> dict:
    assert not any(doc.get("norm_scale", {}).values()), "recommend from an unscaled probe (norm_scale all 0)"
    sites = {}
    for site, d in doc["sites"].items():
        assert "min_variance_all_positions" in d, ("this norm-range json predates the variance fields (round 2): "
                                                   "probe again under a new --tag")
        sites[site] = recommend_one(d["max_sum_of_squares_all_positions"], d["min_variance_all_positions"],
                                    headroom, floor)
        sites[site]["modules"] = d["modules"]
    k = {s: sites[s]["k"] for s in sites}
    return {"what": ("per-site pre-scale k for fp16 storage (" + ("picture tower LayerNorm, information only"
                                                                 if vision else "row graph RMSNorm") + ")"),
            "source_json": doc["_path"], "source": doc["source"], "tag": doc["tag"], "rows": doc["rows"],
            "headroom": headroom, "floor": floor, "fp16_max": FP16_MAX,
            "rule": "k_low = least k with s 4^-k <= 65504 / headroom; k_high = largest k with v 4^-k >= floor x "
                    "headroom; k = the k of [k_low, k_high] nearest 0, none if k_low > k_high (conflict); s, v over "
                    "all positions; k in -15..15",
            "k": k if all(v is not None for v in k.values()) else None, "k_partial": k,
            "conflicts": sorted(s for s, v in sites.items() if v["status"] != "ok"), "sites": sites}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="tiny")
    ap.add_argument("--tag", default="tiny")
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--buckets", default=",".join(map(str, BUCKETS)))
    ap.add_argument("--norm-scale", default="", help="probe the pre-scaled graph (per-site k, JSON or file)")
    ap.add_argument("--vision", action="store_true", help="the picture tower's LayerNorms instead")
    ap.add_argument("--recommend", action="store_true")
    ap.add_argument("--per-module", action="store_true", help="--recommend one k per RMSNorm module (round 6b)")
    ap.add_argument("--out", default="", help="--recommend: the output path (K-relative; default "
                                              "results/norm_scale_<tag>[_vision].json)")
    ap.add_argument("--headroom", type=float, default=4.0)
    ap.add_argument("--floor", type=float, default=FP16_MIN_NORMAL)
    ap.add_argument("--worker", default="")
    a = ap.parse_args()
    buckets = tuple(int(x) for x in a.buckets.split(","))
    if a.worker:
        i, n = (int(x) for x in a.worker.split("/"))
        if a.vision:
            vision_worker(a.source, a.tag, i, n)
        else:
            worker(a.source, a.tag, i, n, buckets, a.norm_scale)
        return 0
    kind = "vision_norm_range" if a.vision else "norm_range"
    out = K / f"results/{a.tag}_{kind}.json"
    if a.recommend and out.exists():
        doc = json.loads(out.read_text())
    else:
        assert not out.exists(), f"refusing to overwrite {out}"
        doc = probe(a, buckets, kind, out)
    doc["_path"] = str(out.relative_to(K))
    if a.recommend:
        rec_out = (K / a.out) if a.out else K / f"results/norm_scale_{a.tag}{'_vision' if a.vision else ''}.json"
        assert not rec_out.exists(), f"refusing to overwrite {rec_out}"
        if a.per_module:
            assert not a.vision, "--per-module is for the row graph"
            rec = recommend_per_module(doc, a.headroom, a.floor)
            rec_out.write_text(json.dumps(rec, indent=1) + "\n")
            print("| module | max sum sq (all) | min variance (all) | k_low | k_high | k | upper margin | lower margin |")
            print("|---|---:|---:|---:|---:|---:|---:|---:|")
            for m, v in rec["modules"].items():
                print(f"| {m} | {v['inputs']['max_sum_of_squares_all_positions']:.4g} | "
                      f"{v['inputs']['min_variance_all_positions']:.4g} | {v['k_low_overflow_side']} | "
                      f"{v['k_high_subnormal_side']} | {v['k']} | {v['upper_margin_at_k']:.4g} | {v['lower_margin_at_k']:.4g} |")
            print(json.dumps({"conflicts": rec["conflicts"], "modules_by_site_and_k": rec["modules_by_site_and_k"],
                              "file": str(rec_out.relative_to(K))}))
            return 0
        rec = recommend(doc, a.headroom, a.floor, a.vision)
        rec_out.write_text(json.dumps(rec, indent=1) + "\n")
        print(f"| site | modules | max sum sq (all) | min variance (all) | k_low (overflow) | k_high (subnormal) | k | "
              f"upper margin at k | lower margin at k (x floor) |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for s, v in rec["sites"].items():
            print(f"| {s} | {v['modules']} | {v['inputs']['max_sum_of_squares_all_positions']:.4g} | "
                  f"{v['inputs']['min_variance_all_positions']:.4g} | {v['k_low_overflow_side']} | "
                  f"{v['k_high_subnormal_side']} | {v['k']} | {v['upper_margin_at_k']:.4g} | {v['lower_margin_at_k']:.4g} |")
        print(json.dumps({"k": rec["k"], "conflicts": rec["conflicts"], "file": str(rec_out.relative_to(K))}))
    return 0


def probe(a, buckets, kind: str, out: Path) -> dict:
    (K / f"cache/{a.tag}").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    env = dict(os.environ, OMP_NUM_THREADS="1")
    procs = []
    for i in range(a.shards):
        log = open(K / f"logs/{a.tag}_{kind}_{i}.log", "w")
        cmd = [sys.executable, __file__, "--source", a.source, "--tag", a.tag, "--buckets", a.buckets,
               "--worker", f"{i}/{a.shards}"] + (["--vision"] if a.vision else []) + \
              (["--norm-scale", a.norm_scale] if a.norm_scale else [])
        procs.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT))
    rcs = [p.wait() for p in procs]
    assert all(rc == 0 for rc in rcs), rcs
    rows, norm_scale, info = [], None, None
    for i in range(a.shards):
        p = K / f"cache/{a.tag}/{kind}_{i}of{a.shards}.json"
        part = json.loads(p.read_text())
        rows += part["rows"]
        norm_scale, info = part.get("norm_scale"), part.get("info")
        p.unlink()
    per_layer, per_site = {}, {}
    for r in rows:
        for name, (site, *vals) in r["sites"].items():
            per_layer.setdefault(name, []).append((*vals, r["key"], r.get("row_len", r.get("patches"))))
            per_site.setdefault(site, []).append((*vals, r["key"], r.get("row_len", r.get("patches")), name))
    nv = 7 if a.vision else 5

    sites = {s: {**summary(v, a.vision), "modules": len({x[nv + 2] for x in v}),
                 "top10": [{"key": x[nv], "length": x[nv + 1], "module": x[nv + 2], "sum_of_squares_real": x[0],
                            "max_abs_real": x[1]} for x in sorted(v, key=lambda x: -x[0])[:10]],
                 "bottom5_variance": [{"key": x[nv], "length": x[nv + 1], "module": x[nv + 2], "variance_all": x[4]}
                                      for x in sorted(v, key=lambda x: x[4])[:5]]}
             for s, v in sorted(per_site.items())}
    doc = {"what": ("LayerNorm reductions of the picture tower vs fp16 storage (sum of squared differences per patch row, "
                    "real patches and all rows)" if a.vision else
                    "RMSNorm reductions of the row graph vs fp16 storage (sum of squares per position along the "
                    "normalised axis; max over real positions and over all positions; min variance)"),
           "source": a.source, "tag": a.tag, "rows": len(rows), "all_finite_fp32": all(r["finite"] for r in rows),
           "fp16_max": FP16_MAX, "fp16_min_normal": FP16_MIN_NORMAL, "shards": a.shards,
           "seconds_wall": round(time.time() - t0, 1), "sites": sites,
           "per_module": {k: summary(v, a.vision) for k, v in sorted(per_layer.items())}}
    if a.vision:
        doc.update(tower=info, pictures=len({r["key"].split("/")[0] for r in rows}))
    else:
        doc.update(buckets=list(buckets), rows_json_sha256=sha256_file(ROWS), norm_scale=norm_scale)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({s: {k: v for k, v in d.items() if k not in ("top10", "bottom5_variance")}
                      for s, d in sites.items()}, indent=1))
    print(json.dumps({"rows": len(rows), "all_finite_fp32": doc["all_finite_fp32"], "seconds": doc["seconds_wall"]}))
    return doc


if __name__ == "__main__":
    sys.exit(main())
