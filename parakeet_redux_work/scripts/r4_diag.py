#!/usr/bin/env python3
"""r4 precision lever, step 1 (venv .venv-ref, CPU torch fp32): where fp16 arithmetic breaks the encoder (the Mac GPU
at its default fp16 arithmetic returns blank-only text, r2) and the per-site 2^-k LayerNorm rescale that keeps it in
range. The graph is r4_graphs.EncB (= the exported Enc30 op for op) with a per-op hook.

1. range pass (fp32, every op output): windows from S32 (r1's out/s32_ii mode-ii mel) interleaved with FLEURS de / fr /
   ru (the 4 longest utterances <= 30 s per language, mel by MelB), until a time budget -> per op: max|x| over all
   frames (padded rows run on the GPU too); per LayerNorm site: max (x - mean)^2 and min row variance.
2. fix: per LayerNorm site k = the larger of ceil(log2(sqrt(max (x - mean)^2) / 32)) (|x - mean| 2^-k <= 32: the
   squared difference <= 1,024, 64x under 65,504) and ceil(log2(1,024 x max row variance / 16,384) / 2) (the row sum
   of the squared difference 4x under 65,504), at least 0; per block m when a residual point passes 2^14 (16,384);
   every other op above 2^14 is listed (the rewrite does not cover it).
3. fp16-storage emulation on 3 windows (weights, constants and every op output rounded to fp16, arithmetic in fp32),
   unfixed and fixed, op by op against the fp32 pass of the same graph: max|d| and max|d| / max|x| on valid frames,
   non-finite counts (valid / all), the first op past 1e-3 / 1e-2 / 1e-1 or non-finite; the encoder output vs fp32
   (max|d|, rows with the same argmax) and the greedy TDT decode (cap 10) vs B_ii's tokens. Also fixed fp32 vs
   unfixed fp32 (the rewrite must not move fp32).
-> results/r4/diag.json, results/r4/fix.json (ln_k / res_m for r4_build.py; "usable" false when no LayerNorm site
needs a rescale and the emulation still breaks = the rewrite is not the answer)
CLI: r4_diag.py [--budget-s 80] [--threads 8] [--small]
  --small = dry run outside the window: r4_small.py's random model instead of B (not the model).
"""
import argparse
import copy
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r4_graphs as G  # noqa: E402
import redux_ref as R  # noqa: E402

OUT = "results/r4"
TARGET = 32.0
ROWSUM_LIMIT = 16384.0
MASKED = (".bd_mask", ".logits")  # attention terms holding the (m3 - 1) x 30,000 key bias (fp16-safe by design)
RES_LIMIT = 2.0 ** 14
DIAG_WINDOWS = ["4507-16021-0047__w0", "4507-16021-0047__w1", "5105-28241-0001"]


def valid_map(l0, t_mel=3000):
    m, t, lv = {}, t_mel, l0
    for _ in range(4):
        m[t] = lv
        t, lv = t // 2, (lv + 1) // 2
    return m


def time_axis(x, vm):
    for ax in range(1, x.dim()):
        if x.shape[ax] in vm:
            return ax, vm[x.shape[ax]]
    return None, None


def amax(x):
    if x.numel() == 0:
        return 0.0
    lo, hi = torch.aminmax(x)
    v = max(-float(lo), float(hi))
    return v


class Hook:
    """q(name, x): mode 'f32' records, mode 'f16' rounds to fp16 then records. q.c(constant) rounds constants in f16."""

    def __init__(self, mode, sink):
        self.mode = mode
        self.sink = sink
        self.order = []

    def __call__(self, name, x):
        if self.mode == "f16":
            x = x.to(torch.float16).to(torch.float32)
        self.order.append(name)
        self.sink(name, x)
        return x

    def c(self, x):
        if self.mode != "f16":
            return x
        if isinstance(x, float):
            return float(torch.tensor(x, dtype=torch.float32).to(torch.float16).to(torch.float32))
        return x.to(torch.float16).to(torch.float32)


def round_model(m):
    m = copy.deepcopy(m)
    with torch.no_grad():
        for t in list(m.parameters()) + list(m.buffers()):
            if t.is_floating_point():
                t.copy_(t.to(torch.float16).to(torch.float32))
    return m


def fixture(budget_n_fleurs=4):
    """[(tag, mel [1,128,3000] torch, l0)] S32 npz windows interleaved with FLEURS windows (mel by MelB)."""
    s32 = sorted(glob.glob("out/s32_ii/*.npz"))
    fl = []
    for lang in ("de", "fr", "ru"):
        rows = R.read_jsonl(f"data/sets/fleurs-{lang}/manifest.jsonl")
        rows = sorted([r for r in rows if r["audio_length_s"] <= 29.9], key=lambda r: -r["audio_length_s"])
        fl += [(lang, r) for r in rows[:budget_n_fleurs]]
    order = []
    i = j = 0
    while i < len(s32) or j < len(fl):
        for _ in range(7):
            if i < len(s32):
                order.append(("s32", s32[i]))
                i += 1
        if j < len(fl):
            order.append(("fl", fl[j]))
            j += 1
    return order


def load_window(kind, item, mel_graph):
    if kind == "s32":
        z = np.load(item)
        return os.path.basename(item)[:-4], torch.from_numpy(z["mel"])[None], int(z["valid_mel"])
    lang, r = item
    audio = R.load_wav(r["path"])[:G.W30]
    n = len(audio)
    x = torch.zeros(1, G.n_audio(3000))
    x[0, :n] = torch.from_numpy(audio)
    with torch.no_grad():
        o = mel_graph(x, torch.tensor([n], dtype=torch.int32))
    return f"fleurs-{lang}:{r['id']}", o["features"], n // G.HOP


def range_pass(enc, mel_graph, budget_s):
    stats = {}
    order_names = None
    t0 = time.time()
    used = []
    for kind, item in fixture():
        if time.time() - t0 > budget_s:
            break
        tag, mel, l0 = load_window(kind, item, mel_graph)
        mm = (torch.arange(3000)[None] < l0).to(torch.float32)

        def sink(name, x):
            s = stats.setdefault(name, {"amax": 0.0})
            s["amax"] = max(s["amax"], amax(x))
            if name.endswith(".var"):
                s["min_var"] = min(s.get("min_var", math.inf), float(x.min()))
            if name.endswith(MASKED):  # these carry the -30,000 key bias by design: also their max without it
                s["amax_unmasked"] = max(s.get("amax_unmasked", 0.0), amax(x[x > -15000.0]))
        h = Hook("f32", sink)
        with torch.no_grad():
            enc(mel, mm, q=h)
        order_names = order_names or list(h.order)
        used.append(tag)
    return stats, order_names, used, round(time.time() - t0, 1)


def decide_fix(stats, order_names):
    ln_k, res, other = [], {}, []
    for name in order_names:
        if name.endswith(".sqdiff"):
            blk, site = name.split(".")[0], name.split(".")[1]
            b = int(blk[1:])
            dev = math.sqrt(stats[name]["amax"])
            var = stats[name.replace(".sqdiff", ".var")]["amax"]
            k_x2 = max(0, math.ceil(math.log2(dev / TARGET))) if dev > TARGET else 0
            # the row sum of the squared difference (1,024 x the row variance) kept 4x under 65,504 too, in case a
            # GPU reduction holds the sum in fp16 (memory fp16-storage-norm-overflow: headroom on x^2 and the row sum)
            k_sum = max(0, math.ceil(0.5 * math.log2(1024.0 * var / ROWSUM_LIMIT))) if 1024.0 * var > ROWSUM_LIMIT else 0
            k = max(k_x2, k_sum)
            ln_k.append({"block": b, "site": site.replace("n_", ""), "max_abs_dev": round(dev, 3), "k": k,
                         "k_x2": k_x2, "k_rowsum": k_sum, "max_row_var": var,
                         "max_sqdiff": stats[name]["amax"], "min_row_var": stats[name.replace(".sqdiff", ".var")]
                         .get("min_var"), "max_abs_in": stats[name.replace(".sqdiff", ".in")]["amax"]})
    for name in order_names:
        a = stats[name].get("amax_unmasked", stats[name]["amax"])
        if a <= RES_LIMIT or name.endswith((".sqdiff", ".var", ".var_eps", ".scaled", ".mean")):
            continue
        blk = name.split(".")[0]
        if blk.startswith("b") and (".r_" in name or name.endswith(".in") or ".res_in" in name):
            b = int(blk[1:])
            res[b] = max(res.get(b, 0), math.ceil(math.log2(a / (RES_LIMIT / 2))))
        else:
            other.append({"op": name, "max_abs": a})
    return ln_k, res, other


def compare(names, ref, emu, vm):
    rows = []
    for n in names:
        a, b = ref[n], emu[n]
        ax, lv = time_axis(a, vm)
        av = a.narrow(ax, 0, lv) if ax is not None else a
        bv = b.narrow(ax, 0, lv) if ax is not None else b
        fin_v = torch.isfinite(bv)
        nf_v = int((~fin_v).sum())
        nf_all = int((~torch.isfinite(b)).sum())
        d = (bv - av).abs()
        dmax = float(d[fin_v].max()) if bool(fin_v.any()) else math.inf
        ref_max = amax(av)
        rows.append({"op": n, "ref_max_valid": ref_max, "ref_max_all": amax(a), "emu_max_all": amax(b[torch.isfinite(b)])
                     if bool(torch.isfinite(b).any()) else None, "dmax_valid": dmax,
                     "rel_valid": (dmax / ref_max if ref_max > 0 else (0.0 if dmax == 0 else math.inf)),
                     "nonfinite_valid": nf_v, "nonfinite_all": nf_all})
    return rows


def first_over(rows, thr):
    """first op (execution order) past thr relative to its own max on valid frames, or non-finite on valid frames; a
    LayerNorm's row mean is skipped (it cancels to ~0 against the row's spread, so its own relative error says
    nothing)."""
    for r in rows:
        if r["op"].endswith(".mean"):
            continue
        if r["nonfinite_valid"] or r["rel_valid"] > thr:
            return {"op": r["op"], "rel_valid": r["rel_valid"], "nonfinite_valid": r["nonfinite_valid"],
                    "ref_max_valid": r["ref_max_valid"]}
    return None


def first_nonfinite(rows):
    """first op whose output holds inf / NaN on any frame (padded rows run on the GPU too)"""
    for r in rows:
        if r["nonfinite_all"]:
            return {"op": r["op"], "nonfinite_all": r["nonfinite_all"], "nonfinite_valid": r["nonfinite_valid"],
                    "ref_max_all": r["ref_max_all"]}
    return None


def run_pass(enc, mel, mm, mode):
    store = {}
    h = Hook(mode, lambda n, x: store.__setitem__(n, x.detach().clone()))
    with torch.no_grad():
        out = enc(mel, mm, q=h)
    return out["enc"][0], store, h.order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-s", type=float, default=80.0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--small", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    R.set_threads(a.threads)
    t0 = time.time()
    if a.small:
        import r4_small
        model = r4_small.small_model_640()
    else:
        model, _ = R.build("B")
    proc = R.processor()
    mel_graph = G.MelB(proc.feature_extractor.mel_filters, 3000).eval()
    enc = G.EncB(model, 3000).eval()
    rep = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "env": R.env_record(), "target": TARGET,
           "res_limit": RES_LIMIT, "build_s": round(time.time() - t0, 1)}
    stats, names, used, rs = range_pass(enc, mel_graph, a.budget_s)
    rep["range"] = {"windows": len(used), "s32": sum(1 for u in used if not u.startswith("fleurs")),
                    "fleurs": [u for u in used if u.startswith("fleurs")], "seconds": rs, "ops": len(names)}
    ln_k, res_m, other = decide_fix(stats, names)
    fix = {"target": TARGET, "rowsum_limit": ROWSUM_LIMIT, "res_limit": RES_LIMIT, "ln_k": ln_k, "res_m": {str(k): v for k, v in res_m.items()},
           "other_over_limit": other, "fixture_windows": len(used)}
    lk, rm = G.fix_from_json(fix)
    enc_fix = G.EncB(model, 3000, ln_k=lk, res_m=rm).eval()
    enc16 = round_model(enc)
    enc16_fix = round_model(enc_fix)
    diag = []
    for stem in DIAG_WINDOWS:
        z = np.load(f"out/s32_ii/{stem}.npz")
        mel = torch.from_numpy(z["mel"])[None]
        l0, ve = int(z["valid_mel"]), int(z["valid_enc"])
        mm = (torch.arange(3000)[None] < l0).to(torch.float32)
        vm = valid_map(l0)
        e32, s32, order = run_pass(enc, mel, mm, "f32")
        e32f, s32f, order_f = run_pass(enc_fix, mel, mm, "f32")
        row = {"stem": stem, "valid_mel": l0, "valid_enc": ve,
               "fixed_vs_unfixed_fp32_max_abs": float((e32f[:ve] - e32[:ve]).abs().max()),
               "fixed_vs_unfixed_fp32_bit_equal": bool(torch.equal(e32f, e32)),
               "fp32_vs_B_max_abs": float((e32[:ve] - torch.from_numpy(z["enc_proj"])[:ve]).abs().max())}
        for tag, m16, ref_store, ref_order in (("unfixed", enc16, s32, order), ("fixed", enc16_fix, s32f, order_f)):
            e16, s16, o16 = run_pass(m16, mel, mm, "f16")
            assert o16 == ref_order, tag
            rows = compare(ref_order, ref_store, s16, vm)
            e_ref = e32 if tag == "unfixed" else e32f
            fin = bool(torch.isfinite(e16[:ve]).all())
            dec = R.tdt_loop(model, e16, ve, cap=10)
            row[tag] = {"ops": rows, "first_over": {str(t): first_over(rows, t) for t in (1e-3, 1e-2, 1e-1)},
                        "first_nonfinite": first_nonfinite(rows), "out_finite": fin,
                        "out_max_abs": float((e16[:ve] - e_ref[:ve]).abs().max()) if fin else None,
                        "out_argmax_rows_equal": int((e16[:ve].argmax(-1) == e_ref[:ve].argmax(-1)).sum()),
                        "tokens_eq_bii": dec["tokens"] == z["tokens"].tolist(), "n_tokens": len(dec["tokens"]),
                        "n_tokens_bii": int(len(z["tokens"]))}
            del s16
        del s32, s32f
        diag.append(row)
        print(json.dumps({"stem": stem, "fixed_vs_unfixed": row["fixed_vs_unfixed_fp32_max_abs"],
                          "unfixed": {k: row["unfixed"][k] for k in ("first_nonfinite", "first_over", "out_max_abs",
                                                                      "tokens_eq_bii", "n_tokens")},
                          "fixed": {k: row["fixed"][k] for k in ("first_nonfinite", "first_over", "out_max_abs",
                                                                  "tokens_eq_bii", "n_tokens")}}, default=str),
              flush=True)
    un_ok = all(r["unfixed"]["tokens_eq_bii"] for r in diag)
    fx_ok = all(r["fixed"]["tokens_eq_bii"] for r in diag)
    fix["emulation"] = {"unfixed_tokens_eq": [r["unfixed"]["tokens_eq_bii"] for r in diag],
                        "fixed_tokens_eq": [r["fixed"]["tokens_eq_bii"] for r in diag],
                        "fixed_vs_unfixed_fp32_bit_equal": [r["fixed_vs_unfixed_fp32_bit_equal"] for r in diag]}
    fix["n_sites_rescaled"] = sum(1 for r in ln_k if r["k"] > 0)
    fix["usable"] = bool(fix["n_sites_rescaled"] > 0 or rm)
    fix["note"] = ("emulation: unfixed breaks / fixed keeps the tokens" if (not un_ok and fx_ok) else
                   "emulation: unfixed keeps the tokens (the fp16-storage model does not reproduce the GPU)" if un_ok
                   else "emulation: the fixed graph still breaks")
    rep["fix"] = fix
    rep["range_ops"] = [{"op": n, "max_abs_all": stats[n]["amax"],
                         **({"min_var": stats[n]["min_var"]} if "min_var" in stats[n] else {}),
                         **({"max_abs_unmasked": stats[n]["amax_unmasked"]} if "amax_unmasked" in stats[n] else {})}
                        for n in names]
    rep["diag"] = diag
    rep["total_s"] = round(time.time() - t0, 1)
    json.dump(rep, open(f"{OUT}/diag.json", "w"), indent=0, default=str)
    json.dump(fix, open(f"{OUT}/fix.json", "w"), indent=1, default=str)
    print(json.dumps({"range": rep["range"], "n_sites_rescaled": fix["n_sites_rescaled"], "res_m": fix["res_m"],
                      "other_over_limit": len(other), "usable": fix["usable"], "note": fix["note"],
                      "total_s": rep["total_s"]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
