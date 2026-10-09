"""Round 8 steps 1-2: the fp16-safe norm rewrite (scripts/d1_graph_f16safe.py) against D1Decision in torch, reference
venv, pinned checkpoint.

    cd d1_omni_work
    venv-ref/bin/python scripts/f16safe_check.py --k-table          # -> results/f16safe_k_table.json (no model run)
    ~/code/standup/tools/quiet/quiet_wait.py -- venv-ref/bin/python scripts/f16safe_check.py
        # -> results/f16safe_eager_check.json

--k-table: the per-site k (round 2's results/norm_range.json `sites.<name>.k`, the smallest k with max sum of squares
x 4^-k <= 65,504 / 4) with the fixture's sum of squares before and after the scale, the margin to 65,504, the largest
element and its square (an fp16 MUL x * x overflows past |x| = 256 before any SUM), the smallest sum after the scale
and its mean (sum / dim: the converter's SUM is followed by MUL 1 / dim, so the mean is the value an fp16 path adds
eps to), and eps x 4^-k with its fp16 value.

Main run (torch fp32, 8 threads):
  1. bit identity: all 448 oracle rows (ref/records_ref.json v2, every row at its smallest bucket, inputs = the host's
     build_inputs + the question type's one-hot) through D1Decision and D1DecisionF16Safe built from the same
     checkpoint state: torch.equal on the scores at the real positions (and at every position); a difference is
     reported with its max |d| and row. Both modules' probabilities vs the oracle (sanity: round 2 gave 1.22e-5).
     While the scaled model runs, forward pre-hooks record per norm site, over the real positions and over every
     position (pads included), the sum of squares the scaled graph forms (x s)^2 summed, its smallest mean, the
     largest |x s| and, for the LayerNorms, |sum(x s)| and the centred sum.
  2. fp16 emulation (a screen for overflow / NaN only: host fp16 emulation is about 5x optimistic about device error,
     memory fp16-storage-norm-overflow): 60 rows (50 text over L128 / 256 / 512 / 2048 / 4096 + 10 media rows: 5 image,
     5 audio), both modules .half() (weights, activations, constants) with the converter's lowering written out:
     RMSNorm = (x s)^2 -> SUM stored in fp16 -> x 1 / dim -> + eps -> rsqrt; LayerNorm = SUM(x s) x 1/dim, the
     centred squares SUM x 1/dim, + eps, rsqrt (MEAN taken as a SUM stored in fp16, the conservative reading); fp32
     islands kept as the launch says: RoPE (cos / sin fp32, rotation in fp32), softmax (fp32), the scores read out in
     fp32 by the host readout. Per module: non-finite rows, uniform rows (every real position the same score = the
     collapse rounds 3-5 saw on the GPU), max / p95 / mean |dp| vs the oracle, argmax, cutoff crossings, and per norm
     site the real positions whose output is all zeros or non-finite. The emulation forwards are checked in fp32
     against the plain modules on 3 rows (max |d scores|) so the harness itself is measured.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d1_src as S  # noqa: E402

K = S.K
FP16_MAX = 65504.0
FP16_MIN_NORMAL = 2.0 ** -14
FP16_MIN_SUBNORMAL = 2.0 ** -24
NEAR_TIE = 0.02
CUTOFFS = (0.5, 0.9)
EMU_PLAN = {128: 28, 256: 13, 512: 5, 2048: 3, 4096: 1}     # text rows per bucket (50)
EMU_MEDIA = {"image": 5, "audio": 5}


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def write_json(path, doc, overwrite=False):
    import os

    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    os.replace(tmp, path)


def fp16_value(v):
    import numpy as np

    return float(np.float16(v))


# ---------------------------------------------------------------- k table

def k_table_main():
    import d1_graph_f16safe as F16

    src = json.loads(F16.NORM_RANGE.read_text())
    sites = {}
    for name, v in src["sites"].items():
        k, dim = int(v["k"]), int(v["dim"])
        s = 2.0 ** -k
        after_max = v["max_sum_sq"] * 4.0 ** -k
        after_max_all = v["max_sum_sq_all_positions"] * 4.0 ** -k
        after_min = v["min_sum_sq"] * 4.0 ** -k
        e = {"kind": v["kind"], "dim": dim, "eps": v["eps"], "k": k, "scale_s": s, "eps_k": v["eps"] * 4.0 ** -k,
             "eps_fp16": fp16_value(v["eps"]), "eps_k_fp16": fp16_value(v["eps"] * 4.0 ** -k),
             "max_sum_sq": v["max_sum_sq"], "max_sum_sq_all_positions": v["max_sum_sq_all_positions"],
             "overflows_fp16_before": v["max_sum_sq"] > FP16_MAX,
             "max_sum_sq_after_k": after_max, "max_sum_sq_after_k_all_positions": after_max_all,
             "margin_after_k": FP16_MAX / after_max, "margin_after_k_all_positions": FP16_MAX / after_max_all,
             "max_abs_x": v["max_abs_x"], "max_abs_x_after_k": v["max_abs_x"] * s,
             "max_element_square_before": v["max_abs_x"] ** 2,
             "element_square_overflows_before": v["max_abs_x"] ** 2 > FP16_MAX,
             "max_element_square_after_k": (v["max_abs_x"] * s) ** 2,
             "min_sum_sq": v["min_sum_sq"], "min_sum_sq_after_k": after_min,
             "min_mean_sq_before": v["min_sum_sq"] / dim, "min_mean_sq_after_k": after_min / dim,
             "min_mean_after_k_below_fp16_min_normal": after_min / dim < FP16_MIN_NORMAL,
             "min_mean_after_k_below_fp16_min_subnormal": after_min / dim < FP16_MIN_SUBNORMAL,
             "top5": v["top5"]}
        if v["kind"] == "layernorm":
            e.update(max_sum_sq_centered=v["max_sum_sq_centered"],
                     max_sum_sq_centered_after_k=v["max_sum_sq_centered"] * 4.0 ** -k,
                     max_abs_sum_x=v["max_abs_sum_x"], max_abs_sum_x_after_k=v["max_abs_sum_x"] * s)
        sites[name] = e
    scaled = sorted(n for n, e in sites.items() if e["k"] > 0)
    kdist = {}
    for e in sites.values():
        kdist[str(e["k"])] = kdist.get(str(e["k"]), 0) + 1
    doc = {"step": "round 8 step 1: the per-site k of the fp16-safe norm rewrite (input x 2^-k, eps x 4^-k)",
           "written": now(), "source": {"file": "results/norm_range.json", "sha256": S.sha256_file(F16.NORM_RANGE),
                                        "written": src["written"], "rows": src["rows"], "by_L": src["by_L"],
                                        "oracle_sha256": src["oracle_sha256"]},
           "rule": "k = the smallest k >= 0 with max sum of squares (fixture, real positions) x 4^-k <= 65,504 / 4 "
                   "(margin >= 4x); the k of round 2's norm_probe.py, used unchanged",
           "fp16": {"max": FP16_MAX, "min_normal": FP16_MIN_NORMAL, "min_subnormal": FP16_MIN_SUBNORMAL},
           "summary": {"sites": len(sites), "sites_k_gt_0": len(scaled), "sites_k_0": len(sites) - len(scaled),
                       "k_histogram": dict(sorted(kdist.items())),
                       "all_margins_ge_4_after_k": all(e["margin_after_k"] >= 4.0 for e in sites.values()),
                       "min_margin_after_k": min(e["margin_after_k"] for e in sites.values()),
                       "min_margin_site": min(sites, key=lambda n: sites[n]["margin_after_k"]),
                       "sites_overflowing_before": sorted(n for n, e in sites.items() if e["overflows_fp16_before"]),
                       "sites_element_square_overflowing_before":
                           sorted(n for n, e in sites.items() if e["element_square_overflows_before"]),
                       "sites_min_mean_after_k_subnormal_fp16":
                           sorted(n for n, e in sites.items() if e["min_mean_after_k_below_fp16_min_normal"]),
                       "sites_eps_k_zero_in_fp16": sorted(n for n, e in sites.items() if e["eps_k_fp16"] == 0.0)},
           "scaled_sites": scaled, "sites": sites}
    assert doc["summary"]["sites"] == 50 and doc["summary"]["sites_k_gt_0"] == 36, doc["summary"]
    assert doc["summary"]["all_margins_ge_4_after_k"], doc["summary"]
    out = K / "results/f16safe_k_table.json"
    write_json(out, doc)
    print(json.dumps(doc["summary"], indent=1))
    return 0


# ---------------------------------------------------------------- rows

def oracle_rows(ref):
    """Every oracle question: (key, record, question, prefix, P, n, L) in oracle order."""
    import numpy as np

    rows = []
    for e in ref["records"]:
        prefix = None
        if e["mode"] != "text":
            with np.load(K / "ref/npz" / f"{e['id']}.npz") as z:
                prefix = np.asarray(z["prefix"], np.float32).copy()
            assert prefix.shape == (e["prefix"], 1024), (e["id"], prefix.shape)
        for q in e["questions"]:
            rows.append({"key": f"{e['id']}/{q['qid']}", "id": e["id"], "mode": e["mode"], "source": e.get("source"),
                         "q": q, "prefix": prefix, "P": int(e["prefix"] or 0), "n": len(q["ids"]),
                         "L": int(q["bucket"])})
    return rows


def emu_rows(rows):
    """60 rows: EMU_PLAN text rows per bucket + EMU_MEDIA media rows, evenly spaced in key order, one question per
    media record first."""
    pick = []
    for L, n in EMU_PLAN.items():
        cand = sorted((r for r in rows if r["mode"] == "text" and r["L"] == L), key=lambda r: r["key"])
        assert len(cand) >= n, (L, len(cand), n)
        step = len(cand) / n
        pick += [cand[int(i * step)] for i in range(n)]
    for mode, n in EMU_MEDIA.items():
        cand = sorted((r for r in rows if r["mode"] == mode), key=lambda r: r["key"])
        firsts, seen = [], set()
        for r in cand:
            if r["id"] not in seen:
                firsts.append(r)
                seen.add(r["id"])
        rest = [r for r in cand if r not in firsts]
        chosen = (firsts + rest)[:n]
        assert len(chosen) == n, (mode, len(cand))
        pick += chosen
    keys = [r["key"] for r in pick]
    assert len(set(keys)) == len(keys) == sum(EMU_PLAN.values()) + sum(EMU_MEDIA.values())
    return pick


def row_inputs(H, r, hq, dtype=None):
    import torch

    x = H.build_inputs(r["q"]["ids"], r["prefix"], r["L"])
    x["qtype_onehot"] = H.qtype_onehot(hq)
    t = {k: torch.from_numpy(v) for k, v in x.items()}
    if dtype is not None:
        t = {k: (v if k == "ids" else v.to(dtype)) for k, v in t.items()}
    return t


def probs_of(H, scores_real, r, hq, temps):
    import numpy as np

    return H.readout(np.asarray(scores_real, np.float32), r["P"], r["q"]["markers"], hq, r["q"]["calibrate"], temps)


def top2_gap(p):
    s = sorted(p, reverse=True)
    return s[0] - s[1] if len(s) > 1 else 1.0


def compare_probs(per):
    """per: list of {probs, ref, nonfinite}; -> statistics like litert_gate.compare (oracle reference)."""
    import numpy as np

    dps, agree, n_main, agree_nt, n_nt, cross, nonfinite = [], 0, 0, 0, 0, {str(c): 0 for c in CUTOFFS}, 0
    flips = []
    for e in per:
        if e["nonfinite"]:
            nonfinite += 1
            continue
        p, q = e["probs"], e["ref"]
        dps.extend(abs(a - b) for a, b in zip(p, q))
        near = top2_gap(q) <= NEAR_TIE
        eq = int(np.argmax(p)) == int(np.argmax(q))
        if near:
            n_nt += 1
            agree_nt += eq
        else:
            n_main += 1
            agree += eq
            if not eq:
                flips.append(e["key"])
        for c in CUTOFFS:
            cross[str(c)] += sum((a >= c) != (b >= c) for a, b in zip(p, q))
    a = np.asarray(dps, np.float64)
    st = {"rows": len(per), "nonfinite_rows": nonfinite,
          "max_abs_dp": float(a.max()) if a.size else None, "p95_abs_dp": float(np.percentile(a, 95)) if a.size else None,
          "mean_abs_dp": float(a.mean()) if a.size else None,
          "argmax_outside_near_tie": f"{agree}/{n_main}", "near_tie": f"{agree_nt}/{n_nt}", "argmax_flips": flips,
          "cutoff_crossings": cross}
    st["bar_pass"] = bool(per and not nonfinite and agree == n_main and st["max_abs_dp"] <= 0.02
                          and st["mean_abs_dp"] <= 0.002)
    return st


# ---------------------------------------------------------------- site ranges (scaled model, fp32)

class SiteRange:
    def __init__(self, name, kind, dim, s):
        self.name, self.kind, self.dim, self.s = name, kind, dim, s
        self.max_real = self.max_all = 0.0
        self.min_real = self.min_all = math.inf
        self.max_abs_real = self.max_abs_all = 0.0
        self.sub_real = self.sub_all = 0          # positions whose mean (sum / dim) < fp16 min normal
        self.zero_all = 0                         # positions whose sum is exactly 0
        self.max_abs_sum = self.max_centered = 0.0
        self.where = None

    def note(self, x, n_real, key):
        xs = x.detach().float() * self.s
        L = xs.shape[1]
        flat = xs.reshape(L, -1, xs.shape[-1])                   # [L, H, d]
        ss = (flat * flat).sum(-1)                                # [L, H]
        real = ss[:n_real]
        mx = float(real.max())
        if mx > self.max_real:
            self.max_real, self.where = mx, key
        self.max_all = max(self.max_all, float(ss.max()))
        self.min_real = min(self.min_real, float(real.min()))
        self.min_all = min(self.min_all, float(ss.min()))
        self.max_abs_real = max(self.max_abs_real, float(flat[:n_real].abs().max()))
        self.max_abs_all = max(self.max_abs_all, float(flat.abs().max()))
        mean = ss / self.dim
        self.sub_real += int((mean[:n_real] < FP16_MIN_NORMAL).sum())
        self.sub_all += int((mean < FP16_MIN_NORMAL).sum())
        self.zero_all += int((ss == 0).sum())
        if self.kind == "layernorm":
            self.max_abs_sum = max(self.max_abs_sum, float(flat[:n_real].sum(-1).abs().max()))
            mu = flat[:n_real].mean(-1, keepdim=True)
            self.max_centered = max(self.max_centered, float(((flat[:n_real] - mu) ** 2).sum(-1).max()))

    def report(self):
        out = {"kind": self.kind, "dim": self.dim, "scale_s": self.s,
               "max_sum_scaled_real": self.max_real, "max_sum_scaled_all_positions": self.max_all,
               "margin_real": FP16_MAX / self.max_real if self.max_real else None,
               "margin_all_positions": FP16_MAX / self.max_all if self.max_all else None,
               "max_row": self.where,
               "min_sum_scaled_real": self.min_real, "min_sum_scaled_all_positions": self.min_all,
               "min_mean_scaled_real": self.min_real / self.dim, "min_mean_scaled_all_positions": self.min_all / self.dim,
               "positions_mean_below_fp16_min_normal_real": self.sub_real,
               "positions_mean_below_fp16_min_normal_all": self.sub_all,
               "positions_sum_exactly_zero_all": self.zero_all,
               "max_abs_scaled_real": self.max_abs_real, "max_abs_scaled_all_positions": self.max_abs_all,
               "max_element_square_scaled_all_positions": self.max_abs_all ** 2}
        if self.kind == "layernorm":
            out.update(max_abs_sum_scaled_real=self.max_abs_sum, max_centered_sum_scaled_real=self.max_centered)
        return out


# ---------------------------------------------------------------- fp16 emulation (the converter's lowering written out)

@contextlib.contextmanager
def lowering_emulation():
    """Patch the norm / attention forwards for the emulation run; restored on exit."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    import d1_graph as G
    import d1_graph_f16safe as F16

    saved = {(G.RMSNorm, "forward"): G.RMSNorm.forward, (F16.ScaledRMSNorm, "forward"): F16.ScaledRMSNorm.forward,
             (nn.LayerNorm, "forward"): nn.LayerNorm.forward,
             (F16.ScaledLayerNorm, "forward"): F16.ScaledLayerNorm.forward,
             (G.Attention, "forward"): G.Attention.forward, (G.HeadLayer, "forward"): G.HeadLayer.forward}

    def rms(self, x):
        s, eps = getattr(self, "s", 1.0), getattr(self, "eps_k", self.eps)
        xs = x * s if s != 1.0 else x
        ms = xs.pow(2).sum(-1, keepdim=True) * (1.0 / xs.shape[-1])       # SUM stored at x's dtype, then MUL 1/dim
        return self.weight * (xs * torch.rsqrt(ms + eps))

    def ln(self, x):
        s, eps = getattr(self, "s", 1.0), getattr(self, "eps_k", self.eps)
        xs = x * s if s != 1.0 else x
        n = xs.shape[-1]
        mu = xs.sum(-1, keepdim=True) * (1.0 / n)
        d = xs - mu
        var = (d * d).sum(-1, keepdim=True) * (1.0 / n)
        return d * torch.rsqrt(var + eps) * self.weight + self.bias

    def attn(self, x, cos, sin, mask2):
        _, length, _ = x.shape
        dt = x.dtype
        q = self.q_layernorm(self.q_proj(x).reshape(1, length, self.heads, self.head_dim)).transpose(1, 2)
        k = self.k_layernorm(self.k_proj(x).reshape(1, length, self.kv_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).reshape(1, length, self.kv_heads, self.head_dim).transpose(1, 2)
        q32, k32 = q.float(), k.float()
        q = (q32 * cos + G.rotate_half(q32) * sin).to(dt)                  # RoPE: fp32 island
        k = (k32 * cos + G.rotate_half(k32) * sin).to(dt)
        qg = q.reshape(1, self.kv_heads, self.groups * length, self.head_dim)
        s = torch.matmul(qg, k.transpose(2, 3)) * self.scale + mask2
        y = torch.matmul(torch.softmax(s.float(), dim=-1).to(dt), v)       # softmax: fp32 island
        y = y.reshape(1, self.heads, length, self.head_dim).transpose(1, 2).reshape(1, length, -1)
        return self.out_proj(y)

    def head_layer(self, x, kmask):
        _, length, _ = x.shape
        dt = x.dtype
        y = self.norm1(x)
        q = self.q_proj(y).reshape(1, length, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(y).reshape(1, length, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(y).reshape(1, length, self.heads, self.head_dim).transpose(1, 2)
        s = torch.matmul(q, k.transpose(2, 3)) * self.scale + kmask
        a = torch.matmul(torch.softmax(s.float(), dim=-1).to(dt), v).transpose(1, 2).reshape(1, length, -1)
        x = x + self.out_proj(a)
        return x + self.linear2(F.relu(self.linear1(self.norm2(x))))

    try:
        G.RMSNorm.forward = rms
        F16.ScaledRMSNorm.forward = rms
        nn.LayerNorm.forward = ln
        F16.ScaledLayerNorm.forward = ln
        G.Attention.forward = attn
        G.HeadLayer.forward = head_layer
        yield
    finally:
        for (cls, name), fn in saved.items():
            setattr(cls, name, fn)


def to_half_keep_rope(m):
    """m.half() with the RoPE tables kept fp32 (the rotation is an fp32 island)."""
    cos, sin = m.encoder.cos.clone(), m.encoder.sin.clone()
    m.half()
    m.encoder.cos, m.encoder.sin = cos, sin
    return m


# ---------------------------------------------------------------- main

def main_check(threads, variant=None, emulation=True):
    import numpy as np
    import torch

    import d1_graph as G
    import d1_graph_f16safe as F16

    t0 = time.time()
    torch.set_num_threads(threads)
    sys.path.insert(0, str(K / "host"))
    import d1_host as H

    out_path = K / (f"results/f16safe_eager_check_{variant}.json" if variant else "results/f16safe_eager_check.json")
    assert not out_path.exists(), f"refusing to overwrite {out_path}"
    ref_path = K / "ref/records_ref.json"
    ref = json.loads(ref_path.read_text())
    fixtures = {r["id"]: r for r in json.loads((K / "fixtures/requests.json").read_text())["records"]}
    cfg_all = S.config()
    cfg, head_layers, temps = cfg_all["text_config"], cfg_all["head_layers"], cfg_all["temperatures"]
    k_table = F16.load_k_table()
    sd = G.checkpoint_state(S.WEIGHTS)
    assert sorted({str(v.dtype) for v in sd.values()}) == ["torch.float32"]
    rows = oracle_rows(ref)
    for r in rows:
        r["hq"] = H.Pm.as_question(fixtures[r["id"]]["request"]["questions"][r["q"]["qid"]])
        assert r["hq"].options == r["q"]["K"], r["key"]
    by_L = {}
    for r in rows:
        by_L.setdefault(r["L"], []).append(r)
    doc = {"step": "round 8 step 2: D1DecisionF16Safe vs D1Decision in torch (fp32 bit identity on every oracle row, "
                   "then an fp16 emulation screen on 60 rows)",
           "written": None, "torch": torch.__version__, "threads": torch.get_num_threads(),
           "python": sys.version.split()[0], "venv": sys.executable,
           "d1_graph_sha256": S.sha256_file(G.__file__), "d1_graph_f16safe_sha256": S.sha256_file(F16.__file__),
           "oracle": {"path": "ref/records_ref.json", "sha256": S.sha256_file(ref_path), "version": ref.get("version")},
           "weights": {"path": str(S.WEIGHTS), "encoder_head_tensors": len(sd)},
           "k_table_source": {"file": "results/norm_range.json", "sha256": S.sha256_file(F16.NORM_RANGE)},
           "rows": len(rows), "rows_by_L": {str(L): len(v) for L, v in sorted(by_L.items())},
           "variant": variant, "variant_sum_form_sites": list(F16.VARIANTS[variant]) if variant else []}

    # ---- 1. fp32 bit identity + site ranges of the scaled model
    t1 = time.time()
    per_row, ranges, report = [], {}, None
    cur = {"n": 0, "key": None}
    with torch.no_grad():
        for L in sorted(by_L):
            A = G.D1Decision(cfg, head_layers, L).eval()
            G.load_state_dict_from_provider(A, sd)
            B = F16.D1DecisionF16Safe(cfg, head_layers, L, k_table, variant=variant).eval()
            G.load_state_dict_from_provider(B, sd)
            report = B.f16safe_report
            hooks = []
            for name, (parent, attr) in F16.norm_sites(B).items():
                mod = getattr(parent, attr)
                dim = mod.weight.shape[0]
                kind = "layernorm" if isinstance(mod, torch.nn.LayerNorm) else "rmsnorm"
                sr = ranges.setdefault(name, SiteRange(name, kind, dim, getattr(mod, "s", 1.0)))
                hooks.append(mod.register_forward_pre_hook(
                    lambda m, args, sr=sr: sr.note(args[0], cur["n"], cur["key"])))
            for r in by_L[L]:
                x = row_inputs(H, r, r["hq"])
                real = r["P"] + r["n"]
                sa = A(**x)["scores"][0]
                cur.update(n=real, key=r["key"])
                sb = B(**x)["scores"][0]
                eq_real = bool(torch.equal(sa[:real], sb[:real]))
                eq_all = bool(torch.equal(sa, sb))
                d_real = float((sa[:real].double() - sb[:real].double()).abs().max())
                pa = probs_of(H, sa[:real].numpy(), r, r["hq"], temps)
                pb = probs_of(H, sb[:real].numpy(), r, r["hq"], temps)
                q = r["q"]
                per_row.append({"key": r["key"], "L": L, "mode": r["mode"], "P": r["P"], "n": r["n"],
                                "bit_equal_real": eq_real, "bit_equal_all_positions": eq_all,
                                "max_abs_dscores_real": d_real,
                                "max_abs_dp_plain_vs_oracle": max(abs(a - b) for a, b in zip(pa, q["probs"])),
                                "max_abs_dp_f16safe_vs_oracle": max(abs(a - b) for a, b in zip(pb, q["probs"])),
                                "probs_bit_equal": pa == pb})
            for h in hooks:
                h.remove()
            del A, B
    neq = [e for e in per_row if not e["bit_equal_real"]]
    doc["fp32_bit_identity"] = {
        "rows": len(per_row), "bit_equal_real_positions": sum(e["bit_equal_real"] for e in per_row),
        "bit_equal_all_positions": sum(e["bit_equal_all_positions"] for e in per_row),
        "probs_bit_equal": sum(e["probs_bit_equal"] for e in per_row),
        "max_abs_dscores_real": max(e["max_abs_dscores_real"] for e in per_row),
        "not_equal_rows": [{"key": e["key"], "L": e["L"], "max_abs_dscores_real": e["max_abs_dscores_real"]}
                           for e in sorted(neq, key=lambda e: -e["max_abs_dscores_real"])][:40],
        "plain_vs_oracle_max_abs_dp": max(e["max_abs_dp_plain_vs_oracle"] for e in per_row),
        "f16safe_vs_oracle_max_abs_dp": max(e["max_abs_dp_f16safe_vs_oracle"] for e in per_row),
        "seconds": round(time.time() - t1, 1)}
    doc["module_report"] = report
    doc["site_ranges_scaled_model"] = {n: ranges[n].report() for n in sorted(ranges)}
    rep = doc["site_ranges_scaled_model"]
    doc["site_ranges_summary"] = {
        "min_margin_all_positions": min(v["margin_all_positions"] for v in rep.values()),
        "min_margin_site": min(rep, key=lambda n: rep[n]["margin_all_positions"]),
        "sites_with_margin_below_4_all_positions": sorted(n for n, v in rep.items()
                                                          if v["margin_all_positions"] < 4.0),
        "max_element_square_scaled": max(v["max_element_square_scaled_all_positions"] for v in rep.values()),
        "sites_with_mean_below_fp16_min_normal_real": {n: v["positions_mean_below_fp16_min_normal_real"]
                                                       for n, v in rep.items()
                                                       if v["positions_mean_below_fp16_min_normal_real"]},
        "sites_with_mean_below_fp16_min_normal_all": {n: v["positions_mean_below_fp16_min_normal_all"]
                                                      for n, v in rep.items()
                                                      if v["positions_mean_below_fp16_min_normal_all"]},
        "sites_with_zero_sum_positions": {n: v["positions_sum_exactly_zero_all"] for n, v in rep.items()
                                          if v["positions_sum_exactly_zero_all"]}}
    doc["per_row_fp32"] = per_row
    print(json.dumps({"fp32_bit_identity": {k: v for k, v in doc["fp32_bit_identity"].items()
                                            if k != "not_equal_rows"},
                      "not_equal_first": doc["fp32_bit_identity"]["not_equal_rows"][:5],
                      "site_ranges_summary": doc["site_ranges_summary"]}, indent=1), flush=True)

    if not emulation:   # the variant run: fp32 identity and site ranges only
        doc["seconds_total"] = round(time.time() - t0, 1)
        doc["written"] = now()
        write_json(out_path, doc)
        return 0

    # ---- 2. fp16 emulation
    t2 = time.time()
    pick = emu_rows(rows)
    emu_by_L = {}
    for r in pick:
        emu_by_L.setdefault(r["L"], []).append(r)
    emu = {"design": __doc__.split("2. fp16 emulation")[1].strip(), "rows": [r["key"] for r in pick],
           "rows_by_L": {str(L): len(v) for L, v in sorted(emu_by_L.items())},
           "rows_by_mode": {m: sum(r["mode"] == m for r in pick) for m in ("text", "image", "audio")}}
    # harness check: the emulation forwards in fp32 vs the plain forwards (3 rows)
    hc = []
    with torch.no_grad():
        L0 = min(emu_by_L)
        for which in ("plain", "f16safe"):
            m = (G.D1Decision(cfg, head_layers, L0) if which == "plain"
                 else F16.D1DecisionF16Safe(cfg, head_layers, L0, k_table)).eval()
            G.load_state_dict_from_provider(m, sd)
            for r in emu_by_L[L0][:3]:
                x = row_inputs(H, r, r["hq"])
                real = r["P"] + r["n"]
                s0 = m(**x)["scores"][0, :real]
                with lowering_emulation():
                    s1 = m(**x)["scores"][0, :real]
                hc.append({"module": which, "key": r["key"], "max_abs_dscores_real": float((s0 - s1).abs().max()),
                           "max_abs_scores_real": float(s0.abs().max())})
            del m
    emu["harness_fp32_vs_plain_forward"] = hc
    results = {"plain": [], "f16safe": []}
    site_out = {"plain": {}, "f16safe": {}}
    with torch.no_grad(), lowering_emulation():
        for L in sorted(emu_by_L):
            for which in ("plain", "f16safe"):
                m = (G.D1Decision(cfg, head_layers, L) if which == "plain"
                     else F16.D1DecisionF16Safe(cfg, head_layers, L, k_table)).eval()
                G.load_state_dict_from_provider(m, sd)
                to_half_keep_rope(m)
                hooks = []
                for name, (parent, attr) in F16.norm_sites(m).items():
                    acc = site_out[which].setdefault(name, {"zero_rows_real": 0, "nonfinite_values_real": 0,
                                                            "rows_with_zero_positions": 0})

                    def out_hook(mod, args, y, acc=acc):
                        n = cur["n"]
                        yy = y.detach()[0, :n].float()
                        flat = yy.reshape(n, -1, yy.shape[-1])
                        z = int((flat.abs().amax(-1) == 0).sum())
                        acc["zero_rows_real"] += z
                        acc["rows_with_zero_positions"] += int(z > 0)
                        acc["nonfinite_values_real"] += int((~torch.isfinite(yy)).sum())
                    hooks.append(getattr(parent, attr).register_forward_hook(out_hook))
                for r in emu_by_L[L]:
                    x = row_inputs(H, r, r["hq"], dtype=torch.float16)
                    real = r["P"] + r["n"]
                    cur.update(n=real, key=r["key"])
                    t = time.time()
                    s = m(**x)["scores"][0, :real].float().numpy().astype(np.float32)
                    finite = bool(np.isfinite(s).all())
                    p = probs_of(H, s, r, r["hq"], temps) if finite else None
                    if p is not None and not all(math.isfinite(v) for v in p):
                        finite, p = False, None
                    results[which].append({"key": r["key"], "L": L, "mode": r["mode"], "P": r["P"], "n": r["n"],
                                           "nonfinite": not finite,
                                           "nonfinite_scores_real": int((~np.isfinite(s)).sum()),
                                           "scores_width_real": (float(np.nanmax(s) - np.nanmin(s))
                                                                 if np.isfinite(s).any() else None),
                                           "uniform_scores": bool(finite and np.nanmax(s) == np.nanmin(s)),
                                           "probs": p, "ref": r["q"]["probs"],
                                           "max_abs_dp": (max(abs(a - b) for a, b in zip(p, r["q"]["probs"]))
                                                          if p is not None else None),
                                           "seconds": round(time.time() - t, 2)})
                for h in hooks:
                    h.remove()
                del m
                print(f"emulation L{L} {which}: {len(emu_by_L[L])} rows", flush=True)
    for which in ("plain", "f16safe"):
        st = compare_probs(results[which])
        st["uniform_rows"] = sum(e["uniform_scores"] for e in results[which])
        st["rows_scores_width_below_1e-3"] = sum(1 for e in results[which]
                                                 if e["scores_width_real"] is not None
                                                 and e["scores_width_real"] < 1e-3)
        st["by_L"] = {}
        for L in sorted(emu_by_L):
            sub = [e for e in results[which] if e["L"] == L]
            dps = [e["max_abs_dp"] for e in sub if e["max_abs_dp"] is not None]
            st["by_L"][str(L)] = {"rows": len(sub), "nonfinite_rows": sum(e["nonfinite"] for e in sub),
                                  "uniform_rows": sum(e["uniform_scores"] for e in sub),
                                  "max_abs_dp": max(dps) if dps else None}
        st["by_mode"] = {}
        for mode in ("text", "image", "audio"):
            sub = [e for e in results[which] if e["mode"] == mode]
            dps = [e["max_abs_dp"] for e in sub if e["max_abs_dp"] is not None]
            st["by_mode"][mode] = {"rows": len(sub), "nonfinite_rows": sum(e["nonfinite"] for e in sub),
                                   "max_abs_dp": max(dps) if dps else None}
        st["norm_sites_with_zero_or_nonfinite_outputs"] = {
            n: v for n, v in site_out[which].items() if v["zero_rows_real"] or v["nonfinite_values_real"]}
        st["seconds"] = round(sum(e["seconds"] for e in results[which]), 1)
        emu[which] = st
    emu["per_row"] = [{"key": a["key"], "L": a["L"], "mode": a["mode"],
                       "plain": {k: a[k] for k in ("nonfinite", "uniform_scores", "scores_width_real", "max_abs_dp")},
                       "f16safe": {k: b[k] for k in ("nonfinite", "uniform_scores", "scores_width_real", "max_abs_dp")},
                       "probs_plain": a["probs"], "probs_f16safe": b["probs"], "probs_oracle": a["ref"]}
                      for a, b in zip(results["plain"], results["f16safe"])]
    emu["seconds"] = round(time.time() - t2, 1)
    doc["fp16_emulation"] = emu
    doc["seconds_total"] = round(time.time() - t0, 1)
    doc["written"] = now()
    write_json(out_path, doc)
    print(json.dumps({"fp16_emulation": {w: {k: emu[w][k] for k in (
        "rows", "nonfinite_rows", "uniform_rows", "max_abs_dp", "p95_abs_dp", "mean_abs_dp",
        "argmax_outside_near_tie", "near_tie", "cutoff_crossings", "bar_pass", "seconds")}
        for w in ("plain", "f16safe")}, "harness": hc, "seconds_total": doc["seconds_total"]}, indent=1))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-table", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--variant", choices=("l0sum",), help="diagnostic form (d1_graph_f16safe.VARIANTS): fp32 identity only")
    a = ap.parse_args()
    if a.k_table:
        return k_table_main()
    return main_check(a.threads, a.variant, emulation=a.variant is None)


if __name__ == "__main__":
    sys.exit(main())
