"""r14_norm_range_probe.py for Kev-4B (--model 4b) and the r15_form.py forms (the r13_kernel tokens + fn<k> / in<k>,
the norm pre-scales). The method and the json shape are r14_norm_range_probe.py's; added for fn<k> / in<k>: the final
norm's / the decoder input norms' sum of squares as their reduction sees it ("final_norm_effective" /
"input_layernorm_effective" = the raw input's sum x 4^-k; the raw sites "final_norm" / "input_layernorm" stay the
unscaled input). The gated norm's site is already the scaled input (vs<k> scales the kernel output that enters it).

    python r15_norm_range_probe.py --model 4b --form R64+sp+ec+dd+vs6+in1+fn5 [--shards 8]
        -> results/r15_norm_range_4b_<form>.json (never overwritten)

Why: the converter lowers every RMS-type norm's mean(x^2) to SUM (+ a constant MUL), so under fp16 storage (GPU default
precision, and FP16_WITH_FP32_ACCUM, whose SUM accumulates in fp32 but stores its result in fp16) a sum of squares above
65,504 becomes inf and rsqrt(inf) = 0 = a silently zeroed row.
Per question (each at the smallest bucket that holds its row, torch fp32, 1 thread per shard, the form applied), over
the real positions, the maximum of: the sum of squares of every Qwen3_5RMSNorm input (input_layernorm /
post_attention_layernorm, the final norm, attention q_norm / k_norm per head), of the GatedDeltaNet output norm's input
(Qwen3_5RMSNormGated, 128 per head; under vs<k> already scaled), and of the kernel's l2norm inputs (q, k: 128 per
head; pads are exact zeros); plus the largest |value| entering each. The json reports the maxima over all questions,
the top 10 questions per site and the margin to 65,504. Shards: --shards; at most 4 while the file named by the
environment variable KEV_GPU_LOCK (another job timing the GPU, when set) holds an owner line with "timing"."""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

from r2_common import MODEL, RESULT_SUFFIX, K, add_model_arg, dump_json, load_oracle, qkey

BUCKETS = (64, 128, 256, 512, 1024, 2048)
FP16_MAX = 65504.0


def ftag(form):
    return form.replace("+", "-")


def worker(form, i, n):
    import torch
    torch.set_num_threads(1)
    import r15_form as R
    from kev_graph import KevPrefill, load_text_model, row_inputs
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    model, _ = load_text_model()
    R.apply(model, form)
    oracle = load_oracle()
    qs = oracle["questions"][i::n]
    cur = {"valid_n": 0, "rec": None}

    def note(site, x):
        """x = [1, L, D] (norms), [1, L, H, D] (q_norm / k_norm / l2norm) or [L * H, D] (gated norm) -> [L, H, D];
        records [max sum(x^2) over real positions, max |x| over real positions, max sum(x^2) over all positions
        (pads included: an inf there can reach real rows through 0 * inf in the masked attention)]."""
        if cur["rec"] is None:
            return
        L, n = cur["L"], cur["valid_n"]
        x = x.detach().float()
        assert x.numel() % L == 0 and (x.dim() == 2 or x.shape[1] == L), (site, tuple(x.shape), L)
        x = x.reshape(L, -1, x.shape[-1])
        ss = (x * x).sum(-1)                                   # [L, H]
        r = cur["rec"].setdefault(site, [0.0, 0.0, 0.0])
        r[0] = max(r[0], float(ss[:n].max()))
        r[1] = max(r[1], float(x[:n].abs().max()))
        r[2] = max(r[2], float(ss.max()))

    fn, inn = R.STATE["fn"], R.STATE["in"]
    hooks = []
    if fn:   # the reduction inside the pre-scaled final norm sees the input times 2^-fn
        hooks.append(model.norm.register_forward_pre_hook(
            lambda m, args: note("final_norm_effective", args[0] * float(2.0 ** -fn))))
    if inn:  # the same for every decoder layer's pre-scaled input_layernorm (in<k>)
        for layer in model.layers:
            hooks.append(layer.input_layernorm.register_forward_pre_hook(
                lambda m, args: note("input_layernorm_effective", args[0] * float(2.0 ** -inn))))
    for name, mod in model.named_modules():
        cls = type(mod).__name__
        if cls == "Qwen3_5RMSNorm":
            site = ("final_norm" if name == "norm" else
                    "input_layernorm" if name.endswith("input_layernorm") else
                    "post_attention_layernorm" if name.endswith("post_attention_layernorm") else
                    "attn_q_norm" if name.endswith("q_norm") else "attn_k_norm" if name.endswith("k_norm") else
                    f"rmsnorm:{name}")
            hooks.append(mod.register_forward_pre_hook(lambda m, args, site=site: note(site, args[0])))
        elif cls == "Qwen3_5RMSNormGated":
            hooks.append(mod.register_forward_pre_hook(lambda m, args: note("gdn_gated_norm", args[0])))
    orig_l2 = M.l2norm

    def l2_rec(x, dim=-1, eps=1e-6):
        note("gdn_l2norm_qk", x)
        return orig_l2(x, dim=dim, eps=eps)
    M.l2norm = l2_rec
    rows, graphs = [], {}
    t0 = time.perf_counter()
    for q in qs:
        L = next(b for b in BUCKETS if b >= q["row_len"])
        if L not in graphs:
            graphs[L] = KevPrefill(model, L).eval().requires_grad_(False)
        cur.update(valid_n=q["row_len"], L=L, rec={})
        with torch.no_grad():
            h = graphs[L](*row_inputs(q["row_ids"], L))["hidden"]
        rows.append({"key": qkey(q), "row_len": q["row_len"], "L": L, "sites": cur["rec"],
                     "finite": bool(torch.isfinite(h).all())})
        cur["rec"] = None
    M.l2norm = orig_l2
    for hk in hooks:
        hk.remove()
    out = K / f"cache/r15/norm_range_{MODEL}_{ftag(form)}_{i}of{n}.json"
    out.write_text(json.dumps({"rows": rows, "seconds": round(time.perf_counter() - t0, 1)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--form", required=True)
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--worker", default="")
    add_model_arg(ap)
    a = ap.parse_args()
    assert a.model == MODEL
    if a.worker:
        i, n = (int(x) for x in a.worker.split("/"))
        return worker(a.form, i, n)
    out = K / f"results/r15_norm_range{RESULT_SUFFIX}_{ftag(a.form)}.json"
    assert not out.exists(), out
    try:
        gl = os.environ.get("KEV_GPU_LOCK", "")
        gl = open(gl).read().strip() if gl else ""
    except OSError:
        gl = ""
    shards = min(a.shards, 4) if "timing" in gl else a.shards     # fewer workers while another job times the GPU
    t0 = time.time()
    env = dict(os.environ, OMP_NUM_THREADS="1")
    procs = [subprocess.Popen([sys.executable, __file__, "--model", MODEL, "--form", a.form, "--worker", f"{i}/{shards}"],
                              env=env, stdout=open(K / f"logs/r15_norm_range_{MODEL}_{ftag(a.form)}_{i}.log", "w"),
                              stderr=subprocess.STDOUT)
             for i in range(shards)]
    rcs = [p.wait() for p in procs]
    assert all(rc == 0 for rc in rcs), rcs
    rows = []
    for i in range(shards):
        p = K / f"cache/r15/norm_range_{MODEL}_{ftag(a.form)}_{i}of{shards}.json"
        rows += json.loads(p.read_text())["rows"]
        p.unlink()
    sites = sorted({s for r in rows for s in r["sites"]})
    summ = {}
    for s in sites:
        vals = [(r["sites"][s][0], r["sites"][s][1], r["sites"][s][2], r["key"], r["row_len"]) for r in rows
                if s in r["sites"]]
        mx = max(v[0] for v in vals)
        mxa = max(v[2] for v in vals)
        summ[s] = {"max_sum_of_squares_real": mx, "max_abs_value_real": max(v[1] for v in vals),
                   "max_sum_of_squares_all_positions": mxa,
                   "margin_to_fp16_max_real": FP16_MAX / mx if mx else None,
                   "margin_to_fp16_max_all_positions": FP16_MAX / mxa if mxa else None,
                   "questions_over_fp16_max_real": sum(v[0] > FP16_MAX for v in vals),
                   "questions_over_fp16_max_all_positions": sum(v[2] > FP16_MAX for v in vals),
                   "top10": [{"key": k, "row_len": n, "sum_of_squares_real": ss, "max_abs_real": aa, "sum_all": sa}
                             for ss, aa, sa, k, n in sorted(vals, key=lambda v: -v[0])[:10]]}
    doc = {"what": "norm reductions vs fp16 storage (sum of squares per position, max over real positions)",
           "model": MODEL, "form": a.form, "copied_from": "scripts/r14_norm_range_probe.py", "questions": len(rows), "all_finite_fp32": all(r["finite"] for r in rows),
           "fp16_max": FP16_MAX, "gpu_lock_at_launch": gl, "shards": shards, "seconds_wall": round(time.time() - t0, 1),
           "sites": summ, "rows": rows}
    dump_json(out, doc)
    print(json.dumps({s: {k: v for k, v in d.items() if k != "top10"} for s, d in summ.items()}, indent=1))


if __name__ == "__main__":
    main()
