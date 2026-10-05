"""The r13_kernel forms on Kev-4B shapes, torch only (random inputs; no full-model forward).

    python r15_kernel_test.py --model 4b --out results/r15_kernel_test_4b.json

The 4B GatedDeltaNet has 16 key heads and 32 value heads (ratio 2): the guarded forward copies every q / k head with
kev_qwen35_patch._litert_interleave_heads (concat + rank-4 reshape, patch guard (6)) before the chunk kernel, so the
kernel sees 32 heads of q / k / v with dk = dv = 128 (the 0.8B: 16 heads, no copy). Two levels:

1. kernel: q / k drawn with 16 heads and interleaved to 32 (every head pair identical, as in the 4B), v with 32 heads,
   in four regimes (R1 random keys + strong decay, R2 weak decay, R3 correlated keys + weak decay, R4 one key for every
   token + no decay), L 256 / 130 (tail pad) / 200, with and without an initial state. Per form, the kernel part of the
   form (A = R64+ec, B = R64+ec+dd+vs8; sp acts in the GatedDeltaNet forward, not in the kernel) vs the unmodified
   kernel (kev_qwen35_patch._rank4_chunk_gated_delta_rule = the loop kernel) and vs the float64 token-by-token
   recurrence; vs8 multiplies v by 2^8 at the kernel entry, so the output AND the recurrent state it carries are exactly
   2^8 larger (the delta rule is linear in (v, state) jointly): an initial state must come in scaled by 2^8 too (= the
   shared-state pair's gdn_state contract under vs8: the state outputs are 2^8 x the stock ones), and the output / final
   state are compared after / 2^8. The unscaled-state run (the wrong contract) is recorded as a control.
2. layer: real 4B GatedDeltaNet layers (weights of merged/kev-4b-v1.0 read from the safetensors, nothing else loaded)
   in the patched class; input = random hidden states with a right pad (valid = 1 then 0: the pad guard); the guarded
   forward with the loop kernel vs with r13_kernel.apply(<holder>, form) (the kernel, the softplus rewrite in the
   guarded forward's F, vs8's gated norm). Reported: out max |diff| at the real positions (absolute and relative to max
   |out|), finiteness, and the interleave calls (2 per forward = q and k).
Never overwrites --out."""
import argparse
import json
import time

import torch

import kev_qwen35_patch as P
import r13_kernel as R
from r2_common import CKPT, K, MODEL, add_model_arg, dump_json

FORMS = {"A": "R64+sp+ec", "B": "R64+sp+ec+dd+vs8"}
KERNELS = {"A": dict(inverse="R64", expclamp=True, decay="cumsum", vscale_log2=0),
           "B": dict(inverse="R64", expclamp=True, decay="direct", vscale_log2=8)}


def regime(name, L, seed, hk, hv, D, init_state=False):
    """q / k with hk heads interleaved to hv (ratio hv / hk), v with hv heads."""
    gen = torch.Generator().manual_seed(seed)
    r = hv // hk
    q = torch.randn(1, L, hk, D, generator=gen)
    if name in ("R1", "R2"):
        k = torch.randn(1, L, hk, D, generator=gen)
    elif name == "R3":
        u = torch.randn(1, 1, hk, D, generator=gen)
        u = u / u.norm(dim=-1, keepdim=True) * D ** 0.5
        k = u + 0.3 * torch.randn(1, L, hk, D, generator=gen)
    else:
        k = torch.randn(1, 1, hk, D, generator=gen).expand(1, L, hk, D).contiguous()
    v = torch.randn(1, L, hv, D, generator=gen)
    if name == "R1":
        beta = 0.2 + 0.7 * torch.rand(1, L, hv, generator=gen)
        g = -(0.05 + 2.95 * torch.rand(1, L, hv, generator=gen))
    elif name in ("R2", "R3"):
        beta = (0.2 if name == "R2" else 0.3) + (0.7 if name == "R2" else 0.7) * torch.rand(1, L, hv, generator=gen)
        g = -0.02 * torch.rand(1, L, hv, generator=gen)
    else:
        beta = torch.full((1, L, hv), 0.5)
        g = torch.zeros(1, L, hv)
    s0 = torch.randn(1, hv, D, D, generator=gen) * 0.1 if init_state else None
    if r > 1:
        q, k = P._litert_interleave_heads(q, r), P._litert_interleave_heads(k, r)
    return q, k, v, g, beta, s0


def truth_f64(q, k, v, g, beta, s0):
    """The gated delta recurrence in float64 (l2norm eps 1e-6, q scaled by D^-0.5), any head count."""
    D = q.shape[-1]
    q, k, v, g, beta = [x.to(torch.float64) for x in (q, k, v, g, beta)]
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    q, k, v = [x.transpose(1, 2) for x in (q, k, v)]          # [1, H, L, D]
    g, beta = g.transpose(1, 2), beta.transpose(1, 2)        # [1, H, L]
    q = q * D ** -0.5
    H = q.shape[1]
    s = torch.zeros(1, H, D, v.shape[-1], dtype=torch.float64) if s0 is None else s0.to(torch.float64).clone()
    outs = []
    for t in range(q.shape[2]):
        s = s * g[:, :, t].exp()[..., None, None]
        kv = (s * k[:, :, t, :, None]).sum(-2)
        delta = (v[:, :, t] - kv) * beta[:, :, t, None]
        s = s + k[:, :, t, :, None] * delta[:, :, None, :]
        outs.append((s * q[:, :, t, :, None]).sum(-2))
    return torch.stack(outs, 2).transpose(1, 2), s


def kernel_level(hk, hv, D):
    rows = []
    for name, L, seed, init in (("R1", 256, 11, False), ("R1", 130, 12, True), ("R2", 256, 21, False),
                                ("R2", 130, 22, True), ("R3", 256, 31, False), ("R3", 130, 32, True),
                                ("R4", 200, 41, True), ("R4", 256, 42, False)):
        q, k, v, g, beta, s0 = regime(name, L, seed, hk, hv, D, init)
        o_t, s_t = truth_f64(q, k, v, g, beta, s0)
        o_ref, s_ref = P._rank4_chunk_gated_delta_rule(q, k, v, g, beta, initial_state=s0, output_final_state=True,
                                                       use_qk_l2norm_in_kernel=True)
        row = {"regime": name, "L": L, "initial_state": init, "heads_q_k_drawn": hk, "heads_kernel": hv, "dim": D,
               "head_pairs_identical": bool(torch.equal(q[:, :, 0::2], q[:, :, 1::2])) if hv > hk else None,
               "max_abs_truth_out": float(o_t.abs().max()), "max_abs_truth_state": float(s_t.abs().max()),
               "stock_vs_truth_out": float((o_ref.double() - o_t).abs().max()),
               "stock_vs_truth_state": float((s_ref.double() - s_t).abs().max()), "forms": {}}
        for f, kw in KERNELS.items():
            kern = R.make_kernel(**kw)
            scale = float(2 ** kw["vscale_log2"])
            # vs<k>: the kernel scales v by 2^k at its entry; the recurrent state it carries (and outputs) is then 2^k x
            # the stock state, so a continued state must come in scaled too (the pair's gdn_state contract under vs<k>)
            s_in = None if s0 is None else s0 * scale
            o, s = kern(q, k, v, g, beta, initial_state=s_in, output_final_state=True, use_qk_l2norm_in_kernel=True)
            o, s = o / scale, s / scale
            if s0 is not None and scale != 1.0:   # the wrong contract (unscaled state) for the record
                o_u, _ = kern(q, k, v, g, beta, initial_state=s0, output_final_state=True, use_qk_l2norm_in_kernel=True)
                unscaled_state_vs_stock = float((o_u / scale - o_ref).abs().max())
            else:
                unscaled_state_vs_stock = None
            row["forms"][f] = {"kernel": kern.__name__, "finite": bool(torch.isfinite(o).all() and torch.isfinite(s).all()),
                               "vs_stock_out": float((o - o_ref).abs().max()),
                               "vs_stock_state": float((s - s_ref).abs().max()),
                               "vs_truth_out": float((o.double() - o_t).abs().max()),
                               "vs_truth_state": float((s.double() - s_t).abs().max()),
                               "initial_state_scaled_by": scale if s0 is not None else None,
                               "unscaled_initial_state_vs_stock_out": unscaled_state_vs_stock}
        rows.append(row)
        print(json.dumps({"regime": name, "L": L, "init": init, "stock_vs_truth": row["stock_vs_truth_out"],
                          **{f: (round(x["vs_stock_out"], 10), round(x["vs_truth_out"], 10)) for f, x in row["forms"].items()}}),
              flush=True)
    return rows


class Holder(torch.nn.Module):
    """r13_kernel.apply / reset walk model.modules() and model.layers."""

    def __init__(self, gdn):
        super().__init__()
        self.layers = torch.nn.ModuleList([gdn])


def load_gdn(layer):
    """One PatchedQwen3_5GatedDeltaNet with the checkpoint's weights for decoder layer `layer`."""
    from safetensors import safe_open
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(str(CKPT))
    cfg._attn_implementation = "eager"
    m = P.PatchedQwen3_5GatedDeltaNet(cfg, layer).eval()
    assert m.chunk_gated_delta_rule is P._rank4_chunk_gated_delta_rule
    idx = json.loads((CKPT / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"layers.{layer}.linear_attn."
    names = [n for n in idx if n.startswith(prefix)]
    sd = {}
    for shard in sorted({idx[n] for n in names}):
        with safe_open(str(CKPT / shard), framework="pt") as f:
            for n in names:
                if idx[n] == shard:
                    sd[n[len(prefix):]] = f.get_tensor(n)
    m.load_state_dict(sd, strict=True)
    return m, {"layer": layer, "tensors": sorted(sd), "num_k_heads": m.num_k_heads, "num_v_heads": m.num_v_heads,
               "head_k_dim": m.head_k_dim, "head_v_dim": m.head_v_dim, "conv_dim": m.conv_dim, "norm": type(m.norm).__name__}


def layer_level(layers, L=200, n_real=173, seeds=(5, 6), scales=(1.0, 3.0)):
    import r11_fp16_safe as S11
    g = P.PatchedQwen3_5GatedDeltaNet.forward.__globals__
    inner = g["_litert_interleave_heads"]
    calls = {"n": 0}

    def counted(x, r):
        calls["n"] += 1
        return inner(x, r)

    g["_litert_interleave_heads"] = counted
    out = []
    try:
        for layer in layers:
            m, info = load_gdn(layer)
            holder = Holder(m)
            valid = torch.zeros(1, L)
            valid[0, :n_real] = 1.0
            for seed in seeds:
                for sc in scales:
                    x = torch.randn(1, L, m.hidden_size, generator=torch.Generator().manual_seed(seed)) * sc
                    m._litert_valid = valid
                    c0 = calls["n"]
                    with torch.no_grad():
                        ref = m(x)
                    stock_calls = calls["n"] - c0
                    row = {**info, "seed": seed, "input_scale": sc, "L": L, "n_real": n_real,
                           "interleave_calls_stock": stock_calls, "max_abs_out_stock": float(ref[0, :n_real].abs().max()),
                           "forms": {}}
                    for f, spec in FORMS.items():
                        applied = R.apply(holder, spec)
                        try:
                            c1 = calls["n"]
                            with torch.no_grad():
                                y = m(x)
                            d = (y - ref)[0, :n_real].abs().max()
                            row["forms"][f] = {"spec": spec, "kernel": applied["kernel"], "r11": applied["r11_applied"],
                                               "finite": bool(torch.isfinite(y).all()),
                                               "interleave_calls": calls["n"] - c1,
                                               "vs_stock_out_max_abs": float(d),
                                               "vs_stock_out_rel": float(d / ref[0, :n_real].abs().max())}
                        finally:
                            R.reset(holder)
                    assert S11.APPLIED == [], S11.APPLIED
                    out.append(row)
                    print(json.dumps({"layer": layer, "seed": seed, "scale": sc, "interleave_stock": stock_calls,
                                      **{f: (x["vs_stock_out_max_abs"], x["vs_stock_out_rel"], x["interleave_calls"])
                                         for f, x in row["forms"].items()}}), flush=True)
    finally:
        g["_litert_interleave_heads"] = inner
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    add_model_arg(ap)
    a = ap.parse_args()
    out = K / a.out
    assert not out.exists(), out
    torch.set_num_threads(4)
    t0 = time.perf_counter()
    cfg = json.loads((CKPT / "config.json").read_text())
    hk, hv, D = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"], cfg["linear_key_head_dim"]
    doc = {"step": "r13_kernel forms on the model's GatedDeltaNet shapes (random inputs)",
           "model": MODEL, "checkpoint": str(CKPT.relative_to(K)), "forms": FORMS, "kernel_parts": KERNELS,
           "heads_k": hk, "heads_v": hv, "dim": D}
    doc["kernel"] = kernel_level(hk, hv, D)
    gdn_layers = [i for i, t in enumerate(cfg["layer_types"]) if t == "linear_attention"]
    doc["layer"] = layer_level([gdn_layers[0], gdn_layers[3]])
    kmax = {f: max(r["forms"][f]["vs_stock_out"] / max(r["max_abs_truth_out"], 1e-30) for r in doc["kernel"]) for f in FORMS}
    smax = {f: max(r["forms"][f]["vs_stock_state"] / max(r["max_abs_truth_state"], 1e-30) for r in doc["kernel"])
            for f in FORMS}
    lmax = {f: max(r["forms"][f]["vs_stock_out_rel"] for r in doc["layer"]) for f in FORMS}
    doc["summary"] = {
        "kernel_vs_stock_out_rel_max": kmax, "kernel_vs_stock_state_rel_max": smax,
        "stock_vs_truth_out_max": max(r["stock_vs_truth_out"] for r in doc["kernel"]),
        "forms_vs_truth_out_max": {f: max(r["forms"][f]["vs_truth_out"] for r in doc["kernel"]) for f in FORMS},
        "layer_vs_stock_out_rel_max": lmax,
        "all_finite": all(x["finite"] for r in doc["kernel"] for x in r["forms"].values())
        and all(x["finite"] for r in doc["layer"] for x in r["forms"].values()),
        "head_pairs_identical": all(r["head_pairs_identical"] for r in doc["kernel"]) if hv > hk else None,
        "interleave_calls_per_forward": sorted({r["interleave_calls_stock"] for r in doc["layer"]}
                                               | {x["interleave_calls"] for r in doc["layer"] for x in r["forms"].values()}),
    }
    doc["seconds"] = round(time.perf_counter() - t0, 1)
    dump_json(out, doc)
    print(json.dumps(doc["summary"], indent=1))


if __name__ == "__main__":
    main()
