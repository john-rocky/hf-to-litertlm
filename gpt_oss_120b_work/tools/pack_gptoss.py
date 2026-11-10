#!/usr/bin/env python3
"""pack_gptoss.py SRC OUT_DIR [--experts xb8|xb4] [--xb4-rule mse|t2|minmax] [--fc fp32|int8|int4]
                  [--lm-head fp32|int8|int4] [--layers N] [--rope-max 8192] [--bias-unit-c 30]

Weight pre-pass for the Tensor API gpt-oss writer (tensor_api/gptoss): reads openai/gpt-oss-120b (or -20b) in its
Hugging Face layout (MXFP4 experts: <proj>_blocks [E, out, in/32, 16] uint8 + <proj>_scales [E, out, in/32] uint8
E8M0; everything else bf16) one layer at a time and writes
  OUT_DIR/weights.bin     every tensor the decoder needs, 64-byte aligned
  OUT_DIR/index.tsv       name, kind, shape, data offset, data bytes, scales offset, scales count, block size
  OUT_DIR/config.tsv      dims and constants the writer reads
  OUT_DIR/moe_options.bin the "moe" custom options (flexbuffer map)
  OUT_DIR/pack.json       what was packed: source files and their sha256 marker, per-layer seconds, peak RSS,
                          the xb4 re-quantization error per layer and projection, scale floor counts

How gpt-oss's expert maps onto the "moe" op (act(gate . x) * (ff1 . x) -> linear, sum of per_expert_scale *
top_w * y; SiLU in the op needs a LiteRT newer than dcb657e8):
  author:  g = x W_g + b_g, u = x W_u + b_u (gate / up interleaved in gate_up: ::2 / 1::2),
           h = (clamp(u, -L, L) + 1) * min(g, L) * sigmoid(alpha * min(g, L)),  y = h W_d + b_d
  op form: x' = [x, 1/S, 0 x 31]  (S = 2^10; model_dim D' = D + 32: one more input block that carries the biases)
           gate' row j = alpha * [W_g[j], S b_g[j], 0...]       -> silu(gate' x') = alpha g sigmoid(alpha g)
           up'   row j = [W_u[j], S (b_u[j] + 1), 0...]           -> up' x' = u + 1
           bias unit j = I: gate' = [0.., S c, 0..], up' = [0.., 1/silu(c), 0..] -> hidden_I = 1/S (c = 30)
           down' [D', E, 1, I'] (I' = I + 32): column I = S alpha b_d, rows D..D'-1 zero
           per_expert_scale = 1 / alpha  -> output = sum_k w_k (h_k W_d + b_d) without the clamps
  The graph feeds x' (CONCATENATION) and slices the first D outputs. The clamps are NOT in this form (the op has
  no clamp): the stats count how often they would bite.
  --act swiglu_oai (the author's math inside the op, which no released LiteRT moe kernel computes: h = (clamp(ff1, -L, L) + 1) g' sigmoid(alpha g'), g' = min(gate, L)
  inside the op): gate' row j = [W_g[j], S b_g[j], 0...] (no alpha), up' row j = [W_u[j], S b_u[j], 0...] (no +1: the op
  adds it after the clamp), bias unit gate' = [0.., 1, 0..] -> g = 1/S, up' row I = 0 -> hidden_I = h0 = (1/S)
  sigmoid(alpha / S) (no cancellation, below the limit), down column I = b_d / h0, per_expert_scale = 1; the moe
  options carry swiglu_alpha / swiglu_limit from the config. Exact against the author's math, clamps included.
Expert kinds:
  xb8  int8, block 32 = the MXFP4 values exactly: q = 2 * fp4 in {0, +-1, +-2, +-3, +-4, +-6, +-8, +-12},
       scale = 2^(e - 128) (times alpha on gate rows). Lossless; 2x the bytes of int4.
  xb4  int4, block 32: q in [-8, 7], scale = t * 2^(e - 128) with t chosen per block: --xb4-rule mse = the t of a
       fixed candidate grid with the least squared error over the block's 32 FP4 codes (exact when the block fits),
       t2 = t of 2 for every block (exact on even codes), minmax = max|v| / 7. The bias block is exact in every
       kind (one nonzero value per block).
Scales float32 [out', E, 1, in'/32] in <name>_scales (the multi-axis blockwise form); a block scale below 2^-14
is raised to 2^-14 (a zero or subnormal block scale breaks the int4 kernels; counted per tensor, only all-zero or tiny blocks).
Attention q/k/v/o (bias kept fp32) follow --fc (fp32 / int8 per channel / int4 blockwise 32), the router is fp32
(weight and bias), sinks / norms fp32, embedding fp32, lm_head --lm-head. RoPE: cos / sin tables [rope_max, 64]
fp32 = transformers' GptOssRotaryEmbedding (yarn, attention scaling folded in), the writer gathers rows by input_pos.
"""
import argparse
import json
import math
import os
import resource
import sys
import time

import numpy as np

ALIGN = 64
FLOOR = np.float32(2.0 ** -14)
BIAS_S = 2.0 ** 10  # the graph feeds 1/BIAS_S in the bias column of x'; the bias columns hold bias * BIAS_S
FP4 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], np.float32)
V2 = (FP4 * 2).astype(np.int8)  # code -> value in half units: q of the lossless int8 form
# t candidates for the xb4 rule (in half units of the E8M0 scale): a fine grid plus the exact ratios
T_CAND = np.unique(np.concatenate([np.linspace(0.6, 2.4, 91), [1.0, 8 / 7, 4 / 3, 1.5, 12 / 7, 2.0]])).astype(np.float32)


def peak_rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30  # bytes on macOS


def xb4_tables(rule):
    """Per (candidate t, code): the int4 q and the squared error in half units."""
    if rule == "t2":
        ts = np.array([2.0], np.float32)
    elif rule == "mse":
        ts = T_CAND
    else:
        ts = None
    if ts is None:
        return None, None, None
    q = np.clip(np.rint(V2[None, :].astype(np.float32) / ts[:, None]), -8, 7).astype(np.int8)  # [T, 16]
    err = (V2[None, :].astype(np.float32) - ts[:, None] * q) ** 2  # [T, 16]
    return ts, q, err.astype(np.float32)


class Src:
    """The checkpoint through tools/st_pread.py (large sequential preads, no mmap page faults: 3 vs 16-25 MB/s on
    the shared HDD). The U8 expert tensors of the current layer are read whole once and sliced per expert (the
    previous layer's are dropped first), so each byte is read once."""

    def __init__(self, src):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from st_pread import StPread
        self.r = StPread(src)
        self.read = set()
        self.cache = {}

    def f32(self, name):
        self.read.add(self.r._fn(name))
        return np.ascontiguousarray(self.r.get(name), dtype=np.float32)

    def u8_expert(self, name, e):
        if name not in self.cache:
            layer = name.split(".mlp.")[0]
            for k in [k for k in self.cache if not k.startswith(layer + ".")]:
                del self.cache[k]
            self.read.add(self.r._fn(name))
            self.cache[name] = self.r.get(name)
        return self.cache[name][e]  # one expert's [out, G, 16] / [out, G] uint8


def codes_of(blocks):
    """[out, G, 16] uint8 -> [out, G, 32] codes (low nibble = even index)."""
    out = np.empty(blocks.shape[:-1] + (32,), np.uint8)
    out[..., 0::2] = blocks & 0x0F
    out[..., 1::2] = blocks >> 4
    return out


def requant_xb4(codes, ts, qtab, errtab, rule):
    """codes [R, G, 32] -> (q int8 [R, G, 32], t [R, G] f32, sq. error [R, G] in half units)."""
    R, G, _ = codes.shape
    if rule == "minmax":
        v = V2[codes].astype(np.float32)
        amax = np.abs(v).max(axis=2)
        t = np.where(amax > 0, amax / 7.0, 1.0).astype(np.float32)
        q = np.clip(np.rint(v / t[..., None]), -8, 7).astype(np.int8)
        e = ((v - q * t[..., None]) ** 2).sum(axis=2)
        return q, t, e
    flat = codes.reshape(R * G, 32)
    ids = (np.arange(R * G, dtype=np.int64)[:, None] * 16 + flat).ravel()
    hist = np.bincount(ids, minlength=R * G * 16).reshape(R * G, 16).astype(np.float32)
    err = hist @ errtab.T  # [RG, T]
    best = err.argmin(axis=1)
    q = qtab[best[:, None], flat]  # [RG, 32]
    return q.reshape(R, G, 32), ts[best].reshape(R, G), err[np.arange(R * G), best].reshape(R, G)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--experts", default="xb4", choices=["xb8", "xb4"])
    ap.add_argument("--xb4-rule", default="mse", choices=["mse", "t2", "minmax"])
    ap.add_argument("--fc", default="int8", choices=["fp32", "int8", "int4"])
    ap.add_argument("--lm-head", default="int8", choices=["fp32", "int8", "int4"])
    ap.add_argument("--layers", type=int, help="keep the first N layers")
    ap.add_argument("--rope-max", type=int, default=8192)
    ap.add_argument("--bias-unit-c", type=float, default=30.0)
    ap.add_argument("--act", default="silu", choices=["silu", "gelu_tanh", "swiglu_oai"],
                    help="silu = alpha folded (exact without the clamps; needs a LiteRT newer than dcb657e8); gelu_tanh = "
                         "g sigmoid(alpha g) approximated by gelu_tanh(g) (runs on the release moe op); swiglu_oai = "
                         "the author's math with the clamps (no released LiteRT moe kernel computes it)")
    a = ap.parse_args()

    src = os.path.abspath(a.src)
    cfg = json.load(open(os.path.join(src, "config.json")))
    assert cfg["model_type"] == "gpt_oss" and cfg["quantization_config"]["quant_method"] == "mxfp4"
    L = a.layers or cfg["num_hidden_layers"]
    D, I, E = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_local_experts"]
    H, KV, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    alpha = float(cfg.get("swiglu_alpha", 1.702))
    limit = float(cfg.get("swiglu_limit", 7.0))
    B = 32
    assert D % B == 0 and I % B == 0
    Dp, Ip = D + B, I + B
    c = np.float32(a.bias_unit_c)
    silu_c = float(c / (1.0 + math.exp(-float(c))))
    # act(c) for the bias unit: silu(30) and gelu_tanh(30) are both 30 in f32
    if a.act == "gelu_tanh":
        silu_c = float(0.5 * c * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (c + 0.044715 * c ** 3))))
    fold = alpha if a.act == "silu" else 1.0  # alpha folded into the gate rows; expert scale = 1 / fold
    # swiglu_oai: the bias unit's hidden value as the kernel computes it (u = 0, g = 2^-10 exact), in float32
    g0 = np.float32(1.0 / BIAS_S)
    h0 = np.float32(np.float32(1.0) * g0 / (np.float32(1.0) + np.exp(-np.float32(alpha) * g0)))
    wtype = {"xb8": "int8", "xb4": "int4"}[a.experts]
    os.makedirs(a.out, exist_ok=True)

    from flatbuffers import flexbuffers
    mo = {"num_experts": E, "num_active_experts": cfg["num_experts_per_tok"], "model_dim": Dp, "hidden_dim": Ip,
          "activation": a.act, "weight_type": wtype, "renormalized_top_weights": True}
    if a.act == "swiglu_oai":
        mo.update({"swiglu_alpha": alpha, "swiglu_limit": limit})
    opts = bytes(flexbuffers.Dumps(mo))
    open(os.path.join(a.out, "moe_options.bin"), "wb").write(opts)

    dims = {"hidden": D, "hidden_p": Dp, "inter": I, "inter_p": Ip, "layers": L, "heads": H, "kv_heads": KV,
            "head_dim": hd, "experts": E, "top_k": cfg["num_experts_per_tok"], "vocab": cfg["vocab_size"],
            "eps": cfg["rms_norm_eps"], "sliding_window": cfg["sliding_window"],
            "layer_types": "".join("S" if t == "sliding_attention" else "F" for t in cfg["layer_types"][:L]),
            "alpha": alpha, "limit": limit, "rope_max": a.rope_max, "expert_kind": a.experts,
            "bias_in": 1.0 / BIAS_S, "act": a.act, "expert_scale": 1.0 / fold}
    if a.act == "swiglu_oai":
        dims["bias_unit_hidden"] = float(h0)

    s = Src(src)
    ts, qtab, errtab = xb4_tables(a.xb4_rule) if a.experts == "xb4" else (None, None, None)
    rows, off = [], 0
    stats = {"floored_scales": 0, "floored_nonzero_blocks": 0, "e8m0_min": 255, "e8m0_max": 0}
    xb4_err = {}
    t0 = time.time()
    layer_seconds = []
    f = open(os.path.join(a.out, "weights.bin"), "wb")

    def put(arr):
        nonlocal off
        pad = (-off) % ALIGN
        if pad:
            f.write(b"\0" * pad)
            off += pad
        start = off
        arr = np.ascontiguousarray(arr)
        f.write(memoryview(arr).cast("B"))
        off += arr.nbytes
        return start, arr.nbytes

    def put_fp32(name, w):
        w = np.ascontiguousarray(w, np.float32)
        d0, dn = put(w)
        rows.append((name, "fp32", w.shape, d0, dn, 0, 0, 0))

    def put_fc(name, kind, w=None):
        w = s.f32(name) if w is None else w
        if kind == "fp32":
            put_fp32(name, w)
        elif kind == "int8":
            amax = np.abs(w).max(axis=1)
            sc = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
            q = np.clip(np.rint(w / sc[:, None]), -127, 127).astype(np.int8)
            d0, dn = put(q)
            s0, _ = put(sc)
            rows.append((name, "pc8", w.shape, d0, dn, s0, sc.size, 0))
        else:
            R, C = w.shape
            wb = w.reshape(R, C // B, B)
            amax = np.abs(wb).max(axis=2)
            sc = np.maximum(np.where(amax > 0, amax / 7.0, 1.0), FLOOR).astype(np.float32)
            q = np.clip(np.rint(wb / sc[..., None]), -8, 7).astype(np.int8).reshape(R, C)
            d0, dn = put(pack4(q))
            s0, _ = put(sc)
            rows.append((name, "bw4", w.shape, d0, dn, s0, sc.size, B))

    def pack4(q):
        q = np.ascontiguousarray(q).reshape(-1)
        assert q.size % 2 == 0
        lo = (q[0::2].astype(np.uint8) & 0x0F)
        hi = (q[1::2].astype(np.uint8) & 0x0F) << 4
        return (lo | hi).astype(np.uint8)

    # The RoPE tables, from transformers' own module (yarn inv_freq and attention scaling).
    import torch
    from transformers import GptOssConfig
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRotaryEmbedding
    tcfg = GptOssConfig(**{k: v for k, v in cfg.items() if k not in ("quantization_config",)})
    rot = GptOssRotaryEmbedding(tcfg)
    pos = torch.arange(a.rope_max, dtype=torch.long)[None, :]
    with torch.no_grad():
        cos, sin = rot(torch.zeros(1, dtype=torch.float32), pos)  # [1, P, hd/2] fp32
    cos = cos[0].numpy()
    sin = sin[0].numpy()
    put_fp32("rope.cos", np.concatenate([cos, cos], axis=1))  # [P, hd] (HF rotate_half form)
    put_fp32("rope.sin", np.concatenate([sin, sin], axis=1))
    dims["attention_scaling"] = float(rot.attention_scaling)

    put_fp32("model.embed_tokens.weight", s.f32("model.embed_tokens.weight"))

    for l in range(L):
        tl = time.time()
        p = f"model.layers.{l}."
        put_fp32(p + "input_layernorm.weight", s.f32(p + "input_layernorm.weight"))
        put_fp32(p + "post_attention_layernorm.weight", s.f32(p + "post_attention_layernorm.weight"))
        put_fp32(p + "self_attn.sinks", s.f32(p + "self_attn.sinks"))
        for n in ("q", "k", "v", "o"):
            put_fc(p + f"self_attn.{n}_proj.weight", a.fc)
            put_fp32(p + f"self_attn.{n}_proj.bias", s.f32(p + f"self_attn.{n}_proj.bias"))
        put_fp32(p + "mlp.router.weight", s.f32(p + "mlp.router.weight"))
        put_fp32(p + "mlp.router.bias", s.f32(p + "mlp.router.bias"))

        b_gu = s.f32(p + "mlp.experts.gate_up_proj_bias")  # [E, 2I]
        b_d = s.f32(p + "mlp.experts.down_proj_bias")  # [E, D]
        G_in_gu, G_in_d = D // B, I // B
        for proj in ("gate", "up", "down"):
            out_f, in_f, G = (Ip, Dp, Dp // B) if proj != "down" else (Dp, Ip, Ip // B)
            if a.experts == "xb8":
                q = np.zeros((out_f, E, in_f), np.int8)
            else:
                q = np.zeros((out_f, E, in_f), np.int8)  # unpacked int4 values, packed at the end
            sc = np.full((out_f, E, G), FLOOR, np.float32)
            err_sum = 0.0
            ref_sum = 0.0
            for e in range(E):
                if proj == "down":
                    blk = s.u8_expert(p + "mlp.experts.down_proj_blocks", e)  # [D, I/32, 16]
                    ex = s.u8_expert(p + "mlp.experts.down_proj_scales", e).astype(np.int32)  # [D, I/32]
                    rsel = slice(None)
                else:
                    sel = 0 if proj == "gate" else 1
                    blk = s.u8_expert(p + "mlp.experts.gate_up_proj_blocks", e)[sel::2]  # [I, D/32, 16]
                    ex = s.u8_expert(p + "mlp.experts.gate_up_proj_scales", e)[sel::2].astype(np.int32)
                stats["e8m0_min"] = min(stats["e8m0_min"], int(ex.min()))
                stats["e8m0_max"] = max(stats["e8m0_max"], int(ex.max()))
                codes = codes_of(blk)  # [R, G0, 32]
                R, G0, _ = codes.shape
                base = np.ldexp(np.float32(1.0), ex - 128).astype(np.float32)  # half-unit scale [R, G0]
                if a.experts == "xb8":
                    qe = V2[codes]
                    se = base
                else:
                    qe, t, e2 = requant_xb4(codes, ts, qtab, errtab, a.xb4_rule)
                    se = (t * base).astype(np.float32)
                    v = V2[codes].astype(np.float64)
                    err_sum += float((e2.astype(np.float64) * base.astype(np.float64) ** 2).sum())
                    ref_sum += float(((v ** 2).sum(axis=2) * base.astype(np.float64) ** 2).sum())
                if proj == "gate":
                    se = (se * np.float32(fold)).astype(np.float32)
                q[:R, e, :G0 * B] = qe.reshape(R, G0 * B)
                sc[:R, e, :G0] = se
            # The bias block / bias unit (exact in every kind: one nonzero value per block). The graph's constant
            # input is 1/S (S = 2^10, exact), so the bias columns hold bias * S and their block scales stay above the
            # 2^-14 floor for |bias| >= ~7e-6 (int8) / ~4e-7 (int4); the bias unit's hidden value is 1/S, so the
            # down column holds alpha * b_d * S. (With a constant 1 the floor re-quantized 27,157 bias blocks of the
            # first 2 layers, 19:54.)
            qmax = 127 if a.experts == "xb8" else 7
            S = np.float32(BIAS_S)
            if proj in ("gate", "up"):
                plus1 = np.float32(0.0 if a.act == "swiglu_oai" else 1.0)  # swiglu_oai: the op adds the +1
                bias = b_gu[:, 0::2] * np.float32(fold) if proj == "gate" else b_gu[:, 1::2] + plus1
                bias = (bias * S).T  # [I, E]
                q[:I, :, D] = np.where(bias != 0, np.sign(bias) * qmax, 0).astype(np.int8)
                sc[:I, :, D // B] = np.where(bias != 0, np.abs(bias) / qmax, FLOOR).astype(np.float32)
                # bias unit: gate' x' = c, up' x' = 1 / (S silu(c)) -> hidden_I = 1/S
                unit = np.float32(c) * S if proj == "gate" else np.float32(1.0 / silu_c)
                if a.act == "swiglu_oai":  # gate' x' = 1/S, up' x' = 0 -> hidden_I = h0
                    unit = np.float32(1.0) if proj == "gate" else np.float32(0.0)
                q[I, :, D] = qmax if unit != 0 else 0
                sc[I, :, D // B] = unit / qmax if unit != 0 else FLOOR
            else:
                bias = (b_d * np.float32(fold) * S).T  # [D, E]
                if a.act == "swiglu_oai":
                    bias = (b_d / h0).T  # hidden_I = h0
                q[:D, :, I] = np.where(bias != 0, np.sign(bias) * qmax, 0).astype(np.int8)
                sc[:D, :, I // B] = np.where(bias != 0, np.abs(bias) / qmax, FLOOR).astype(np.float32)
            low = sc < FLOOR
            if low.any():
                # Raise the scale to the floor and re-quantize those blocks' values on it (mostly to 0).
                qv = q.reshape(out_f, E, G, B)
                idx = np.nonzero(low)
                vals = qv[idx].astype(np.float32) * sc[idx][:, None]
                stats["floored_nonzero_blocks"] += int((np.abs(qv[idx]).max(axis=1) > 0).sum())
                stats["floored_scales"] += int(low.sum())
                lo_q = -127 if a.experts == "xb8" else -8
                qv[idx] = np.clip(np.rint(vals / FLOOR), lo_q, qmax).astype(np.int8)
                sc[idx] = FLOOR
            name = p + f"mlp.experts.{proj}"
            d0, dn = put(pack4(q) if a.experts == "xb4" else q)
            rows.append((name, a.experts, (out_f, E, 1, in_f), d0, dn, 0, 0, B))
            put_fp32(name + "_scales", sc.reshape(out_f, E, 1, G))
            if a.experts == "xb4":
                xb4_err[name] = {"rel_rmse": math.sqrt(err_sum / max(ref_sum, 1e-30))}
            del q, sc
            print(f"{name} {a.experts} [{out_f}, {E}, 1, {in_f}]"
                  + (f" rel_rmse {xb4_err[name]['rel_rmse']:.4f}" if name in xb4_err else ""), flush=True)
        layer_seconds.append(round(time.time() - tl, 1))
        print(f"LAYER {l} {layer_seconds[-1]} s, peak RSS {peak_rss_gib():.2f} GiB", flush=True)

    put_fp32("model.norm.weight", s.f32("model.norm.weight"))
    put_fc("lm_head.weight", a.lm_head)
    f.close()

    with open(os.path.join(a.out, "index.tsv"), "w") as fo:
        for r in rows:
            fo.write("\t".join([r[0], r[1], ",".join(str(x) for x in r[2])] + [str(x) for x in r[3:]]) + "\n")
    with open(os.path.join(a.out, "config.tsv"), "w") as fo:
        for k, v in dims.items():
            fo.write(f"{k}\t{v}\n")
    ok = {fn: (open(os.path.join(src, fn + ".ok")).read().strip() if os.path.exists(os.path.join(src, fn + ".ok"))
               else None) for fn in sorted(s.read)}
    meta = {"source": src, "shards_read_sha256_ok": ok, "dims": dims, "experts": a.experts,
            "xb4_rule": a.xb4_rule if a.experts == "xb4" else None, "fc": a.fc, "lm_head": a.lm_head,
            "bias_unit_c": float(c), "silu_c": silu_c, "stats": stats, "xb4_err": xb4_err, "bytes": off,
            "moe_options_hex": opts.hex(), "seconds": round(time.time() - t0, 1), "layer_seconds": layer_seconds,
            "tensors": len(rows), "peak_rss_gib": round(peak_rss_gib(), 2)}
    json.dump(meta, open(os.path.join(a.out, "pack.json"), "w"), indent=1)
    print(f"PACK_DONE {len(rows)} tensors {off} B in {meta['seconds']} s; experts {a.experts}; stats {stats}; "
          f"peak RSS {meta['peak_rss_gib']} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
