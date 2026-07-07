"""GPTQ int4 (blockwise-32, symmetric) for the folded Qwen3-TTS MTP.

Data-free int4 collapses the MTP (residual-code errors compound through the
in-graph AR loop). This runs classic GPTQ in torch on every FC weight of
MTPFolded using real calibration activations, bakes the DEQUANTIZED weights
(exact int4 blockwise-32 grid) into the graph, re-exports, and the sibling
recovery step (quantize_mtp_folded_v2.py-style DEQUANTIZED_WEIGHT_RECOVERY)
turns the grid into a true int4 flatbuffer.

Calibration = per-frame MTP inputs (past_hidden, cb0): the 52 e2e reference
frames + out/qwen3tts-mtp/calib/calib_*.npz dumped by hostloop_e2e.py
(TEXT=… DUMP_MTP=…). Hessians are accumulated online (no activation storage).

  .venv/bin/python qwen3tts_work/gptq_mtp_folded.py
Outputs:
  out/qwen3tts-mtp/mtp_folded_gptqfq.tflite   (fp32 values on the int4 grid)

Env switches (ablation): QUANT_BODY=0 skip body, QUANT_HEADS=0 skip heads,
EXPORT=0 verify-only (no tflite), TAG=x -> output mtp_folded_gptqfq_x.tflite.
"""
import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SRC = "/Users/majimadaisuke/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-Base/snapshots/5d83992436eae1d760afd27aff78a71d676296fc"
OUT = "out/qwen3tts-mtp"

L, H, KV, HD, FF, V, STEPS = 5, 16, 8, 128, 3072, 2048, 16
EPS, THETA = 1e-6, 1e6
BITS = int(os.environ.get("BITS", "4"))
GROUP = int(os.environ.get("GROUP", "32"))  # 0 => channelwise (one scale/row)
QMAX = 2 ** (BITS - 1) - 1  # symmetric narrow range
PERCDAMP = 0.01


def rms(x, w):
    v = (x.float().pow(2)).mean(-1, keepdim=True)
    return (x * torch.rsqrt(v + EPS)) * w


def rot_half(x):
    a, b = x[..., : HD // 2], x[..., HD // 2:]
    return torch.cat((-b, a), dim=-1)


class MTPFolded(nn.Module):
    """Same graph as export_mtp_folded.MTPFolded + optional Hessian recording."""

    def __init__(self, w, tables):
        super().__init__()
        for k, t in w.items():
            self.register_buffer(k.replace(".", "_"), t, persistent=False)
        self.register_buffer("tables", tables, persistent=False)
        inv = 1.0 / (THETA ** (torch.arange(0, HD, 2, dtype=torch.float32) / HD))
        ang = torch.arange(STEPS, dtype=torch.float32).reshape(-1, 1) * inv
        emb = torch.cat((ang, ang), dim=-1)
        self.register_buffer("cos_t", emb.cos().reshape(STEPS, 1, 1, 1, HD),
                             persistent=False)
        self.register_buffer("sin_t", emb.sin().reshape(STEPS, 1, 1, 1, HD),
                             persistent=False)
        self.hess = None  # {key: [in,in] float64} when recording
        self._buf = {}    # {key: [rows]} pending activation rows
        self._CHUNK = 512  # flush cadence: rank-512 GEMMs, not rank-1
        # (rank-1 fp64 updates are memory-bound death: every update rewrites
        #  the full H matrix, ~123 MB traffic per layer-step -> hours. Buffer
        #  rows and accumulate in rank-CHUNK GEMMs instead: ~500x less traffic.)

    def _rec(self, key, x):
        if self.hess is None:
            return
        rows = self._buf.setdefault(key, [])
        rows.append(x.reshape(-1, x.shape[-1]).float())
        if len(rows) >= self._CHUNK:
            self._flush(key)

    def _flush(self, key):
        rows = self._buf.get(key)
        if not rows:
            return
        a = torch.cat(rows, dim=0).double()
        h = self.hess.setdefault(key, torch.zeros(a.shape[-1], a.shape[-1],
                                                  dtype=torch.float64))
        h += a.T @ a
        rows.clear()

    def flush_all(self):
        for key in list(self._buf):
            self._flush(key)

    def _step(self, x, t, ks, vs):
        cos, sin = self.cos_t[t], self.sin_t[t]
        for i in range(L):
            g = lambda n: getattr(self, f"layers_{i}_{n}".replace(".", "_"))
            h = rms(x, g("input_layernorm_weight"))
            self._rec(f"attn_in_{i}", h)
            q = F.linear(h, g("self_attn_q_proj_weight")).view(1, 1, H, HD)
            k = F.linear(h, g("self_attn_k_proj_weight")).view(1, 1, KV, HD)
            v = F.linear(h, g("self_attn_v_proj_weight")).view(1, 1, KV, HD)
            q = rms(q, g("self_attn_q_norm_weight")).transpose(1, 2)
            k = rms(k, g("self_attn_k_norm_weight")).transpose(1, 2)
            v = v.transpose(1, 2)
            q = q * cos + rot_half(q) * sin
            k = k * cos + rot_half(k) * sin
            kc = k if ks[i] is None else torch.cat((ks[i], k), dim=2)
            vc = v if vs[i] is None else torch.cat((vs[i], v), dim=2)
            ks[i], vs[i] = kc, vc
            kr = kc.repeat_interleave(H // KV, dim=1)
            vr = vc.repeat_interleave(H // KV, dim=1)
            att = torch.matmul(q, kr.transpose(2, 3)) * (HD ** -0.5)
            att = att.softmax(dim=-1)
            o = torch.matmul(att, vr)
            o = o.transpose(1, 2).reshape(1, 1, H * HD)
            self._rec(f"o_in_{i}", o)
            x = x + F.linear(o, g("self_attn_o_proj_weight"))
            h2 = rms(x, g("post_attention_layernorm_weight"))
            self._rec(f"mlp_in_{i}", h2)
            dn = F.silu(F.linear(h2, g("mlp_gate_proj_weight"))) * \
                F.linear(h2, g("mlp_up_proj_weight"))
            self._rec(f"down_in_{i}", dn)
            x = x + F.linear(dn, g("mlp_down_proj_weight"))
        return rms(x, self.norm_weight)

    def forward(self, past_hidden, cb0_embed, noise):
        ks = [None] * L
        vs = [None] * L
        codes, logits_seq = [], []
        embed = past_hidden
        for t in range(STEPS):
            hidden = self._step(embed, t, ks, vs)
            if t == 0:
                embed = cb0_embed
                continue
            head = t - 1
            self._rec(f"head_in_{head}", hidden)
            logits = F.linear(hidden, self.heads[head]).reshape(V)
            logits_seq.append(logits)
            code = (logits + noise[head]).argmax(-1)
            codes.append(code)
            if t < STEPS - 1:
                embed = self.tables[head].index_select(
                    0, code.reshape(1)).reshape(1, 1, 1024)
        return (torch.stack(codes).to(torch.int32), torch.stack(logits_seq))


def load_weights():
    from safetensors import safe_open
    f = safe_open(f"{SRC}/model.safetensors", framework="pt")
    pre = "talker.code_predictor."
    w = {}
    for k in f.keys():
        if k.startswith(pre + "model.layers.") or k == pre + "model.norm.weight":
            w[k[len(pre + "model."):]] = f.get_tensor(k).to(torch.float32)
    heads = [f.get_tensor(f"{pre}lm_head.{i}.weight").to(torch.float32)
             for i in range(15)]
    w["heads"] = torch.stack(heads)
    embs = [f.get_tensor(f"{pre}model.codec_embedding.{i}.weight").to(torch.float32)
            for i in range(15)]
    return w, torch.stack(embs)


def gptq_quantize(W, Hs, group=GROUP, qmax=QMAX, percdamp=PERCDAMP):
    """Classic GPTQ (no act-order) with per-(row, group-of-32) symmetric scales.

    W: [out, in] fp32 tensor. Hs: [in, in] float64 Hessian (X^T X).
    Returns the dequantized weight (exact multiples of per-group scales).
    """
    Wd = W.clone().double()
    n = Wd.shape[1]
    if not group:
        group = n  # channelwise: one scale per output row
    Hd = Hs.clone()
    dead = torch.diag(Hd) == 0
    Hd[dead, dead] = 1.0
    Wd[:, dead] = 0.0
    damp = percdamp * torch.mean(torch.diag(Hd))
    Hd += torch.eye(n, dtype=torch.float64) * damp
    Lc = torch.linalg.cholesky(Hd)
    Hinv = torch.cholesky_inverse(Lc)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)  # upper-triangular factor

    Q = torch.zeros_like(Wd)
    scales = None
    BLK = 128
    for bs in range(0, n, BLK):
        be = min(bs + BLK, n)
        Werr = torch.zeros_like(Wd[:, bs:be])
        for j in range(bs, be):
            if j % group == 0:
                ge = min(j + group, n)
                scales = Wd[:, j:ge].abs().max(dim=1).values / qmax
                scales = torch.clamp(scales, min=1e-10)
            w = Wd[:, j]
            q = torch.clamp(torch.round(w / scales), -qmax, qmax) * scales
            Q[:, j] = q
            d = Hinv[j, j]
            err = (w - q) / d
            if j + 1 < be:
                Wd[:, j + 1: be] -= err.reshape(-1, 1) * Hinv[j, j + 1: be].reshape(1, -1)
            Werr[:, j - bs] = err
        if be < n:
            Wd[:, be:] -= Werr @ Hinv[bs:be, be:]
    return Q.float()


# ---------------- run ----------------
print("loading weights…")
w, tables = load_weights()
m = MTPFolded(w, tables).eval()
codec_emb = np.load("out/qwen3tts-host/codec_embedding.npy")
zero_noise = torch.zeros(15, V)

# calibration inputs: 52 ref frames + all dumped sentences
ref = np.load("qwen3tts_work/ref/e2e_ref.npz")
pairs = [(ref["hiddens"][i], int(ref["codes"][i, 0]))
         for i in range(ref["codes"].shape[0])]
for f in sorted(glob.glob(f"{OUT}/calib/calib_*.npz")):
    d = np.load(f)
    # cap per sentence so one greedy-runaway trajectory can't dominate
    n = min(len(d["cb0s"]), 150)
    pairs += [(d["hiddens"][i], int(d["cb0s"][i])) for i in range(n)]
print(f"calibration: {len(pairs)} frames")

HESS_CACHE = f"{OUT}/hessians_{len(pairs)}.pt"
if os.path.exists(HESS_CACHE):
    hess = torch.load(HESS_CACHE)
    print(f"hessians: loaded {len(hess)} keys from cache")
else:
    m.hess = {}
    import time as _time
    _t0 = _time.time()
    with torch.no_grad():
        for pi, (hid, cb0) in enumerate(pairs):
            m(torch.tensor(hid.reshape(1, 1, 1024), dtype=torch.float32),
              torch.tensor(codec_emb[cb0].reshape(1, 1, 1024),
                           dtype=torch.float32),
              zero_noise)
            if (pi + 1) % 100 == 0:
                print(f"  calib {pi + 1}/{len(pairs)} "
                      f"({_time.time() - _t0:.0f}s)", flush=True)
    m.flush_all()
    print(f"hessians: {len(m.hess)} keys "
          f"(~{len(pairs) * STEPS} samples/body matrix, ~{len(pairs)} per head)")
    hess = m.hess
    m.hess = None
    torch.save(hess, HESS_CACHE)

# GPTQ every FC weight (ablatable per group)
QUANT_BODY = os.environ.get("QUANT_BODY", "1") == "1"
QUANT_HEADS = os.environ.get("QUANT_HEADS", "1") == "1"
if QUANT_BODY:
    plan = []
    for i in range(L):
        plan += [
            (f"layers_{i}_self_attn_q_proj_weight", f"attn_in_{i}"),
            (f"layers_{i}_self_attn_k_proj_weight", f"attn_in_{i}"),
            (f"layers_{i}_self_attn_v_proj_weight", f"attn_in_{i}"),
            (f"layers_{i}_self_attn_o_proj_weight", f"o_in_{i}"),
            (f"layers_{i}_mlp_gate_proj_weight", f"mlp_in_{i}"),
            (f"layers_{i}_mlp_up_proj_weight", f"mlp_in_{i}"),
            (f"layers_{i}_mlp_down_proj_weight", f"down_in_{i}"),
        ]
    for name, hkey in plan:
        Wt = getattr(m, name)
        Qw = gptq_quantize(Wt, hess[hkey])
        rel = (Qw - Wt).norm() / Wt.norm()
        getattr(m, name).copy_(Qw)
        print(f"  {name}: rel err {rel:.4f}")
if QUANT_HEADS:
    heads_q = torch.empty_like(m.heads)
    for k in range(15):
        Qh = gptq_quantize(m.heads[k], hess[f"head_in_{k}"])
        rel = (Qh - m.heads[k]).norm() / m.heads[k].norm()
        heads_q[k] = Qh
        print(f"  head_{k}: rel err {rel:.4f}")
    m.heads.copy_(heads_q)

# ---------------- verify tokens vs fp32 trajectory ----------------
ok = 0
tok_ok = 0
tot = ref["codes"].shape[0]
with torch.no_grad():
    for i in range(tot):
        cb0 = int(ref["codes"][i, 0])
        c, _ = m(torch.tensor(ref["hiddens"][i].reshape(1, 1, 1024)),
                 torch.tensor(codec_emb[cb0].reshape(1, 1, 1024)),
                 zero_noise)
        eq = (c.numpy() == ref["codes"][i, 1:])
        tok_ok += int(eq.sum())
        if eq.all():
            ok += 1
print(f"gptq torch-vs-ref: {ok}/{tot} frames all-15 exact, "
      f"per-token {tok_ok}/{tot * 15} = {tok_ok / (tot * 15):.1%}")

# ---------------- export ----------------
if os.environ.get("EXPORT", "1") == "1":
    import litert_torch  # noqa: E402

    sample = (torch.zeros(1, 1, 1024), torch.zeros(1, 1, 1024),
              torch.zeros(15, V))
    ed = litert_torch.convert(m, sample)
    tag = os.environ.get("TAG")
    path = f"{OUT}/mtp_folded_gptqfq{'_' + tag if tag else ''}.tflite"
    ed.export(path)
    print(f"exported -> {path} ({os.path.getsize(path)/1e6:.1f} MB)")
print("GPTQ_DONE")
