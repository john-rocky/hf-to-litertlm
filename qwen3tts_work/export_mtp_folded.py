"""Folded (single-invoke) graph for the Qwen3-TTS MTP inner loop.

Replaces the 16-invoke-per-frame decode-step graph (export_mtp.py) with ONE
fixed-shape graph that unrolls all 16 positions x 5 layers, does the greedy
argmax + embedding-table gather in-graph, and keeps KV as internal tensors
(no cache I/O, no mask/pos inputs, no host-side table lookups):

  inputs : past_hidden [1,1,1024] fp32   (talker last hidden state)
           cb0_embed   [1,1,1024] fp32   (codec_embedding row of sampled cb0)
           noise       [15,2048]  fp32   (0 => greedy; T*Gumbel => sampling:
                                          argmax(logits + T*G) ~ softmax(l/T))
  outputs: codes      [15] int32         (residual codebooks 1..15)
           logits_all [15,2048] fp32     (pre-noise, for verification)

The 15 MTP embedding tables ([15,2048,1024]) and 15 lm_heads become graph
constants. Position embeddings (cos/sin for pos 0..15) fold into constants.
Attention is causal by construction: step t concatenates its K/V onto the
running per-layer lists, so no mask is needed.

Verifies torch-vs-reference (single-frame npz + all 52 e2e frames), converts,
then verifies tflite the same way and prints a steady-state benchmark.
  .venv/bin/python qwen3tts_work/export_mtp_folded.py
"""
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SRC = "/Users/majimadaisuke/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-Base/snapshots/5d83992436eae1d760afd27aff78a71d676296fc"
OUT = "out/qwen3tts-mtp"
os.makedirs(OUT, exist_ok=True)

L, H, KV, HD, FF, V, STEPS = 5, 16, 8, 128, 3072, 2048, 16
EPS, THETA = 1e-6, 1e6


def rms(x, w):
    v = (x.float().pow(2)).mean(-1, keepdim=True)
    return (x * torch.rsqrt(v + EPS)) * w


def rot_half(x):
    a, b = x[..., : HD // 2], x[..., HD // 2:]
    return torch.cat((-b, a), dim=-1)


class MTPFolded(nn.Module):
    def __init__(self, w, tables):
        super().__init__()
        for k, t in w.items():
            self.register_buffer(k.replace(".", "_"), t, persistent=False)
        self.register_buffer("tables", tables, persistent=False)  # [15,2048,1024]
        inv = 1.0 / (THETA ** (torch.arange(0, HD, 2, dtype=torch.float32) / HD))
        ang = torch.arange(STEPS, dtype=torch.float32).reshape(-1, 1) * inv  # [16,64]
        emb = torch.cat((ang, ang), dim=-1)
        self.register_buffer("cos_t", emb.cos().reshape(STEPS, 1, 1, 1, HD),
                             persistent=False)
        self.register_buffer("sin_t", emb.sin().reshape(STEPS, 1, 1, 1, HD),
                             persistent=False)

    def _step(self, x, t, ks, vs):
        """One transformer pass at position t; appends K/V in place."""
        cos, sin = self.cos_t[t], self.sin_t[t]
        for i in range(L):
            g = lambda n: getattr(self, f"layers_{i}_{n}".replace(".", "_"))
            h = rms(x, g("input_layernorm_weight"))
            q = F.linear(h, g("self_attn_q_proj_weight")).view(1, 1, H, HD)
            k = F.linear(h, g("self_attn_k_proj_weight")).view(1, 1, KV, HD)
            v = F.linear(h, g("self_attn_v_proj_weight")).view(1, 1, KV, HD)
            q = rms(q, g("self_attn_q_norm_weight")).transpose(1, 2)  # [1,16,1,128]
            k = rms(k, g("self_attn_k_norm_weight")).transpose(1, 2)  # [1,8,1,128]
            v = v.transpose(1, 2)
            q = q * cos + rot_half(q) * sin
            k = k * cos + rot_half(k) * sin

            kc = k if ks[i] is None else torch.cat((ks[i], k), dim=2)  # [1,8,t+1,128]
            vc = v if vs[i] is None else torch.cat((vs[i], v), dim=2)
            ks[i], vs[i] = kc, vc

            kr = kc.repeat_interleave(H // KV, dim=1)  # [1,16,t+1,128]
            vr = vc.repeat_interleave(H // KV, dim=1)
            att = torch.matmul(q, kr.transpose(2, 3)) * (HD ** -0.5)
            att = att.softmax(dim=-1)  # causal: keys 0..t only exist
            o = torch.matmul(att, vr)  # [1,16,1,128]
            o = o.transpose(1, 2).reshape(1, 1, H * HD)
            x = x + F.linear(o, g("self_attn_o_proj_weight"))

            h2 = rms(x, g("post_attention_layernorm_weight"))
            ff = F.linear(F.silu(F.linear(h2, g("mlp_gate_proj_weight"))) *
                          F.linear(h2, g("mlp_up_proj_weight")),
                          g("mlp_down_proj_weight"))
            x = x + ff
        return rms(x, self.norm_weight)  # [1,1,1024]

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
            logits = F.linear(hidden, self.heads[head]).reshape(V)  # [2048]
            logits_seq.append(logits)
            code = (logits + noise[head]).argmax(-1)  # scalar int64
            codes.append(code)
            if t < STEPS - 1:
                embed = self.tables[head].index_select(
                    0, code.reshape(1)).reshape(1, 1, 1024)
        return (torch.stack(codes).to(torch.int32),
                torch.stack(logits_seq))


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
    w["heads"] = torch.stack(heads)  # [15,2048,1024]
    embs = [f.get_tensor(f"{pre}model.codec_embedding.{i}.weight").to(torch.float32)
            for i in range(15)]
    return w, torch.stack(embs)  # tables [15,2048,1024]


w, tables = load_weights()
m = MTPFolded(w, tables).eval()

zero_noise = np.zeros((15, V), np.float32)


def torch_frame(past_hidden, cb0_embed):
    with torch.no_grad():
        c, lg = m(torch.tensor(past_hidden), torch.tensor(cb0_embed),
                  torch.tensor(zero_noise))
    return c.numpy(), lg.numpy()


# ---- verify torch vs single-frame reference ----
d = np.load("qwen3tts_work/ref/mtp_equiv_ref.npz")
codes_t, logits_t = torch_frame(d["past_hidden"], d["last_id_hidden"])
ref_seq = d["seq"][0]
ref_scores = d["scores"][:, 0, :]
match = int((codes_t == ref_seq).sum())
c = float(np.corrcoef(logits_t.ravel(), ref_scores.ravel())[0, 1])
print(f"torch-vs-ref (1 frame): {match}/15 tokens, logits corr {c:.8f}, "
      f"max|d| {np.abs(logits_t - ref_scores).max():.3e}")
assert match == 15, "torch folded reimplementation mismatch"

# ---- verify torch vs all 52 e2e frames ----
e = np.load("qwen3tts_work/ref/e2e_ref.npz")
codec_emb = None
from safetensors import safe_open  # noqa: E402
with safe_open(f"{SRC}/model.safetensors", framework="pt") as f:
    codec_emb = f.get_tensor(
        "talker.model.codec_embedding.weight").to(torch.float32).numpy()
ok_frames = 0
for i in range(e["codes"].shape[0]):
    cb0 = int(e["codes"][i, 0])
    ph = e["hiddens"][i].reshape(1, 1, 1024).astype(np.float32)
    ce = codec_emb[cb0].reshape(1, 1, 1024).astype(np.float32)
    cf, _ = torch_frame(ph, ce)
    if (cf == e["codes"][i, 1:]).all():
        ok_frames += 1
print(f"torch-vs-e2e: {ok_frames}/{e['codes'].shape[0]} frames all-15 exact")
assert ok_frames == e["codes"].shape[0]

# ---- convert ----
import litert_torch  # noqa: E402

sample = (torch.zeros(1, 1, 1024), torch.zeros(1, 1, 1024),
          torch.zeros(15, V))
ed = litert_torch.convert(m, sample)
path = f"{OUT}/mtp_folded.tflite"
ed.export(path)
print(f"converted -> {path} ({os.path.getsize(path) / 1e6:.1f} MB)")

# ---- verify tflite ----
from ai_edge_litert.interpreter import Interpreter  # noqa: E402

it = Interpreter(model_path=path, num_threads=4)
sig = it.get_signature_runner()
print("sig inputs :", {n: v["shape"].tolist()
                       for n, v in sig.get_input_details().items()})
print("sig outputs:", {n: v["shape"].tolist()
                       for n, v in sig.get_output_details().items()})


def tfl_frame(past_hidden, cb0_embed):
    out = sig(args_0=past_hidden.astype(np.float32),
              args_1=cb0_embed.astype(np.float32), args_2=zero_noise)
    return out["output_0"], out["output_1"]


codes_f, logits_f = tfl_frame(d["past_hidden"], d["last_id_hidden"])
match_f = int((codes_f == ref_seq).sum())
cf_ = float(np.corrcoef(logits_f.ravel(), ref_scores.ravel())[0, 1])
print(f"tflite-vs-ref (1 frame): {match_f}/15 tokens, logits corr {cf_:.8f}, "
      f"max|d| {np.abs(logits_f - ref_scores).max():.3e}")

ok_frames_f = 0
for i in range(e["codes"].shape[0]):
    cb0 = int(e["codes"][i, 0])
    ph = e["hiddens"][i].reshape(1, 1, 1024).astype(np.float32)
    ce = codec_emb[cb0].reshape(1, 1, 1024).astype(np.float32)
    cf2, _ = tfl_frame(ph, ce)
    if (cf2 == e["codes"][i, 1:]).all():
        ok_frames_f += 1
print(f"tflite-vs-e2e: {ok_frames_f}/{e['codes'].shape[0]} frames all-15 exact")

# ---- steady-state bench (compare vs 16-invoke step graph ~148 ms on M4 Max) ----
ph = d["past_hidden"].astype(np.float32)
ce = d["last_id_hidden"].astype(np.float32)
for _ in range(3):
    tfl_frame(ph, ce)
n = 20
t0 = time.perf_counter()
for _ in range(n):
    tfl_frame(ph, ce)
ms = (time.perf_counter() - t0) / n * 1000
print(f"folded MTP steady-state: {ms:.1f} ms/frame (4 threads)")
print("MTP_FOLDED_DONE")
