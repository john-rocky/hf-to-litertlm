"""A reference forward of NVIDIA-Nemotron-Labs-Dory1.0-4B in plain torch, read straight from the safetensors.

The arithmetic follows vLLM PR #60149 @55076950 (`vllm/model_executor/models/dory.py`) line by line; no transformers
model class is used. Three number modes:
  fp64  every tensor float64 (weights are exact bf16 values)
  fp32  every tensor float32
  bf16  activations bf16 at vLLM's cast positions: a linear layer = bf16 input x float32 weight with float32
        accumulation, output rounded to bf16 (a bf16 GEMM with fp32 accumulate); the `.float()` / `.to(x.dtype)` casts
        of dory.py are kept where they are; RoPE cos/sin are bf16 (vLLM casts its cos/sin cache to the model dtype);
        attention: fp32 scores and softmax, P rounded to bf16 before P@V (FlashAttention-2), bf16 output; KV cache bf16.
Loops 1..4 and the two recurrent KV-cache modes:
  per_loop   one cache per (loop, recurrent attention layer)
  last_loop  one cache per recurrent attention layer; every loop writes this chunk's KV at the chunk's positions, so
             loop i sees loop-i KV for this chunk's tokens and the last loop's KV for earlier chunks (vLLM dory.py:151-153)
Sliding window: key j is visible to the query at position p iff 0 <= p - j <= W - 1 (W keys including the query's own),
the definition of vLLM flash_attn.py:1082 `(sliding_window - 1, 0)` and llama.cpp src/llama-hparams.h:507.
Device: cpu, or mps for fp32 / bf16 (no float64 on MPS); logits always come back on the CPU.
"""
import json
import math
import os

import torch
from safetensors import safe_open

EPS = 1e-6


class Num:
    """The dtype policy of one run."""

    def __init__(self, mode):
        assert mode in ("fp64", "fp32", "bf16"), mode
        self.mode = mode
        self.acc = torch.float64 if mode == "fp64" else torch.float32   # vLLM's `.float()`
        self.act = torch.bfloat16 if mode == "bf16" else self.acc       # activation / KV storage dtype

    def lin(self, x, w):
        return (x.to(self.acc) @ w.T).to(self.act)


class Stats:
    """Per-site maxima of value ranges; only filled when enabled."""

    def __init__(self):
        self.d = {}

    def put(self, site, value, where):
        v = float(value)
        cur = self.d.get(site)
        if cur is None or v > cur[0]:
            self.d[site] = (v, where)

    def put_min(self, site, value, where):
        v = float(value)
        cur = self.d.get(site)
        if cur is None or v < cur[0]:
            self.d[site] = (v, where)


class DoryRef:
    def __init__(self, model_dir, mode="fp32", loops=None, kv_mode="per_loop", threads=12, max_len=4096, device="cpu"):
        torch.set_num_threads(threads)
        self.num = Num(mode)
        self.dev = torch.device(device)
        assert not (mode == "fp64" and self.dev.type == "mps"), "MPS has no float64"
        with open(os.path.join(model_dir, "config.json")) as f:
            c = json.load(f)
        self.c = c
        self.H = c["hidden_size"]
        self.nh, self.nkv, self.hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        self.ffn = c["intermediate_size"]
        self.std = c["initializer_range"]
        self.n_in, self.n_rec, self.n_out = c["n_input_layers"], c["n_recurrent_layers"], c["n_output_layers"]
        self.loops = c["n_recurrent_loops"] if loops is None else loops
        assert kv_mode in ("per_loop", "last_loop")
        self.kv_mode = kv_mode
        self.types = c["layer_types"]
        self.window = c["sliding_window"]
        self.max_len = max_len
        # vLLM's Python-float factors (dory.py:94-98, 186, 207, 327)
        self.sqk_factor = self.hd ** 0.5 / self.std
        g0 = c["attention_output_gate_logit_scale_init"]
        if g0 is None:
            g0 = self.H ** 0.5
        self.s_gate_factor = g0 ** 0.5 / self.std
        self.suv_factor = self.H ** 0.5
        self.alpha_scale = c["ngpt_alpha_init"] / self.std
        self.stats = None
        # fp16-storage hooks (default off = the arithmetic above is unchanged): f16 = site names whose values are stored in fp16
        # the way a GPU fp16 path stores op outputs (round to nearest, overflow to inf); f16_r1 = {"L<i>": k} shifts
        # the input of block i's MLP justnorm by 2^-k (the rewrite export_dory.py applies); jn_range / f16_counts collect per-layer justnorm
        # ranges and fp16 overflow rows when they are dicts.
        self.f16 = frozenset()
        self.f16_r1 = {}
        self.jn_range = None
        self.f16_counts = None
        self._load(model_dir)
        self._rope_tables()
        self.reset()

    # ---------------------------------------------------------------- weights
    def _load(self, model_dir):
        acc = self.num.acc
        L = []
        with safe_open(os.path.join(model_dir, "model.safetensors"), framework="pt") as f:
            g = lambda n: f.get_tensor(n).to(acc).to(self.dev)
            self.emb = f.get_tensor("backbone.embeddings.weight")          # bf16; rows are cast at lookup
            self.lm_head = g("lm_head.weight")
            self.logit_scale = g("lm_head.logit_scale")
            for i, t in enumerate(self.types):
                p = f"backbone.layers.{i}."
                if t == "mlp":
                    L.append(dict(
                        kind="mlp",
                        w_gu=torch.cat([g(p + "mixer.gate_proj.weight"), g(p + "mixer.up_proj.weight")], 0).contiguous(),
                        w_down=g(p + "mixer.down_proj.weight"),
                        suv_gate=g(p + "mixer.suv_gate"), suv_up=g(p + "mixer.suv_up"),
                        alpha=g(p + "mlp_alpha")))
                else:
                    L.append(dict(
                        kind=t,
                        w_qkvg=torch.cat([g(p + "mixer.q_proj.weight"), g(p + "mixer.k_proj.weight"),
                                          g(p + "mixer.v_proj.weight"), g(p + "mixer.g_proj.weight")], 0).contiguous(),
                        w_o=g(p + "mixer.o_proj.weight"),
                        sqk=g(p + "mixer.sqk"), s_gate=g(p + "mixer.s_gate"),
                        alpha=g(p + "attn_alpha")))
        self.layers = L

    def _rope_tables(self):
        """vLLM RotaryEmbedding: inv_freq and t in float32, freqs = t * inv_freq, cos/sin in float32, then cast to the
        model dtype (bf16 in vLLM's bf16 model). fp64 mode computes everything in float64."""
        self.rope_tab = {}
        rp = self.c["rope_parameters"]
        dt = torch.float64 if self.num.mode == "fp64" else torch.float32
        for kind in ("sliding_attention", "full_attention"):
            base = rp[kind]["rope_theta"]
            rot = int(self.hd * rp[kind].get("partial_rotary_factor", 1.0))
            assert rot == self.hd
            inv_freq = 1.0 / (base ** (torch.arange(0, rot, 2, dtype=dt) / rot))
            t = torch.arange(self.max_len, dtype=dt)
            freqs = t[:, None] * inv_freq[None, :]
            cos, sin = freqs.cos(), freqs.sin()
            self.rope_tab[kind] = (cos.to(self.num.act).to(self.dev), sin.to(self.num.act).to(self.dev))

    # ---------------------------------------------------------------- cache
    def reset(self):
        """Empty every KV cache. Slot key: physical layer index, plus the loop index for per_loop recurrent layers."""
        self.cache = {}

    def _slot(self, i, loop):
        rec = self.n_in <= i < self.n_in + self.n_rec
        return (i, loop) if (rec and self.kv_mode == "per_loop") else (i, 0)

    def _kv(self, slot):
        if slot not in self.cache:
            z = torch.zeros(self.max_len, self.nkv, self.hd, dtype=self.num.act, device=self.dev)
            self.cache[slot] = (z, z.clone())
        return self.cache[slot]

    # ---------------------------------------------------------------- math
    def justnorm(self, x, site=None):
        """x * rsqrt(sum(x^2) + eps) with accumulation in the run's float dtype; returns x's dtype (dory.py:37-41)."""
        xf = x.to(self.num.acc)
        if site is not None and ("jn:" + site[0]) in self.f16:
            return self._justnorm16(xf, site).to(x.dtype)
        ss = xf.pow(2).sum(dim=-1, keepdim=True)
        if self.stats is not None and site is not None:
            self.stats.put("sumsq:" + site[0], ss.max(), site[1])
            self.stats.put("absmax:" + site[0], xf.abs().max(), site[1])
        if self.jn_range is not None and site is not None:
            key = (site[0], site[1].split(".")[0])
            cur = self.jn_range.get(key)
            new = (float(ss.max()), float(ss.min()), float(xf.abs().max()), int(ss.numel()))
            self.jn_range[key] = new if cur is None else (max(cur[0], new[0]), min(cur[1], new[1]),
                                                          max(cur[2], new[2]), cur[3] + new[3])
        y = xf * torch.rsqrt(ss + EPS)
        return y.to(x.dtype)

    def _h(self, x, site):
        """fp16-storage hook: x rounded to fp16 storage when `site` is in self.f16 (overflow -> inf), else x itself."""
        return x.to(torch.float16).to(x.dtype) if site in self.f16 else x

    def _lin(self, x, w):
        """num.lin with the fp16-storage hook: input and output stored in fp16 when "lin" is in self.f16."""
        if "lin" not in self.f16:
            return self.num.lin(x, w)
        return self._h(self.num.lin(self._h(x, "lin"), w), "lin")

    def _justnorm16(self, xf, site):
        """justnorm with every op output stored in fp16 (MUL x*x, SUM, ADD eps, RSQRT, MUL = the exported graph's
        ops), the sum accumulated in fp32. With a shift k for this layer in f16_r1 (MLP site only), the input is first
        multiplied by 2^-k, eps by 4^-k, and rsqrt is capped at 2^k / sqrt(eps) (the rewrite export_dory.py applies; exact in fp32)."""
        h = lambda t: t.to(torch.float16).to(xf.dtype)
        k = self.f16_r1.get(site[1].split(".")[0], 0) if site[0] == "mlp.m" else 0
        x16 = h(xf * (2.0 ** -k)) if k else h(xf)
        ss = h(h(x16 * x16).sum(dim=-1, keepdim=True))
        r = h(torch.rsqrt(h(ss + EPS * (4.0 ** -k))))
        if k:
            r = torch.clamp(r, max=2.0 ** k / math.sqrt(EPS))
        if self.f16_counts is not None:
            c = self.f16_counts.setdefault(site[0], [0, 0, 0])
            c[0] += int(torch.isinf(ss).sum())
            c[1] += int(ss.numel())
            c[2] += int((r == 0).sum())
        return h(x16 * r)

    def slerp_alpha(self, a, b, alpha, where):
        acc = self.num.acc
        lr = (alpha.to(acc) * self.alpha_scale).abs()
        ya = self.justnorm(a, ("slerp.a", where)).to(acc)
        yb = self.justnorm(b, ("slerp.b", where)).to(acc)
        if "slerp" in self.f16:
            mix = self._h(ya + self._h(lr * self._h(yb - ya, "slerp"), "slerp"), "slerp")
            return self.justnorm(mix, ("slerp.mix", where)).to(a.dtype)
        return self.justnorm(ya + lr * (yb - ya), ("slerp.mix", where)).to(a.dtype)

    def rope(self, x, pos, kind):
        """neox-style rotation (vLLM _apply_rotary_emb_torch, is_neox_style=True); x: (T, heads, hd)."""
        cos, sin = self.rope_tab[kind]
        cos = cos[pos].unsqueeze(-2).to(x.dtype)
        sin = sin[pos].unsqueeze(-2).to(x.dtype)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)

    def attention(self, i, l, h, start, loop):
        num, acc = self.num, self.num.acc
        T = h.shape[0]
        where = f"L{i}" + (f".loop{loop}" if self.n_in <= i < self.n_in + self.n_rec else "")
        qkvg = self._lin(h, l["w_qkvg"])
        qs, ks = self.nh * self.hd, self.nkv * self.hd
        q, k, v, g = qkvg.split([qs, ks, ks, qs], dim=-1)
        pos = torch.arange(start, start + T, device=self.dev)
        q = self.rope(q.reshape(T, self.nh, self.hd), pos, l["kind"])
        k = self.rope(k.reshape(T, self.nkv, self.hd), pos, l["kind"])
        sqk = (l["sqk"].to(acc) * self.sqk_factor).view(1, self.nh, self.hd)
        q = self._h((sqk.square() * self.justnorm(q, ("attn.q", where)).to(acc)).to(num.act), "attn.qk")
        k = self._h(self.justnorm(k, ("attn.k", where)), "attn.qk")
        v = v.reshape(T, self.nkv, self.hd)
        K, V = self._kv(self._slot(i, loop))
        K[start:start + T] = self._h(k, "kv")
        V[start:start + T] = self._h(v, "kv")
        Lk = start + T
        rep = self.nh // self.nkv
        qf = q.to(acc).transpose(0, 1)                                         # (nh, T, hd)
        kf = K[:Lk].to(acc).transpose(0, 1).repeat_interleave(rep, dim=0)       # (nh, Lk, hd)
        vf = V[:Lk].to(acc).transpose(0, 1).repeat_interleave(rep, dim=0)
        scores = self._h((qf @ kf.transpose(-1, -2)) * (1.0 / math.sqrt(self.hd)), "attn.logit")
        qpos = pos[:, None]
        kpos = torch.arange(Lk, device=self.dev)[None, :]
        mask = kpos <= qpos
        if l["kind"] == "sliding_attention":
            mask &= (qpos - kpos) <= self.window - 1
        scores = scores.masked_fill(~mask, float("-inf"))
        if self.stats is not None:
            self.stats.put("absmax:attn.q_scaled", q.to(acc).abs().max(), where)
            self.stats.put("max:attn.logit", scores.max(), where)
        p = self._h(torch.softmax(scores, dim=-1), "attn.p")
        if num.mode == "bf16":
            p = p.to(torch.bfloat16).to(acc)
        out = self._h((p @ vf).transpose(0, 1).reshape(T, qs).to(num.act), "attn.pv")
        sg = (l["s_gate"].to(acc) * self.s_gate_factor).square()
        gpre = self._h(g.to(acc) * sg, "gate")
        if self.stats is not None:
            self.stats.put("max:gate.pre", gpre.max(), where)
            self.stats.put_min("min:gate.pre", gpre.min(), where)
            self.stats.put("absmax:qkvg", qkvg.to(acc).abs().max(), where)
        out = self._h(out.to(acc) * self._h(torch.sigmoid(gpre), "gate"), "gate").to(num.act)
        out = self.justnorm(out, ("attn.out", where))
        out = self._lin(out, l["w_o"])
        if self.stats is not None:
            self.stats.put("absmax:o_proj", out.to(acc).abs().max(), where)
        return self.slerp_alpha(h, out, l["alpha"], where)

    def mlp(self, i, l, h, loop):
        num, acc = self.num, self.num.acc
        where = f"L{i}" + (f".loop{loop}" if self.n_in <= i < self.n_in + self.n_rec else "")
        m = self._lin(h, l["w_gu"])
        gt, up = m.chunk(2, dim=-1)
        gt = (gt.to(acc) * (l["suv_gate"].to(acc) * self.suv_factor)).to(m.dtype)
        up = (up.to(acc) * (l["suv_up"].to(acc) * self.suv_factor)).to(m.dtype)
        if self.stats is not None:
            self.stats.put("absmax:mlp.gate_scaled", gt.to(acc).abs().max(), where)
            self.stats.put("absmax:mlp.up_scaled", up.to(acc).abs().max(), where)
        m = self._h((self._h(torch.nn.functional.silu(self._h(gt.to(acc), "mlp.act")), "mlp.act") * self._h(up.to(acc), "mlp.act")).to(m.dtype), "mlp.act")
        m = self.justnorm(m, ("mlp.m", where))
        out = self._lin(m, l["w_down"])
        if self.stats is not None:
            self.stats.put("absmax:mlp.down", out.to(acc).abs().max(), where)
        return self.slerp_alpha(h, out, l["alpha"], where)

    def layer(self, i, h, start, loop=0):
        l = self.layers[i]
        if l["kind"] == "mlp":
            return self.mlp(i, l, h, loop)
        return self.attention(i, l, h, start, loop)

    @torch.inference_mode()
    def forward(self, ids, start, all_logits=False, rows=None):
        """ids: list/1-D tensor of T token ids at positions start..start+T-1 (one chunk). Returns float logits
        (T, vocab) if all_logits else (vocab,) for the chunk's last position.
        rows (default off): chunk-relative positions whose logits are returned, (len(rows), vocab),
        left on the run's device; lm_head runs on those rows only. The arithmetic of every row is unchanged."""
        num, acc = self.num, self.num.acc
        ids = torch.as_tensor(ids, dtype=torch.long)
        assert start + ids.numel() <= self.max_len
        h = self._h(self.emb[ids].to(num.act).to(self.dev), "emb")
        if self.stats is not None:
            self.stats.put("absmax:embedding", h.to(acc).abs().max(), "emb")
        for i in range(self.n_in):
            h = self.layer(i, h, start)
        input_h = h
        state = input_h
        for lp in range(self.loops):
            h = self.justnorm(self._h(state + input_h, "loop"), ("loop.entry", f"loop{lp}"))
            for i in range(self.n_in, self.n_in + self.n_rec):
                h = self.layer(i, h, start, lp)
            state = self.justnorm(h, ("loop.exit", f"loop{lp}"))
        h = state
        for i in range(self.n_in + self.n_rec, len(self.layers)):
            h = self.layer(i, h, start)
        if rows is not None:
            h = h[torch.as_tensor(rows, dtype=torch.long, device=h.device)]
        elif not all_logits:
            h = h[-1:]
        logits = self._h(self._lin(h, self.lm_head).to(acc) * (self.logit_scale / self.std), "logits")
        if self.stats is not None:
            self.stats.put("absmax:logits", logits.abs().max(), "lm_head")
            self.stats.put("max:logits", logits.max(), "lm_head")
        if rows is not None:
            return logits
        logits = logits.cpu()
        return logits if all_logits else logits[0]
