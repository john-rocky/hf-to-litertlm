"""Exportable re-implementation of Audio8-TTS-Preview-0.6b (ArkTTS DualAR) for LiteRT.

Weights come straight from model.safetensors / codec.pth (no transformers from_pretrained, so the
tf-5.x meta-load buffer trap cannot bite). Numerics follow the vendor code:
  * RoPE tables are the vendor's bf16-rounded cos/sin (interleaved pairs). We permute the q/k rows
    of wqkv to [even dims, odd dims] so the standard rotate-half form is exact (dot products are
    invariant to a shared permutation of head dims).
  * slow lm_head is sliced to the 4097 rows the sampler can ever pick (semantic 151678..155773, eos
    151645) -- same layout as the publisher's ONNX ("semantic_then_eos").
  * KV caches are graph inputs/outputs updated with dynamic_update_slice (positions contiguous).
"""
import importlib.util
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

try:
    from litert_torch.generative.custom_ops import dynamic_update_slice as _dus
    def dus(cache, update, pos):
        zero = torch.zeros([], dtype=torch.int32)
        return _dus.dynamic_update_slice(cache, update, [zero, zero, pos.reshape([]).to(torch.int32), zero])
except Exception:  # plain torch fallback (verification without litert_torch)
    def dus(cache, update, pos):
        p = int(pos.reshape(-1)[0])
        return cache.index_copy(2, torch.arange(p, p + update.shape[2]), update)


def load_main_weights():
    sd = load_file(os.path.join(C.SNAP, "model.safetensors"))
    return {k: v.to(torch.float32) for k, v in sd.items()}


def half_perm(head_dim):
    """new index j -> old index: [0,2,4,...,hd-2, 1,3,...,hd-1]."""
    return torch.cat([torch.arange(0, head_dim, 2), torch.arange(1, head_dim, 2)])


def rope_tables(length, head_dim, base):
    """cos/sin [length, hd/2] with the vendor's bf16 rounding (values feed fp32 math)."""
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2).float()[: head_dim // 2] / head_dim))
    phases = torch.outer(torch.arange(length).float(), freqs)
    values = torch.polar(torch.ones_like(phases), phases)
    cs = torch.stack((values.real, values.imag), dim=-1).to(torch.bfloat16).float()
    return cs[..., 0].contiguous(), cs[..., 1].contiguous()


def rms(x, w, eps):
    v = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(v + eps) * w


def rot_half(x):
    hd = x.shape[-1]
    a, b = x[..., : hd // 2], x[..., hd // 2:]
    return torch.cat((-b, a), dim=-1)


class ArBlock(nn.Module):
    """One transformer block (attention with external KV cache + gated SiLU FFN)."""

    def __init__(self, w, prefix, heads, kv_heads, head_dim, bias, eps):
        super().__init__()
        self.h, self.kvh, self.hd, self.eps = heads, kv_heads, head_dim, eps
        q_size, kv_size = heads * head_dim, kv_heads * head_dim
        wqkv = w[f"{prefix}.attention.wqkv.weight"]
        perm = half_perm(head_dim)
        rows = torch.cat([
            torch.arange(0, q_size).view(heads, head_dim)[:, perm].reshape(-1),
            q_size + torch.arange(0, kv_size).view(kv_heads, head_dim)[:, perm].reshape(-1),
            torch.arange(q_size + kv_size, q_size + 2 * kv_size),
        ])
        self.register_buffer("wqkv", wqkv[rows].contiguous(), persistent=False)
        if bias:
            self.register_buffer("bqkv", w[f"{prefix}.attention.wqkv.bias"][rows].contiguous(), persistent=False)
        else:
            self.bqkv = None
        for n in ("attention.wo.weight", "feed_forward.w1.weight", "feed_forward.w2.weight",
                  "feed_forward.w3.weight", "attention_norm.weight", "ffn_norm.weight"):
            self.register_buffer(n.replace(".", "_"), w[f"{prefix}.{n}"].contiguous(), persistent=False)

    def forward(self, x, cos, sin, mask, pos0, k_cache, v_cache):
        B, T, _ = x.shape
        h = rms(x, self.attention_norm_weight, self.eps)
        qkv = F.linear(h, self.wqkv, self.bqkv)
        q, k, v = qkv.split((self.h * self.hd, self.kvh * self.hd, self.kvh * self.hd), dim=-1)
        q = q.view(B, T, self.h, self.hd).transpose(1, 2)      # [B,H,T,hd]
        k = k.view(B, T, self.kvh, self.hd).transpose(1, 2)    # [B,KV,T,hd]
        v = v.view(B, T, self.kvh, self.hd).transpose(1, 2)
        q = q * cos + rot_half(q) * sin
        k = k * cos + rot_half(k) * sin
        k_cache = dus(k_cache, k, pos0)
        v_cache = dus(v_cache, v, pos0)
        # GQA without repeat/broadcast: fold the `rep` query heads that share a kv head into the row dim
        # (same head mapping as repeat_interleave(rep, dim=1): kv head j serves q heads j*rep .. j*rep+rep-1).
        rep = self.h // self.kvh
        L = k_cache.shape[2]
        qg = q.reshape(B, self.kvh, rep * T, self.hd)
        att = torch.matmul(qg, k_cache.transpose(2, 3)) * (self.hd ** -0.5)      # [B,KV,rep*T,L]
        att = (att.reshape(B, self.kvh, rep, T, L) + mask.unsqueeze(1)).reshape(B, self.kvh, rep * T, L)
        att = att.softmax(dim=-1)
        o = torch.matmul(att, v_cache).reshape(B, self.h, T, self.hd).transpose(1, 2).reshape(B, T, self.h * self.hd)
        x = x + F.linear(o, self.attention_wo_weight)
        h2 = rms(x, self.ffn_norm_weight, self.eps)
        ff = F.linear(F.silu(F.linear(h2, self.feed_forward_w1_weight)) * F.linear(h2, self.feed_forward_w3_weight),
                      self.feed_forward_w2_weight)
        return x + ff, k_cache, v_cache


class SlowAR(nn.Module):
    """codes [1,11,T] int32, input_pos [T] int32, mask [1,1,T,CACHE] fp32, 24x(k,v) [1,2,CACHE,64]
    -> logits [1,4097] (last position; semantic range then eos), hidden [1,1,896] (normed), 24x(k,v)."""

    def __init__(self, w, cache_len=C.MAX_SEQ):
        super().__init__()
        self.cache_len = cache_len
        self.register_buffer("emb", w["embeddings.weight"].contiguous(), persistent=False)
        self.register_buffer("cb_emb", w["codebook_embeddings.weight"].contiguous(), persistent=False)
        head = torch.cat([w["embeddings.weight"][C.SEM_BEGIN:C.SEM_END + 1], w["embeddings.weight"][C.EOS:C.EOS + 1]], 0)
        self.register_buffer("lm_head", head.contiguous(), persistent=False)
        self.register_buffer("norm_weight", w["norm.weight"].contiguous(), persistent=False)
        self.register_buffer("cb_offsets", (torch.arange(C.NUM_CB) * C.CB_SIZE).to(torch.int32).view(1, C.NUM_CB, 1), persistent=False)
        cos, sin = rope_tables(C.MAX_SEQ, C.HEAD_DIM, C.ROPE_BASE)
        self.register_buffer("cos_tab", torch.cat([cos, cos], -1), persistent=False)  # [2048,64]
        self.register_buffer("sin_tab", torch.cat([sin, sin], -1), persistent=False)
        self.blocks = nn.ModuleList([
            ArBlock(w, f"layers.{i}", C.HEADS, C.KV_HEADS, C.HEAD_DIM, bias=True, eps=C.EPS) for i in range(C.N_LAYER)
        ])

    def embed(self, codes):
        # clamp every index so an out-of-range id can never fault a gather (benchmark tools feed random ints)
        row0 = codes[:, 0].clamp(0, C.VOCAB - 1)                            # [1,T]
        text = F.embedding(row0, self.emb)                                  # [1,T,896]
        idx = codes[:, 1:].clamp(0, C.CB_SIZE - 1) + self.cb_offsets        # [1,10,T]
        cb = F.embedding(idx, self.cb_emb).sum(dim=1)                       # [1,T,896]
        is_sem = (row0 >= C.SEM_BEGIN) & (row0 <= C.SEM_END)
        cb = torch.where(is_sem.unsqueeze(-1), cb, torch.zeros_like(cb))
        return text + cb

    def forward(self, codes, input_pos, mask, *kv):
        x = self.embed(codes)
        input_pos = input_pos.clamp(0, self.cache_len - 1)
        cos = self.cos_tab[input_pos].unsqueeze(0).unsqueeze(0)             # [1,1,T,64]
        sin = self.sin_tab[input_pos].unsqueeze(0).unsqueeze(0)
        pos0 = input_pos[0]
        new_kv = []
        for i, blk in enumerate(self.blocks):
            x, k, v = blk(x, cos, sin, mask, pos0, kv[2 * i], kv[2 * i + 1])
            new_kv += [k, v]
        last = x[:, -1:]
        hidden = rms(last, self.norm_weight, C.EPS)                         # [1,1,896]
        logits = F.linear(hidden, self.lm_head)[:, 0]                       # [1,4097]
        return (logits, hidden, *new_kv)


class FastAR(nn.Module):
    """hidden [1,1,896], token [1] int32, use_hidden [1] fp32, pos [1] int32, mask [1,1,1,10],
    k_all/v_all [4,1,2,10,64] -> logits [1,4096], k_all, v_all."""

    def __init__(self, w):
        super().__init__()
        self.register_buffer("fast_emb", w["fast_embeddings.weight"].contiguous(), persistent=False)
        self.register_buffer("fast_out", w["fast_output.weight"].contiguous(), persistent=False)
        self.register_buffer("fast_norm_weight", w["fast_norm.weight"].contiguous(), persistent=False)
        cos, sin = rope_tables(C.NUM_CB, C.HEAD_DIM, C.ROPE_BASE)
        self.register_buffer("cos_tab", torch.cat([cos, cos], -1), persistent=False)  # [10,64]
        self.register_buffer("sin_tab", torch.cat([sin, sin], -1), persistent=False)
        self.blocks = nn.ModuleList([
            ArBlock(w, f"fast_layers.{i}", C.HEADS, C.KV_HEADS, C.HEAD_DIM, bias=False, eps=C.EPS) for i in range(C.N_FAST)
        ])

    def forward(self, hidden, token, use_hidden, pos, mask, k_all, v_all):
        token = token.clamp(0, C.CB_SIZE - 1)
        pos = pos.clamp(0, C.NUM_CB - 1)
        emb = F.embedding(token, self.fast_emb).unsqueeze(0)                # [1,1,896]
        u = use_hidden.reshape(1, 1, 1)
        x = hidden * u + emb * (1.0 - u)
        cos = self.cos_tab[pos].unsqueeze(0).unsqueeze(0)
        sin = self.sin_tab[pos].unsqueeze(0).unsqueeze(0)
        pos0 = pos[0]
        ks, vs = [], []
        for i, blk in enumerate(self.blocks):
            x, k, v = blk(x, cos, sin, mask, pos0, k_all[i], v_all[i])
            ks.append(k); vs.append(v)
        h = rms(x, self.fast_norm_weight, C.EPS)
        logits = F.linear(h, self.fast_out)[:, 0]                           # [1,4096]
        return logits, torch.stack(ks), torch.stack(vs)


# ---------------- codec (vendor module, patched for export) ----------------
def load_codec_module():
    spec = importlib.util.spec_from_file_location("arktts_codec_vendor", os.path.join(C.SNAP, "modeling_arktts_codec.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses need the module registered (PEP 563 annotations)
    spec.loader.exec_module(mod)

    def _snake(x, alpha):  # the vendor's @torch.jit.script function, as plain python
        shape = x.shape
        x = x.reshape(shape[0], shape[1], -1)
        x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
        return x.reshape(shape)

    _rope_cache = {}

    def _rope(length, head_dim, base, device=None):
        # Real-valued twin of the vendor's torch.polar table (bf16-rounded, fed to fp32 math). The table is
        # computed eagerly once per static length and served from a cache, so torch.export never traces
        # complex ops: run one eager forward at the export length before converting.
        key = (int(length), int(head_dim), float(base))
        if key not in _rope_cache:
            frequencies = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
            phases = torch.outer(torch.arange(length).float(), frequencies)
            values = torch.polar(torch.ones_like(phases), phases)
            _rope_cache[key] = torch.stack((values.real, values.imag), dim=-1).to(torch.bfloat16).float()
        return _rope_cache[key]

    mod._arktts_snake = _snake
    mod._rope = _rope
    mod._rope_cache = _rope_cache
    return mod


class _Cfg:  # the codec reads 4 post-transformer fields from the model config
    codec_post_n_layer = 8
    codec_post_n_head = 16
    codec_post_n_local_heads = 8
    codec_post_intermediate_size = 1216


def load_codec():
    mod = load_codec_module()
    codec = mod.ArkttsCodec(_Cfg())
    state = torch.load(os.path.join(C.SNAP, "codec.pth"), map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    state = {k: v for k, v in state.items() if not k.endswith(("freqs_cis", "causal_mask"))}
    codec.load_state_dict(state, strict=True)
    codec.eval()
    # fold weight norm (both the parametrizations API and the legacy weight_g/weight_v form)
    from torch.nn.utils import parametrize, remove_weight_norm
    for m in list(codec.modules()):
        if hasattr(m, "parametrizations") and "weight" in m.parametrizations:
            parametrize.remove_parametrizations(m, "weight", leave_parametrized=True)
        elif hasattr(m, "weight_g"):
            remove_weight_norm(m)
    if os.environ.get("CODEC_GPU_ATTN", "1") == "1":
        _gpu_clean_codec_attention(codec, mod)
    return codec, mod


def _gpu_clean_codec_attention(codec, mod):
    """Rewrite the codec transformers' attention for the mobile GPU delegate (ML Drift rejects BROADCAST_TO
    and any tensor with rank >= 5): interleaved RoPE -> rotate-half on permuted q/k rows (exact), GQA
    repeat_interleave -> query heads folded into the matmul rows, boolean sdpa mask -> additive constant.
    Numerics: same math, fp32 reassociation only (measured max|d| vs the vendor path ~1e-6 on wav)."""
    import types
    _mask_cache = {}
    mod._attn_mask_cache = _mask_cache

    def forward(self, x, rope_values, mask):
        B, L, _ = x.shape
        H, KV, hd = self.n_head, self.n_local_heads, self.head_dim
        q, k, v = self.wqkv(x).split((H * hd, KV * hd, KV * hd), dim=-1)
        q = q.view(B, L, H, hd).transpose(1, 2)
        k = k.view(B, L, KV, hd).transpose(1, 2)
        v = v.view(B, L, KV, hd).transpose(1, 2)
        c = rope_values[..., 0]; s_ = rope_values[..., 1]                       # [L, hd/2] (bf16-rounded consts)
        cos = torch.cat([c, c], -1).view(1, 1, L, hd); sin = torch.cat([s_, s_], -1).view(1, 1, L, hd)
        q = q * cos + rot_half(q) * sin
        k = k * cos + rot_half(k) * sin
        rep = H // KV
        qg = q.reshape(B * KV, rep * L, hd)                                     # head h = kv*rep + r
        kg = k.reshape(B * KV, L, hd); vg = v.reshape(B * KV, L, hd)
        att = torch.matmul(qg, kg.transpose(1, 2)) * (hd ** -0.5)              # [B*KV, rep*L, L]
        # additive mask tiled over the folded heads, built eagerly once per (L, rep, window) and served as a
        # constant (a traced repeat() lowers to BROADCAST_TO, which the GPU delegate rejects)
        key = (int(L), int(rep), self._window)   # python ints only: a traced mask.sum() is data-dependent for export
        if key not in _mask_cache:
            _mask_cache[key] = torch.where(mask[0, 0], 0.0, -1e9).float().repeat(rep, 1).unsqueeze(0).contiguous()
        att = (att + _mask_cache[key]).softmax(dim=-1)
        o = torch.matmul(att, vg).reshape(B, KV * rep, L, hd).transpose(1, 2).reshape(B, L, H * hd)
        return self.wo(o)

    for wt in codec.modules():
        if isinstance(wt, mod.ArkttsCodecWindowTransformer):
            for layer in wt.layers:
                layer.attention._window = wt.window_size if wt.window_size is not None else -1
    perm = half_perm(64)
    for m in codec.modules():
        if isinstance(m, mod.ArkttsCodecAttention):
            H, KV, hd = m.n_head, m.n_local_heads, m.head_dim
            assert hd == 64
            q_size, kv_size = H * hd, KV * hd
            rows = torch.cat([
                torch.arange(0, q_size).view(H, hd)[:, perm].reshape(-1),
                q_size + torch.arange(0, kv_size).view(KV, hd)[:, perm].reshape(-1),
                torch.arange(q_size + kv_size, q_size + 2 * kv_size),
            ])
            with torch.no_grad():
                m.wqkv.weight.copy_(m.wqkv.weight[rows].clone())
            m.forward = types.MethodType(forward, m)


class CodecDecoder(nn.Module):
    """codes [1,10,T] int32 -> wav [1,1,T*2048] fp32 (vendor decode path, weight norm folded)."""

    def __init__(self, codec):
        super().__init__()
        self.q = codec.quantizer
        self.dec = codec.decoder

    def forward(self, codes):
        idx = codes.clone()
        idx0 = idx[:, :1].clamp(0, self.q.semantic_quantizer.codebook_size - 1)
        idx1 = idx[:, 1:].clamp(0, self.q.quantizer.codebook_size - 1)
        semantic = self.q.semantic_quantizer.from_codes(idx0)
        residual = self.q.quantizer.from_codes(idx1)
        z = self.q.upsample(self.q.post_module(semantic + residual))
        return self.dec(z)


class CodecEncoder(nn.Module):
    """audio [1,1,N] fp32 (N multiple of 2048) -> codes [1,10,N/2048] int32."""

    def __init__(self, codec):
        super().__init__()
        self.enc = codec.encoder
        self.q = codec.quantizer

    def forward(self, audio):
        z = self.enc(audio)
        z = self.q.pre_module(self.q.downsample(z))
        semantic, semantic_codes = self.q.semantic_quantizer(z)
        residual, residual_codes = self.q.quantizer(z - semantic)
        return torch.cat((semantic_codes, residual_codes), dim=1).to(torch.int32)
