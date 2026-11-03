"""The static 1-page DASH decoder of uocr_dash.py with a pinned + ring KV cache.

One-page decode reads only the positions 0..287 (the 9 image blocks, with their DASH bias, and the local window while
p < 416) and the last 128 tokens (the local window of 4 blocks of 32). So the K / V caches hold
SLOTS = 288 + 128 = 416 slots:
  slot(p) = p                          for p < 288   (pinned: the prompt 0..278 and the first reply tokens)
  slot(p) = 288 + (p - 288) mod 128    for p >= 288  (ring: always the last 128 positions >= 288)
computed in the graph from input_pos[0] (float p - 128 * floor(relu(p - 288) / 128), exact below 2^24, cast to int32 =
the DYNAMIC_UPDATE_SLICE index; a small model of this form ran to position 100 on 24 slots on the CPU and WebGPU).
Everything a position decides stays a function of the absolute p from input_pos: the f(p) page2d cos / sin table keeps
CTX = 4,096 rows (the one-hot FC of RopeTable), the mask input keeps [1, 1, T, 4096] (the runtime's context: LiteRT-LM
v0.19.0 fills [0, step] of each row and, with LOCAL-typed caches, takes no context limit from the KV: state.cc
MetadataBasedCreate counts only the GLOBAL caches).
Attention mask over the 416 slots, against the C 4,096 graph of uocr_dash.py:
  slots 0..287: the same terms as before on positions 0..287 (the runtime's causal gate mask > -1, the local window
                (p - 128, p], the image-block bias of the candidate blocks, NEG otherwise);
  ring slot j:  visible (0) iff j + 288 <= p, else NEG. The visible ring slots then hold exactly the positions
                max(288, p - 127) .. p = the C graph's visible positions >= 288 (never a candidate block, all local).
So each query sees the same keys with the same mask values as in the C graph, in another order. Prefill writes stay in
the pinned slots (the fixed prompt = one prefill_280 call at step 0, rows 0..279 = slots 0..279); a prefill that
starts at p0 >= 288 or crosses 288 is outside this contract (as a second turn is outside the fixed layout already).
The block summaries read the router keys at slots 0..287 = positions 0..287 (unchanged).
"""
import importlib.util
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
  if name in sys.modules:
    return sys.modules[name]
  spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
  mod = importlib.util.module_from_spec(spec)
  sys.modules[name] = mod            # uocr_dash does `import uocr_decoder as ud`
  spec.loader.exec_module(mod)
  return mod


ud = _load("uocr_decoder")
UD = _load("uocr_dash")

PIN = UD.IMG                         # 288
RING = UD.WL                         # 128
SLOTS = PIN + RING                   # 416
NEG = UD.NEG


def ring_slot_f(input_pos):
  """[1] float32: slot of input_pos[0]."""
  p0 = input_pos[:1].to(torch.float32)
  return p0 - float(RING) * torch.floor(F.relu(p0 - float(PIN)) * (1.0 / RING))


class RingShared(UD.DashShared):
  """DashShared with the 416-slot mask terms and the slot index; cos / sin and the candidates as before."""

  def __init__(self, cfg):
    super().__init__(cfg)            # the RoPE table [cfg.cache = CTX, 256], jcol [1, CTX], thr
    self.register_buffer("jimg", torch.arange(PIN, dtype=torch.float32).reshape(1, 1, 1, PIN), persistent=False)
    self.register_buffer("jring", torch.arange(RING, dtype=torch.float32).reshape(1, 1, 1, RING) + float(PIN),
                         persistent=False)

  def forward(self, input_pos, mask):
    T = input_pos.shape[0]
    posf = input_pos.to(torch.float32).reshape(1, 1, T, 1)
    onehot = (self.jcol == input_pos.to(torch.float32).reshape(T, 1)).to(torch.float32)   # [T, CTX]
    cs = self.table(onehot).reshape(1, 1, T, 2 * UD.DH)
    cos, sin = cs[..., :UD.DH], cs[..., UD.DH:]
    causal = (self.jimg <= posf).to(torch.float32)                         # [1, 1, T, 288]
    local = causal * (self.jimg > (posf - float(UD.WL))).to(torch.float32)
    keep = (mask[..., :PIN] > -1.0).to(torch.float32)                       # the runtime's causal gate, pinned part
    a = keep * (1.0 - local)
    b = (1.0 - keep) * NEG
    valid = (self.jring <= posf).to(torch.float32)                          # [1, 1, T, 128]
    rest = (1.0 - valid) * NEG
    cand = (posf >= self.thr).to(torch.float32)
    slot = ring_slot_f(input_pos).to(torch.int32)                           # [1]
    return {"cos": cos, "sin": sin, "a_img": a, "b_img": b, "rest": rest, "cand": cand, "slot": slot}


class RingAttention(UD.DashAttention):
  """DashAttention writing K / V at the ring slot; the rest of the math is uocr_dash.DashAttention.forward."""

  def forward(self, xn, sh, input_pos, kc, vc):
    cfg = self.cfg
    T = xn.shape[1]
    n, d = cfg.heads, cfg.head_dim
    inv = 1.0 / (d ** 0.5)
    cos, sin = sh["cos"], sh["sin"]
    q = self.q_proj(xn).reshape(1, T, n, d).permute(0, 2, 1, 3)
    k = self.k_proj(xn).reshape(1, T, n, d).permute(0, 2, 1, 3)
    v = self.v_proj(xn).reshape(1, T, n, d).permute(0, 2, 1, 3)
    q, k = ud.apply_rope(q, cos, sin), ud.apply_rope(k, cos, sin)
    iq = ud.apply_rope(UD.gemma_norm(self.rq(xn), self.rq_norm).reshape(1, 1, T, UD.DH), cos, sin)
    ik = ud.apply_rope(UD.gemma_norm(self.rk(xn), self.rk_norm).reshape(1, 1, T, UD.DH), cos, sin)
    zero = torch.zeros((1,), dtype=torch.int32)
    s0 = sh["slot"]
    k_idx = torch.cat([zero, zero, s0, zero], dim=0)
    v_idx = torch.cat([zero, zero, zero, s0], dim=0)
    kc = ud.dus(kc, torch.cat([k, ik], dim=1), [x for x in k_idx])      # [1, 11, 416, 128]
    vc = ud.dus(vc, v.permute(0, 1, 3, 2), [x for x in v_idx])           # [1, 10, 128, 416]
    kk = kc[:, n:n + 1, :PIN, :].reshape(1, UD.NIMG, UD.BS, UD.DH)     # router keys, slots = positions 0..287
    aw = torch.clamp((kk * self.head_cls.reshape(1, 1, 1, UD.DH)).sum(-1, keepdim=True) * inv, -UD.ACT, UD.ACT)
    e = torch.exp(aw - UD.amax_last(aw.reshape(1, 1, UD.NIMG, UD.BS)).reshape(1, UD.NIMG, 1, 1))
    w = e / torch.clamp(e.sum(2, keepdim=True), min=1e-6)
    kbar = (w * kk).sum(2, keepdim=True).reshape(1, 1, UD.NIMG, UD.DH)
    z = torch.clamp(ud.bmm4(iq, kbar) * inv, -UD.ACT, UD.ACT)
    cand = sh["cand"]
    xs = cand * (z * self.idx_scale) + (1.0 - cand) * UD.ENT_NEG
    prior = UD.entmax15_r4(xs)
    logp = torch.log(torch.clamp(prior, min=self.floor))
    cc = cand.reshape(1, T, UD.NIMG, 1)
    cnt = torch.clamp(cc.sum(2, keepdim=True), min=1.0)
    mu = (cc * logp).sum(2, keepdim=True) / cnt
    bias = cc * ((logp - mu) * self.inv_sigma) + (1.0 - cc) * NEG
    full = bias
    for _ in range(5):
      full = torch.cat([full, full], dim=-1)
    full = full.reshape(1, 1, T, PIN)
    m = torch.cat([sh["a_img"] * full + sh["b_img"], sh["rest"]], dim=-1)   # [1, 1, T, 416]
    q11 = torch.cat([q * inv, torch.zeros((1, 1, T, d), dtype=xn.dtype)], dim=1)
    s = ud.bmm4(q11, kc) + m                                               # [1, 11, T, 416]
    p = ud.softmax_last(s)
    o = ud.bmm4(p[:, :n], vc)
    o = o.permute(0, 2, 1, 3).reshape(1, T, n * d)
    return self.o_proj(o), kc, vc


class RingLayer(UD.DashLayer):
  def __init__(self, cfg, idx, consts):
    super().__init__(cfg, idx, consts)
    old = self.attn
    self.attn = RingAttention(cfg, consts)
    assert sorted(n for n, _ in old.named_parameters()) == sorted(n for n, _ in self.attn.named_parameters())


class RingDecoder(UD.DashDecoder):
  def __init__(self, cfg, n_layers=None):
    nn.Module.__init__(self)
    self.cfg = cfg
    consts = UD.dash_consts()
    self.shared = RingShared(cfg)
    self.layers = nn.ModuleList([RingLayer(cfg, i, consts) for i in range(n_layers or cfg.n_layers)])
    self.norm = nn.Parameter(torch.ones(cfg.hidden))
    self.lm_head = ud.LMHead(cfg.hidden, cfg.vocab, bias=False)


def sample_kwargs(cfg, T, n_layers=None):
  """The ring signature inputs: KV of SLOTS slots, the mask of cfg.cache (= CTX) columns."""
  kv = {}
  for i in range(n_layers or cfg.n_layers):
    kv[f"kv_cache_k_{i}"] = torch.zeros(1, cfg.kv_heads + cfg.extra_k_heads, SLOTS, cfg.head_dim)
    kv[f"kv_cache_v_{i}"] = torch.zeros(1, cfg.kv_heads, cfg.head_dim, SLOTS)
  return {"embeddings": torch.zeros(1, T, cfg.hidden), "input_pos": torch.arange(T, dtype=torch.int32),
          "mask": torch.zeros(1, 1, T, cfg.cache), **kv}


class Sig(nn.Module):
  def __init__(self, dec, with_logits, n_layers):
    super().__init__()
    self.dec, self.with_logits, self.L = dec, with_logits, n_layers

  def forward(self, embeddings, input_pos, mask, **kv):
    kcs = [kv[f"kv_cache_k_{i}"] for i in range(self.L)]
    vcs = [kv[f"kv_cache_v_{i}"] for i in range(self.L)]
    logits, nk, nv = self.dec(embeddings, input_pos, mask, kcs, vcs, self.with_logits)
    out = {} if logits is None else {"logits": logits}
    for i in range(self.L):
      out[f"kv_cache_k_{i}"] = nk[i]
      out[f"kv_cache_v_{i}"] = nv[i]
    return out


def export_tflite(dec, emb_mod, cfg, prefills, out_path, emb_path=None):
  import litert_torch
  L = len(dec.layers)
  conv = None
  for T in prefills:
    sig = Sig(dec, False, L).eval()
    kw = sample_kwargs(cfg, T, L)
    conv = (litert_torch.signature(f"prefill_{T}", sig, sample_kwargs=kw) if conv is None
            else conv.signature(f"prefill_{T}", sig, sample_kwargs=kw))
  conv = conv.signature("decode", Sig(dec, True, L).eval(), sample_kwargs=sample_kwargs(cfg, 1, L))
  conv.convert().export(out_path)
  if emb_path is not None:
    litert_torch.signature("embedder", emb_mod.eval(),
                           sample_kwargs={"token_ids": torch.zeros(1, 1, dtype=torch.int32)}).convert().export(emb_path)
