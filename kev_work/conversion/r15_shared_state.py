"""The shared-state pair (two signatures sharing the weights of one file) for Kev-4B on the final kernel: the
r14_shared_state.py contract generalized to the model's config (r14_shared_state.py stays the 0.8B file and is not
imported here).

    state_prefill_<Ls>(ids int32 [1, Ls], valid float32 [1, Ls]) ->
        gdn_state_<l>  float32 [1, Hv, dk, dv]        recurrent state of GatedDeltaNet layer l after the last real token
                                                      (4B: [1, 32, 128, 128] x 24)
        conv_tail_<l>  float32 [1, 3, conv_dim]       that layer's conv input at the last 3 real state tokens, oldest
                                                      first, zero rows before the first token (4B: [1, 3, 8192] x 24)
        k_<l>, v_<l>   float32 [1, Hkv, Ls, head_dim] full-attention layer l: keys after k_norm and RoPE, values, every
                                                      position (4B: [1, 4, Ls, 256] x 8 each)
    question_step_<Ls>_<Lq>(ids int32 [1, Lq], valid float32 [1, Lq], state_valid float32 [1, Ls], + the state tensors)
        -> hidden float32 [1, Lq, d]   (after the final RMSNorm; row j = token j of the question part; 4B d = 2560)

What changes against r14_shared_state.py (everything else is that file's code, comments kept where they apply):
- the head layout comes from the config: GatedDeltaNet value heads Hv / key heads Hk (4B: 32 / 16, ratio 2: q and k are
  copied with the patch's head interleave, guard (6), at the same place as in the guarded forward, through the guarded
  forward's own global so a call counter sees it), attention kv heads, head_dim, rotary dim (head_dim x
  partial_rotary_factor), conv kernel, hidden size;
- the chunk kernel is the module's own `chunk_gated_delta_rule` (the loop kernel, or the form installed by
  r13_kernel.apply / r15_form.apply) and softplus / silu are the guarded forward's `F` (r11_fp16_safe's softplus
  rewrite when the form has `sp`), so with the same form installed the pair computes what KevPrefill computes; RowCheck
  (StatePrefill with with_hidden on a whole row) must equal KevPrefill bit for bit;
- under vs<k> (the kernel scales v by 2^k at its entry) the gdn_state outputs carry 2^k x the stock state: the question
  step feeds them back into the same kernel (initial_state), so the factor is consistent; the host never reads the
  state. A pair file is therefore tied to its form (never mix a state of one form with the question step of another).
Positions, the conv tail one-hot, the RoPE selection (RopeSelect, one FULLY_CONNECTED with the constant table on the
right) and the attention mask are r14_shared_state.py's."""
import torch
from torch import nn

import kev_qwen35_patch as P
from kev_graph import NEG, causal_constant


def _F():
    """torch.nn.functional as the guarded GatedDeltaNet forward sees it (r11_fp16_safe's shim when softplus is rewritten)."""
    return P.PatchedQwen3_5GatedDeltaNet.forward.__globals__["F"]


def _interleave():
    return P.PatchedQwen3_5GatedDeltaNet.forward.__globals__["_litert_interleave_heads"]


class _KV:
    """What transformers' attention forward calls on its cache: update(k, v, layer) -> the K / V to attend over."""

    def __init__(self, state=None):
        self.state = state or {}
        self.new = {}

    def update(self, key, value, layer_idx, *args, **kwargs):
        self.new[layer_idx] = (key, value)
        if layer_idx in self.state:
            ks, vs = self.state[layer_idx]
            return torch.cat([ks, key], dim=2), torch.cat([vs, value], dim=2)
        return key, value


def gdn_forward(m, h, valid, initial_state=None, conv_tail=None, tail_onehot=None):
    """One PatchedQwen3_5GatedDeltaNet on h [1, L, d] (after input_layernorm) with the pad guard from valid [1, L].
    initial_state [1, Hv, dk, dv] and conv_tail [1, K-1, C] continue a state (question step); tail_onehot [1, 1, K-1, L]
    asks for the state ends (state step). -> (out [1, L, d], final recurrent state or None, conv tail or None)."""
    F = _F()
    tail_n = m.conv_kernel_size - 1
    batch_size, seq_len, _ = h.shape
    valid = valid.to(h.dtype)
    h = h * valid[..., None]                                                # guard (1)+(2)
    x = m.in_proj_qkv(h)                                                    # conv input [1, L, C]
    z = m.in_proj_z(h)
    z = z.reshape(batch_size, seq_len, -1, m.head_v_dim)
    b = m.in_proj_b(h)
    a = m.in_proj_a(h)
    a = a * valid[..., None] + (1.0 - valid[..., None]) * (-30.0)           # guard (3)
    tail = None
    if tail_onehot is not None:                                             # conv input at the last K-1 real tokens
        tail = torch.matmul(tail_onehot, x.reshape(1, 1, seq_len, -1)).reshape(1, tail_n, -1)
    if conv_tail is None:
        mixed = x.transpose(1, 2)
        mixed = F.silu(m.conv1d(mixed)[:, :, : mixed.shape[-1]])
    else:                                                                   # cached chunked continuation
        mixed = torch.cat([conv_tail, x], dim=1).transpose(1, 2)            # [1, C, K-1 + L]
        mixed = F.silu(m.conv1d(mixed)[:, :, : mixed.shape[-1]])
        mixed = mixed[:, :, -seq_len:]
    mixed = mixed.transpose(1, 2)
    mixed = mixed * valid[..., None]                                        # guard (4)
    query, key, value = torch.split(mixed, [m.key_dim, m.key_dim, m.value_dim], dim=-1)
    query = query.reshape(batch_size, seq_len, -1, m.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, m.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, m.head_v_dim)
    beta = b.sigmoid()
    g = -m.A_log.float().exp() * F.softplus(a.float() + m.dt_bias)
    ratio = m.num_v_heads // m.num_k_heads
    if ratio > 1:                                                           # guard (6): the 4B head copy
        il = _interleave()
        query = il(query, ratio)
        key = il(key, ratio)
    out, last = m.chunk_gated_delta_rule(query, key, value, g=g, beta=beta, initial_state=initial_state,
                                         output_final_state=tail_onehot is not None, use_qk_l2norm_in_kernel=True)
    out = out.reshape(-1, m.head_v_dim)
    z = z.reshape(-1, m.head_v_dim)
    out = m.norm(out, z)
    out = out.reshape(batch_size, seq_len, -1)
    return m.out_proj(out), last, tail


class _Base(nn.Module):
    def __init__(self, text_model):
        super().__init__()
        self.tm = text_model
        cfg = text_model.config
        types = cfg.layer_types
        self.gdn_layers = [i for i, t in enumerate(types) if t == "linear_attention"]
        self.attn_layers = [i for i, t in enumerate(types) if t == "full_attention"]
        self.tail_n = cfg.linear_conv_kernel_dim - 1
        self.rotary = int(cfg.head_dim * cfg.partial_rotary_factor)

    def layer(self, i, h, pos_emb, mask, valid, kv, gdn_kwargs):
        """Qwen3_5DecoderLayer.forward with our GatedDeltaNet forward."""
        lyr = self.tm.layers[i]
        residual = h
        x = lyr.input_layernorm(h)
        last = tail = None
        if lyr.block_type == "linear_attention":
            x, last, tail = gdn_forward(lyr.linear_attn, x, valid, **gdn_kwargs)
        else:
            x, _ = lyr.self_attn(hidden_states=x, attention_mask=mask, position_ids=None, past_key_values=kv,
                                 position_embeddings=pos_emb)
        h = residual + x
        residual = h
        x = lyr.post_attention_layernorm(h)
        x = lyr.mlp(x)
        h = residual + x
        return h, last, tail

    def state_names(self):
        names = []
        for i in range(len(self.tm.layers)):
            names += [f"gdn_state_{i}", f"conv_tail_{i}"] if i in self.gdn_layers else [f"k_{i}", f"v_{i}"]
        return names


class StatePrefill(_Base):
    """state_prefill: ids [1, Ls] + valid [1, Ls] -> the state tensors (with_hidden: + hidden, for RowCheck)."""

    def __init__(self, text_model, Ls, with_hidden=False):
        super().__init__(text_model)
        self.Ls, self.with_hidden = Ls, with_hidden
        self.register_buffer("position_ids", torch.arange(Ls, dtype=torch.long).reshape(1, Ls), persistent=False)
        self.register_buffer("causal_const", causal_constant(Ls), persistent=False)
        self.register_buffer("cols", torch.tensor([float(c) for c in range(Ls)]).reshape(1, 1, 1, Ls), persistent=False)
        self.register_buffer("tail_offsets", torch.tensor([float(-self.tail_n + j) for j in range(self.tail_n)])
                             .reshape(1, 1, self.tail_n, 1), persistent=False)

    def forward(self, ids, valid):
        h = self.tm.embed_tokens(ids)
        pos_emb = self.tm.rotary_emb(h, self.position_ids)
        mask = self.causal_const + (1.0 - valid)[:, None, None, :] * NEG
        n = valid.sum(-1, keepdim=True)                                     # [1, 1] (keepdim: no rank-0 scalar)
        idx = n.reshape(1, 1, 1, 1) + self.tail_offsets                     # [1, 1, K-1, 1] = n-3, n-2, n-1
        onehot = torch.relu(1.0 - torch.abs(self.cols - idx))               # [1, 1, K-1, Ls]; idx < 0 -> zero row
        kv = _KV()
        out = {}
        for i in range(len(self.tm.layers)):
            gdn = i in self.gdn_layers
            h, last, tail = self.layer(i, h, pos_emb, mask, valid, kv, {"tail_onehot": onehot} if gdn else {})
            if gdn:
                out[f"gdn_state_{i}"] = last
                out[f"conv_tail_{i}"] = tail
            else:
                out[f"k_{i}"], out[f"v_{i}"] = kv.new[i]
        if self.with_hidden:
            out["hidden"] = self.tm.norm(h)
        return out


class RopeSelect(nn.Module):
    """cos, sin [1, Lq, R] = one-hot [1, Lq, Lmax] @ table [Lmax, 2R] (cos | sin) as one FULLY_CONNECTED."""

    def __init__(self, cos, sin):
        super().__init__()
        self.rotary = cos.shape[-1]
        self.register_buffer("table_t", torch.cat([cos, sin], dim=-1).t().contiguous(), persistent=False)  # [2R, Lmax]

    def forward(self, sel):
        cs = torch.nn.functional.linear(sel, self.table_t)                  # [1, Lq, 2R]
        return cs[..., :self.rotary], cs[..., self.rotary:]


class _QuestionStepBase(_Base):
    def __init__(self, text_model, Ls, Lq):
        super().__init__(text_model)
        self.Ls, self.Lq = Ls, Lq
        Lmax = Ls + Lq
        with torch.no_grad():
            dummy = torch.zeros(1, Lmax, text_model.config.hidden_size)
            cos, sin = text_model.rotary_emb(dummy, torch.arange(Lmax, dtype=torch.long).reshape(1, Lmax))
        assert cos.shape == (1, Lmax, self.rotary), (cos.shape, self.rotary)
        self.rope = RopeSelect(cos[0], sin[0])
        self.register_buffer("pos_row", torch.tensor([float(p) for p in range(Lmax)]).reshape(1, 1, 1, Lmax),
                             persistent=False)
        self.register_buffer("q_col", torch.tensor([float(j) for j in range(Lq)]).reshape(1, 1, Lq, 1),
                             persistent=False)
        # [state part: 0 | question part: causal 0 / -1e4]; a non-zero constant, so `const + row` stays an ADD (round
        # 10 attempt 1 added an all-zero constant to the state row and the converter folded that into BROADCAST_TO +
        # an INT64 shape)
        self.register_buffer("mask_const", torch.cat([torch.zeros(1, 1, Lq, Ls), causal_constant(Lq)], dim=-1),
                             persistent=False)

    def input_names(self):
        return ["ids", "valid", "state_valid"] + self.state_names()

    def _forward(self, ids, valid, state_valid, **states):
        Lq = self.Lq
        h = self.tm.embed_tokens(ids)
        n = state_valid.sum(-1, keepdim=True)                               # [1, 1] real state tokens
        sel = torch.relu(1.0 - torch.abs(self.pos_row - (n.reshape(1, 1, 1, 1) + self.q_col)))  # [1, 1, Lq, Lmax]
        cos, sin = self.rope(sel.reshape(1, Lq, -1))                                    # [1, Lq, R] each
        keys_valid = torch.cat([state_valid, valid], dim=-1)                             # [1, Ls + Lq]
        mask = self.mask_const + (1.0 - keys_valid)[:, None, None, :] * NEG              # [1, 1, Lq, Ls + Lq]
        kv = _KV({i: (states[f"k_{i}"], states[f"v_{i}"]) for i in self.attn_layers})
        for i in range(len(self.tm.layers)):
            kw = ({"initial_state": states[f"gdn_state_{i}"], "conv_tail": states[f"conv_tail_{i}"]}
                  if i in self.gdn_layers else {})
            h, _, _ = self.layer(i, h, (cos, sin), mask, valid, kv, kw)
        return {"hidden": self.tm.norm(h)}


def QuestionStep(text_model, Ls, Lq):
    """question_step module whose forward takes every input by its own name (torch.export / the signature need the
    names; a **kwargs forward would hide them)."""
    names = ["ids", "valid", "state_valid"] + _Base(text_model).state_names()
    src = (f"def forward(self, {', '.join(names)}):\n"
           f"    return self._forward({', '.join(f'{n}={n}' for n in names)})\n")
    ns = {}
    exec(compile(src, "<question_step_forward>", "exec"), ns)  # pylint: disable=exec-used
    cls = type(f"QuestionStep_{Ls}_{Lq}", (_QuestionStepBase,), {"forward": ns["forward"]})
    return cls(text_model, Ls, Lq)


def state_inputs(state_ids, Ls, pad_id):
    """state token ids -> (ids int32 [1, Ls], valid float32 [1, Ls]), right-padded."""
    n = len(state_ids)
    assert n <= Ls, (n, Ls)
    ids = torch.full((1, Ls), pad_id, dtype=torch.int32)
    ids[0, :n] = torch.tensor(state_ids, dtype=torch.int32)
    valid = torch.zeros((1, Ls), dtype=torch.float32)
    valid[0, :n] = 1.0
    return ids, valid


def state_shapes(cfg, Ls):
    """{name: shape} of the state tensors for a text config (Qwen3_5TextConfig or its dict)."""
    g = cfg if isinstance(cfg, dict) else cfg.to_dict()
    hv, dk, dv = g["linear_num_value_heads"], g["linear_key_head_dim"], g["linear_value_head_dim"]
    conv_dim = 2 * g["linear_num_key_heads"] * dk + hv * dv
    tail_n = g["linear_conv_kernel_dim"] - 1
    out = {}
    for i, t in enumerate(g["layer_types"]):
        if t == "linear_attention":
            out[f"gdn_state_{i}"] = [1, hv, dk, dv]
            out[f"conv_tail_{i}"] = [1, tail_n, conv_dim]
        else:
            out[f"k_{i}"] = [1, g["num_key_value_heads"], Ls, g["head_dim"]]
            out[f"v_{i}"] = [1, g["num_key_value_heads"], Ls, g["head_dim"]]
    return out


def contract(cfg, Ls, Lq, form=None):
    g = cfg if isinstance(cfg, dict) else cfg.to_dict()
    shapes = state_shapes(g, Ls)
    states = [{"name": n, "shape": s, "dtype": "float32"} for n, s in shapes.items()]
    vs = None
    if form:
        import re
        m = [int(t[2:]) for t in form.split("+") if re.fullmatch(r"vs\d+", t)]
        vs = m[0] if m else 0
    return {
        "graph": "Kev shared-state pair (two signatures, shared weights)", "Ls": Ls, "Lq": Lq, "form": form,
        "signatures": {
            f"state_prefill_{Ls}": {
                "inputs": [{"name": "ids", "shape": [1, Ls], "dtype": "int32",
                            "meaning": "[248060] + state tokens, right-padded with 248044"},
                           {"name": "valid", "shape": [1, Ls], "dtype": "float32", "meaning": "1.0 real / 0.0 pad"}],
                "outputs": states},
            f"question_step_{Ls}_{Lq}": {
                "inputs": [{"name": "ids", "shape": [1, Lq], "dtype": "int32",
                            "meaning": "one question's branch [248061] + instructions + options + [248062], right-padded"},
                           {"name": "valid", "shape": [1, Lq], "dtype": "float32", "meaning": "1.0 real / 0.0 pad"},
                           {"name": "state_valid", "shape": [1, Ls], "dtype": "float32",
                            "meaning": "the valid of the state_prefill call whose outputs are passed"}] + states,
                "outputs": [{"name": "hidden", "shape": [1, Lq, g["hidden_size"]], "dtype": "float32",
                             "meaning": "last_hidden_state of the question tokens (after the final RMSNorm)"}]}},
        "positions": "state 0..Ls-1; question n..n+Lq-1 with n = sum(state_valid) (the row form's positions)",
        "readout": "decide = the question's last real token, opt_k = each option's 248050; head on the host as the row "
                   "form (indices relative to the question start)",
        "gdn_state_scale": (f"2^{vs} x the stock recurrent state (vs{vs}: the kernel scales v at its entry); pass it "
                            "through unchanged" if vs else "the stock recurrent state"),
        "state_tensors": len(states),
        "state_bytes": sum(4 * int(torch.tensor(s["shape"]).prod()) for s in states),
    }
