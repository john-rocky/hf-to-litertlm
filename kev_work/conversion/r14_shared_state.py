"""The shared-state pair of Kev-0.8B graphs (two signatures sharing the weights of one file), with the GatedDeltaNet
forward following the form installed by r13_kernel.apply(model, spec) on the text model, as the row form does:
  - the chunk kernel is the instance's `m.chunk_gated_delta_rule` (r13_kernel binds R64 / ec / dd / vs<k> there);
  - softplus / silu come from the `F` of the patch's guarded forward (r11_fp16_safe's softplus shim when `sp` is
    applied, torch.nn.functional otherwise);
  - the gated norm is `m.norm` (class forward: r13_kernel's eps-scaled norm under vs<k>) and beta = b.sigmoid(). With
    vs<k> the delta rule runs on 2^k v, so gdn_state_<l> leaving state_prefill is 2^k times the stock state and
    question_step consumes it in the same scale (both signatures carry the same form).
RowCheck (StatePrefill with_hidden on a whole row) must equal KevPrefill with the same form bit for bit.

    state_prefill_<Ls>(ids int32 [1, Ls], valid float32 [1, Ls]) ->
        gdn_state_<l>  float32 [1, 16, 128, 128]  x 18   recurrent state of GatedDeltaNet layer l after the last real
                                                         state token
        conv_tail_<l>  float32 [1, 3, 6144]       x 18   that layer's conv input (in_proj_qkv output) at the last 3 real
                                                         state tokens, oldest first (zero rows before the first token)
        k_<l>, v_<l>   float32 [1, 2, Ls, 256]    x 6    full-attention layer l: keys after k_norm and RoPE, values;
                                                         every position (the question step masks the pad positions)
    question_step_<Ls>_<Lq>(ids int32 [1, Lq], valid float32 [1, Lq], state_valid float32 [1, Ls], + the 48 tensors
        above) -> hidden float32 [1, Lq, 1024]   (after the final RMSNorm; row j = token j of the question part)

l = the decoder layer index (GatedDeltaNet 0 1 2 4 5 6 ... 22; attention 3 7 11 15 19 23).

The pair computes the row form (kev_graph.KevPrefill on state + question as one causal row, the author's
`forward_rows_batch`), with the state run once (the author's cached path `probs_and_prefix` / `_branch_rows_from_prefix`:
positions continue after the state, the state's caches are carried):
- positions: state 0..Ls-1 (constant, as in the row form); question n + 0..Lq-1 with n = sum(state_valid). The question's
  RoPE comes from one constant table [Ls + Lq, 128] = cos | sin (rotary dim = head_dim 256 x partial_rotary_factor
  0.25 = 64, computed by the model's rotary module for positions 0..Ls+Lq-1, so the values are the row form's) selected
  by a one-hot built from n in float arithmetic (relu(1 - |p - (n + j)|); no GATHER, EQUAL or CAST) times the table as the
  constant RIGHT operand = one FULLY_CONNECTED in the module `RopeSelect` (Metal refuses a BATCH_MATMUL whose LEFT
  operand is constant: "Not supported batched mat mul case: non-constant tensor"). quantize_shared_state.py keeps the
  RopeSelect scope out of the float16 cast, so the table stays float32 and the selection is exact.
- GatedDeltaNet: the state step is the row form's guarded forward (pad hidden zeroed before the projections, pad decay
  -> identity with a -> -30 pre-softplus, pad q / k / v zeroed after the conv) with zero initial state and
  output_final_state = True, so with right padding the recurrent state stops at the last real token. The conv tail is
  cut from the conv input with a one-hot matmul from n (the float one-hot of the patch's guard (5)). The question step
  runs the rank-4 chunk kernel with initial_state = gdn_state and the causal conv over concat(conv_tail, question conv
  input), keeping the last Lq outputs (transformers' cached chunked-continuation path, with the 3 columns kernel 4 needs).
- attention: K / V = concat(state, question) on the sequence axis (length Ls + Lq); the additive mask is
  [state pads -1e4] ++ [question causal + question pads -1e4] (the row form's -1e4 convention), built as the row form
  builds it: constant [0 | causal] + (1 - concat(state_valid, valid)) * -1e4.
Everything else is the loaded (patched) text model's own modules: embedding, norms, MLPs, the attention module
(transformers' forward with `kev_eager`, through a minimal cache object whose update() concatenates the state K / V)
and the patch's rank-4 chunk kernel. Only the GatedDeltaNet forward is written out here; it mirrors the patch's guarded
forward line by line (kev_qwen35_patch._build_guarded_forward on transformers 5.14.1) plus the two state ends."""
import torch
import torch.nn.functional as F
from torch import nn

import kev_qwen35_patch as P
from kev_graph import NEG, causal_constant

TAIL = 3          # conv kernel 4 -> the 3 previous conv inputs
ROTARY = 64       # head_dim 256 x partial_rotary_factor 0.25


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
    initial_state [1, H, dk, dv] and conv_tail [1, 3, C] continue a state (question step); tail_onehot [1, 1, 3, L] asks
    for the state ends (state step). -> (out [1, L, d], final recurrent state or None, conv tail [1, 3, C] or None)."""
    Fg = P.PatchedQwen3_5GatedDeltaNet.forward.__globals__["F"]            # the guarded forward's F
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
    if tail_onehot is not None:                                             # conv input at the last 3 real tokens
        tail = torch.matmul(tail_onehot, x.reshape(1, 1, seq_len, -1)).reshape(1, TAIL, -1)
    if conv_tail is None:
        mixed = x.transpose(1, 2)
        mixed = Fg.silu(m.conv1d(mixed)[:, :, : mixed.shape[-1]])
    else:                                                                   # cached chunked continuation
        mixed = torch.cat([conv_tail, x], dim=1).transpose(1, 2)            # [1, C, 3 + L]
        mixed = Fg.silu(m.conv1d(mixed)[:, :, : mixed.shape[-1]])
        mixed = mixed[:, :, -seq_len:]
    mixed = mixed.transpose(1, 2)
    mixed = mixed * valid[..., None]                                        # guard (4)
    query, key, value = torch.split(mixed, [m.key_dim, m.key_dim, m.value_dim], dim=-1)
    query = query.reshape(batch_size, seq_len, -1, m.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, m.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, m.head_v_dim)
    beta = b.sigmoid()
    g = -m.A_log.float().exp() * Fg.softplus(a.float() + m.dt_bias)
    assert m.num_v_heads == m.num_k_heads, "0.8B only (4B interleaves heads, patch guard (6))"
    out, last = m.chunk_gated_delta_rule(query, key, value, g=g, beta=beta, initial_state=initial_state,
                                         output_final_state=tail_onehot is not None,
                                         use_qk_l2norm_in_kernel=True)
    out = out.reshape(-1, m.head_v_dim)
    z = z.reshape(-1, m.head_v_dim)
    out = m.norm(out, z)
    out = out.reshape(batch_size, seq_len, -1)
    return m.out_proj(out), last, tail


class _Base(nn.Module):
    def __init__(self, text_model):
        super().__init__()
        self.tm = text_model
        types = text_model.config.layer_types
        self.gdn_layers = [i for i, t in enumerate(types) if t == "linear_attention"]
        self.attn_layers = [i for i, t in enumerate(types) if t == "full_attention"]

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
    """state_prefill: ids [1, Ls] + valid [1, Ls] -> the 48 state tensors (with_hidden: + hidden, for RowCheck)."""

    def __init__(self, text_model, Ls, with_hidden=False):
        super().__init__(text_model)
        self.Ls, self.with_hidden = Ls, with_hidden
        self.register_buffer("position_ids", torch.arange(Ls, dtype=torch.long).reshape(1, Ls), persistent=False)
        self.register_buffer("causal_const", causal_constant(Ls), persistent=False)
        self.register_buffer("cols", torch.tensor([float(c) for c in range(Ls)]).reshape(1, 1, 1, Ls), persistent=False)
        self.register_buffer("tail_offsets", torch.tensor([-3.0, -2.0, -1.0]).reshape(1, 1, TAIL, 1), persistent=False)

    def forward(self, ids, valid):
        h = self.tm.embed_tokens(ids)
        pos_emb = self.tm.rotary_emb(h, self.position_ids)
        mask = self.causal_const + (1.0 - valid)[:, None, None, :] * NEG
        n = valid.sum(-1, keepdim=True)                                     # [1, 1] (keepdim: no rank-0 scalar)
        idx = n.reshape(1, 1, 1, 1) + self.tail_offsets                     # [1, 1, 3, 1] = n-3, n-2, n-1
        onehot = torch.relu(1.0 - torch.abs(self.cols - idx))               # [1, 1, 3, Ls]; idx < 0 -> zero row
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
    """cos, sin [1, Lq, 64] = one-hot [1, Lq, Lmax] @ table [Lmax, 128] (cos | sin) as one FULLY_CONNECTED."""

    def __init__(self, cos, sin):
        super().__init__()
        self.register_buffer("table_t", torch.cat([cos, sin], dim=-1).t().contiguous(), persistent=False)  # [128, Lmax]

    def forward(self, sel):
        cs = F.linear(sel, self.table_t)                                    # [1, Lq, 128]
        return cs[..., :ROTARY], cs[..., ROTARY:]


class _QuestionStepBase(_Base):
    def __init__(self, text_model, Ls, Lq):
        super().__init__(text_model)
        self.Ls, self.Lq = Ls, Lq
        Lmax = Ls + Lq
        with torch.no_grad():
            dummy = torch.zeros(1, Lmax, text_model.config.hidden_size)
            cos, sin = text_model.rotary_emb(dummy, torch.arange(Lmax, dtype=torch.long).reshape(1, Lmax))
        assert cos.shape == (1, Lmax, ROTARY), cos.shape
        self.rope = RopeSelect(cos[0], sin[0])
        self.register_buffer("pos_row", torch.tensor([float(p) for p in range(Lmax)]).reshape(1, 1, 1, Lmax),
                             persistent=False)
        self.register_buffer("q_col", torch.tensor([float(j) for j in range(Lq)]).reshape(1, 1, Lq, 1),
                             persistent=False)
        # [state part: 0 | question part: causal 0 / -1e4]; a non-zero constant, so `const + row` stays an ADD (attempt 1
        # added an all-zero constant to the state row and the converter folded that into BROADCAST_TO + an INT64 shape)
        self.register_buffer("mask_const", torch.cat([torch.zeros(1, 1, Lq, Ls), causal_constant(Lq)], dim=-1),
                             persistent=False)

    def input_names(self):
        return ["ids", "valid", "state_valid"] + self.state_names()

    def _forward(self, ids, valid, state_valid, **states):
        Lq = self.Lq
        h = self.tm.embed_tokens(ids)
        n = state_valid.sum(-1, keepdim=True)                               # [1, 1] real state tokens
        sel = torch.relu(1.0 - torch.abs(self.pos_row - (n.reshape(1, 1, 1, 1) + self.q_col)))  # [1, 1, Lq, Lmax]
        cos, sin = self.rope(sel.reshape(1, Lq, -1))                                    # [1, Lq, 64] each
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


def contract(Ls, Lq, gdn_layers, attn_layers, conv_dim=6144, hidden=1024):
    states = []
    for i in sorted(gdn_layers + attn_layers):
        if i in gdn_layers:
            states += [{"name": f"gdn_state_{i}", "shape": [1, 16, 128, 128], "dtype": "float32"},
                       {"name": f"conv_tail_{i}", "shape": [1, TAIL, conv_dim], "dtype": "float32"}]
        else:
            states += [{"name": f"k_{i}", "shape": [1, 2, Ls, 256], "dtype": "float32"},
                       {"name": f"v_{i}", "shape": [1, 2, Ls, 256], "dtype": "float32"}]
    return {
        "graph": "Kev-0.8B shared-state pair (two signatures, shared weights)", "Ls": Ls, "Lq": Lq,
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
                "outputs": [{"name": "hidden", "shape": [1, Lq, hidden], "dtype": "float32",
                             "meaning": "last_hidden_state of the question tokens (after the final RMSNorm)"}]}},
        "positions": "state 0..Ls-1; question n..n+Lq-1 with n = sum(state_valid) (the row form's positions)",
        "readout": "decide = the question's last real token, opt_k = each option's 248050; head on the host as the row "
                   "form (indices relative to the question start)",
        "state_bytes": sum(4 * int(torch.tensor(s["shape"]).prod()) for s in states),
    }
