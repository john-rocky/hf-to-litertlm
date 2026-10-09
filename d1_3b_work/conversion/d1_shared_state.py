"""Round 9: the shared-state pair of d1-3B (two signatures over one copy of the weights) and its check against the row
form in torch.

    state_prefill_<Ls>(ids int32 [1, Ls], valid float32 [1, Ls]) ->
        conv_tail_<l>  float32 [1, 2048, 2]    x 22  ShortConv layer l: its conv input (b * u, the tensor the provider's
                                                     ShortConv.forward hands to causal_conv) at the last 2 real state
                                                     tokens, oldest first; zero columns before the first token
        k_<l>, v_<l>   float32 [1, 8, Ls, 64]  x 8   attention layer l: keys after k_layernorm and RoPE, values; every
                                                     position (question_step masks the state's pads with state_valid)
    question_step_<Ls>_<Lq>(ids int32 [1, Lq], valid float32 [1, Lq], state_valid float32 [1, Ls], the 38 tensors above)
        -> {"hidden": float32 [1, Lq, 2048]}   (after the final RMSNorm; row j = token j of the question part)

l = the decoder layer index (attention: 2 5 9 13 17 21 24 27, ShortConv: the 22 others); the state tensors are listed in
layer order (conv_tail_0, conv_tail_1, k_2, v_2, conv_tail_3, ...).

The pair computes the row form (d1_prefill_graph.D1Prefill on state + question as one causal row from position 0) with
the state run once, the way the provider's tree path continues a trunk (hybrid.Tree: positions continue after the trunk,
a branch's conv starts from the trunk's last inputs, a branch attends over the trunk's keys and its own):
- positions: state 0..Ls-1 (constant tables, as D1Prefill); question n + 0..Lq-1 with n = sum(state_valid). The
  question's RoPE comes from one constant table [Ls + Lq, 128] = cos | sin (d1_prefill_graph.rope_tables at positions
  0..Ls+Lq-1, so every value is the row form's) selected by a float one-hot from n (relu(1 - |p - (n + j)|): no GATHER,
  EQUAL or CAST) times the table as the constant RIGHT operand = one FULLY_CONNECTED in the module `RopeSelect` (Kev r10
  trap: Metal refuses a BATCH_MATMUL whose LEFT operand is constant); `d1_storage.py --pair` keeps RopeSelect out of the
  float16 cast, so the table stays float32 and the selection exact.
- attention: K / V = concat(state, question) on the sequence axis (Ls + Lq keys), GQA by concat as RowAttention; the
  additive mask is built as the row form builds its own: a non-zero constant [0 (state part) | causal (question part)] +
  (1 - concat(state_valid, valid)) * -1e4 (an all-zero constant + a row folds to BROADCAST_TO + INT64, Kev r10 trap 1).
- ShortConv: the state step is the provider's forward (causal_conv with tree None) plus the tail, cut from the conv input
  with a float one-hot from n (one BATCH_MATMUL of two activations); the question step runs the depthwise conv over
  concat(conv_tail, question conv input) and keeps its Lq outputs (the provider's Tree.conv_order: a branch led by the
  trunk's last `keep` = 2 inputs).
Everything else is the provider's own code, as in D1Prefill: the embedding, DecoderLayer.forward (it hands its `pos`
argument, here a PairCtx, to the operator untouched), RMSNorm, the MLP, the operators' projections and norms. The
operators get pair-aware classes (`to_pair_form`): the attention class keeps d1_prefill_graph's RowAttention forward for
the row form's (cos, sin, mask) context, and its name (D1Prefill checks it); the ShortConv class keeps the provider's
forward for every context but a PairCtx. So one loaded model serves the row form and the pair.

--check (acceptance 1; float32, CPU, one process): every text row of fixtures/rows.json (415) as its request's
question (the state = the row's first state_len tokens, the same in every row of a request; every row is split_equal =
the provider's trunk + branch ids), through
  (a) each export pair it fits: Ls256 + Lq128 and Ls128 + Lq64 (the state once per request and pair);
  (b) when it fits neither, the smallest bucket pair (Ls in 128 .. 4096, Lq in 64 / 128 / 192),
against D1Prefill on the whole row at the host's bucket (pick_L over 256 .. 4096): max |dh| over the question's real
positions and at the answer slot, |dp| of the host's read-out (host/d1_litert.py readout with
cache/real/tables/readout_table.safetensors), argmax; both also against the provider's float32 reference
(results/reference_real.json). Edge cases per export pair: an empty state (n = 0: a whole row <= Lq as the question;
both tail columns zero), n = 1 and 2 (tv4_000's question after its first 1 / 2 state tokens: 1 / 0 zero tail columns),
and n = Ls exactly (the first row with Ls < length <= Ls + Lq, split at Ls). RowCheck: the state step with its hidden
output on whole rows (the first 24 requests' first rows) equals D1Prefill bit for bit. State checks (the first 24
requests, Ls256): pads holding another id leave the state bit-equal (k / v at the real positions, the tails); valid = 1
on the pads moves the tails (control); Ls128 vs Ls256 for n <= 128. A layer trace (max |dh| of the residual stream after
every layer at the question positions) on 2 requests. Bars: max |dh| <= 1e-4 over the question positions and
|dp| <= 1e-5 vs the row form on every run, RowCheck bit-equal, every value finite.
Outputs (never overwritten): results/real_sharedstate_torch_check.json (summary); cache/real/pair/: torch_check_rows.json
(one row per question and pair), torch_pair_hsel.npz (answer-slot hidden: `row/<id>/<qid>`, `Ls<Ls>_Lq<Lq>/<id>/<qid>`),
torch_state_Ls<Ls>_<id>.npz (the 38 state tensors of card_text_001, own_fiveq_09 and tv4_000, for the LiteRT state
check); progress parts in cache/real/pair/torch_parts/ (--resume skips the requests already done; deleted at the end).

    scripts/d1_guarded.sh logs/r9_torch_check.guard.log $EXPORT scripts/d1_shared_state.py --check

Round 10, the embeds pair (--check --embeds): state_prefill_<Ls>(embeds float32 [1, Ls, d], valid) and
question_step_<Ls>_<Lq>(embeds float32 [1, Lq, d], valid, state_valid, the 38 state tensors) = StatePrefillEmbeds and
QuestionStep(embeds=True): the table lookup leaves the graph (the host writes the float32 rows of the bfloat16
embed_table.safetensors at the padded ids, the pad id's row on the pads, as host/d1_litert.py does for the embeds row
graphs); everything else is the ids pair's. The check is --check's with: the pairs Ls256+Lq128, Ls128+Lq64 and the new
Ls64+Lq64; the row form = D1PrefillEmbeds on the same rows; the table's rows first compared bit for bit with the
model's float32 table at every id of the fixture rows; the gate of the round-9 ruling (answer slot <= 1e-4, |dp| <= 1e-5,
the question positions' hidden recorded with the alert line 2.5e-4); `vs_ids_round9` = every run against round 9's ids
run of the same question and shape (Ls64+Lq64 against round 9's Ls128+Lq64), |dp| and the answer-slot hidden, and the
row form against round 9's (stop line |dp| 1e-5). Outputs: results/real_sharedstate_embeds_torch_check.json,
cache/real/pair/embeds/ (rows, answer-slot npz, the state dumps for the LiteRT state check).

    scripts/d1_guarded.sh logs/r10_torch_check_embeds.guard.log $EXPORT scripts/d1_shared_state.py \
        --check --embeds
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d1_prefill_graph as G  # noqa: E402
from d1_common import K, ROWS, provider, sha256_file  # noqa: E402

SNAP = Path.home() / ".cache/huggingface/hub/models--LiquidAI--d1-3B/snapshots/da1fe36a861f24690f27f622dca1d8688503d113"
PAD_ID = 124893
NEG = G.NEG
TAIL = 2                                  # conv_L_cache 3 -> the 2 previous conv inputs
PAIRS = ((256, 128), (128, 64))           # the export sizes (design 2)
PAIRS_EMBEDS = ((256, 128), (128, 64), (64, 64))   # round 10 (--embeds): + the small pair Ls64+Lq64
BAR_HIDDEN_ALERT = 2.5e-4                 # round 9 ruling: the question positions' hidden is recorded, alert above
LS_BUCKETS = (128, 256, 512, 1024, 2048, 4096)
LQ_BUCKETS = (64, 128, 192)
ROW_BUCKETS = (256, 512, 1024, 2048, 4096)   # the host's row buckets
BAR_HIDDEN, BAR_DP = 1e-4, 1e-5
CHECK_N = 24
STATE_DUMP = ("card_text_001", "own_fiveq_09", "tv4_000")
TRACE = ("tv4_000", "own_fiveq_09")
PAIR_DIR = K / "cache/real/pair"
OUT = K / "results/real_sharedstate_torch_check.json"
OUT_EMBEDS = K / "results/real_sharedstate_embeds_torch_check.json"     # round 10 (--embeds)
EMBED_TABLE = K / "cache/real/tables/embed_table.safetensors"
TABLE = K / "cache/real/tables/readout_table.safetensors"
REFERENCE = K / "results/reference_real.json"


# --------------------------------------------------------------------------- #
# the pair
# --------------------------------------------------------------------------- #


class PairCtx:
    """The pair's context, passed through DecoderLayer's `pos` argument (the provider's DecoderLayer.forward hands it to
    the operator untouched). mode "state": cos / sin [1, 1, Ls, 64], mask [1, 1, Ls, Ls], tail_sel [1, Ls, 2]; the
    operators write their state into `out`. mode "question": cos / sin [1, 1, Lq, 64], mask [1, 1, Lq, Ls + Lq]; the
    operators read `state`."""

    def __init__(self, mode, cos, sin, mask, tail_sel=None, state=None):
        assert mode in ("state", "question"), mode
        self.mode, self.cos, self.sin, self.mask, self.tail_sel = mode, cos, sin, mask, tail_sel
        self.state = state or {}
        self.out = {}


_CLASSES: dict = {}
_SOFTMAX = {"dtype": torch.float32}       # --fp64 sets float64 (both forms); the export form is float32


def _pair_classes():
    """(attention class, ShortConv class), made once per process."""
    if not _CLASSES:
        lfm2_vl, hybrid = provider("lfm2_vl"), provider("hybrid")
        row = G._row_attention_class()

        class RowAttention(row):
            """d1_prefill_graph's RowAttention, unchanged for its (cos, sin, mask) context, plus the pair's PairCtx. The
            name stays RowAttention: D1Prefill checks it."""

            def forward(self, x, ctx, tree):
                if not isinstance(ctx, PairCtx):
                    if _SOFTMAX["dtype"] is torch.float32:
                        return super().forward(x, ctx, tree)
                    return self._row64(x, ctx)
                assert tree is None, "the pair runs no tree"
                i = self._d1_layer
                b, length, _ = x.shape
                hd, h, hkv = self.head_dim, self.heads, self.kv_heads
                q = self.q_layernorm(self.q_proj(x).view(b, length, h, hd)).transpose(1, 2)
                k = self.k_layernorm(self.k_proj(x).view(b, length, hkv, hd)).transpose(1, 2)
                v = self.v_proj(x).view(b, length, hkv, hd).transpose(1, 2)
                q, k = hybrid.rotate(q, ctx.cos, ctx.sin), hybrid.rotate(k, ctx.cos, ctx.sin)
                if ctx.mode == "state":
                    ctx.out[f"k_{i}"], ctx.out[f"v_{i}"] = k, v
                else:                                       # the state's keys and values, then the question's own
                    k = torch.cat([ctx.state[f"k_{i}"], k], dim=2)
                    v = torch.cat([ctx.state[f"v_{i}"], v], dim=2)
                keys = k.shape[2]
                n_rep = h // hkv
                k = torch.cat([k.reshape(hkv, 1, keys, hd)] * n_rep, dim=1).reshape(1, h, keys, hd)
                v = torch.cat([v.reshape(hkv, 1, keys, hd)] * n_rep, dim=1).reshape(1, h, keys, hd)
                scores = torch.matmul(q, k.transpose(2, 3)) * self.scale + ctx.mask
                p = torch.softmax(scores, dim=-1, dtype=_SOFTMAX["dtype"])
                y = torch.matmul(p, v)
                return self.out_proj(y.transpose(1, 2).flatten(2))

            def _row64(self, x, ctx):
                """--fp64 only: RowAttention.forward with the softmax in _SOFTMAX's dtype (round 6b's forward64)."""
                cos, sin, mask = ctx
                b, length, _ = x.shape
                hd, h, hkv = self.head_dim, self.heads, self.kv_heads
                q = self.q_layernorm(self.q_proj(x).view(b, length, h, hd)).transpose(1, 2)
                k = self.k_layernorm(self.k_proj(x).view(b, length, hkv, hd)).transpose(1, 2)
                v = self.v_proj(x).view(b, length, hkv, hd).transpose(1, 2)
                q, k = hybrid.rotate(q, cos, sin), hybrid.rotate(k, cos, sin)
                n_rep = h // hkv
                k = torch.cat([k.reshape(hkv, 1, length, hd)] * n_rep, dim=1).reshape(1, h, length, hd)
                v = torch.cat([v.reshape(hkv, 1, length, hd)] * n_rep, dim=1).reshape(1, h, length, hd)
                scores = torch.matmul(q, k.transpose(2, 3)) * self.scale + mask
                p = torch.softmax(scores, dim=-1, dtype=_SOFTMAX["dtype"])
                return self.out_proj(torch.matmul(p, v).transpose(1, 2).flatten(2))

        class PairShortConv(lfm2_vl.ShortConv):
            """The provider's ShortConv; with a PairCtx its forward also writes the conv tail (state step) or continues
            one (question step)."""

            def forward(self, x, pos, tree):
                if not isinstance(pos, PairCtx):
                    return super().forward(x, pos, tree)
                ctx, i = pos, self._d1_layer
                b, c, u = self.in_proj(x).transpose(1, 2).chunk(3, dim=1)
                bu = b * u                                                    # the conv input [1, C, T]
                if ctx.mode == "state":
                    ctx.out[f"conv_tail_{i}"] = torch.matmul(bu, ctx.tail_sel)   # [1, C, 2], oldest first
                    z = hybrid.causal_conv(bu, self.conv.weight, None)
                else:
                    z = F.conv1d(torch.cat([ctx.state[f"conv_tail_{i}"], bu], dim=-1), self.conv.weight,
                                 groups=bu.shape[1])
                if self.conv.bias is not None:
                    z = z + self.conv.bias[:, None]
                return self.out_proj((c * z).transpose(1, 2))

        _CLASSES.update(attention=RowAttention, conv=PairShortConv)
    return _CLASSES["attention"], _CLASSES["conv"]


def to_pair_form(lm: nn.Module) -> nn.Module:
    """Swap the pair-aware classes onto the operators (same objects, parameters and names) and number them."""
    attn_cls, conv_cls = _pair_classes()
    lfm2_vl = provider("lfm2_vl")
    for i, layer in enumerate(lm.layers):
        if layer.operator_name == "self_attn":
            assert isinstance(layer.self_attn, lfm2_vl.Attention), type(layer.self_attn)
            layer.self_attn.__class__ = attn_cls
            layer.self_attn._d1_layer = i
        else:
            assert type(layer.conv) in (lfm2_vl.ShortConv, conv_cls), type(layer.conv)
            assert layer.conv.conv.weight.shape[-1] - 1 == TAIL, layer.conv.conv.weight.shape
            layer.conv.__class__ = conv_cls
            layer.conv._d1_layer = i
    lm._d1_pair_form = True
    return lm


def load_pair_model(source: str | Path = SNAP) -> tuple[nn.Module, dict]:
    lm, info = G.load(str(source))
    return to_pair_form(lm), info


def state_names(lm: nn.Module) -> list[str]:
    names = []
    for i, layer in enumerate(lm.layers):
        names += [f"k_{i}", f"v_{i}"] if layer.operator_name == "self_attn" else [f"conv_tail_{i}"]
    return names


def _attention(lm):
    return next(layer.self_attn for layer in lm.layers if layer.operator_name == "self_attn")


class StatePrefill(nn.Module):
    """state_prefill_<Ls>: ids int32 [1, Ls] + valid float32 [1, Ls] -> the 38 state tensors (with_hidden: + "hidden"
    [1, Ls, d], for the RowCheck)."""

    def __init__(self, lm: nn.Module, Ls: int, with_hidden: bool = False):
        super().__init__()
        assert getattr(lm, "_d1_pair_form", False), "call to_pair_form(lm) first"
        attn = _attention(lm)
        self.lm, self.Ls, self.with_hidden, self.names = lm, Ls, with_hidden, state_names(lm)
        cos, sin = G.rope_tables(Ls, attn.head_dim, attn.theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("causal_const", G.causal_constant(Ls), persistent=False)
        self.register_buffer("cols", torch.tensor([float(c) for c in range(Ls)]).reshape(1, Ls, 1), persistent=False)
        self.register_buffer("tail_offsets", torch.tensor([-2.0, -1.0]).reshape(1, 1, TAIL), persistent=False)

    def forward(self, ids, valid):
        return self._run(self.lm.embed_tokens(ids), valid)

    def _run(self, h, valid):
        mask = self.causal_const + (1.0 - valid)[:, None, None, :] * NEG                   # D1Prefill.run's mask
        n = valid.sum(-1, keepdim=True)                                                    # [1, 1] real state tokens
        sel = torch.relu(1.0 - torch.abs(self.cols - (n.reshape(1, 1, 1) + self.tail_offsets)))   # [1, Ls, 2]
        ctx = PairCtx("state", self.cos, self.sin, mask, tail_sel=sel)
        for layer in self.lm.layers:
            h = layer(h, ctx, None)
        out = {name: ctx.out[name] for name in self.names}
        if self.with_hidden:
            out["hidden"] = self.lm.embedding_norm(h)
        return out


class StatePrefillEmbeds(StatePrefill):
    """Round 10, the embeds pair: state_prefill_<Ls>(embeds float32 [1, Ls, d], valid float32 [1, Ls]) -> the same 38
    state tensors. `embeds` = the host's float32 rows of the bfloat16 table (embed_table.safetensors, d1_tables.
    EmbedTable) at the state's tokens, the pad id's row on the pads; the graph holds no table (no EMBEDDING_LOOKUP)."""

    def forward(self, embeds, valid):
        return self._run(embeds, valid)


class RopeSelect(nn.Module):
    """cos, sin [1, Lq, 64] = one-hot [1, Lq, Lmax] @ table [Lmax, 128] (cos | sin): one FULLY_CONNECTED whose weights
    are the constant table."""

    def __init__(self, cos: torch.Tensor, sin: torch.Tensor):
        super().__init__()
        self.half = int(cos.shape[-1])
        self.register_buffer("table_t", torch.cat([cos, sin], dim=-1).t().contiguous(), persistent=False)  # [128, Lmax]

    def forward(self, sel):
        cs = F.linear(sel, self.table_t)
        return cs[..., : self.half], cs[..., self.half:]


class _QuestionStepBase(nn.Module):
    def __init__(self, lm: nn.Module, Ls: int, Lq: int, embeds: bool = False):
        super().__init__()
        assert getattr(lm, "_d1_pair_form", False), "call to_pair_form(lm) first"
        attn = _attention(lm)
        self.lm, self.Ls, self.Lq, self.head_dim, self.names = lm, Ls, Lq, attn.head_dim, state_names(lm)
        self.embeds_input = embeds          # round 10: `embeds` float32 [1, Lq, d] in place of `ids` (no table)
        lmax = Ls + Lq
        cos, sin = G.rope_tables(lmax, attn.head_dim, attn.theta)                         # [1, 1, Lmax, 64]
        self.rope = RopeSelect(cos[0, 0], sin[0, 0])
        self.register_buffer("pos_row", torch.tensor([float(p) for p in range(lmax)]).reshape(1, 1, lmax),
                             persistent=False)
        self.register_buffer("q_col", torch.tensor([float(j) for j in range(Lq)]).reshape(1, Lq, 1), persistent=False)
        # [state part: 0 | question part: causal 0 / -1e4]; a non-zero constant, so `const + row` stays an ADD
        self.register_buffer("mask_const", torch.cat([torch.zeros(1, 1, Lq, Ls), G.causal_constant(Lq)], dim=-1),
                             persistent=False)

    def input_names(self) -> list[str]:
        return ["embeds" if self.embeds_input else "ids", "valid", "state_valid"] + self.names

    def _forward(self, valid, state_valid, ids=None, embeds=None, **states):
        Lq, hd = self.Lq, self.head_dim
        h = embeds if self.embeds_input else self.lm.embed_tokens(ids)
        n = state_valid.sum(-1, keepdim=True)                                              # [1, 1]
        sel = torch.relu(1.0 - torch.abs(self.pos_row - (n.reshape(1, 1, 1) + self.q_col)))  # [1, Lq, Lmax]
        cos, sin = self.rope(sel)                                                          # [1, Lq, 64] each
        cos, sin = cos.reshape(1, 1, Lq, hd), sin.reshape(1, 1, Lq, hd)
        mask = self.mask_const + (1.0 - torch.cat([state_valid, valid], dim=-1))[:, None, None, :] * NEG
        ctx = PairCtx("question", cos, sin, mask, state=states)
        for layer in self.lm.layers:
            h = layer(h, ctx, None)
        return {"hidden": self.lm.embedding_norm(h)}


def QuestionStep(lm: nn.Module, Ls: int, Lq: int, embeds: bool = False) -> nn.Module:
    """question_step module whose forward takes every input by its own name (torch.export and the signature need the
    names; a **kwargs forward would hide them). embeds (round 10): the first input is `embeds` float32 [1, Lq, d] (the
    host's float32 rows of the question's own tokens, the pad id's row on the pads) in place of `ids`."""
    names = ["embeds" if embeds else "ids", "valid", "state_valid"] + state_names(lm)
    src = (f"def forward(self, {', '.join(names)}):\n"
           f"    return self._forward({', '.join(f'{n}={n}' for n in names)})\n")
    ns: dict = {}
    exec(compile(src, "<question_step_forward>", "exec"), ns)  # noqa: S102 - generated from the layer names
    cls = type(f"QuestionStep{'Embeds' if embeds else ''}_{Ls}_{Lq}", (_QuestionStepBase,), {"forward": ns["forward"]})
    return cls(lm, Ls, Lq, embeds)


def pad_inputs(ids, L: int, pad_id: int = PAD_ID) -> tuple[torch.Tensor, torch.Tensor]:
    """ids (may be empty) -> (int32 [1, L] right-padded with pad_id, valid float32 [1, L])."""
    n = len(ids)
    assert n <= L, (n, L)
    t = torch.full((1, L), pad_id, dtype=torch.int32)
    v = torch.zeros((1, L), dtype=torch.float32)
    if n:
        t[0, :n] = torch.tensor(list(ids), dtype=torch.int32)
        v[0, :n] = 1.0
    return t, v


def contract(lm: nn.Module, Ls: int, Lq: int, embeds: bool = False) -> dict:
    """The pair's signatures (names, shapes, dtypes) and rules, for the export check and host/contract.json. embeds
    (round 10): the first input of both signatures is `embeds` float32 [1, L, d] in place of `ids` int32 [1, L]."""
    attn = _attention(lm)
    d, hkv, hd = lm.embedding_norm.weight.shape[0], attn.kv_heads, attn.head_dim
    conv_dim = next(layer.conv.conv.weight.shape[0] for layer in lm.layers if layer.operator_name == "conv")
    states = []
    for i, layer in enumerate(lm.layers):
        if layer.operator_name == "self_attn":
            states += [{"name": f"k_{i}", "shape": [1, hkv, Ls, hd], "dtype": "float32"},
                       {"name": f"v_{i}", "shape": [1, hkv, Ls, hd], "dtype": "float32"}]
        else:
            states.append({"name": f"conv_tail_{i}", "shape": [1, conv_dim, TAIL], "dtype": "float32"})
    nbytes = sum(4 * int(np.prod(s["shape"])) for s in states)

    def first(L, what):
        if embeds:
            return {"name": "embeds", "shape": [1, L, d], "dtype": "float32",
                    "meaning": f"the float32 rows of embed_table.safetensors (bfloat16) at {what}, the token_ids.pad "
                               "row on the pads (right padding)"}
        return {"name": "ids", "shape": [1, L], "dtype": "int32",
                "meaning": f"{what}, right-padded with token_ids.pad"}

    return {
        "Ls": Ls, "Lq": Lq, "input": "embeds" if embeds else "ids",
        "signatures": {
            f"state_prefill_{Ls}": {
                "inputs": [first(Ls, "the state's tokens (the request's prefix text: BOS, user turn, state block, "
                                     "QUESTION line)"),
                           {"name": "valid", "shape": [1, Ls], "dtype": "float32", "meaning": "1.0 real / 0.0 pad"}],
                "outputs": states},
            f"question_step_{Ls}_{Lq}": {
                "inputs": [first(Lq, "one question's own tokens (the row after the state)"),
                           {"name": "valid", "shape": [1, Lq], "dtype": "float32", "meaning": "1.0 real / 0.0 pad"},
                           {"name": "state_valid", "shape": [1, Ls], "dtype": "float32",
                            "meaning": "the valid of the state_prefill call whose outputs are passed"}] + states,
                "outputs": [{"name": "hidden", "shape": [1, Lq, d], "dtype": "float32",
                             "meaning": "hidden states after the final RMSNorm at the question's positions; the answer "
                                        "slot is the question's last real token"}]}},
        "state": {"tensors": len(states), "bytes": nbytes,
                  "conv_tail": "the ShortConv conv input (b * u) at the last 2 real state tokens, oldest first, zero "
                               "columns before the first token",
                  "k_v": "attention keys after k_layernorm and RoPE, and values, at every state position (the pads are "
                         "masked by state_valid in question_step)"},
        "positions": "state 0..Ls-1; question n..n+Lq-1 with n = sum(state_valid) (the row form's positions)",
    }


# --------------------------------------------------------------------------- #
# --check
# --------------------------------------------------------------------------- #


def maxabs(a, b) -> float:
    return float(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).max())


def rel(p: Path) -> str:
    return str(p.relative_to(K)) if p.is_relative_to(K) else str(p)


def pick(n: int, sizes) -> int:
    return next(s for s in sizes if n <= s)


class Mods:
    """The torch modules by shape, built once (the buffers are small; the weights are the one model's). table (round 10,
    --embeds): a d1_tables.EmbedTable = the embeds pair (StatePrefillEmbeds / QuestionStep(embeds=True)) and the embeds
    row form (D1PrefillEmbeds), fed the host's float32 rows of the bfloat16 table at the padded ids (pads included)."""

    def __init__(self, lm, table=None):
        self.lm, self.cache, self.dtype, self.table = lm, {}, torch.float32, table

    def get(self, kind, *shape):
        key = (kind,) + shape
        if key not in self.cache:
            emb = self.table is not None
            make = {"state": lambda: (StatePrefillEmbeds if emb else StatePrefill)(self.lm, shape[0]),
                    "rowcheck": lambda: (StatePrefillEmbeds if emb else StatePrefill)(self.lm, shape[0], with_hidden=True),
                    "question": lambda: QuestionStep(self.lm, *shape, embeds=emb),
                    "row": lambda: G.graph(self.lm, shape[0], embeds=emb)}[kind]
            self.cache[key] = make().eval().requires_grad_(False).to(self.dtype)
        return self.cache[key]

    def to(self, dtype):
        """--fp64: the model, every cached module's buffers, and the modules built later."""
        self.dtype = dtype
        self.lm.to(dtype)
        for m in self.cache.values():
            m.to(dtype)

    def inp(self, ids_t):
        """The first graph input for padded ids int32 [1, L]: the ids, or (embeds) the table's float32 rows [1, L, d]."""
        if self.table is None:
            return ids_t
        return torch.from_numpy(np.ascontiguousarray(self.table.rows(ids_t[0].numpy())[None])).to(self.dtype)

    def first_name(self) -> str:
        return "ids" if self.table is None else "embeds"

    def state(self, ids, Ls):
        s_ids, s_valid = pad_inputs(ids, Ls)
        return self.get("state", Ls)(self.inp(s_ids), s_valid), s_valid

    def question(self, q_ids, Ls, Lq, s_valid, st):
        ids, valid = pad_inputs(q_ids, Lq)
        kw = {self.first_name(): self.inp(ids)}
        return self.get("question", Ls, Lq)(**kw, valid=valid, state_valid=s_valid, **st)["hidden"][0].numpy()

    def row(self, ids):
        L = pick(len(ids), ROW_BUCKETS)
        r_ids, r_valid = pad_inputs(ids, L)
        return self.get("row", L)(self.inp(r_ids), r_valid)["hidden"][0].numpy(), L


def compare(hq, hr, n, m, groups, ref_probs, H, table) -> dict:
    """The pair's question hidden hq [>= m, d] against the row form's hr [>= n + m, d] at the question's positions."""
    hq, hr_q = hq[:m], hr[n: n + m]
    p_pair, p_row = H.readout(hq[m - 1], table, groups), H.readout(hr_q[m - 1], table, groups)
    out = {"hidden_question_max_abs_vs_row": maxabs(hq, hr_q), "h_sel_max_abs_vs_row": maxabs(hq[m - 1], hr_q[m - 1]),
           "max_abs_dp_vs_row": maxabs(p_pair, p_row), "argmax_equal_row": int(np.argmax(p_pair)) == int(np.argmax(p_row)),
           "max_abs_hidden_question": float(np.abs(hq).max()), "finite": bool(np.isfinite(hq).all()),
           "probs_pair": [float(x) for x in p_pair], "probs_row": [float(x) for x in p_row]}
    if ref_probs is not None:
        out.update(max_abs_dp_vs_reference=maxabs(p_pair, ref_probs), row_max_abs_dp_vs_reference=maxabs(p_row, ref_probs),
                   argmax_equal_reference=int(np.argmax(p_pair)) == int(np.argmax(ref_probs)))
    return out


def state_diff(sa: dict, sb: dict, n: int) -> dict:
    """max |diff| per kind; k / v at the real positions 0..n-1 only."""
    d = {"conv_tail": 0.0, "kv_real": 0.0}
    for name in sa:
        a, b = sa[name].numpy(), sb[name].numpy()
        if name.startswith(("k_", "v_")):
            if n:
                d["kv_real"] = max(d["kv_real"], maxabs(a[:, :, :n], b[:, :, :n]))
        else:
            d["conv_tail"] = max(d["conv_tail"], maxabs(a, b))
    return d


def layer_trace(mods: Mods, lm, state_ids, q_ids, Ls, Lq) -> list[dict]:
    """max |dh| of the residual stream after every layer at the question's positions: question step vs row form."""
    got: dict = {}
    hooks = [layer.register_forward_hook(lambda mod, args, out, i=i: got.setdefault(i, []).append(out[0].numpy().copy()))
             for i, layer in enumerate(lm.layers)]
    try:
        st, s_valid = mods.state(state_ids, Ls)
        got.clear()
        mods.question(q_ids, Ls, Lq, s_valid, st)
        pair = {i: v[-1] for i, v in got.items()}
        got.clear()
        mods.row(list(state_ids) + list(q_ids))
        row = {i: v[-1] for i, v in got.items()}
    finally:
        for hk in hooks:
            hk.remove()
    n, m = len(state_ids), len(q_ids)
    return [{"layer": i, "kind": lm.layers[i].operator_name, "max_abs": maxabs(pair[i][:m], row[i][n: n + m]),
             "max_abs_h": float(np.abs(row[i][n: n + m]).max())} for i in sorted(pair)]


def run_request(mods: Mods, lm, rid: str, rs: list, ref: dict, H, table, idx: int, pairs=PAIRS) -> tuple[list, dict, dict]:
    n = rs[0]["state_len"]
    state = rs[0]["ids"][:n]
    assert all(r["state_len"] == n and r["ids"][:n] == state and r["split_equal"] for r in rs), rid
    rows, hsel, extra = [], {}, {}
    row_h = {}
    t0 = time.perf_counter()
    for r in rs:
        h, L = mods.row(r["ids"])
        row_h[r["qid"]] = (h, L)
        hsel[f"row/{rid}/{r['qid']}"] = h[r["row_len"] - 1].astype(np.float32)
    t_row = time.perf_counter() - t0
    ran = {r["qid"]: [] for r in rs}
    configs = [(Ls, Lq, f"Ls{Ls}_Lq{Lq}") for Ls, Lq in pairs if n <= Ls]
    todo = []
    for Ls, Lq, label in configs:
        qs = [r for r in rs if r["row_len"] - n <= Lq]
        if qs:
            todo.append((Ls, Lq, label, qs))
            for r in qs:
                ran[r["qid"]].append(label)
    left = [r for r in rs if not ran[r["qid"]]]
    by_bucket: dict = {}
    for r in left:
        by_bucket.setdefault((pick(n, LS_BUCKETS), pick(r["row_len"] - n, LQ_BUCKETS)), []).append(r)
    for (Ls, Lq), qs in sorted(by_bucket.items()):
        todo.append((Ls, Lq, "bucket", qs))
    for Ls, Lq, label, qs in todo:
        t = time.perf_counter()
        st, s_valid = mods.state(state, Ls)
        t_state = time.perf_counter() - t
        if rid in STATE_DUMP and label != "bucket":
            extra.setdefault("state_dumps", {})[f"Ls{Ls}"] = {k: v.numpy().copy() for k, v in st.items()}
        for r in qs:
            m = r["row_len"] - n
            t = time.perf_counter()
            hq = mods.question(r["ids"][n:], Ls, Lq, s_valid, st)
            t_q = time.perf_counter() - t
            key = f"{rid}/{r['qid']}"
            q_ref = ref.get(key)
            rec = {"key": key, "request": rid, "source": r["source"], "config": label, "Ls": Ls, "Lq": Lq, "n_state": n,
                   "n_question": m, "row_len": r["row_len"], "L_row": row_h[r["qid"]][1],
                   "near_tie_reference": bool(q_ref["near_tie"]) if q_ref else None,
                   "seconds_state": round(t_state, 3), "seconds_question": round(t_q, 3)}
            rec.update(compare(hq, row_h[r["qid"]][0], n, m, r["readout_ids"], q_ref["probs"] if q_ref else None, H, table))
            rows.append(rec)
            hsel[f"Ls{Ls}_Lq{Lq}/{key}"] = hq[m - 1].astype(np.float32)
    extra["seconds_row_form"] = round(t_row, 3)
    if idx < CHECK_N:
        extra["checks"] = request_checks(mods, rs[0], n, state)
    if rid in TRACE:
        Ls, Lq = next(((Ls, Lq) for Ls, Lq, label, qs in todo if label != "bucket"), (todo[0][0], todo[0][1]))
        extra["layer_trace"] = {"request": rid, "question": rs[0]["qid"], "Ls": Ls, "Lq": Lq,
                                "layers": layer_trace(mods, lm, state, rs[0]["ids"][n:], Ls, Lq)}
    return rows, hsel, extra


def request_checks(mods: Mods, r0: dict, n: int, state: list) -> dict:
    """RowCheck on the request's first row; state checks at Ls256 (and Ls128 vs Ls256 when n <= 128)."""
    out = {"request": r0["id"], "n_state": n}
    L = pick(r0["row_len"], ROW_BUCKETS)
    ids, valid = pad_inputs(r0["ids"], L)
    x = mods.inp(ids)
    hc = mods.get("rowcheck", L)(x, valid)["hidden"][0].numpy()
    hr = mods.get("row", L)(x, valid)["hidden"][0].numpy()
    out["rowcheck"] = {"L": L, "bit_equal_all_positions": bool(np.array_equal(hc, hr)), "max_abs": maxabs(hc, hr)}
    if n <= 256:
        st, s_valid = mods.state(state, 256)
        s_ids, _ = pad_inputs(state, 256)
        alt = s_ids.clone()
        alt[0, n:] = 124899                                  # <|im_start|> in every pad slot
        out["pad_content"] = state_diff(st, mods.get("state", 256)(mods.inp(alt), s_valid), n)
        out["pad_content_bit_equal"] = all(v == 0.0 for v in out["pad_content"].values())
        if n < 256:
            out["guard_off"] = state_diff(st, mods.get("state", 256)(mods.inp(s_ids), torch.ones_like(s_valid)), n)
        if n <= 128:
            out["pad_amount_128_vs_256"] = state_diff(st, mods.state(state, 128)[0], n)
        if n <= 64 and mods.table is not None:    # round 10: the Ls64 pair
            out["pad_amount_64_vs_256"] = state_diff(st, mods.state(state, 64)[0], n)
    return out


def edge_cases(mods: Mods, rows_all: list, H, table, pairs=PAIRS) -> list[dict]:
    """Per export pair: n = 0, 1, 2 and n = Ls (module docstring)."""
    out = []
    first = rows_all[0]
    assert first["id"] == "tv4_000"
    n0 = first["state_len"]
    for Ls, Lq in pairs:
        cases = []
        r = next(x for x in rows_all if x["row_len"] <= Lq)
        cases.append(("empty_state", r["id"] + "/" + r["qid"], [], r["ids"], r["readout_ids"]))
        for k in (1, 2):
            q = first["ids"][n0:]
            if len(q) <= Lq:
                cases.append((f"state_{k}_token", f"tv4_000/answer (first {k} state tokens)", first["ids"][:k], q,
                              first["readout_ids"]))
        r = next(x for x in rows_all if Ls < x["row_len"] <= Ls + Lq)
        cases.append(("state_exactly_Ls", f"{r['id']}/{r['qid']} (split at {Ls})", r["ids"][:Ls], r["ids"][Ls:],
                      r["readout_ids"]))
        for name, src, s_ids, q_ids, groups in cases:
            st, s_valid = mods.state(s_ids, Ls)
            hq = mods.question(q_ids, Ls, Lq, s_valid, st)
            hr, L = mods.row(list(s_ids) + list(q_ids))
            zero_cols = sorted({int((st[k][0].abs().sum(0) == 0).sum()) for k in st if k.startswith("conv_tail")})
            rec = {"key": f"edge {name}: {src}", "config": f"Ls{Ls}_Lq{Lq}", "Ls": Ls, "Lq": Lq, "case": name,
                   "from": src, "n_state": len(s_ids), "n_question": len(q_ids), "L_row": L,
                   "conv_tail_zero_columns_per_layer": zero_cols}
            rec.update(compare(hq, hr, len(s_ids), len(q_ids), groups, None, H, table))
            out.append(rec)
    return out


def agg(sel: list) -> dict:
    if not sel:
        return {"runs": 0}
    keys = {r["key"] for r in sel}
    with_ref = [r for r in sel if "max_abs_dp_vs_reference" in r]
    non = [r for r in with_ref if not r["near_tie_reference"]]
    return {"runs": len(sel), "questions": len(keys), "requests": len({r["request"] for r in sel}),
            "hidden_question_max_abs_vs_row": max(r["hidden_question_max_abs_vs_row"] for r in sel),
            "h_sel_max_abs_vs_row": max(r["h_sel_max_abs_vs_row"] for r in sel),
            "max_abs_dp_vs_row": max(r["max_abs_dp_vs_row"] for r in sel),
            "argmax_equal_row": f"{sum(r['argmax_equal_row'] for r in sel)}/{len(sel)}",
            "max_abs_dp_vs_reference": max((r["max_abs_dp_vs_reference"] for r in with_ref), default=None),
            "row_form_max_abs_dp_vs_reference": max((r["row_max_abs_dp_vs_reference"] for r in with_ref), default=None),
            "argmax_equal_reference_non_near_tie": f"{sum(r['argmax_equal_reference'] for r in non)}/{len(non)}",
            "nonfinite_runs": sum(not r["finite"] for r in sel),
            "runs_hidden_question_over_bar": sum(r["hidden_question_max_abs_vs_row"] > BAR_HIDDEN for r in sel),
            "runs_h_sel_over_bar": sum(r["h_sel_max_abs_vs_row"] > BAR_HIDDEN for r in sel),
            "runs_dp_over_bar": sum(r["max_abs_dp_vs_row"] > BAR_DP for r in sel),
            "hidden_question_p50_p99": [float(np.percentile([r["hidden_question_max_abs_vs_row"] for r in sel], q))
                                        for q in (50, 99)],
            "max_abs_hidden_question": max(r["max_abs_hidden_question"] for r in sel),
            "worst_hidden_key": max(sel, key=lambda r: r["hidden_question_max_abs_vs_row"])["key"],
            "seconds_question_median": float(np.median([r["seconds_question"] for r in sel]))}


def vs_ids_round9(rows: list, hsel: dict) -> dict:
    """Round 10: the embeds runs against round 9's ids runs (results/real_sharedstate_torch_check.json, its rows file
    and answer-slot npz): same (question, Ls, Lq) = |dp| and answer-slot max |dh| of the pair and of the row form; the
    new Ls64+Lq64 runs against round 9's Ls128+Lq64 run of the same question (other pad amount). Stop line:
    |dp| > 1e-5 = a difference beyond taking the table lookup out = a design error."""
    r9 = json.loads(OUT.read_text())
    rows_path = K / r9["rows_file"]
    assert sha256_file(rows_path) == r9["rows_file_sha256"], f"{rows_path.name} differs from round 9's record"
    by = {(r["key"], r["Ls"], r["Lq"]): r for r in json.loads(rows_path.read_text())["rows"]}
    h9 = np.load(K / r9["hsel_file"])
    same, cross = [], []
    for r in rows:
        key, Ls, Lq = r["key"], r["Ls"], r["Lq"]
        o = by.get((key, Ls, Lq))
        tgt = same
        if o is None and (Ls, Lq) == (64, 64):
            o, tgt = by.get((key, 128, 64)), cross
        if o is None:
            continue
        hk = f"Ls{o['Ls']}_Lq{o['Lq']}/{key}"
        tgt.append({"key": key, "config": r["config"], "vs": f"Ls{o['Ls']}_Lq{o['Lq']}",
                    "dp_pair": maxabs(r["probs_pair"], o["probs_pair"]),
                    "h_sel_pair": maxabs(hsel[f"Ls{Ls}_Lq{Lq}/{key}"], h9[hk]) if hk in h9.files else None})
    row_keys = sorted(k for k in hsel if k.startswith("row/"))
    row_h = [maxabs(hsel[k], h9[k]) for k in row_keys if k in h9.files]
    row_dp = [maxabs(r["probs_row"], by[(r["key"], r["Ls"], r["Lq"])]["probs_row"]) for r in rows
              if (r["key"], r["Ls"], r["Lq"]) in by]

    def stats(xs):
        dp = [x["dp_pair"] for x in xs]
        hs = [x["h_sel_pair"] for x in xs if x["h_sel_pair"] is not None]
        return {"runs": len(xs), "max_abs_dp_pair": max(dp, default=None), "max_abs_h_sel_pair": max(hs, default=None),
                "bit_equal_runs": sum(1 for x in xs if x["dp_pair"] == 0.0 and x["h_sel_pair"] == 0.0),
                "worst": max(xs, key=lambda x: x["dp_pair"])["key"] if xs else None}

    out = {"round9": rel(OUT), "round9_rows_sha256": r9["rows_file_sha256"], "same_shape": stats(same),
           "Ls64_Lq64_vs_round9_Ls128_Lq64": stats(cross),
           "row_form": {"questions": len(row_h), "max_abs_h_sel": max(row_h, default=None),
                        "bit_equal": sum(1 for x in row_h if x == 0.0), "max_abs_dp": max(row_dp, default=None)},
           "bar_max_abs_dp": BAR_DP}
    out["pass"] = all(v is not None and v <= BAR_DP for v in (out["same_shape"]["max_abs_dp_pair"],
                                                             out["Ls64_Lq64_vs_round9_Ls128_Lq64"]["max_abs_dp_pair"],
                                                             out["row_form"]["max_abs_dp"]))
    return out


def kp(x) -> Path:
    return Path(x) if Path(x).is_absolute() else K / x


def check(a) -> int:
    emb = bool(a.embeds)                                    # round 10: the embeds pair and row form (module docstring)
    pairs = PAIRS_EMBEDS if emb else PAIRS
    pair_dir = PAIR_DIR / "embeds" if emb else PAIR_DIR
    out = kp(a.out) if a.out else (OUT_EMBEDS if emb else OUT)
    parts = kp(a.parts) if a.parts else pair_dir / "torch_parts"
    final = {"rows": pair_dir / "torch_check_rows.json", "hsel": pair_dir / "torch_pair_hsel.npz"}
    if a.limit:                                             # a smoke run writes next to its parts, never the real names
        final = {"rows": parts / "smoke_rows.json", "hsel": parts / "smoke_hsel.npz"}
    for p in (out, *final.values()):
        assert not p.exists(), f"refusing to overwrite {p}"
    parts.mkdir(parents=True, exist_ok=True)
    pair_dir.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    t_all = time.time()
    torch.set_num_threads(a.threads)
    sys.path.insert(0, str(K / "host"))
    import d1_litert as H

    table = H.ReadoutTable.from_file(TABLE)
    ref = {f"{q['id']}/{q['qid']}": q for q in json.loads(REFERENCE.read_text())["questions"]}
    rows_all = [r for r in json.loads(ROWS.read_text())["rows"] if not (r.get("images") or r.get("image_expansion_pending"))]
    reqs: dict = {}
    for r in rows_all:
        reqs.setdefault(r["id"], []).append(r)
    order = list(reqs)[: a.limit or None]
    t0 = time.time()
    lm, info = load_pair_model(SNAP)
    load_s = time.time() - t0
    embed_check = None
    if emb:
        from d1_tables import EmbedTable

        et = EmbedTable(EMBED_TABLE)
        used = sorted({i for r in rows_all for i in r["ids"]} | {PAD_ID, 124899})
        w = lm.embed_tokens.weight.detach()[used].float().numpy()
        t_rows = et.rows(used)
        embed_check = {"table": rel(EMBED_TABLE), "table_sha256": sha256_file(EMBED_TABLE), "ids_checked": len(used),
                       "rows_bit_equal_to_the_model_s_float32_table": bool(np.array_equal(w.view(np.uint32),
                                                                                         t_rows.view(np.uint32)))}
        assert embed_check["rows_bit_equal_to_the_model_s_float32_table"], embed_check
        mods = Mods(lm, et)
    else:
        mods = Mods(lm)
    with torch.inference_mode():
        for idx, rid in enumerate(order):
            part = parts / f"{idx:04d}.json"
            if a.resume and part.exists() and (parts / f"{idx:04d}.npz").exists():
                continue
            t = time.time()
            rows, hsel, extra = run_request(mods, lm, rid, reqs[rid], ref, H, table, idx, pairs)
            dumps = extra.pop("state_dumps", {})
            for lab, st in dumps.items():
                np.savez((parts if a.limit else pair_dir) / f"torch_state_{lab}_{rid}.npz",
                         n_state=np.int64(reqs[rid][0]["state_len"]),
                         state_ids=np.asarray(reqs[rid][0]["ids"][: reqs[rid][0]["state_len"]], dtype=np.int32), **st)
            np.savez(parts / f"{idx:04d}.npz", **hsel)
            part.write_text(json.dumps({"request": rid, "rows": rows, "extra": extra, "seconds": round(time.time() - t, 2)})
                            + "\n")
            worst = max((r["hidden_question_max_abs_vs_row"] for r in rows), default=None)
            print(json.dumps({"i": idx, "req": rid, "q": len(reqs[rid]), "runs": len(rows), "max_dh": worst,
                              "max_dp": max((r["max_abs_dp_vs_row"] for r in rows), default=None),
                              "s": round(time.time() - t, 1)}), flush=True)
        edges = edge_cases(mods, rows_all, H, table, pairs)
    rows, extras, hsel = [], [], {}
    for idx, rid in enumerate(order):
        d = json.loads((parts / f"{idx:04d}.json").read_text())
        assert d["request"] == rid, (idx, d["request"], rid)
        rows += d["rows"]
        extras.append(d["extra"])
        hsel.update(dict(np.load(parts / f"{idx:04d}.npz")))
    checks = [e["checks"] for e in extras if "checks" in e]
    traces = [e["layer_trace"] for e in extras if "layer_trace" in e]
    by_cfg = {lab: agg([r for r in rows if r["config"] == lab]) for lab in [f"Ls{Ls}_Lq{Lq}" for Ls, Lq in pairs] + ["bucket"]}
    covered = {r["key"] for r in rows}
    want = {f"{r['id']}/{r['qid']}" for rid in order for r in reqs[rid]}
    ga = [c["guard_off"]["conv_tail"] for c in checks if "guard_off" in c]
    pa = [c["pad_amount_128_vs_256"] for c in checks if "pad_amount_128_vs_256" in c]
    pa64 = [c["pad_amount_64_vs_256"] for c in checks if "pad_amount_64_vs_256" in c]
    summary = {
        "all": agg(rows), "by_config": by_cfg, "questions_wanted": len(want), "questions_covered": len(covered & want),
        "edge_cases": [{k: v for k, v in e.items() if k not in ("probs_pair", "probs_row")} for e in edges],
        "rowcheck": {"rows": len(checks), "bit_equal": sum(c["rowcheck"]["bit_equal_all_positions"] for c in checks),
                     "max_abs": max((c["rowcheck"]["max_abs"] for c in checks), default=None)},
        "state_checks": {
            "pad_content": {"requests": sum("pad_content" in c for c in checks),
                            "bit_equal": sum(bool(c.get("pad_content_bit_equal")) for c in checks)},
            "guard_off_control": {"requests": len(ga), "min_conv_tail_move": min(ga, default=None)},
            "pad_amount_128_vs_256": {"requests": len(pa), "conv_tail": max((x["conv_tail"] for x in pa), default=None),
                                      "kv_real": max((x["kv_real"] for x in pa), default=None)},
            "pad_amount_64_vs_256": {"requests": len(pa64), "conv_tail": max((x["conv_tail"] for x in pa64), default=None),
                                     "kv_real": max((x["kv_real"] for x in pa64), default=None)}},
        "layer_traces": traces,
        "bar": {"hidden_question_max_abs_vs_row": BAR_HIDDEN, "max_abs_dp_vs_row": BAR_DP,
                "rowcheck": "bit-equal", "nonfinite": 0},
        "over_bar_hidden": sorted(({"key": r["key"], "config": r["config"],
                                    "hidden_question": r["hidden_question_max_abs_vs_row"],
                                    "h_sel": r["h_sel_max_abs_vs_row"], "dp": r["max_abs_dp_vs_row"],
                                    "max_abs_hidden_question": r["max_abs_hidden_question"]}
                                   for r in rows + edges if r["hidden_question_max_abs_vs_row"] > BAR_HIDDEN),
                                  key=lambda x: -x["hidden_question"]),
    }
    runs = rows + edges
    others = (summary["questions_covered"] == len(want)
              and summary["rowcheck"]["bit_equal"] == len(checks)
              and summary["state_checks"]["pad_content"]["bit_equal"] == summary["state_checks"]["pad_content"]["requests"]
              and all(x > 0 for x in ga))
    if emb:     # round 10: the round-9 ruling's gate (answer slot, |dp|; the question positions recorded, alert line)
        summary["embed_table_check"] = embed_check
        summary["bar"] = {"h_sel_max_abs_vs_row": BAR_HIDDEN, "max_abs_dp_vs_row": BAR_DP,
                          "hidden_question_alert": BAR_HIDDEN_ALERT, "rowcheck": "bit-equal", "nonfinite": 0,
                          "vs_ids_round9_max_abs_dp": BAR_DP,
                          "why": "round 9 ruling (2026-10-09 03:2x): the question positions' hidden is "
                                 "this model's float32 rounding (results/real_sharedstate_torch_fp64.json), recorded "
                                 "with an alert at 2.5e-4; the gate is |dp| and the answer slot"}
        summary["vs_ids_round9"] = vs_ids_round9(rows, hsel)
        summary["pass"] = bool(
            others and all(r["h_sel_max_abs_vs_row"] <= BAR_HIDDEN and r["max_abs_dp_vs_row"] <= BAR_DP and r["finite"]
                           and r["argmax_equal_row"] and r["hidden_question_max_abs_vs_row"] <= BAR_HIDDEN_ALERT
                           for r in runs)
            and summary["vs_ids_round9"]["pass"])
    else:
        summary["pass"] = bool(
            others and all(r["hidden_question_max_abs_vs_row"] <= BAR_HIDDEN and r["max_abs_dp_vs_row"] <= BAR_DP
                           and r["finite"] and r["argmax_equal_row"] for r in runs))
    final["rows"].write_text(json.dumps({"what": ("round 10: the embeds pair (torch) vs the embeds row form" if emb else
                                                  "round 9: the pair (torch) vs the row form") + ", one row per question "
                                                 "and pair (scripts/d1_shared_state.py --check" + (" --embeds)" if emb
                                                                                                     else ")"),
                                         "rows": rows,
                                         "edge_cases": edges, "request_checks": checks}) + "\n")
    np.savez(final["hsel"], **hsel)
    doc = {"what": ("round 10 design 3: the embeds shared-state pair in torch (StatePrefillEmbeds once, then "
                    "QuestionStep(embeds=True) per question, fed the host's float32 rows of the bfloat16 table) vs the "
                    "embeds row form (D1PrefillEmbeds on state + question as one row), and vs round 9's ids pair, "
                    "float32 CPU") if emb else
                   ("round 9 acceptance 1: the shared-state pair in torch (two phases: state once, then each question) "
                    "vs the row form (D1Prefill on state + question as one row), float32 CPU"),
           "started_at": started, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "seconds_wall": round(time.time() - t_all, 1), "load_seconds": round(load_s, 1), "threads": a.threads,
           "limit": a.limit or None, "source": info, "pairs": [list(p) for p in pairs],
           "input": "embeds" if emb else "ids",
           "buckets": {"Ls": LS_BUCKETS, "Lq": LQ_BUCKETS, "row": ROW_BUCKETS},
           "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "numpy")},
           "rows_json_sha256": sha256_file(ROWS), "reference_sha256": sha256_file(REFERENCE),
           "table_sha256": sha256_file(TABLE), "summary": summary,
           "rows_file": rel(final["rows"]), "rows_file_sha256": sha256_file(final["rows"]),
           "hsel_file": rel(final["hsel"]),
           "state_dumps": sorted(rel(p) for p in (parts if a.limit else pair_dir).glob("torch_state_*.npz"))}
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: summary.get(k) for k in ("all", "questions_covered", "rowcheck", "state_checks", "vs_ids_round9",
                                                   "pass")}, indent=1))
    print(json.dumps(by_cfg, indent=1))
    if not a.limit:
        shutil.rmtree(parts)
    return 0 if summary["pass"] else 1


def _rmsnorm_dtype(self, x):
    """--fp64 only: hybrid.RMSNorm.forward in x's own dtype (the provider's code computes it in float32)."""
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)) * self.weight


def _rotate_dtype(x, cos, sin):
    """--fp64 only: hybrid.rotate in x's own dtype (the provider's code computes it in float32)."""
    a, b = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-b, a), dim=-1) * sin


def fp64_check(a) -> int:
    """--fp64 <request ids>: is a float32 difference between the pair and the row form rounding or a design error?
    For each question (the first --fp64-questions of each request; the first export pair that holds it, else the
    smallest bucket pair): the pair and the row form in float32 (as --check), then the same in pure float64 (the model
    and every module buffer in float64; the provider's RMSNorm and rotate, which compute in float32, and the softmax run
    in float64 for this test only; the RoPE tables are the float32 values in float64 in both forms). Equal functions
    agree in float64 to float64 rounding; a design error shows the same size in float64 as in float32. Also each form's
    float32 against the row form's float64 (= each form's own float32 rounding), the layer traces in both precisions,
    and the read-out |dp|. -> --out64 (results/real_sharedstate_torch_fp64.json, never overwritten)."""
    out = Path(a.out64) if Path(a.out64).is_absolute() else K / a.out64
    assert not out.exists(), f"refusing to overwrite {out}"
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    torch.set_num_threads(a.threads)
    sys.path.insert(0, str(K / "host"))
    import d1_litert as H

    table = H.ReadoutTable.from_file(TABLE)
    rows_all = [r for r in json.loads(ROWS.read_text())["rows"] if not (r.get("images") or r.get("image_expansion_pending"))]
    reqs: dict = {}
    for r in rows_all:
        reqs.setdefault(r["id"], []).append(r)
    cases = []
    for rid in [x for x in a.fp64.split(",") if x]:
        rs = reqs[rid]
        n = rs[0]["state_len"]
        for r in rs[: a.fp64_questions]:
            m = r["row_len"] - n
            Ls, Lq = next(((Ls, Lq) for Ls, Lq in PAIRS if n <= Ls and m <= Lq),
                          (pick(n, LS_BUCKETS), pick(m, LQ_BUCKETS)))
            cases.append({"key": f"{rid}/{r['qid']}", "state": r["ids"][:n], "own": r["ids"][n:],
                          "groups": r["readout_ids"], "Ls": Ls, "Lq": Lq})
    t0 = time.time()
    lm, info = load_pair_model(SNAP)
    mods = Mods(lm)
    hybrid = provider("hybrid")
    res: dict = {32: {}, 64: {}}

    def run_all(bits):
        for c in cases:
            st, s_valid = mods.state(c["state"], c["Ls"])
            hq = mods.question(c["own"], c["Ls"], c["Lq"], s_valid, st)
            hr, L = mods.row(list(c["state"]) + list(c["own"]))
            tr = layer_trace(mods, lm, c["state"], c["own"], c["Ls"], c["Lq"])
            res[bits][c["key"]] = (hq.astype(np.float64), hr.astype(np.float64), L, tr)
            print(json.dumps({"bits": bits, "key": c["key"], "s": round(time.time() - t0, 1)}), flush=True)

    with torch.inference_mode():
        run_all(32)
    saved = (hybrid.RMSNorm.forward, hybrid.rotate)
    hybrid.RMSNorm.forward, hybrid.rotate, _SOFTMAX["dtype"] = _rmsnorm_dtype, _rotate_dtype, torch.float64
    try:
        mods.to(torch.float64)
        with torch.inference_mode():
            run_all(64)
    finally:
        hybrid.RMSNorm.forward, hybrid.rotate = saved
        _SOFTMAX["dtype"] = torch.float32
    out_rows = []
    for c in cases:
        k, n, m = c["key"], len(c["state"]), len(c["own"])
        q32, r32, L, tr32 = res[32][k]
        q64, r64, _, tr64 = res[64][k]
        pq32, rq32, pq64, rq64 = q32[:m], r32[n: n + m], q64[:m], r64[n: n + m]
        probs = {name: H.readout(h[m - 1], table, c["groups"]) for name, h in
                 (("pair32", pq32), ("row32", rq32), ("pair64", pq64), ("row64", rq64))}
        out_rows.append({
            "key": k, "Ls": c["Ls"], "Lq": c["Lq"], "n_state": n, "n_question": m, "L_row": L,
            "hidden_question": {"pair32_vs_row32": maxabs(pq32, rq32), "pair64_vs_row64": maxabs(pq64, rq64),
                                "row32_vs_row64": maxabs(rq32, rq64), "pair32_vs_row64": maxabs(pq32, rq64),
                                "max_abs_h_row64": float(np.abs(rq64).max())},
            "h_sel": {"pair32_vs_row32": maxabs(pq32[m - 1], rq32[m - 1]),
                      "pair64_vs_row64": maxabs(pq64[m - 1], rq64[m - 1]),
                      "row32_vs_row64": maxabs(rq32[m - 1], rq64[m - 1]),
                      "pair32_vs_row64": maxabs(pq32[m - 1], rq64[m - 1])},
            "max_abs_dp": {"pair32_vs_row32": maxabs(probs["pair32"], probs["row32"]),
                           "pair64_vs_row64": maxabs(probs["pair64"], probs["row64"]),
                           "row32_vs_row64": maxabs(probs["row32"], probs["row64"]),
                           "pair32_vs_row64": maxabs(probs["pair32"], probs["row64"])},
            "layer_trace": [{"layer": x["layer"], "kind": x["kind"], "float32": x["max_abs"], "float64": y["max_abs"],
                             "max_abs_h": x["max_abs_h"]} for x, y in zip(tr32, tr64)]})
    worst = lambda part, key: max(r[part][key] for r in out_rows)
    summary = {part: {key: worst(part, key) for key in ("pair32_vs_row32", "pair64_vs_row64", "row32_vs_row64",
                                                        "pair32_vs_row64")} for part in ("hidden_question", "h_sel",
                                                                                         "max_abs_dp")}
    doc = {"what": "round 9: the pair vs the row form in float32 and in pure float64 (scripts/d1_shared_state.py --fp64)",
           "started_at": started, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "seconds_wall": round(time.time() - t0, 1), "threads": a.threads, "source": info,
           "float64": "lm.to(float64) + every module buffer; RMSNorm, rotate and the softmax in float64 for this test "
                      "(the provider's RMSNorm / rotate compute in float32, the export form's softmax is float32); the "
                      "RoPE tables are the float32 values in both forms",
           "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "numpy")},
           "summary": summary, "rows": out_rows}
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="the torch check (module docstring)")
    ap.add_argument("--fp64", default="", help="request ids for the float64 test (fp64_check)")
    ap.add_argument("--fp64-questions", type=int, default=1)
    ap.add_argument("--out64", default="results/real_sharedstate_torch_fp64.json")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="the first N requests only (a smoke run; --out elsewhere)")
    ap.add_argument("--resume", action="store_true", help="skip the requests whose parts exist")
    ap.add_argument("--out", default="", help="default results/real_sharedstate[_embeds]_torch_check.json")
    ap.add_argument("--parts", default="", help="default cache/real/pair[/embeds]/torch_parts")
    ap.add_argument("--embeds", action="store_true", help="round 10: the embeds pair and row form (module docstring)")
    a = ap.parse_args()
    if a.fp64:
        return fp64_check(a)
    if a.check:
        return check(a)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
