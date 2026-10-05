"""KevPrefill: the exported graph = one state-free causal row prefill.

    forward(ids: int32 [1, L], valid: float32 [1, L]) -> {"hidden": float32 [1, L, d]}   (final RMSNorm applied; d = 1024 on
    0.8B, 2560 on 4B: `--model 4b` in the calling script selects merged/kev-4b-v1.0 through r2_common)

Inside: position_ids = 0..L-1 (constant buffer, [1, L] int64; the patched text model stacks it to the 4-plane M-RoPE
form), mask4d = causal_const + (1 - valid)[:, None, None, :] * -1e4 (causal_const [1, 1, L, L] float32 from a python
list: 0 on and below the diagonal, -1e4 above), then
    text_model(input_ids=ids, attention_mask=mask4d, position_ids=position_ids, use_cache=False, past_key_values=None)
A 4-D float mask passes create_causal_mask as-is and makes create_recurrent_attention_mask return None, so the
GatedDeltaNet layers get the pad guard from `valid` instead: the wrapper sets text_model._kev_valid = valid, and the
patched text model stashes it on every PatchedQwen3_5GatedDeltaNet (as `_litert_valid`, the name the guarded forward
reads). Pads (id 248044) sit only after the real tokens. No KV / conv / recurrent state I/O.

Model = the merged fp32 checkpoint loaded through AutoModel after the three classes of kev_qwen35_patch.py are swapped
into transformers' modeling module; attention = `kev_eager` (GQA repeat by concat, rank-4 BMM, additive mask input)
registered in ALL_ATTENTION_FUNCTIONS. On 4B the GatedDeltaNet has 32 value heads over 16 key heads (ratio 2), so the
patch's head interleave runs; `count_interleave()` wraps `_litert_interleave_heads` in the guarded forward's globals to
prove it is called (twice per linear-attention layer: query and key)."""
import sys

import torch
from torch import nn

import kev_qwen35_patch as P   # import runs the rank-4 kernel gate and the five anchor asserts
from r2_common import CKPT, HIDDEN, PAD_ID
from transformers import AutoModel
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_5 import modeling_qwen3_5 as M

NEG = -1e4
ATTN = "kev_eager"


def kev_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """eager_attention_forward without BROADCAST_TO: key/value [1, Hkv, L, D] -> [Hkv, 1, L, D] -> concat n_rep copies on
    dim 1 -> [Hkv, n_rep, L, D] -> [1, Hkv * n_rep, L, D] (the order of repeat_kv: kv head h serves query heads
    h*n_rep .. h*n_rep+n_rep-1). scores = q @ k^T * scaling + mask (rank-4 BMM, mask [1, 1, L, L] additive), softmax in
    float32, out = p @ v -> [1, L, H, D]."""
    n_rep = module.num_key_value_groups
    b, hkv, length, dim = key.shape
    assert b == 1, "batch 1 only (the batch dim is folded to keep the copy axis inside rank 4)"
    key_states = torch.cat([key.reshape(hkv, 1, length, dim)] * n_rep, dim=1).reshape(1, hkv * n_rep, length, dim)
    value_states = torch.cat([value.reshape(hkv, 1, length, dim)] * n_rep, dim=1).reshape(1, hkv * n_rep, length, dim)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


def count_interleave():
    """Wrap the guarded forward's `_litert_interleave_heads` with a call counter (same function underneath, so the
    traced graph and the numbers are unchanged). -> the counter dict {"calls", "ratios", "shapes"}."""
    g = P.PatchedQwen3_5GatedDeltaNet.forward.__globals__
    inner = getattr(g["_litert_interleave_heads"], "__wrapped__", g["_litert_interleave_heads"])
    counter = {"calls": 0, "ratios": [], "shapes": []}

    def counted(x, r):
        counter["calls"] += 1
        if len(counter["ratios"]) < 4:
            counter["ratios"].append(int(r))
            counter["shapes"].append(list(x.shape))
        return inner(x, r)

    counted.__wrapped__ = inner
    g["_litert_interleave_heads"] = counted
    return counter


def causal_constant(L):
    rows = [[0.0 if j <= i else NEG for j in range(L)] for i in range(L)]
    return torch.tensor(rows, dtype=torch.float32).reshape(1, 1, L, L)


def swap_classes():
    """Swap the patch classes into the modeling module (and into the lazy package if it already cached a name)."""
    pkg = sys.modules["transformers.models.qwen3_5"]
    swaps = {"Qwen3_5TextModel": P.PatchedQwen3_5TextModel, "Qwen3_5GatedDeltaNet": P.PatchedQwen3_5GatedDeltaNet,
             "Qwen3_5TextRotaryEmbedding": P.PatchedQwen3_5TextRotaryEmbedding}
    cached = []
    for name, cls in swaps.items():
        setattr(M, name, cls)
        if name in vars(pkg):
            cached.append(name)
            setattr(pkg, name, cls)
    return {"swapped": sorted(swaps), "package_cache_also_swapped": cached}


def load_text_model(ckpt=CKPT):
    ALL_ATTENTION_FUNCTIONS.register(ATTN, kev_eager)
    swap_info = swap_classes()
    model = AutoModel.from_pretrained(str(ckpt), dtype=torch.float32, attn_implementation=ATTN).eval()
    if model.config._attn_implementation != ATTN:   # fallback: set the attribute directly
        model.config._attn_implementation = ATTN
    assert type(model) is P.PatchedQwen3_5TextModel, type(model)
    assert type(model.rotary_emb) is P.PatchedQwen3_5TextRotaryEmbedding, type(model.rotary_emb)
    gdn = [m for m in model.modules() if isinstance(m, M.Qwen3_5GatedDeltaNet)]
    attn = [m for m in model.modules() if isinstance(m, M.Qwen3_5Attention)]
    assert gdn and all(type(m) is P.PatchedQwen3_5GatedDeltaNet for m in gdn)
    assert all(m.chunk_gated_delta_rule is P._rank4_chunk_gated_delta_rule for m in gdn)
    assert all(m.config._attn_implementation == ATTN for m in attn)
    info = {**swap_info, "model_class": type(model).__name__, "gdn_layers": len(gdn), "attention_layers": len(attn),
            "attn_implementation": model.config._attn_implementation, "num_key_value_groups": attn[0].num_key_value_groups,
            "chunk_kernel": P._rank4_chunk_gated_delta_rule.__name__, "hidden_size": model.config.hidden_size,
            "gdn_heads": {"k": gdn[0].num_k_heads, "v": gdn[0].num_v_heads, "ratio": gdn[0].num_v_heads // gdn[0].num_k_heads},
            "attention_heads": {"q": model.config.num_attention_heads, "kv": model.config.num_key_value_heads,
                                "head_dim": attn[0].head_dim}}
    assert model.config.hidden_size == HIDDEN, (model.config.hidden_size, HIDDEN)
    return model, info


class KevPrefill(nn.Module):
    def __init__(self, text_model, L):
        super().__init__()
        self.text_model = text_model
        self.L = L
        self.register_buffer("position_ids", torch.arange(L, dtype=torch.long).reshape(1, L), persistent=False)
        self.register_buffer("causal_const", causal_constant(L), persistent=False)

    def forward(self, ids, valid):
        self.text_model._kev_valid = valid
        mask4d = self.causal_const + (1.0 - valid)[:, None, None, :] * NEG
        out = self.text_model(input_ids=ids, attention_mask=mask4d, position_ids=self.position_ids, use_cache=False,
                              past_key_values=None)
        return {"hidden": out.last_hidden_state}


def row_inputs(row_ids, L):
    """One question row -> (ids int32 [1, L], valid float32 [1, L]); right pad with 248044."""
    n = len(row_ids)
    assert n <= L, (n, L)
    ids = torch.full((1, L), PAD_ID, dtype=torch.int32)
    ids[0, :n] = torch.tensor(row_ids, dtype=torch.int32)
    valid = torch.zeros((1, L), dtype=torch.float32)
    valid[0, :n] = 1.0
    return ids, valid


def contract(L, head=None):
    doc = {
        "graph": "KevPrefill (state-free causal row prefill, fp32)", "L": L,
        "inputs": [{"name": "ids", "shape": [1, L], "dtype": "int32", "meaning": "row token ids, right-padded with 248044"},
                   {"name": "valid", "shape": [1, L], "dtype": "float32", "meaning": "1.0 = real token, 0.0 = pad"}],
        "outputs": [{"name": "hidden", "shape": [1, L, HIDDEN], "dtype": "float32",
                     "meaning": "last_hidden_state (after the final RMSNorm), every position"}],
        "pad_id": PAD_ID, "pad_rule": "pads only after the last real token (right padding)",
        "positions": "constant 0..L-1 (row form: each question row starts at position 0)",
        "attention_mask": "mask4d = causal_const + (1 - valid)[:, None, None, :] * -1e4; causal_const = 0 where key <= query else -1e4",
        "gated_deltanet_pad_guard": "valid zeroes pad hidden states before the projections, sets pad decay to identity "
                                    "(a -> -30 pre-softplus) and re-zeroes pad q/k/v after the causal conv",
        "state_io": "none (every row starts from zero conv / recurrent state)",
    }
    if head is not None:
        doc["readout"] = {
            "where": "host, float32", "rows": "h_dec = hidden[0, decide_idx]; h_opts = hidden[0, opt_idx] (the </opt> tokens)",
            "formula": "z = ((h_opts @ Wk^T + bk) @ (h_dec @ Wq^T + bq)) / sqrt(head_dim) / T; probs = softmax(z)",
            "head_file": head.info["path"], "head_sha256": head.info["sha256"], "head_shapes": head.info["shapes"],
            "head_dim": head.head_dim, "temperature": head.T}
    return doc
