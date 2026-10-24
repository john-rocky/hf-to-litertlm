"""Process-local patches that turn litert-torch 0.10.0's export_hf text rail (externalize_embedder +
single_token_embedder) into WeVisOne-VL-2B's wide-input fast_vlm decoder (docs/deepstack-wide-embedding.md).
Nothing on disk is modified: install() rebinds methods on the export_hf classes and on the loaded model instance.

Embedding width WIDE = 8196 per token:
    [0, 2048)      decoder input embedding (text lookup, or the image merger output)
    [2048, 4096)   deepstack 0 (merger of ViT block 5),  added after decoder layer 0
    [4096, 6144)   deepstack 1 (block 11),               added after decoder layer 1
    [6144, 8192)   deepstack 2 (block 17),               added after decoder layer 2
    8192           image-token flag (the adapter writes 1, the EMBEDDER 0)
    8193 .. 8195   0
Text tokens carry zeros in [2048, 8196): adding them is the identity, so the wide form computes HF's
`_deepstack_process` (add at the image positions only) exactly.

Positions (the runtime's 1-D input_pos p, the flag f in {0, 1}; P = 15 = the fixed system turn + "user\\n" +
<|vision_start|>, a G x G merged grid, N = G * G, D = N - G):
    image token (f = 1):  (t, h, w) = (P, P + (p - P) // G, P + (p - P) % G)
    text token  (f = 0):  t = h = w = p - D * [p >= P + N]
= transformers' Qwen3VLModel.get_rope_index for one image after P tokens (checked on 85,542 positions), and for a text-only
conversation the plain positions up to P + N - 1 (then p - D: the documented cost). Steps are relu(x) - relu(x - 1)
on integer-valued floats (never clamp: RELU_0_TO_1 is refused by the GPU delegates); the floor division by G is a sum
of G - 1 steps; the channels recombine per frequency with constant 0/1-masked inv_freq vectors as
Qwen3VLTextRotaryEmbedding.apply_interleaved_mrope (mrope_section [24, 20, 20], interleaved). Everything is rank 3.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# litert-torch 0.10.0's export_hf imports tensorflow.lite.python.schema_py_generated (model_ext/bonsai_flux2.py:41)
# without declaring TensorFlow, and TensorFlow cannot share a process with litert-converter: the one-line shim
# `from ai_edge_litert.schema_py_generated import *` stands in (memory litert-torch-head-export-tf-shim; the copy of
# qwen3_gpuopt_work/reexport_20261013/shim). On sys.path here because the tools run with python -I.
TF_SHIM = str(Path(__file__).resolve().parent / "tfshim")
if TF_SHIM not in sys.path:
    sys.path.insert(0, TF_SHIM)

HID = 2048
WIDE = 8196
FLAG = 8192
P_PREFIX = 15
MROPE_SECTION = (24, 20, 20)

STATE = {"emb": None}


def _step(x):
    return torch.relu(x) - torch.relu(x - 1.0)


class Grid:
    def __init__(self, g, prefix=P_PREFIX):
        self.P, self.G = int(prefix), int(g)
        self.N = self.G * self.G
        self.D = self.N - self.G

    def __repr__(self):
        return f"Grid(P={self.P}, G={self.G}, N={self.N}, D={self.D})"


def flagged_thw(p, flag, grid):
    """p, flag: float tensors of one shape (integer-valued p, flag in {0, 1}). Returns (t, h, w)."""
    P, N, G, D = float(grid.P), float(grid.N), grid.G, float(grid.D)
    text = p - D * _step(p - (P + N - 1.0))
    q = p - P
    row = _step(q - float(G - 1))
    for k in range(2, G):
        row = row + _step(q - float(k * G - 1))
    col = q - float(G) * row
    keep = 1.0 - flag
    t = flag * P + keep * text
    h = flag * (P + row) + keep * text
    w = flag * (P + col) + keep * text
    return t, h, w


def reference_thw(p, f, grid):
    """Integer reference of flagged_thw (plain piecewise formula)."""
    if f:
        q = p - grid.P
        return grid.P, grid.P + q // grid.G, grid.P + q % grid.G
    v = p - grid.D if p >= grid.P + grid.N else p
    return v, v, v


def channel_masks(n_freq, section=MROPE_SECTION):
    owner = [0] * n_freq
    for dim, offset in enumerate((1, 2), start=1):
        for j in range(offset, section[dim] * 3, 3):
            owner[j] = dim
    return [[1.0 if owner[j] == c else 0.0 for j in range(n_freq)] for c in range(3)]


def flagged_cos_sin(position_ids, flag, inv_freq, grid, attention_scaling=1.0, dtype=torch.float32):
    """position_ids [1, T] (int), flag [1, T, 1] -> cos, sin [1, T, 2 * n_freq]."""
    p = position_ids.float()[:, :, None]                           # [1, T, 1]
    t, h, w = flagged_thw(p, flag.float(), grid)
    m = channel_masks(inv_freq.shape[0])
    inv = inv_freq.float()
    inv_t = inv * torch.tensor(m[0], dtype=torch.float32, device=inv.device)
    inv_h = inv * torch.tensor(m[1], dtype=torch.float32, device=inv.device)
    inv_w = inv * torch.tensor(m[2], dtype=torch.float32, device=inv.device)
    freqs = t * inv_t + h * inv_h + w * inv_w                      # [1, T, n_freq]
    emb = torch.cat((freqs, freqs), dim=-1)
    return (emb.cos() * attention_scaling).to(dtype), (emb.sin() * attention_scaling).to(dtype)


def install_model(model, grid):
    """On the loaded Qwen3ForCausalLM: the flagged derived rotary + the deepstack adds after layers 0..2."""
    inner = model.model
    rotary = inner.rotary_emb
    assert float(rotary.inv_freq.abs().max()) > 0, "rotary inv_freq is zero (meta-load trap)"

    def rotary_forward(x, position_ids):
        emb = STATE["emb"]
        assert emb is not None, "the wide forward did not set the embeddings"
        return flagged_cos_sin(position_ids, emb[:, :, FLAG:FLAG + 1], rotary.inv_freq, grid,
                               rotary.attention_scaling, dtype=x.dtype)
    rotary.forward = rotary_forward
    handles = []
    for i in range(3):
        lo, hi = HID * (i + 1), HID * (i + 2)

        def hook(module, args, out, lo=lo, hi=hi):
            return out + STATE["emb"][:, :, lo:hi]
        handles.append(inner.layers[i].register_forward_hook(hook))

    def undo():
        rotary.__dict__.pop("forward", None)
        for h in handles:
            h.remove()
    return undo


def install_classes():
    """Wide inputs on the external-embedder prefill / decode modules and the EMBEDDER's zero fill."""
    from litert_torch.generative.export_hf.core.external_emb import exportable_module as ext
    undo = []
    for cls in (ext.LiteRTExportableModuleForDecoderOnlyLMPrefillExternalEmbedder,
                ext.LiteRTExportableModuleForDecoderOnlyLMGenerateExternalEmbedder):
        orig_fwd, orig_inp = cls.forward, cls._get_input

        def forward(self, embeddings, input_pos, kv_cache, mask, local_mask=None, _orig=orig_fwd, **kwargs):
            STATE["emb"] = embeddings
            try:
                return _orig(self, embeddings[:, :, :HID], input_pos, kv_cache, mask, local_mask=local_mask,
                             **kwargs)
            finally:
                STATE["emb"] = None

        def get_input(self, batch_size, length, length_dim, model_config, _orig=orig_inp):
            inputs, dyn = _orig(self, batch_size, length, length_dim, model_config)
            e = inputs["embeddings"]
            inputs["embeddings"] = torch.ones((e.shape[0], e.shape[1], WIDE), dtype=e.dtype, device=e.device)
            return inputs, dyn
        cls.forward, cls._get_input = forward, get_input
        undo.append((cls, orig_fwd, orig_inp))

    emb_cls = ext.LiteRTExportableModuleForEmbedder
    orig_emb = emb_cls.forward

    def emb_forward(self, token_ids, _orig=orig_emb):
        e = _orig(self, token_ids)["embeddings"]
        return {"embeddings": F.pad(e, (0, WIDE - e.shape[-1]))}
    emb_cls.forward = emb_forward

    def restore():
        for cls, f, g in undo:
            cls.forward, cls._get_input = f, g
        emb_cls.forward = orig_emb
    return restore


def install_on_load(grid):
    """Wrap export_lib.load_model so the model export() loads gets install_model(); returns (restore, record)."""
    from litert_torch.generative.export_hf.core import export_lib
    orig = export_lib.load_model
    record = {}

    def load_model(*a, **k):
        art = orig(*a, **k)
        record["undo_model"] = install_model(art.model, grid)
        record["model_class"] = type(art.model).__name__
        record["inv_freq"] = art.model.model.rotary_emb.inv_freq.detach().clone()
        return art
    export_lib.load_model = load_model

    def restore():
        export_lib.load_model = orig
    return restore, record
