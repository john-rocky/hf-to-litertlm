"""Rank-4 respelling of transformers' GQA `repeat_kv`.

The stock spelling — `x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(...)`
— lowers to a rank-5 BROADCAST_TO + rank-5 RESHAPE, which the mobile GPU
delegates (Metal / OpenCL / WebGPU) refuse, leaving every LFM2.5 encoder-family
graph only partially delegated. The replacement is an outer product against a
ones row through a matmul: the same tensor through supported rank<=4 ops,
bitwise-identical (x*1.0 is exact in IEEE float) and asserted so at import.

Usage in a convert script:

    import _rank4_repeat_kv
    _rank4_repeat_kv.install()          # BEFORE from_pretrained
    model = AutoModel.from_pretrained(...)
    _rank4_repeat_kv.rebind_all()       # AFTER load — catches remote-code copies

`rebind_all` exists because remote-code repos import symbols BY VALUE into
their own module namespace: patching transformers by module name can miss the
live call site. The downstream graph gate is `assert "BROADCAST_TO" not in hist`.
"""
import sys

import torch


def repeat_kv_rank4(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(b, h_kv, s, d) -> (b, h_kv*n_rep, s, d) with every tensor rank<=4."""
    if n_rep == 1:
        return hidden_states
    b, h, s, d = hidden_states.shape
    ones_rep = torch.ones(1, n_rep, device=hidden_states.device, dtype=hidden_states.dtype)
    x = hidden_states.reshape(b * h, s * d, 1)   # rank 3
    x = torch.matmul(x, ones_rep)                # (b*h, s*d, n_rep)
    x = x.transpose(1, 2)                        # (b*h, n_rep, s*d)
    return x.reshape(b, h * n_rep, s, d)


def _reference_rank5(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    b, h, s, d = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    x = hidden_states[:, :, None, :, :].expand(b, h, n_rep, s, d)
    return x.reshape(b, h * n_rep, s, d)


def _semantic_gate():
    torch.manual_seed(0)
    for shape, rep in (((1, 8, 512, 64), 2), ((2, 3, 17, 5), 4), ((1, 4, 64, 64), 1)):
        t = torch.randn(*shape)
        a, b = repeat_kv_rank4(t, rep), _reference_rank5(t, rep)
        assert a.shape == b.shape and torch.equal(a, b), f"rewrite mismatch at {shape} rep={rep}"
    print("repeat_kv_rank4 semantic gate: bitwise-identical to stock rank-5 spelling")


def install():
    """Patch the two known transformers copies. Call BEFORE from_pretrained."""
    _semantic_gate()
    import transformers.integrations.sdpa_attention as _sdpa_mod
    from transformers.models.lfm2 import modeling_lfm2 as _lfm2_mod

    _sdpa_mod.repeat_kv = repeat_kv_rank4   # live call site (attn_implementation="sdpa")
    _lfm2_mod.repeat_kv = repeat_kv_rank4   # eager fallback


def rebind_all():
    """Rebind `repeat_kv` in EVERY loaded module (remote-code by-value imports).
    Call AFTER from_pretrained. Returns the module names that were rebound."""
    hit = []
    for name, mod in list(sys.modules.items()):
        fn = getattr(mod, "repeat_kv", None)
        if callable(fn) and fn is not repeat_kv_rank4:
            setattr(mod, "repeat_kv", repeat_kv_rank4)
            hit.append(name)
    print(f"rank-4 repeat_kv rebound in: {hit if hit else '(none beyond install())'}")
    return hit
