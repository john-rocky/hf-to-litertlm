"""The Gated DeltaNet chunk kernel with the in-chunk inverse (I - A)^-1 computed without the 63-step row loop.

The loop kernel (kev_qwen35_patch._rank4_chunk_gated_delta_rule, the kernel of the earlier release) builds, per head and
chunk,
    A = -((k_beta @ k^T) * decay_mask) * tril1                       (strictly lower, [H, n, c, c])
and then the forward substitution
    for i in 1..c-1:  row_i <- A_i + sum_m A_im row_m                 (stack / slice / mul / sum / cat per step)
    T = rows + I = (I - A)^-1
= 63 steps x 18 GatedDeltaNet layers, most of the 17,417 ops of the loop kernel's 128-token graph.
A is nilpotent (A^c = 0), so (I - A)^-1 = sum_{j<c} A^j. The inverse is computed in one of these forms:

  loop     the forward substitution of the loop kernel (control; at chunk 64 bit-identical to it)
  product  (I + A)(I + A^2)(I + A^4)...(I + A^(c/2)):  X <- I + A; P <- A; then log2(c) - 1 times
           P <- P @ P;  X <- X + X @ P        (2 BATCH_MATMUL + 1 ADD per factor; X keeps an exact unit diagonal
           because X @ P is strictly lower)
  block8   8-row blocks of the 64 chunk: the block-diagonal part D = A * blockmask (strictly lower inside each
           8 x 8 diagonal block) is inverted with the 3-factor product (I + D)(I + D^2)(I + D^4) (D^8 = 0), then
           the block rows are substituted in order (7 steps):  X[I, :I] = X_II @ (A[I, :I] @ X[:I, :I]),
           concatenated with X_D's own row block (the diagonal block and the zeros right of it)
  doubling recursive doubling of the block forward substitution:
           X = the block-diagonal matrix of the inverses of the s x s diagonal blocks of T = I - A; for the 2s
           block [[P, 0], [-R, Q]] the inverse is [[P^-1, 0], [Q^-1 R P^-1, Q^-1]], i.e.
               X <- I + A * M_1;  then for s = 2, 4, ..., c/2:  X <- X + X @ (A * M_s) @ X
           with M_s the constant mask of the lower-left s x s sub-block of every 2s diagonal block (MUL + 2
           BATCH_MATMUL + ADD per level, log2(c) levels). Every intermediate X is made of blocks of the true
           inverse, so nothing grows like the powers A^j do in the product form.

FORMS: loop64 (= the loop kernel), P64 / P32 / P16 (product at chunk 64 / 32 / 16), B8 (block8 at chunk 64), R64
(doubling at chunk 64), C32 / C16 (loop at chunk 32 / 16 = op-count controls). A chunk other than 64 changes the
chunking itself (the tail concat pad, the tril / eye constants of _chunk_rule_constants, n = L / c chunks and the chunk
loop): the same math, other rounding. The final-kernel files use R64, copied into r13_kernel.py (this file is the
reference and is not imported there). The product forms lose precision on real activations (the powers A^j grow far
beyond the inverse's entries), so they are not used.

expclamp=True clamps every EXP argument in the kernel at r11_fp16_safe.EXP_FLOOR (-80) exactly as
r11_fp16_safe.rank4_kernel_expclamp does (same op order): loop64 + expclamp is bit-identical to that kernel, and with
r11_fp16_safe.apply("softplus") (the GatedDeltaNet forward's softplus, not the kernel) it gives the
"softplus + exp clamp + product" composition.

    import r12_kernel_product as R
    R.apply(model, "P64")                 # every PatchedQwen3_5GatedDeltaNet of a kev_graph.load_text_model() model
    R.apply(model, "P64", expclamp=True)
    R.reset(model)                        # back to the loop kernel

Nothing else is modified (kev_qwen35_patch and r11_fp16_safe are imported, never edited).
`capture` (a list, or None): when set, every kernel call appends {"form", "chunk", "A": float32 [H, n, c, c]} before
the inverse (a probe of the powers' growth reads it)."""
import math

import torch

import kev_qwen35_patch as P

EXP_FLOOR = -80.0

FORMS = {
    "loop64": (64, "loop"),
    "P64": (64, "product"),
    "P32": (32, "product"),
    "P16": (16, "product"),
    "B8": (64, "block8"),
    "R64": (64, "doubling"),
    "C32": (32, "loop"),
    "C16": (16, "loop"),
}
BLOCK = 8

capture = None


def _cexp(x):
    return torch.clamp_min(x, EXP_FLOOR).exp()


def _texp(x):
    return x.exp()


def inverse_loop(attn, eye):
    """The shipped forward substitution (kev_qwen35_patch, same ops in the same order)."""
    c = attn.shape[-1]
    rows = [attn[..., 0, :]]
    for i in range(1, c):
        prev = torch.stack(rows, dim=-2)                    # [H, n, i, c]
        row_full = attn[..., i, :]                          # [H, n, c]
        row_lead = row_full[..., :i]                        # [H, n, i]
        sub = prev[..., :i]                                 # [H, n, i, i]
        upd = row_lead + (row_lead.unsqueeze(-1) * sub).sum(-2)
        rows.append(torch.cat([upd, row_full[..., i:]], dim=-1))
    return torch.stack(rows, dim=-2) + eye                # [H, n, c, c]


def n_factors(nilpotency):
    """Factors (I + A^(2^k)), k < m, so that 2^m >= nilpotency (the sum then holds every nonzero power)."""
    return max(1, math.ceil(math.log2(nilpotency)))


def inverse_product(attn, eye, nilpotency=None):
    """(I - A)^-1 = (I + A)(I + A^2)...(I + A^(2^(m-1))) for A strictly lower (A^nilpotency = 0)."""
    m = n_factors(nilpotency or attn.shape[-1])
    x = eye + attn
    p = attn
    for _ in range(m - 1):
        p = p @ p
        x = x + x @ p
    return x


def block_mask(c, b, dtype, device):
    """1.0 where row and column sit in the same b-row block and col < row (python data -> a graph constant)."""
    return torch.tensor([[1.0 if (i // b == j // b and j < i) else 0.0 for j in range(c)] for i in range(c)],
                        dtype=dtype, device=device)


def inverse_block(attn, eye, b=BLOCK):
    """Block forward substitution over b-row blocks; the diagonal blocks by the log2(b)-factor product."""
    c = attn.shape[-1]
    nb = c // b
    assert nb * b == c, (c, b)
    xd = inverse_product(attn * block_mask(c, b, attn.dtype, attn.device), eye, nilpotency=b)   # (I - D)^-1
    rows = [xd[..., 0:b, :]]                                # [H, n, b, c]
    for blk in range(1, nb):
        r0, r1 = blk * b, (blk + 1) * b
        a_i = attn[..., r0:r1, :r0]                         # [H, n, b, r0]
        x_prev = torch.cat(rows, dim=-2)[..., :r0]          # [H, n, r0, r0]
        x_ii = xd[..., r0:r1, r0:r1]                        # [H, n, b, b]
        left = x_ii @ (a_i @ x_prev)                        # [H, n, b, r0]
        rows.append(torch.cat([left, xd[..., r0:r1, r0:]], dim=-1))
    return torch.cat(rows, dim=-2)                          # [H, n, c, c]


def level_masks(c, dtype, device):
    """M_s for s = 1, 2, 4, ..., c/2: 1.0 where (row, col) is in the lower-left s x s sub-block of a 2s diagonal block."""
    masks, s = [], 1
    while s < c:
        masks.append(torch.tensor([[1.0 if (i // (2 * s) == j // (2 * s) and (i // s) % 2 == 1 and (j // s) % 2 == 0)
                                    else 0.0 for j in range(c)] for i in range(c)], dtype=dtype, device=device))
        s *= 2
    return masks


def inverse_doubling(attn, eye):
    """(I - A)^-1 by recursive doubling of the block forward substitution (module docstring)."""
    masks = level_masks(attn.shape[-1], attn.dtype, attn.device)
    x = eye + attn * masks[0]
    for m in masks[1:]:
        x = x + x @ (attn * m) @ x
    return x


INVERSES = {"loop": inverse_loop, "product": inverse_product, "block8": inverse_block, "doubling": inverse_doubling}


def make_kernel(form, expclamp=False):
    """The shipped rank-4 kernel with the form's chunk default and inverse, EXP optionally clamped (r11 order)."""
    chunk, inv_name = FORMS[form]
    inverse = INVERSES[inv_name]
    ex = _cexp if expclamp else _texp

    def kernel(query, key, value, g, beta, chunk_size=chunk, initial_state=None, output_final_state=False,
               use_qk_l2norm_in_kernel=False, **kwargs):
        from transformers.models.qwen3_5 import modeling_qwen3_5
        initial_dtype = query.dtype
        if use_qk_l2norm_in_kernel:
            query = modeling_qwen3_5.l2norm(query, dim=-1, eps=1e-6)
            key = modeling_qwen3_5.l2norm(key, dim=-1, eps=1e-6)
        q, k, v = [x.squeeze(0).transpose(0, 1).contiguous().to(torch.float32) for x in (query, key, value)]
        b_t, g_t = [x.squeeze(0).transpose(0, 1).contiguous().to(torch.float32) for x in (beta, g)]
        num_heads, seq_len, k_dim = k.shape
        v_dim = v.shape[-1]
        pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
        if pad_size:   # tail pad as concat (the shipped kernel's GPU-safe form)
            zero_qk = torch.zeros(num_heads, pad_size, k_dim, dtype=q.dtype, device=q.device)
            zero_v = torch.zeros(num_heads, pad_size, v_dim, dtype=v.dtype, device=v.device)
            zero_t = torch.zeros(num_heads, pad_size, dtype=g_t.dtype, device=g_t.device)
            q = torch.cat([q, zero_qk], dim=1)
            k = torch.cat([k, zero_qk], dim=1)
            v = torch.cat([v, zero_v], dim=1)
            b_t = torch.cat([b_t, zero_t], dim=-1)
            g_t = torch.cat([g_t, zero_t], dim=-1)
        total = seq_len + pad_size
        n = total // chunk_size
        q = q * (q.shape[-1] ** -0.5)
        v_beta = v * b_t.unsqueeze(-1)
        k_beta = k * b_t.unsqueeze(-1)
        q, k, v, k_beta, v_beta = [x.reshape(num_heads, n, chunk_size, x.shape[-1]) for x in (q, k, v, k_beta, v_beta)]
        g_t = g_t.reshape(num_heads, n, chunk_size)
        eye, tril0_f, tril1_f = P._chunk_rule_constants(chunk_size, torch.float32, q.device)
        g_c = g_t.cumsum(dim=-1)
        diff = (g_c.unsqueeze(-1) - g_c.unsqueeze(-2)) * tril0_f
        decay_mask = ex(diff).float() * tril0_f
        attn = -((k_beta @ k.transpose(-1, -2)) * decay_mask) * tril1_f
        if capture is not None:
            capture.append({"form": form, "chunk": chunk_size, "A": attn.detach().clone()})
        attn = inverse(attn, eye)                             # (I - A)^-1, [H, n, c, c]
        v2 = attn @ v_beta
        k_cumdecay = attn @ (k_beta * ex(g_c).unsqueeze(-1))
        if initial_state is None:
            state = torch.zeros(num_heads, k_dim, v_dim, dtype=v2.dtype, device=v2.device)
        else:
            state = initial_state.reshape(-1, initial_state.shape[-2], initial_state.shape[-1]).to(v2)
        outs = []
        for i in range(n):
            q_i, k_i, v_i = q[:, i], k[:, i], v2[:, i]                       # [H, c, d]
            attn_i = q_i @ k_i.transpose(-1, -2) * decay_mask[:, i]
            v_prime = k_cumdecay[:, i] @ state
            v_new = v_i - v_prime
            attn_inter = (q_i * ex(g_c[:, i, :, None])) @ state
            outs.append(attn_inter + attn_i @ v_new)
            state = (state * ex(g_c[:, i, -1, None, None])
                     + (k_i * ex(g_c[:, i, -1, None] - g_c[:, i])[..., None]).transpose(-1, -2) @ v_new)
        last_state = state.unsqueeze(0) if output_final_state else None
        out = torch.stack(outs, dim=1)                        # [H, n, c, dv]
        out = out.reshape(num_heads, total, v_dim)[:, :seq_len]
        out = out.unsqueeze(0).transpose(1, 2).contiguous().to(initial_dtype)
        return out, last_state

    kernel.__name__ = f"r12_kernel_{form}" + ("_expclamp" if expclamp else "")
    kernel.r12_form = form
    kernel.r12_expclamp = expclamp
    return kernel


KERNELS = {}


def get_kernel(form, expclamp=False):
    key = (form, bool(expclamp))
    if key not in KERNELS:
        KERNELS[key] = make_kernel(form, expclamp)
    return KERNELS[key]


def _gdn_modules(model):
    return [m for m in model.modules() if isinstance(m, P.PatchedQwen3_5GatedDeltaNet)]


def _ours_or_stock(fn):
    return fn is P._rank4_chunk_gated_delta_rule or getattr(fn, "r12_form", None) is not None


def apply(model, form, expclamp=False):
    mods = _gdn_modules(model)
    assert mods and all(_ours_or_stock(m.chunk_gated_delta_rule) for m in mods), \
        "a GatedDeltaNet holds a kernel that is neither the shipped one nor an r12 form (reset other rewrites first)"
    kern = get_kernel(form, expclamp)
    for m in mods:
        m.chunk_gated_delta_rule = kern
    return {"form": form, "expclamp": bool(expclamp), "modules": len(mods), "kernel": kern.__name__}


def reset(model):
    for m in _gdn_modules(model):
        if getattr(m.chunk_gated_delta_rule, "r12_form", None) is not None:
            m.chunk_gated_delta_rule = P._rank4_chunk_gated_delta_rule


def current(model):
    names = sorted({getattr(m.chunk_gated_delta_rule, "__name__", "?") for m in _gdn_modules(model)})
    return names
