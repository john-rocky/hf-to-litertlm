"""fp16-safe rewrites of the Kev row-prefill graph that do not change the fp32 result.

    import r11_fp16_safe; r11_fp16_safe.apply("softplus")      # before the model's first forward / before export

The final-kernel files use "softplus" (through r13_kernel.py, token `sp`); r13_kernel.py has its own form of "expclamp"
(token `ec`). "gnorm" and "l2norm" are kept for reference.

Rewrite "softplus": the GatedDeltaNet decay g = -exp(A_log) * softplus(a + dt_bias) lowers to
SELECT(GREATER(x, 20), x, LOG(ADD(EXP(x), 1))); in fp16 the EXP is inf for 11.09 < x <= 20, softplus becomes inf,
g = -inf and the masked chunk products turn every position NaN. The rewrite is the log-space identity
    softplus(x) = relu(x) + log1p(exp(x - 2 relu(x)))        (= relu(x) + log1p(exp(-|x|)))
whose EXP argument is <= 0, so it stays inside (0, 1] in any precision. In fp32:
  x <= 0       relu(x) = 0 and x - 0 = x exactly: log1p(exp(x)) + 0, the same arithmetic as torch's softplus (and as the
               graph's EXP / ADD / LOG) = bit-identical;
  x > 20       exp(-x) < 2.1e-9, log1p(...) < ulp(x) / 2 = x exactly, as the thresholded branch;
  0 < x <= 20  equal up to rounding (~1 ulp of the result).
No GREATER / SELECT is left (RELU, MUL, SUB, EXP, ADD, LOG instead).

Rewrite "gnorm" (GatedDeltaNet output norm, Qwen3_5RMSNormGated): its input (the delta-rule output) is small, so
mean(x^2) and eps = 1e-6 sit in or below the fp16 subnormal range (min normal 6.1e-5): a runtime that flushes subnormals
gets rsqrt(0) = inf -> x * inf. Pre-scale by S = 2^7:
    h = x * S;  h * rsqrt(mean(h^2) + eps * S^2)        (= x * rsqrt(mean(x^2) + eps), exact scaling in fp32)
Rewrite "l2norm" (q / k l2norm inside the chunk kernel, eps 1e-6): pad positions are exact zero vectors (the guard zeroes
q / k after the conv), so in fp16 with eps flushed rsqrt(0 + 0) = inf and 0 * inf = NaN. The inverse norm is capped at
rsqrt(eps): MINIMUM(rsqrt(s + eps), rsqrt(eps)) is the identity in fp32 (s >= 0, rsqrt decreasing; the cap is the fp32
value torch computes for rsqrt(fp32(1e-6))), and turns the flushed case's inf into 1000 (0 * 1000 = 0).

Rewrite "expclamp" (every EXP inside the chunked delta rule): on the Galaxy S26 NPU (Qualcomm HTP) the EXP returned -inf
for inputs in about [-181.6, -87.6], and the decay mask exp(g_c_i - g_c_j) of the lower triangle reaches that band,
which made every row non-finite there. The kernel's EXP arguments are clamped from below at EXP_FLOOR = -80 (MAXIMUM,
accepted by Mac Metal at fp32 and fp16): exp(max(x, -80)) differs from exp(x) only where exp(x) < 1.8e-35 (fp32
subnormal / zero territory), terms below fp32 resolution of the sums they enter. Applied per GatedDeltaNet instance
(apply(..., model=...)): the kernel copy below replaces the instance's chunk_gated_delta_rule.

Mechanism: the guarded GatedDeltaNet forward (kev_qwen35_patch.py, exec'd from the installed transformers source) looks
`F` up in its own globals at call time; apply() points that `F` at a shim whose softplus is the rewrite and whose every
other attribute is torch.nn.functional's. No source file is modified; nothing changes unless a script calls apply()."""
import types

import torch
import torch.nn.functional as F_torch

import kev_qwen35_patch as P

APPLIED = []


def softplus_f16safe(x, beta=1.0, threshold=20.0):
    assert beta == 1.0 and threshold == 20.0, "the graph uses the defaults"
    r = torch.relu(x)
    return r + torch.log1p(torch.exp(x - 2.0 * r))


GN_SCALE = 128.0                       # 2^7
RSQRT_EPS = None                       # fp32 rsqrt(fp32(1e-6)), set on first use


def gated_norm_f16safe(self, hidden_states, gate=None):
    """Qwen3_5RMSNormGated.forward with the input pre-scaled by 2^7 (see module docstring)."""
    input_dtype = hidden_states.dtype
    h = hidden_states.to(torch.float32) * GN_SCALE
    variance = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(variance + self.variance_epsilon * GN_SCALE * GN_SCALE)
    h = self.weight * h.to(input_dtype)
    h = h * F_torch.silu(gate.to(torch.float32))
    return h.to(input_dtype)


def l2norm_f16safe(x, dim=-1, eps=1e-6):
    """modeling_qwen3_5.l2norm with the inverse norm capped at rsqrt(eps) (an identity in fp32)."""
    global RSQRT_EPS
    assert eps == 1e-6
    if RSQRT_EPS is None:
        RSQRT_EPS = float(torch.rsqrt(torch.tensor(eps, dtype=torch.float32)))
    inv = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    inv = torch.minimum(inv, torch.tensor(RSQRT_EPS, dtype=inv.dtype))
    return x * inv


EXP_FLOOR = -80.0


def _cexp(x):
    return torch.clamp_min(x, EXP_FLOOR).exp()


def rank4_kernel_expclamp(query, key, value, g, beta, chunk_size=64, initial_state=None, output_final_state=False,
                          use_qk_l2norm_in_kernel=False, **kwargs):
    """kev_qwen35_patch._rank4_chunk_gated_delta_rule with every EXP argument clamped at EXP_FLOOR (same order)."""
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
    if pad_size:
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
    decay_mask = _cexp(diff).float() * tril0_f
    attn = -((k_beta @ k.transpose(-1, -2)) * decay_mask) * tril1_f
    rows = [attn[..., 0, :]]
    for i in range(1, chunk_size):
        prev = torch.stack(rows, dim=-2)
        row_full = attn[..., i, :]
        row_lead = row_full[..., :i]
        sub = prev[..., :i]
        upd = row_lead + (row_lead.unsqueeze(-1) * sub).sum(-2)
        rows.append(torch.cat([upd, row_full[..., i:]], dim=-1))
    attn = torch.stack(rows, dim=-2) + eye
    v2 = attn @ v_beta
    k_cumdecay = attn @ (k_beta * _cexp(g_c).unsqueeze(-1))
    if initial_state is None:
        state = torch.zeros(num_heads, k_dim, v_dim, dtype=v2.dtype, device=v2.device)
    else:
        state = initial_state.reshape(-1, initial_state.shape[-2], initial_state.shape[-1]).to(v2)
    outs = []
    for i in range(n):
        q_i, k_i, v_i = q[:, i], k[:, i], v2[:, i]
        attn_i = q_i @ k_i.transpose(-1, -2) * decay_mask[:, i]
        v_prime = k_cumdecay[:, i] @ state
        v_new = v_i - v_prime
        attn_inter = (q_i * _cexp(g_c[:, i, :, None])) @ state
        outs.append(attn_inter + attn_i @ v_new)
        state = (state * _cexp(g_c[:, i, -1, None, None])
                 + (k_i * _cexp(g_c[:, i, -1, None] - g_c[:, i])[..., None]).transpose(-1, -2) @ v_new)
    last_state = state.unsqueeze(0) if output_final_state else None
    out = torch.stack(outs, dim=1)
    out = out.reshape(num_heads, total, v_dim)[:, :seq_len]
    out = out.unsqueeze(0).transpose(1, 2).contiguous().to(initial_dtype)
    return out, last_state


_ORIG = {}


class _FShim(types.ModuleType):
    def __getattr__(self, name):
        return getattr(F_torch, name)


def _gdn_modules(model):
    return [m for m in model.modules() if isinstance(m, P.PatchedQwen3_5GatedDeltaNet)]


def apply(*names, model=None):
    g = P.PatchedQwen3_5GatedDeltaNet.forward.__globals__
    for name in names:
        if name in APPLIED:
            continue
        if name == "softplus":
            shim = _FShim("F_f16safe")
            shim.softplus = softplus_f16safe
            assert g["F"] is F_torch, "F in the guarded forward's globals is not torch.nn.functional"
            g["F"] = shim
        elif name == "gnorm":
            from transformers.models.qwen3_5 import modeling_qwen3_5 as M
            _ORIG.setdefault("gnorm", M.Qwen3_5RMSNormGated.forward)
            M.Qwen3_5RMSNormGated.forward = gated_norm_f16safe
        elif name == "l2norm":
            from transformers.models.qwen3_5 import modeling_qwen3_5 as M
            _ORIG.setdefault("l2norm", M.l2norm)
            M.l2norm = l2norm_f16safe
        elif name == "expclamp":
            assert model is not None, "expclamp is set per GatedDeltaNet instance: pass model="
            mods = _gdn_modules(model)
            assert mods and all(m.chunk_gated_delta_rule is P._rank4_chunk_gated_delta_rule for m in mods)
            for m in mods:
                m.chunk_gated_delta_rule = rank4_kernel_expclamp
        else:
            raise ValueError(name)
        APPLIED.append(name)
    return list(APPLIED)


def reset(model=None):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    if model is not None:
        for m in _gdn_modules(model):
            if m.chunk_gated_delta_rule is rank4_kernel_expclamp:
                m.chunk_gated_delta_rule = P._rank4_chunk_gated_delta_rule
    g = P.PatchedQwen3_5GatedDeltaNet.forward.__globals__
    g["F"] = F_torch
    if "gnorm" in _ORIG:
        M.Qwen3_5RMSNormGated.forward = _ORIG["gnorm"]
    if "l2norm" in _ORIG:
        M.l2norm = _ORIG["l2norm"]
    APPLIED.clear()


def selftest():
    """softplus_f16safe vs F.softplus on a dense grid (fp32): bit-equal for x <= 0 and x > 20, max abs diff otherwise."""
    x = torch.linspace(-60, 60, 1_200_001, dtype=torch.float32)
    ref, new = F_torch.softplus(x), softplus_f16safe(x)
    neg, big, mid = x <= 0, x > 20, (x > 0) & (x <= 20)
    return {"grid": "linspace(-60, 60, 1,200,001) fp32",
            "bit_equal_x_le_0": bool(torch.equal(ref[neg], new[neg])),
            "bit_equal_x_gt_20": bool(torch.equal(ref[big], new[big])),
            "max_abs_diff_0_to_20": float((ref[mid] - new[mid]).abs().max()),
            "max_rel_diff_0_to_20": float(((ref[mid] - new[mid]).abs() / ref[mid]).max()),
            "fp16_new_finite": bool(torch.isfinite(softplus_f16safe(x.half())).all()),
            "fp16_ref_finite": bool(torch.isfinite(F_torch.softplus(x.half().float()).half()).all()),
            "fp16_graph_form_nonfinite_x_range": _graph_form_fp16_nonfinite(x)}


def _graph_form_fp16_nonfinite(x):
    """The graph's lowering SELECT(x > 20, x, LOG(EXP(x) + 1)) evaluated in fp16: the x range that gives inf."""
    h = x.half()
    y = torch.where(h > 20, h, torch.log(torch.exp(h) + 1))
    bad = ~torch.isfinite(y)
    return [float(x[bad].min()), float(x[bad].max())] if bool(bad.any()) else None


if __name__ == "__main__":
    import json
    print(json.dumps(selftest(), indent=1))
