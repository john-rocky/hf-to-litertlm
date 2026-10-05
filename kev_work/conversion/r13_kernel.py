"""fp32-invariant rewrites of the Kev row graph that keep fp16 compute (GPU float16 activations or float16 storage,
an NPU) closer to the fp32 result. The final-kernel files are the form R64+sp+ec+dd+vs6 (Kev-0.8B) and, through
r15_form.py, R64+sp+ec+dd+vs6+in1+fn5 (Kev-4B).

A form is a '+'-joined spec applied to a kev_graph.load_text_model() model:

    import r13_kernel as R
    R.apply(model, "R64+sp+ec+dd+vs6")
    R.reset(model)

tokens
  loop | R64   the in-chunk inverse (I - A)^-1: the loop kernel's 63-step forward substitution, or recursive doubling
               (X <- I + A*M_1; X <- X + X (A*M_s) X for s = 2..32). Both copied from r12_kernel_product.py, which is
               not imported.
  sp           the softplus rewrite (r11_fp16_safe.apply("softplus"); the guarded GatedDeltaNet forward's F)
  ec           the exp clamp: every EXP argument in the kernel clamped at -80 (r11_fp16_safe.EXP_FLOOR)
  dd           direct decay sums: the kernel builds its decays from the cumulative sum g_c = cumsum(g) as differences,
               exp(g_c[i] - g_c[j]) (decay mask) and exp(g_c[-1] - g_c[i]) (state update). g_c sums up to 64 decays
               and grows large inside a chunk; where |g_c| >= 128, fp16 holds steps of 0.125 or more, so the
               difference of two such numbers carries an absolute error of that size exactly where it is small and
               matters. dd forms the window sums directly:
                   diff[i, j] = sum_{t=j+1..i} g_t = g @ W   (W constant 0/1 [c, c*c], W[t, i*c+j] = [j < t <= i])
                   suffix[i]  = sum_{t>i} g_t      = g @ S   (S constant 0/1 [c, c],   S[t, i] = [t > i])
               so every partial sum holds only the terms inside the window (the masked terms are exact zeros) and the
               absolute error scales with the window sum, not with g_c. exp(g_c[i]) (prefix decays) keeps the cumsum:
               there the value itself is the large number. fp32: the same sums in another order (rounding level).
  vs<k>        the kernel scales v by 2^k at its entry (the delta rule is linear in v, so the output is exactly 2^k
               larger) and the gated norm after it takes eps * 4^k: the norm's input moves up out of the fp16
               subnormal range by 2^k and its sum of squares by 4^k, an exact power-of-two scaling in fp32. The
               recurrent state the kernel carries is 2^k times larger too (a shared-state pair passes it unchanged
               between its two signatures). vs6 is the final files' value (fp16 headroom of the gated norm's sum of
               squares: r14_norm_range_probe.py).
  gn, gs<j>, nr, rs, fc<n>k<k>, bmc<chain>, sg, sga, sg2
               further rewrites tried for float16 compute (the gated norm pre-scale, a Newton step after RSQRT,
               two-stage reductions, FULLY_CONNECTED splits, sigmoid as EXP / DIV); not used by the final files.

Capture: CAPTURE["on"] and CAPTURE["layers"] select GatedDeltaNet layers whose kernel intermediates are appended to
CAPTURE["out"] as (f"L{layer:02d}_{name}", tensor) in execution order (a probe reads them). Off by default.

Nothing outside this file is modified (kev_qwen35_patch, r11_fp16_safe, r12_kernel_product are read only)."""
import torch

import kev_qwen35_patch as P

EXP_FLOOR = -80.0                     # = r11_fp16_safe.EXP_FLOOR
CAPTURE = {"on": False, "layers": (), "out": []}


def _cap(layer, name, t):
    if CAPTURE["on"] and layer in CAPTURE["layers"]:
        CAPTURE["out"].append((f"L{layer:02d}_{name}", t))


# ---- in-chunk inverse (copied from scripts/r12_kernel_product.py, unchanged) -------------------------------------------
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


def level_masks(c, dtype, device):
    """M_s for s = 1, 2, 4, ..., c/2: 1.0 where (row, col) is in the lower-left s x s sub-block of a 2s diagonal block."""
    masks, s = [], 1
    while s < c:
        masks.append(torch.tensor([[1.0 if (i // (2 * s) == j // (2 * s) and (i // s) % 2 == 1 and (j // s) % 2 == 0)
                                    else 0.0 for j in range(c)] for i in range(c)], dtype=dtype, device=device))
        s *= 2
    return masks


def inverse_doubling(attn, eye):
    """(I - A)^-1 by recursive doubling of the block forward substitution (r12 module docstring)."""
    masks = level_masks(attn.shape[-1], attn.dtype, attn.device)
    x = eye + attn * masks[0]
    for m in masks[1:]:
        x = x + x @ (attn * m) @ x
    return x


INVERSES = {"loop": inverse_loop, "R64": inverse_doubling}


# ---- direct decay sums ------------------------------------------------------------------------------------------------
def window_matrix(c, dtype, device):
    """W [c, c*c]: W[t, i*c + j] = 1 if j < t <= i (python data -> a graph constant)."""
    return torch.tensor([[1.0 if (j < t <= i) else 0.0 for i in range(c) for j in range(c)] for t in range(c)],
                        dtype=dtype, device=device)


def suffix_matrix(c, dtype, device):
    """S [c, c]: S[t, i] = 1 if t > i."""
    return torch.tensor([[1.0 if t > i else 0.0 for i in range(c)] for t in range(c)], dtype=dtype, device=device)


def _cexp(x):
    return torch.clamp_min(x, EXP_FLOOR).exp()


def _texp(x):
    return x.exp()


def make_kernel(inverse="R64", expclamp=True, decay="cumsum", dtype=torch.float32, vscale_log2=0):
    """kev_qwen35_patch._rank4_chunk_gated_delta_rule (batch 1, rank <= 4) with the chosen inverse / exp / decay.
    dtype = the internal compute dtype (float32 in every export; float64 only for the selftest's reference)."""
    inv = INVERSES[inverse]
    ex = _cexp if expclamp else _texp
    assert decay in ("cumsum", "direct"), decay

    def kernel(query, key, value, g, beta, chunk_size=64, initial_state=None, output_final_state=False,
               use_qk_l2norm_in_kernel=False, _layer=-1, **kwargs):
        from transformers.models.qwen3_5 import modeling_qwen3_5
        cap = lambda name, t: _cap(_layer, name, t)
        initial_dtype = query.dtype
        cap("k_in_q", query)
        cap("k_in_k", key)
        cap("k_in_v", value)
        if vscale_log2:      # vs<k>: the delta rule is linear in v, so out(2^k v) = 2^k out(v) exactly (power of 2)
            value = value * float(2 ** vscale_log2)
        cap("k_in_g", g)
        cap("k_in_beta", beta)
        if use_qk_l2norm_in_kernel:
            query = modeling_qwen3_5.l2norm(query, dim=-1, eps=1e-6)
            key = modeling_qwen3_5.l2norm(key, dim=-1, eps=1e-6)
            cap("l2_q", query)
            cap("l2_k", key)
        q, k, v = [x.squeeze(0).transpose(0, 1).contiguous().to(dtype) for x in (query, key, value)]
        b_t, g_t = [x.squeeze(0).transpose(0, 1).contiguous().to(dtype) for x in (beta, g)]
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
        eye, tril0_f, tril1_f = P._chunk_rule_constants(chunk_size, dtype, q.device)
        g_c = g_t.cumsum(dim=-1)
        cap("g_cumsum", g_c)
        if decay == "direct":
            W = window_matrix(chunk_size, dtype, q.device)
            S = suffix_matrix(chunk_size, dtype, q.device)
            diff = torch.matmul(g_t, W).reshape(num_heads, n, chunk_size, chunk_size)
            suffix = torch.matmul(g_t, S)                       # [H, n, c] = g_c[-1] - g_c[i]
            cap("decay_diff", diff)
            cap("decay_suffix", suffix)
            decay_mask = ex(diff) * tril0_f
        else:
            diff = (g_c.unsqueeze(-1) - g_c.unsqueeze(-2)) * tril0_f
            cap("decay_diff", diff)
            decay_mask = ex(diff).float() * tril0_f
            suffix = None
        cap("decay_mask", decay_mask)
        attn = -((k_beta @ k.transpose(-1, -2)) * decay_mask) * tril1_f
        cap("A_strict", attn)
        attn = inv(attn, eye)                                   # (I - A)^-1, [H, n, c, c]
        cap("T_solve", attn)
        v2 = attn @ v_beta
        cap("T_v_beta", v2)
        k_cumdecay = attn @ (k_beta * ex(g_c).unsqueeze(-1))
        cap("k_cumdecay", k_cumdecay)
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
            o = attn_inter + attn_i @ v_new
            cap(f"chunk{i}_out", o)
            outs.append(o)
            if suffix is None:
                sdec = ex(g_c[:, i, -1, None] - g_c[:, i])
            else:
                sdec = ex(suffix[:, i])
            state = (state * ex(g_c[:, i, -1, None, None])
                     + (k_i * sdec[..., None]).transpose(-1, -2) @ v_new)
            cap(f"chunk{i}_state", state)
        last_state = state.unsqueeze(0) if output_final_state else None
        out = torch.stack(outs, dim=1)                        # [H, n, c, dv]
        out = out.reshape(num_heads, total, v_dim)[:, :seq_len]
        out = out.unsqueeze(0).transpose(1, 2).contiguous().to(initial_dtype)
        cap("k_out", out)
        return out, last_state

    kernel.__name__ = (f"r13_kernel_{inverse}" + ("_ec" if expclamp else "") + ("_dd" if decay == "direct" else "")
                       + (f"_vs{vscale_log2}" if vscale_log2 else ""))
    kernel.r13 = {"inverse": inverse, "expclamp": expclamp, "decay": decay, "vscale_log2": vscale_log2}
    return kernel


# ---- FC split ---------------------------------------------------------------------------------------------------------
class SplitLinear(torch.nn.Module):
    """A bias-free nn.Linear as N-slices of <= nmax outputs (CONCATENATION), each the sum of m K-slices of kslice inputs
    (pairwise ADD tree). Same math; in fp16 the Metal FULLY_CONNECTED accumulates over a shorter chain: its kernel for
    N <= 1024 splits K in 4, the one for N >= 1280 does not (a micro test on the real gate_proj input: the N-split
    and 4 K-slices reduce the relative RMS error several fold). The slices are separate constant weights (pre-sliced here), so the V2 recipe casts each one to fp16 as before."""

    def __init__(self, lin, nmax, kslice):
        super().__init__()
        W = lin.weight.data
        assert lin.bias is None
        N, Kd = W.shape
        m = max(1, Kd // kslice) if kslice else 1
        assert Kd % m == 0, (Kd, m)
        self.k = Kd // m
        self.m = m
        self.nslices = []
        self.parts = torch.nn.ModuleList()
        for j in range(0, N, nmax):
            row = torch.nn.ModuleList()
            for i in range(m):
                sub = torch.nn.Linear(self.k, min(nmax, N - j), bias=False)
                sub.weight.data = W[j:j + nmax, i * self.k:(i + 1) * self.k].clone()
                row.append(sub)
            self.parts.append(row)
        self.in_features, self.out_features = Kd, N
        self.orig = [lin]                      # a list: not a registered submodule (no duplicate weights in the export)

    def forward(self, x):
        ys = []
        for row in self.parts:
            ps = [sub(x[..., i * self.k:(i + 1) * self.k]) if self.m > 1 else sub(x) for i, sub in enumerate(row)]
            while len(ps) > 1:
                ps = [ps[i] + ps[i + 1] if i + 1 < len(ps) else ps[i] for i in range(0, len(ps), 2)]
            ys.append(ps[0])
        return torch.cat(ys, dim=-1) if len(ys) > 1 else ys[0]


class BmmLinear(torch.nn.Module):
    """A bias-free nn.Linear as one BATCH_MATMUL with a constant right operand + a SUM over the K-slices:
        y = sum_i x[:, K_i] W[:, K_i]^T,  x [1, L, K] -> [L, m, K/m] -> [m, L, K/m] @ Wb [m, K/m, N] -> sum over m
    = the K-split of SplitLinear in three to four ops instead of m FCs and m - 1 ADDs (the same error as the 16-FC
    split in a Metal fp16 micro test on the real gate_proj input). Wb is a buffer (a graph constant);
    r13_export.py casts it to fp16 with the V2 recipe's FLOAT_CASTING, which it registers for BATCH_MATMUL in its own
    process (the scope regex "BmmLinear" keeps the kernel's non-constant BATCH_MATMULs untouched)."""

    def __init__(self, lin, chain):
        super().__init__()
        W = lin.weight.data
        assert lin.bias is None
        N, Kd = W.shape
        self.m = max(1, Kd // chain)
        assert Kd % self.m == 0, (Kd, self.m)
        self.k = Kd // self.m
        self.register_buffer("Wb", W.t().reshape(self.m, self.k, N).clone().contiguous())
        self.in_features, self.out_features = Kd, N
        self.orig = [lin]

    def forward(self, x):
        b, L, Kd = x.shape
        assert b == 1
        xs = x.reshape(L, self.m, self.k).transpose(0, 1)          # [m, L, K/m]
        return torch.matmul(xs, self.Wb).sum(0, keepdim=True)        # [1, L, N]


def bmm_linears(model, chain):
    count = 0
    for layer in model.layers:
        for parent in list(layer.modules()):
            for name, child in list(parent.named_children()):
                if type(child) is torch.nn.Linear and child.bias is None:
                    setattr(parent, name, BmmLinear(child, chain))
                    count += 1
    return count


def split_linears(model, nmax, kslice):
    """Replace every bias-free nn.Linear of the text model's decoder layers by a SplitLinear; returns the count."""
    count = 0
    for layer in model.layers:
        for parent in list(layer.modules()):
            for name, child in list(parent.named_children()):
                if type(child) is torch.nn.Linear and child.bias is None:
                    setattr(parent, name, SplitLinear(child, nmax, kslice))
                    count += 1
    return count


def unsplit_linears(model):
    for layer in model.layers:
        for parent in list(layer.modules()):
            for name, child in list(parent.named_children()):
                if isinstance(child, (SplitLinear, BmmLinear)):
                    setattr(parent, name, child.orig[0])


# ---- gated norm / Newton rsqrt ----------------------------------------------------------------------------------------
GN = {"in_log2": 0, "eps_log4": 0, "newton": False, "rsplit": False}
NORM = {"newton": False, "rsplit": False}


def _two_stage(x2, op):
    """mean / sum over the last dim in two stages of <= 32 terms (rs): [*, D] -> [-1, D/c, c] -> op(-1) twice -> [*, 1].
    Same sum in another order (fp32: rounding level); an fp16 reduction then chains <= 32 terms per stage instead of D."""
    D = x2.shape[-1]
    c = 32 if D % 32 == 0 and D >= 512 else 16
    assert D % c == 0, D
    y = x2.reshape(-1, D // c, c)
    y = op(op(y, -1), -1)                     # rank 3 -> 2 -> 1
    return y.reshape(*x2.shape[:-1], 1)


def _mean(x2):
    return _two_stage(x2, lambda t, d: t.mean(d)) if NORM["rsplit"] else x2.mean(-1, keepdim=True)


def _sum(x2):
    return _two_stage(x2, lambda t, d: t.sum(d)) if NORM["rsplit"] else x2.sum(-1, keepdim=True)


def rsqrt_newton(v):
    """rsqrt(v) followed by one Newton step y <- y (1.5 - 0.5 v y^2) (v y first: no fp16 overflow for v up to 65504).
    fp32: the step moves an rsqrt that is already correctly rounded by <= 1 ulp. fp16 / HTP: an approximate RSQRT with
    relative error e leaves ~1.5 e^2."""
    y = torch.rsqrt(v)
    return y * (1.5 - 0.5 * ((v * y) * y))
_ORIG_GN = {}


def gated_norm_scaled(self, hidden_states, gate=None):
    """Qwen3_5RMSNormGated.forward for an input that the kernel already scaled by 2^v (vs<v>) and that is pre-scaled
    here by 2^j (gn<j>): x' = 2^(v+j) x, x' * rsqrt(mean(x'^2) + eps * 4^(v+j)) = x * rsqrt(mean(x^2) + eps) exactly up
    to the rsqrt rounding (powers of two; eps * 4^(v+j) is the fp32 eps scaled exactly). In fp16 the kernel output
    (a large share of its values below the fp16 normal range 6.1e-5, and almost all of its squares) leaves the
    subnormal range."""
    input_dtype = hidden_states.dtype
    h = hidden_states.to(torch.float32)
    if GN["in_log2"]:
        h = h * float(2 ** GN["in_log2"])
    variance = _mean(h.pow(2))
    eps = torch.tensor(self.variance_epsilon, dtype=torch.float32) * float(4 ** GN["eps_log4"])
    h = h * (rsqrt_newton(variance + eps) if NORM["newton"] else torch.rsqrt(variance + eps))
    h = self.weight * h.to(input_dtype)
    h = h * torch.nn.functional.silu(gate.to(torch.float32))
    return h.to(input_dtype)


_ORIG_NR = {}
_ORIG_SG = {}


def sigmoid_exp(x):
    """sg: sigmoid(x) = 1 / (1 + exp(max(-x, -80))) — EXP / ADD / DIV instead of LOGISTIC, for an NPU whose
    LOGISTIC is less accurate than its EXP (a layer-0 probe on the Galaxy S26 NPU, Qualcomm HTP). The clamp keeps the
    EXP argument out of the band where that NPU's EXP returned -inf, [-181.6, -87.6] (exp(-80) = 1.8e-35 -> the result
    is 1 in any precision); a negative x below -11 makes exp overflow fp16 to inf -> 1 / inf = 0 (the sigmoid's own
    limit). fp32: within an ulp of torch.sigmoid. Not used by the final files."""
    return 1.0 / (1.0 + torch.exp(torch.clamp_min(-x, EXP_FLOOR)))


def _sigmoid_method(self):
    return SIGMOID_IMPL[0](self)


def sigmoid_exp2(x):
    """sg2: sigmoid(x) = exp(min(x, 0)) / (1 + exp(-|x|)) with min(x, 0) = x - relu(x) and -|x| = x - 2 relu(x) (the
    softplus rewrite's form) and both EXP arguments clamped at -80: every EXP argument is <= 0, so nothing overflows
    fp16 (sg's exp(-x) does for x < -11) and nothing enters the NPU band [-181.6, -87.6]. fp32: the same value up to
    rounding (x >= 0: 1 / (1 + e^-x); x < 0: e^x / (1 + e^x)). Not used by the final files."""
    r = torch.relu(x)
    num = torch.exp(torch.clamp_min(x - r, EXP_FLOOR))
    den = 1.0 + torch.exp(torch.clamp_min(x - 2.0 * r, EXP_FLOOR))
    return num / den


SIGMOID_IMPL = [sigmoid_exp]


class _TorchShim(__import__("types").ModuleType):
    """modeling_qwen3_5's `torch` global with only `sigmoid` replaced (the attention output gate's torch.sigmoid(gate));
    torch._decomp's SiLU decomposition (x * torch.sigmoid(x)) keeps the real torch, so the 60 SiLU stay LOGISTIC x."""

    def __getattr__(self, name):
        return getattr(torch, name)


def rmsnorm_norm_newton(self, x):
    """Qwen3_5RMSNorm._norm with the Newton-refined rsqrt (nr) and / or the two-stage mean (rs)."""
    v = _mean(x.pow(2)) + self.eps
    return x * (rsqrt_newton(v) if NORM["newton"] else torch.rsqrt(v))


def l2norm_newton(x, dim=-1, eps=1e-6):
    """modeling_qwen3_5.l2norm with the Newton-refined rsqrt (nr) and / or the two-stage sum (rs)."""
    assert dim == -1
    v = _sum(x * x) + eps
    return x * (rsqrt_newton(v) if NORM["newton"] else torch.rsqrt(v))


# ---- apply / reset ----------------------------------------------------------------------------------------------------
KNOWN = ("loop", "R64", "sp", "ec", "dd", "gn", "nr", "rs", "sg", "sga", "sg2")   # + fc<n>k<k>, vs<k>, gs<j>
STATE = {"spec": None}


def parse(spec):
    """tokens: KNOWN, plus fc<nmax>k<kslice> (e.g. fc1024k256 = N-slices <= 1024, K-slices of 256 inputs)."""
    import re
    toks = [t for t in spec.split("+") if t]
    fc = [t for t in toks if re.fullmatch(r"fc\d+k\d+", t)]
    vs = [int(t[2:]) for t in toks if re.fullmatch(r"vs\d+", t)]
    bm = [int(t[3:]) for t in toks if re.fullmatch(r"bmc\d+", t)]
    gs = [int(t[2:]) for t in toks if re.fullmatch(r"gs\d+", t)]
    bad = [t for t in toks if t not in KNOWN and t not in fc and not re.fullmatch(r"(vs|gs|bmc)\d+", t)]
    assert len(bm) <= 1 and not (bm and fc), "one FC rewrite: fc<n>k<k> or bmc<chain>"
    assert not bad, f"unknown tokens {bad} (known {KNOWN}, fc<n>k<k>)"
    assert len(fc) <= 1
    inverse = "R64" if "R64" in toks else "loop"
    assert not ("R64" in toks and "loop" in toks)
    fcs = None
    if fc:
        mt = re.fullmatch(r"fc(\d+)k(\d+)", fc[0])
        fcs = (int(mt.group(1)), int(mt.group(2)))
    assert len(vs) <= 1 and len(gs) <= 1
    assert not ("gn" in toks and (vs or gs)), "gn = r11_fp16_safe's gated norm; vs / gs use this file's"
    return {"inverse": inverse, "softplus": "sp" in toks, "expclamp": "ec" in toks,
            "decay": "direct" if "dd" in toks else "cumsum", "gnorm": "gn" in toks, "fc_split": fcs,
            "vscale_log2": vs[0] if vs else 0, "gn_in_log2": gs[0] if gs else 0, "newton": "nr" in toks,
            "rsplit": "rs" in toks, "sigmoid_exp": "sg" in toks, "sigmoid_exp_all": "sga" in toks,
            "sigmoid_exp2": "sg2" in toks,
            "bmm_chain": bm[0] if bm else 0,
            "tokens": toks}


def _gdn_modules(model):
    return [m for m in model.modules() if isinstance(m, P.PatchedQwen3_5GatedDeltaNet)]


class _Bound:
    """A kernel bound to one GatedDeltaNet layer (passes _layer for the capture; same call signature otherwise)."""

    def __init__(self, kern, layer):
        self.kern, self.layer = kern, layer
        self.__name__ = kern.__name__
        self.r13 = kern.r13

    def __call__(self, *args, **kwargs):
        return self.kern(*args, _layer=self.layer, **kwargs)


def apply(model, spec):
    """Install the form: the kernel on every PatchedQwen3_5GatedDeltaNet; sp / gn through r11_fp16_safe (softplus = the
    guarded forward's F, gnorm = Qwen3_5RMSNormGated.forward with the 2^7 pre-scale); fc = SplitLinear swaps."""
    import r11_fp16_safe as S11
    assert STATE["spec"] is None, f"form {STATE['spec']} already applied: reset first"
    f = parse(spec)
    mods = _gdn_modules(model)
    assert mods and all(m.chunk_gated_delta_rule is P._rank4_chunk_gated_delta_rule for m in mods), \
        "a GatedDeltaNet does not hold the shipped kernel (reset other rewrites first)"
    kern = make_kernel(f["inverse"], f["expclamp"], f["decay"], vscale_log2=f["vscale_log2"])
    for m in mods:
        m.chunk_gated_delta_rule = _Bound(kern, m.layer_idx)
    if f["softplus"]:
        S11.apply("softplus")
    if f["gnorm"]:
        S11.apply("gnorm")
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    if f["vscale_log2"] or f["gn_in_log2"] or f["newton"] or f["rsplit"]:
        assert not f["gnorm"], "nr / rs / vs / gs replace the gated norm: do not combine with r11_fp16_safe's gn"
        _ORIG_GN.setdefault("forward", M.Qwen3_5RMSNormGated.forward)
        GN["in_log2"] = f["gn_in_log2"]
        GN["eps_log4"] = f["vscale_log2"] + f["gn_in_log2"]
        GN["newton"] = f["newton"]
        M.Qwen3_5RMSNormGated.forward = gated_norm_scaled
    NORM["newton"], NORM["rsplit"] = f["newton"], f["rsplit"]
    if f["newton"] or f["rsplit"]:
        _ORIG_NR.setdefault("_norm", M.Qwen3_5RMSNorm._norm)
        _ORIG_NR.setdefault("l2norm", M.l2norm)
        M.Qwen3_5RMSNorm._norm = rmsnorm_norm_newton
        M.l2norm = l2norm_newton
    assert sum((f["sigmoid_exp"], f["sigmoid_exp_all"], f["sigmoid_exp2"])) <= 1
    SIGMOID_IMPL[0] = sigmoid_exp2 if f["sigmoid_exp2"] else sigmoid_exp
    if f["sigmoid_exp"] or f["sigmoid_exp2"]:      # sg: the 24 standalone sigmoids only — beta = b.sigmoid() (Tensor method, guarded GDN
        # forward) and torch.sigmoid(gate) (attention, through modeling_qwen3_5's torch global); the converter's SiLU
        # decomposition calls torch.sigmoid from torch._decomp, which keeps the real torch (60 SiLU stay LOGISTIC x)
        _ORIG_SG.setdefault("method", torch.Tensor.sigmoid)
        _ORIG_SG.setdefault("mod_torch", M.torch)
        shim = _TorchShim("torch_r13_sg")
        shim.sigmoid = SIGMOID_IMPL[0]
        torch.Tensor.sigmoid = _sigmoid_method
        M.torch = shim
    if f["sigmoid_exp_all"]:  # sga (first sg version): torch.sigmoid itself patched = all 84 LOGISTIC incl. the SiLU
        _ORIG_SG.setdefault("fn", torch.sigmoid)
        _ORIG_SG.setdefault("method", torch.Tensor.sigmoid)
        torch.sigmoid = sigmoid_exp
        torch.Tensor.sigmoid = _sigmoid_method
    nsplit = split_linears(model, *f["fc_split"]) if f["fc_split"] else 0
    if f["bmm_chain"]:
        nsplit = bmm_linears(model, f["bmm_chain"])
    STATE["spec"] = spec
    return {"spec": spec, **f, "kernel": kern.__name__, "gdn_modules": len(mods), "split_linears": nsplit,
            "r11_applied": list(S11.APPLIED)}


def reset(model):
    import r11_fp16_safe as S11
    for m in _gdn_modules(model):
        if isinstance(m.chunk_gated_delta_rule, _Bound):
            m.chunk_gated_delta_rule = P._rank4_chunk_gated_delta_rule
    unsplit_linears(model)
    S11.reset()
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    if "forward" in _ORIG_GN:
        M.Qwen3_5RMSNormGated.forward = _ORIG_GN["forward"]
        GN["in_log2"] = GN["eps_log4"] = 0
        GN["newton"] = False
    NORM["newton"] = NORM["rsplit"] = False
    if "fn" in _ORIG_SG:
        torch.sigmoid = _ORIG_SG.pop("fn")
    if "method" in _ORIG_SG:
        torch.Tensor.sigmoid = _ORIG_SG.pop("method")
    if "mod_torch" in _ORIG_SG:
        M.torch = _ORIG_SG.pop("mod_torch")
    SIGMOID_IMPL[0] = sigmoid_exp
    if "_norm" in _ORIG_NR:
        M.Qwen3_5RMSNorm._norm = _ORIG_NR["_norm"]
        M.l2norm = _ORIG_NR["l2norm"]
    STATE["spec"] = None


def selftest():
    """Random kernel inputs: the window / suffix sums equal the cumsum differences (fp64), and every fp32 form vs the
    kernel computed in float64 (decays down to -6 / token, cumsum to -390; the float64 reference has no exp clamp)."""
    import json
    torch.manual_seed(0)
    c = 64
    g = -torch.rand(16, 2, c, dtype=torch.float64) * 6.0
    gc = g.cumsum(-1)
    W = window_matrix(c, torch.float64, "cpu")
    S = suffix_matrix(c, torch.float64, "cpu")
    diff_ref = (gc.unsqueeze(-1) - gc.unsqueeze(-2)) * torch.tril(torch.ones(c, c, dtype=torch.float64))
    diff_dd = (g @ W).reshape(16, 2, c, c)
    suf_ref = gc[..., -1:] - gc
    suf_dd = g @ S
    res = {"window_vs_cumsum_diff_fp64_max_abs": float((diff_dd - diff_ref).abs().max()),
           "suffix_vs_cumsum_fp64_max_abs": float((suf_dd - suf_ref).abs().max())}
    # fp32 kernels vs an fp64 kernel on the same inputs
    H, L, D = 4, 130, 16
    q = torch.randn(1, L, H, D, dtype=torch.float64)
    k = torch.randn(1, L, H, D, dtype=torch.float64)
    v = torch.randn(1, L, H, D, dtype=torch.float64)
    gg = -torch.rand(1, L, H, dtype=torch.float64) * 6.0
    b = torch.rand(1, L, H, dtype=torch.float64)
    ref, _ = make_kernel("loop", False, "direct", dtype=torch.float64)(q, k, v, gg, b, use_qk_l2norm_in_kernel=True)
    for inv in ("loop", "R64"):
        for dec in ("cumsum", "direct"):
            out, _ = make_kernel(inv, True, dec)(*(x.float() for x in (q, k, v, gg, b)), use_qk_l2norm_in_kernel=True)
            res[f"{inv}_ec_{dec}_fp32_vs_fp64_max_abs"] = float((out.double() - ref).abs().max())
    # fp16 emulation of the decay alone on real-looking magnitudes: abs error of the decay mask, cumsum vs direct
    gh = (-torch.rand(16, 2, c) * 4.0).half()
    gc_h = gh.float().cumsum(-1).half()                  # fp16 storage of the prefix sums
    diff_h = (gc_h.unsqueeze(-1).float() - gc_h.unsqueeze(-2).float()).half()
    dd_h = (gh.float() @ window_matrix(c, torch.float32, "cpu")).half().reshape(16, 2, c, c)
    true = (gh.double() @ window_matrix(c, torch.float64, "cpu")).reshape(16, 2, c, c)
    tri = torch.tril(torch.ones(c, c, dtype=torch.bool))
    res["fp16_storage_decay_mask_abs_err_cumsum_diff"] = float((diff_h.double().exp() - true.exp())[..., tri].abs().max())
    res["fp16_storage_decay_mask_abs_err_direct"] = float((dd_h.double().exp() - true.exp())[..., tri].abs().max())
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    selftest()
