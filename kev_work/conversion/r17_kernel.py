"""FC rewrites on top of r13_kernel.py's forms (imported; its forms and kernel are used unchanged). The NPU-ready files use
`bk1024@in_proj_z.q_proj`.

    import r17_kernel as R
    R.apply(model, "R64+sp+ec+dd+vs6+bk1024@in_proj_z.q_proj")
    R.reset(model)

A spec is r13_kernel's spec plus at most one of these FC tokens (r13's own fc<n>k<k> / bmc<chain> stay available):
  kc<kmax>[@owners]  K-chunk split: an FC with K > kmax becomes m = ceil(K / kmax) FCs over equal K-slices (K / m
                     inputs each) summed by a pairwise ADD tree; N is not split; FCs with K <= kmax are not touched.
  bk<kmax>[@owners]  BATCH_MATMUL form: an FC becomes one BATCH_MATMUL with a constant right operand [m, K / m, N]
                     (m = ceil(K / kmax); m = 1 for K <= kmax = the whole K-long sum in one batch) + SUM over m
                     (r13_kernel.BmmLinear's forward; the class name keeps "BmmLinear", so r13_export's FLOAT_CASTING
                     scope regex casts its weights to fp16). The converter turns a batch-1 BATCH_MATMUL with a constant
                     right operand back into FULLY_CONNECTED and keeps the size-1 SUM after it, so with m = 1 the file
                     gets an FC followed by a SUM over an axis of size 1 (an exact identity).
  os<k>[@owners]     outlier split: the k input channels with the largest |x| over calibration rows
                     (results/r17_outlier_channels.json) are summed in their own small FC: y = (x * m) W^T + (x E^T)
                     W_top^T, m = a constant 0/1 vector that zeroes them, E = one-hot rows that select them.
  @owners            optional filter: only FCs whose attribute name is in the list (in_proj_qkv, in_proj_z, in_proj_b,
                     in_proj_a, out_proj, q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj), separated by
                     ',' or '.' ('.' inside a file tag), e.g. bk1024@in_proj_z.q_proj.
`bk1024@in_proj_z.q_proj` puts a size-1 SUM between the in_proj_z FC and the Gated DeltaNet gated norm's sigmoid, and
between the q_proj FC and the attention output gate's sigmoid: on the Galaxy S26's Qualcomm HTP a sigmoid that reads an
FC output directly carried an absolute error of about 2.3e-3, and with the SUM in between the files pass the
tolerance there. The rewrites keep the math: fp32 changes at most the summation order (the size-1 SUM changes nothing).
sgn / sga2 (sigmoid as exp(min(x, 0)) / (1 + exp(-|x|)) in the gated norm / everywhere) were tried and are not used."""
import math
import re

import torch

import r13_kernel as R13

R17_TOKEN = re.compile(r"(kc|bk|os)(\d+)(?:@([a-z_.,]+))?")
OWNERS = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj",
          "gate_proj", "up_proj", "down_proj")
STATE = {"spec": None, "swaps": []}


class KSplitLinear(torch.nn.Module):
    """y = sum_i x[..., K_i] W[:, K_i]^T over m equal K-slices (separate constant FCs + pairwise ADD tree)."""

    def __init__(self, lin, kmax):
        super().__init__()
        W = lin.weight.data
        assert lin.bias is None
        N, Kd = W.shape
        self.m = math.ceil(Kd / kmax)
        assert Kd % self.m == 0, (Kd, self.m)
        self.k = Kd // self.m
        self.parts = torch.nn.ModuleList()
        for i in range(self.m):
            sub = torch.nn.Linear(self.k, N, bias=False)
            sub.weight.data = W[:, i * self.k:(i + 1) * self.k].clone()
            self.parts.append(sub)
        self.in_features, self.out_features = Kd, N
        self.orig = [lin]                      # a list: not a registered submodule (no duplicate weights in the export)

    def forward(self, x):
        ps = [sub(x[..., i * self.k:(i + 1) * self.k]) for i, sub in enumerate(self.parts)]
        while len(ps) > 1:
            ps = [ps[i] + ps[i + 1] if i + 1 < len(ps) else ps[i] for i in range(0, len(ps), 2)]
        return ps[0]


class OutlierSplitLinear(torch.nn.Module):
    """y = (x * m) W^T + (x E^T) W[:, top]^T: the top channels' products summed apart from the rest (fp32: the same sum in
    another order). m [K] constant 0/1, E [k, K] one-hot (an FC whose sum has one non-zero term per output)."""

    def __init__(self, lin, top):
        super().__init__()
        W = lin.weight.data
        assert lin.bias is None
        N, Kd = W.shape
        top = [int(c) for c in top]
        m = torch.ones(Kd, dtype=W.dtype)
        m[top] = 0.0
        self.register_buffer("mask", m)
        self.sel = torch.nn.Linear(Kd, len(top), bias=False)
        E = torch.zeros(len(top), Kd, dtype=W.dtype)
        E[torch.arange(len(top)), torch.tensor(top)] = 1.0
        self.sel.weight.data = E
        self.rest = torch.nn.Linear(Kd, N, bias=False)
        self.rest.weight.data = W.clone()
        self.topfc = torch.nn.Linear(len(top), N, bias=False)
        self.topfc.weight.data = W[:, top].clone()
        self.in_features, self.out_features, self.k = Kd, N, len(top)
        self.m = 1
        self.orig = [lin]

    def forward(self, x):
        return self.rest(x * self.mask) + self.topfc(self.sel(x))


def outlier_channels(path=None):
    import json
    from pathlib import Path
    p = Path(path) if path else Path(__file__).resolve().parents[1] / "results/r17_outlier_channels.json"
    return json.loads(p.read_text())["channels"]          # {"<layer>.<fc>": [channel, ...] by |x| descending}


class BmmLinear(R13.BmmLinear):
    """r13_kernel.BmmLinear with m = ceil(K / kmax) (r13_kernel takes K // chain); same forward, same class name."""

    def __init__(self, lin, kmax):
        torch.nn.Module.__init__(self)
        W = lin.weight.data
        assert lin.bias is None
        N, Kd = W.shape
        self.m = math.ceil(Kd / kmax)
        assert Kd % self.m == 0, (Kd, self.m)
        self.k = Kd // self.m
        self.register_buffer("Wb", W.t().reshape(self.m, self.k, N).clone().contiguous())
        self.in_features, self.out_features = Kd, N
        self.orig = [lin]


SIG_TOKENS = ("sgn", "sga2")


def gated_norm_sgn(self, hidden_states, gate=None):
    """r13_kernel.gated_norm_scaled (vs / gs / nr / rs aware) with silu(gate) = gate * sigmoid_exp2(gate) instead of
    F.silu (token sgn): a probe of every layer on the HTP put the whole-graph loss in this op (local error 2.7-6.4e-2
    relative RMS at layers 0 / 12 / 22, every FC at 1 ulp): its sigmoid carries an absolute error of ~2.3e-3 RMS
    (5.4e-3 max), 30-38% relative where sigmoid(gate) < 0.05. sigmoid_exp2 uses EXP arguments <= 0 only (EXP is ~1 ulp
    on the HTP). fp32: the same value within rounding."""
    input_dtype = hidden_states.dtype
    h = hidden_states.to(torch.float32)
    if R13.GN["in_log2"]:
        h = h * float(2 ** R13.GN["in_log2"])
    variance = R13._mean(h.pow(2))
    eps = torch.tensor(self.variance_epsilon, dtype=torch.float32) * float(4 ** R13.GN["eps_log4"])
    h = h * (R13.rsqrt_newton(variance + eps) if R13.NORM["newton"] else torch.rsqrt(variance + eps))
    h = self.weight * h.to(input_dtype)
    g = gate.to(torch.float32)
    h = h * (g * R13.sigmoid_exp2(g))
    return h.to(input_dtype)


def silu_exp2(x, inplace=False):
    """sga2: every SiLU (MLP act, GatedDeltaNet conv) as x * sigmoid_exp2(x)."""
    return x * R13.sigmoid_exp2(x)


_ORIG_SIG17 = {}


def parse(spec):
    toks = [t for t in spec.split("+") if t]
    sig = [t for t in toks if t in SIG_TOKENS]
    toks = [t for t in toks if t not in SIG_TOKENS]
    rest, fc = _parse(toks)
    return rest, fc, sig


def _parse(toks):
    r17 = [t for t in toks if R17_TOKEN.fullmatch(t)]
    assert len(r17) <= 1, f"one FC token: {r17}"
    rest = "+".join(t for t in toks if not R17_TOKEN.fullmatch(t))
    fc = None
    if r17:
        mt = R17_TOKEN.fullmatch(r17[0])
        default = ("down_proj", "out_proj", "o_proj") if mt.group(1) == "os" else OWNERS
        owners = tuple(re.split(r"[.,]", mt.group(3))) if mt.group(3) else default
        assert all(o in OWNERS for o in owners), owners
        fc = {"kind": mt.group(1), "kmax": int(mt.group(2)), "owners": owners, "token": r17[0]}
        r13f = R13.parse(rest)
        assert not r13f["fc_split"] and not r13f["bmm_chain"], "fc<n>k<k> / bmc<chain> next to a kc / bk / os token"
    return rest, fc


def swap_linears(model, fc):
    """Replace the selected bias-free nn.Linear of every decoder layer; returns [(parent, name, new module)]."""
    swaps = []
    chans = outlier_channels() if fc["kind"] == "os" else None
    for li, layer in enumerate(model.layers):
        for parent in list(layer.modules()):
            for name, child in list(parent.named_children()):
                if type(child) is not torch.nn.Linear or child.bias is not None or name not in fc["owners"]:
                    continue
                Kd = child.weight.shape[1]
                if fc["kind"] == "kc":
                    if Kd <= fc["kmax"]:
                        continue
                    new = KSplitLinear(child, fc["kmax"])
                elif fc["kind"] == "os":
                    new = OutlierSplitLinear(child, chans[f"{li}.{name}"][: fc["kmax"]])
                else:
                    new = BmmLinear(child, fc["kmax"])
                setattr(parent, name, new)
                swaps.append((parent, name, new))
    return swaps


def apply(model, spec):
    assert STATE["spec"] is None, f"form {STATE['spec']} already applied: reset first"
    rest, fc, sig = parse(spec)
    info = R13.apply(model, rest)
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    if sig:          # after R13.apply: R13 kept the original forward in R13._ORIG_GN when it patched the norm
        R13._ORIG_GN.setdefault("forward", M.Qwen3_5RMSNormGated.forward)
        M.Qwen3_5RMSNormGated.forward = gated_norm_sgn
    if "sga2" in sig:
        _ORIG_SIG17.setdefault("silu", torch.nn.functional.silu)
        torch.nn.functional.silu = silu_exp2
    swaps = swap_linears(model, fc) if fc else []
    STATE["spec"], STATE["swaps"] = spec, swaps
    shapes = {}
    for parent, name, new in swaps:
        key = f"{name} K{new.in_features} N{new.out_features} -> m{new.m} x {new.k}"
        shapes[key] = shapes.get(key, 0) + 1
    info = dict(info, spec=spec, r17_fc=fc, r17_sig=sig, r17_swapped=len(swaps), r17_swap_shapes=shapes,
                # r13_export.py reads these two: BATCH_MATMUL count for its V2 recipe and check
                bmm_chain=(fc["kmax"] if fc and fc["kind"] == "bk" else info.get("bmm_chain", 0)),
                split_linears=(len(swaps) if fc else info.get("split_linears", 0)))
    return info


def reset(model):
    for parent, name, new in STATE["swaps"]:
        setattr(parent, name, new.orig[0])
    STATE["swaps"] = []
    if "silu" in _ORIG_SIG17:
        torch.nn.functional.silu = _ORIG_SIG17.pop("silu")
    R13.reset(model)          # restores Qwen3_5RMSNormGated.forward from R13._ORIG_GN
    STATE["spec"] = None


def selftest():
    """fp64: every rewrite equals the Linear it replaces (random weights / inputs at the real shapes)."""
    import json
    torch.manual_seed(0)
    res = {}
    for K, N in ((1024, 6144), (2048, 1024), (3584, 1024), (1024, 16)):
        lin = torch.nn.Linear(K, N, bias=False).double()
        x = torch.randn(1, 128, K, dtype=torch.float64)
        ref = lin(x)
        for kind, kmax in (("kc", 1024), ("kc", 512), ("bk", 1024), ("bk", 256), ("bk", 64)):
            if kind == "kc" and K <= kmax:
                continue
            mod = KSplitLinear(lin, kmax) if kind == "kc" else BmmLinear(lin, kmax)
            res[f"K{K}_N{N}_{kind}{kmax}_m{mod.m}_fp64_max_abs"] = float((mod(x) - ref).abs().max())
    lin = torch.nn.Linear(3584, 1024, bias=False).double()
    x = torch.randn(1, 128, 3584, dtype=torch.float64)
    x[..., 46] *= 80.0
    mod = OutlierSplitLinear(lin, [46, 22, 84, 72])
    res["K3584_N1024_os4_fp64_max_abs"] = float((mod(x) - lin(x)).abs().max())
    print(json.dumps(res, indent=1))
    assert all(v < 1e-9 for v in res.values()), res
    return res


if __name__ == "__main__":
    selftest()
