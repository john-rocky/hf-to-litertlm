"""A form = the r13_kernel.py tokens + two pre-scale tokens for Kev-4B's norms, `fn<k>` and `in<k>`.

    import r15_form as F
    F.apply(model, "R64+sp+ec+dd+vs6+in1+fn5")      # r13_kernel.apply(model, "R64+sp+ec+dd+vs6") + the pre-scales
    F.reset(model)

fn<k>: the text model's FINAL RMSNorm (model.norm, the norm in front of the hidden output) runs on 2^-k x with eps
4^-k eps:
    x' = 2^-k x;  out = x' * rsqrt(mean(x'^2) + 4^-k eps) * (1 + w)
which equals x * rsqrt(mean(x^2) + eps) * (1 + w) exactly in fp32 (a power-of-two scale commutes with the rounding of
the products, the mean and the sum; rsqrt(4^-k v) = 2^k rsqrt(v); outside the subnormal range). Why: on Kev-4B the
final norm's input sum of squares over 2,560 values goes above the fp16 maximum 65,504 on ordinary rows
(r15_norm_range_probe.py), so with fp16 activations mean(x^2) overflows to inf, rsqrt(inf) = 0 and the readout rows
come out exactly 0 (Metal default precision). fn<k> changes only the final norm.
in<k>: the same pre-scale on every decoder layer's input_layernorm (2^-k x, eps 4^-k eps; fp32 bit-identical, same
argument), for that norm's fp16 headroom.
Nothing outside this file is modified (r13_kernel is imported, never edited)."""
import re
import types

import torch

import r13_kernel as R

STATE = {"spec": None, "fn": 0, "in": 0}


def split(spec):
    """-> (the r13_kernel spec, fn k, in k)."""
    toks = [t for t in spec.split("+") if t]
    fn = [int(t[2:]) for t in toks if re.fullmatch(r"fn\d+", t)]
    inn = [int(t[2:]) for t in toks if re.fullmatch(r"in\d+", t)]
    assert len(fn) <= 1 and len(inn) <= 1, spec
    rest = "+".join(t for t in toks if not re.fullmatch(r"(fn|in)\d+", t))
    return rest, (fn[0] if fn else 0), (inn[0] if inn else 0)


def _final_norm_prescaled(self, x):
    s = float(2.0 ** -self._r15_fn)
    xf = x.float() * s
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps * (s * s))
    out = out * (1.0 + self.weight.float())
    return out.type_as(x)


def _prescale(norm, k):
    assert type(norm).__name__ == "Qwen3_5RMSNorm" and not hasattr(norm, "_r15_fn"), type(norm)
    norm._r15_fn = k
    norm.forward = types.MethodType(_final_norm_prescaled, norm)


def _unscale(norm):
    del norm.forward            # back to the class method
    del norm._r15_fn


def apply(model, spec):
    assert STATE["spec"] is None, f"form {STATE['spec']} already applied: reset first"
    rest, fn, inn = split(spec)
    info = R.apply(model, rest) if rest else {"spec": "", "r11_applied": [], "kernel": "stock"}
    if fn:
        _prescale(model.norm, fn)
    if inn:
        for layer in model.layers:
            _prescale(layer.input_layernorm, inn)
    STATE.update(spec=spec, fn=fn, **{"in": inn})
    return {**info, "spec": spec, "r13_spec": rest, "final_norm_prescale_log2": -fn if fn else 0,
            "input_layernorm_prescale_log2": -inn if inn else 0, "expclamp": info.get("expclamp", False)}


def reset(model):
    if STATE["fn"]:
        _unscale(model.norm)
    if STATE["in"]:
        for layer in model.layers:
            _unscale(layer.input_layernorm)
    if STATE["spec"] is not None and split(STATE["spec"])[0]:
        R.reset(model)
    STATE.update(spec=None, fn=0, **{"in": 0})


def selftest():
    """fp32: the pre-scaled norm vs the stock norm on random and large inputs (bit equality expected)."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    torch.manual_seed(0)
    norm = M.Qwen3_5RMSNorm(2560, eps=1e-6)
    with torch.no_grad():
        norm.weight.normal_(0, 0.1)
    res = {}
    for name, x in (("randn", torch.randn(3, 64, 2560)), ("x300", torch.randn(3, 64, 2560) * 6.0),
                    ("outlier", torch.randn(3, 64, 2560).index_fill_(-1, torch.tensor([82, 762]), 300.0))):
        ref = norm(x)
        norm._r15_fn = 5
        norm.forward = types.MethodType(_final_norm_prescaled, norm)
        got = norm(x)
        del norm.forward
        del norm._r15_fn
        res[name] = {"bit_equal": bool(torch.equal(ref, got)), "max_abs": float((ref - got).abs().max())}
    return res


if __name__ == "__main__":
    import json
    print(json.dumps(selftest(), indent=1))
