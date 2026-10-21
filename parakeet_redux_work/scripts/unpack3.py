#!/usr/bin/env python3
"""Unpack moondream/parakeet-redux ternary weights (thrush-ternary-v2) and run the self-checks.

ternary.json rule: w = scales[row, col // group_size] * (code - 1), code in {0, 1, 2}; five base-3 digits per
byte, element i of a row = digit i % 5 of byte i // 5 (least significant digit first), row_bytes = ceil(in / 5),
pad digits 0, row major; `as_conv1d` modules are Conv1d weights of shape [out, in, 1].

Library use: `unpack_module(qweight_u8, scales_f16, in_features)` -> (codes uint8 [out, in], w float32 [out, in]);
`load_redux_state_dict(st_path, ternary_json)` -> {name: torch.float32 tensor} with every tensor of the file
(ternary modules expanded to `<name>.weight`, F16 dense tensors raised to fp32, I64 kept).

CLI: unpack3.py <model.safetensors> <ternary.json> <out.json>   (self-checks a-f, exit 1 on any FAIL)
"""
import json
import math
import sys

import numpy as np

# LUT[b, k] = base-3 digit k (least significant first) of byte value b, for b in 0..255
_LUT = np.zeros((256, 5), np.uint8)
for _b in range(256):
    _v = _b
    for _k in range(5):
        _LUT[_b, _k] = _v % 3
        _v //= 3


def digits(qweight):
    """uint8 [out, row_bytes] -> base-3 digits uint8 [out, row_bytes * 5] (pad digits included)."""
    out, rb = qweight.shape
    return _LUT[qweight].reshape(out, rb * 5)


def unpack_module(qweight, scales, in_features, group_size=128):
    d = digits(qweight)
    codes = d[:, :in_features]
    s = scales.astype(np.float32)
    s_full = np.repeat(s, group_size, axis=1)[:, :in_features]
    w = (codes.astype(np.float32) - 1.0) * s_full
    return codes, w


def _read_safetensors(path):
    from safetensors.numpy import load_file
    return load_file(path)


def load_redux_state_dict(st_path, ternary_json):
    import torch
    meta = json.load(open(ternary_json))
    tens = _read_safetensors(st_path)
    gs = meta["quant"]["group_size"]
    sd = {}
    used = set()
    for m in meta["quantized_modules"]:
        n = m["name"]
        q, s = tens[n + ".qweight"], tens[n + ".scales"]
        _, w = unpack_module(q, s, m["in_features"], gs)
        assert w.shape == (m["out_features"], m["in_features"]), (n, w.shape)
        if m["as_conv1d"]:
            w = w[:, :, None]
        sd[n + ".weight"] = torch.from_numpy(np.array(w, copy=True, order="C"))
        used.update([n + ".qweight", n + ".scales"])
    for k, v in tens.items():
        if k in used:
            continue
        # np.array keeps a 0-d tensor 0-d (np.ascontiguousarray turns BatchNorm's num_batches_tracked [] into [1])
        t = torch.from_numpy(np.array(v, copy=True, order="C"))
        if t.dtype == torch.float16:
            t = t.float()
        sd[k] = t
    return sd


def main(argv):
    st_path, tj, out_json = argv[1], argv[2], argv[3]
    meta = json.load(open(tj))
    gs = meta["quant"]["group_size"]
    tens = _read_safetensors(st_path)
    sub = np.float32(2.0 ** -14)
    rep = {"format": meta.get("format"), "group_size": gs, "modules": [], "checks": {}}
    total = 0
    a_max_byte = 0
    b_bad_pad = 0
    b_pad_digits_total = 0
    c_worst = 0.0
    c_fail = []
    s_min = np.inf
    s_zero = s_sub = s_neg = s_nonfinite = s_total = 0
    f_allzero = f_allzero_nonzero_scale = f_groups = 0
    code_hist = np.zeros(3, np.int64)
    for m in meta["quantized_modules"]:
        n, o, i = m["name"], m["out_features"], m["in_features"]
        q, s = tens[n + ".qweight"], tens[n + ".scales"]
        rb = math.ceil(i / 5)
        assert q.dtype == np.uint8 and q.shape == (o, rb), (n, q.dtype, q.shape)
        ng = math.ceil(i / gs)
        assert s.dtype == np.float16 and s.shape == (o, ng), (n, s.dtype, s.shape)
        a_max_byte = max(a_max_byte, int(q.max()))
        d = digits(q)
        pad = d[:, i:]
        b_pad_digits_total += pad.size
        b_bad_pad += int((pad != 0).sum())
        codes = d[:, :i]
        hist = np.bincount(codes.reshape(-1), minlength=3)
        code_hist += hist[:3]
        zf = float(hist[1]) / float(o * i)
        diff = abs(zf - m["zero_fraction"])
        c_worst = max(c_worst, diff)
        if diff > 1e-9:
            c_fail.append((n, zf, m["zero_fraction"]))
        total += o * i
        sf = s.astype(np.float32)
        s_total += sf.size
        fin = np.isfinite(sf)
        s_nonfinite += int((~fin).sum())
        s_zero += int((sf == 0).sum())
        s_sub += int(((sf > 0) & (sf < sub)).sum())
        s_neg += int((sf < 0).sum())
        s_min = min(s_min, float(sf[fin].min()))
        grp_allzero = (codes.reshape(o, ng, gs) == 1).all(axis=2)
        f_groups += grp_allzero.size
        f_allzero += int(grp_allzero.sum())
        f_allzero_nonzero_scale += int((grp_allzero & (sf != 0)).sum())
        rep["modules"].append({
            "name": n, "out": o, "in": i, "as_conv1d": m["as_conv1d"],
            "zero_fraction": zf, "zero_fraction_json": m["zero_fraction"],
            "codes_0_1_2": [int(x) for x in hist[:3]],
            "scale_min": float(sf.min()), "scale_max": float(sf.max()),
            "allzero_groups": int(grp_allzero.sum()),
        })
    checks = rep["checks"]
    checks["a_max_byte"] = {"value": a_max_byte, "pass": a_max_byte <= 242}
    checks["b_pad_digits"] = {"pad_digits": b_pad_digits_total, "nonzero": b_bad_pad, "pass": b_bad_pad == 0}
    checks["c_zero_fraction"] = {"worst_abs_diff": c_worst, "fails": c_fail[:10], "pass": not c_fail}
    checks["d_total_elements"] = {"value": total, "want": meta["n_quantized_params"],
                                  "pass": total == meta["n_quantized_params"] == 603979776}
    checks["e_scales"] = {"count": s_total, "min": s_min, "zero": s_zero, "subnormal_lt_2^-14": s_sub,
                          "negative": s_neg, "nonfinite": s_nonfinite,
                          "pass": s_nonfinite == 0}
    checks["f_allzero_groups"] = {"groups": f_groups, "allzero": f_allzero,
                                  "allzero_with_nonzero_scale": f_allzero_nonzero_scale, "pass": True}
    rep["code_hist_0_1_2"] = [int(x) for x in code_hist]
    rep["n_modules"] = len(meta["quantized_modules"])
    rep["all_pass"] = all(v["pass"] for v in checks.values())
    json.dump(rep, open(out_json, "w"), indent=1)
    for k, v in checks.items():
        print(k, json.dumps(v))
    print("code histogram (0,1,2):", rep["code_hist_0_1_2"], "modules", rep["n_modules"])
    print("ALL PASS" if rep["all_pass"] else "FAIL")
    return 0 if rep["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
