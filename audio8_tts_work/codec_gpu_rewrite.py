#!/usr/bin/env python3
"""codec_gpu_rewrite.py IN OUT [--promote] [--mask-floor F --mask-ops K,...] [--rms-scale S --rms-ops K,...|all]
[--report R.json]: rewrite an Audio8 codec decoder (codec_decoder_fp16_T128 / _T192.tflite) so the LiteRT GPU
accelerator computes it right, without changing what the CPU computes.

The fp16 decoders of the Audio8-TTS-Preview-0.6b conversion give a wrong output on the LiteRT GPU accelerator (ML
Drift), at fp16 and at fp32: measured on a Pixel 11 Pro and a Galaxy S26 (OpenCL; LiteRT 2.2.0 and 2.3.2) and on an
Apple M4 Max (Metal; ai-edge-litert 2.2.0). Three faults, one change each:
--promote      placement: a binary elementwise op whose runtime operand has a lower rank than its output and the same
               element count (codec_decoder_fp16_T128 ops 677 / 711: [N,1024] x const [1,1,1024] -> [1,N,1024]) gets a
               RESHAPE of that operand to the output shape first. ML Drift places a runtime rank-2 [a,b] as
               BHWC(a,1,1,b) and a rank-3 [1,a,b] as BHWC(1,1,a,b) (tools/gpu_graph/tfl_rank4.py), so without it the
               op multiplies the wrong elements, at fp32 too. A RESHAPE keeps the row-major order: the same values.
--rms-scale S  fp16 range: each RMSNorm square x * x (followed by SUM / MEAN, a MUL by a scalar, the ADD of eps, a
--rms-ops      RESHAPE, then RSQRT or SQRT; single consumers, read op by op, refused otherwise) becomes (x*S) * (x*S),
               eps becomes eps * S^2 and the RSQRT output is multiplied by S (a SQRT output by 1 / S). With S = 2^-4
               the 1,024-channel sums of squares stay inside fp16 (unscaled they reach about 76,500 > 65,504, rsqrt
               of the inf is 0 and the output stops depending on the codes). Powers of two only move the exponent:
               exact while nothing under- or overflows. "all" = every x * x MUL whose chain reads that way.
--mask-floor F the attention mask: the constant operand of each listed ADD is replaced by max(const, F). -1e9 is -inf
--mask-ops     in fp16, and in T192 a softmax row with no finite score turns into NaN; -30000 fits fp16, and exp() of
               a floored score is 0 in fp32 as well, so every row with one unmasked score gives the same softmax.
The Mac CPU output of the rewritten T128 and T192 decoders equals the original's byte for byte (15 and 17 test code
sequences). On the GPU at its default precision (fp16), on both phones, the rewritten decoders pass the speech checks
of the Audio8-TTS-Realistic-0.6b model card.
Uses tools/gpu_graph/tfl_rank4.py of this repository (Graph, compact, write), loaded by path.

  python codec_gpu_rewrite.py codec_decoder_fp16_T128.tflite out/codec_decoder_fp16_T128_gpu.tflite --promote \\
      --mask-floor -30000 --mask-ops 155,220,285,350,415,480,545,610 --rms-scale 0.0625 --rms-ops all
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np

T4P = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "tools", "gpu_graph", "tfl_rank4.py")
spec = importlib.util.spec_from_file_location("tfl_rank4", T4P)
t4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t4)
s = t4.s


def consumers(sg, ti):
    return [k for k, op in enumerate(sg.operators) if ti in op.inputs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--promote", action="store_true")
    ap.add_argument("--mask-floor", type=float)
    ap.add_argument("--mask-ops", default="")
    ap.add_argument("--rms-scale", type=float)
    ap.add_argument("--rms-ops", default="")
    ap.add_argument("--report")
    a = ap.parse_args()
    g = t4.Graph(a.src)
    sg = g.sg
    report = []
    pre = {}   # op index -> ops to insert before it
    post = {}  # op index -> ops to insert after it
    if a.promote:
        for k, op in enumerate(sg.operators):
            if g.opname(op) not in t4.ELEMWISE or len(op.inputs) != 2:
                continue
            c = [g.is_const(i) for i in op.inputs]
            if c[0] == c[1]:
                continue
            ri = 1 if c[0] else 0
            rt, ot = sg.tensors[op.inputs[ri]], sg.tensors[op.outputs[0]]
            if len(rt.shape) < len(ot.shape) and int(np.prod(rt.shape)) == int(np.prod(ot.shape)):
                nt = g.add_tensor(ot.shape, rt.type, rt.name + b"/promoted")
                pre.setdefault(k, []).append(g.reshape_op(op.inputs[ri], nt))
                ins = list(op.inputs)
                ins[ri] = nt
                op.inputs = ins
                report.append({"rule": "promote", "op": k, "name": g.opname(op), "runtime": rt.shape,
                               "const": sg.tensors[op.inputs[1 - ri]].shape, "out": ot.shape})
    if a.mask_floor is not None:
        floored = {}
        for k in [int(x) for x in a.mask_ops.split(",") if x]:
            op = sg.operators[k]
            assert g.opname(op) == "ADD", (k, g.opname(op))
            ci = 0 if g.is_const(op.inputs[0]) else 1
            assert g.is_const(op.inputs[ci]) and not g.is_const(op.inputs[1 - ci]), k
            arr = g.const(op.inputs[ci])
            assert arr.dtype == np.float32, (k, arr.dtype)
            new = np.maximum(arr, np.float32(a.mask_floor)).astype(np.float32)
            if op.inputs[ci] not in floored:
                floored[op.inputs[ci]] = g.add_const(new, sg.tensors[op.inputs[ci]].name + b"/floor")
            nt = floored[op.inputs[ci]]
            ins = list(op.inputs)
            ins[ci] = nt
            op.inputs = ins
            report.append({"rule": "mask_floor", "op": k, "const": list(arr.shape), "min_before": float(arr.min()),
                           "min_after": float(new.min()), "changed": int((arr < a.mask_floor).sum())})
    if a.rms_scale is not None:
        S = np.float32(a.rms_scale)
        cons = {}
        for k, op in enumerate(sg.operators):
            for i in op.inputs:
                cons.setdefault(i, []).append(k)

        def rms_chain(k):
            """[(op index, name), ...] from the square's consumer to the RSQRT / SQRT, or None."""
            t, out = sg.operators[k].outputs[0], []
            for _ in range(6):
                cs = cons.get(t, [])
                if len(cs) != 1:
                    return None
                nm = g.opname(sg.operators[cs[0]])
                if nm not in ("SUM", "MEAN", "MUL", "ADD", "RESHAPE", "RSQRT", "SQRT"):
                    return None
                out.append((cs[0], nm))
                if nm in ("RSQRT", "SQRT"):
                    return out if out[0][1] in ("SUM", "MEAN") and [n for _, n in out].count("ADD") == 1 else None
                t = sg.operators[cs[0]].outputs[0]
            return None

        if a.rms_ops == "all":
            ks = [k for k, op in enumerate(sg.operators) if g.opname(op) == "MUL" and op.inputs[0] == op.inputs[1]
                  and rms_chain(k) is not None]
        else:
            ks = [int(x) for x in a.rms_ops.split(",") if x]
        for k in ks:
            op = sg.operators[k]
            assert g.opname(op) == "MUL" and op.inputs[0] == op.inputs[1], (k, g.opname(op), op.inputs)
            chain = rms_chain(k)
            assert chain is not None, f"op {k}: the chain after the square is not SUM/MEAN ... ADD eps ... RSQRT/SQRT"
            kadd = next(c for c, n in chain if n == "ADD")
            krs, rsn = chain[-1]
            x = op.inputs[0]
            sc = g.add_const(np.array([S], np.float32), sg.tensors[x].name + b"/rms_s")
            xs = g.add_tensor(sg.tensors[x].shape, sg.tensors[x].type, sg.tensors[x].name + b"/rms_xs")
            mop = s.OperatorT()
            mop.opcodeIndex = g.opcode("MUL")
            mop.inputs, mop.outputs = [x, sc], [xs]
            mop.builtinOptionsType = s.BuiltinOptions.MulOptions
            mop.builtinOptions = s.MulOptionsT()
            pre.setdefault(k, []).append(mop)
            op.inputs = [xs, xs]
            aop = sg.operators[kadd]
            ei = 0 if g.is_const(aop.inputs[0]) else 1
            assert g.is_const(aop.inputs[ei]) and not g.is_const(aop.inputs[1 - ei]), kadd
            eps = g.const(aop.inputs[ei])
            assert eps.size == 1 and eps.dtype == np.float32, (kadd, eps.shape, eps.dtype)
            ne = g.add_const((eps * S * S).astype(np.float32).reshape(eps.shape),
                             sg.tensors[aop.inputs[ei]].name + b"/rms_s2")
            ins = list(aop.inputs)
            ins[ei] = ne
            aop.inputs = ins
            rop = sg.operators[krs]
            rout = rop.outputs[0]
            r2 = g.add_tensor(sg.tensors[rout].shape, sg.tensors[rout].type, sg.tensors[rout].name + b"/rms_pre")
            rop.outputs = [r2]
            f = S if rsn == "RSQRT" else np.float32(1.0) / S
            sc2 = g.add_const(np.array([f], np.float32), sg.tensors[rout].name + b"/rms_f")
            fop = s.OperatorT()
            fop.opcodeIndex = g.opcode("MUL")
            fop.inputs, fop.outputs = [r2, sc2], [rout]
            fop.builtinOptionsType = s.BuiltinOptions.MulOptions
            fop.builtinOptions = s.MulOptionsT()
            post.setdefault(krs, []).append(fop)
            report.append({"rule": "rms_scale", "op": k, "chain": [f"{c}:{n}" for c, n in chain], "S": float(S),
                           "eps_before": float(eps.reshape(-1)[0]), "eps_after": float((eps * S * S).reshape(-1)[0]),
                           "out_factor": float(f), "x": sg.tensors[x].shape})
    ops = []
    for k, op in enumerate(sg.operators):
        ops.extend(pre.get(k, []) + [op] + post.get(k, []))
    sg.operators = ops
    t4.compact(g)
    hlen, size = t4.write(g, a.dst)
    by = {}
    for r in report:
        by[r["rule"]] = by.get(r["rule"], 0) + 1
    print(f"wrote {a.dst}: ops {len(sg.operators)} tensors {len(sg.tensors)} header {hlen} B file {size} B; sites {by}", flush=True)
    if a.report:
        json.dump({"src": a.src, "dst": a.dst, "sites": report}, open(a.report, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
