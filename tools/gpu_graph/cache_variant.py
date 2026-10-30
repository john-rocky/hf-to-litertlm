#!/usr/bin/env python3
"""cache_variant.py SRC.tflite DST.tflite --cache 2048 --report R.json [--dry-run]

A KV-cache-length variant of an exported LiteRT-LM decoder (.tflite with prefill_<N> / decode signatures, KV caches as
signature inputs and outputs), without a re-export. In every signature's subgraph the cache length C is propagated
symbolically through the ops with the Propagation of prefill_bucket_clone.py, seeded at
  kv_cache_k_* [1, KV, C, hd] (dim 2), kv_cache_v_* [1, KV, hd, C] (dim 3), mask [1, 1, T, C] (dim 3);
every tensor shape, RESHAPE shape constant and SLICE size constant that carries C is rewritten for the new length (the
propagation never replaces a number because it equals C: Muse's 32 heads x 128 = 4096 = its exported C). Each subgraph
is replaced by its rewritten copy at the same index; SignatureDefs keep their keys, IO names and tensor indices; the
weight buffers (external region) are copied verbatim and checked byte for byte. A graph with STABLEHLO_COMPOSITE ops, a
constant whose size along C is not 1 (an arange over the cache) or a fixed SLICE window of the C dim larger than the
new length stops the tool.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gpu_graph_retrofit as R  # noqa: E402
import prefill_bucket_clone as PB  # noqa: E402
PB.ALLOW_CONTRACT = True

name_of = R.name_of


def seeds_for(sg, sd, C0):
    seeds, found = {}, {}
    for tm in sd.inputs:
        n, t = name_of(tm.name), tm.tensorIndex
        s = PB.shp(sg, t)
        if n.startswith("kv_cache_k_"):
            if len(s) != 4 or s[2] != C0:
                raise PB.CloneError(f"{n} {s}: dim 2 is not C0={C0}")
            seeds[t] = (2, 1, 1)
        elif n.startswith("kv_cache_v_"):
            if len(s) != 4 or s[3] != C0:
                raise PB.CloneError(f"{n} {s}: dim 3 is not C0={C0}")
            seeds[t] = (3, 1, 1)
        elif n in ("mask", "mask_local", "attn_mask"):
            if len(s) != 4 or s[3] != C0:
                raise PB.CloneError(f"{n} {s}: dim 3 is not C0={C0}")
            seeds[t] = (3, 1, 1)
        else:
            seeds[t] = None
        found[n] = s
    return seeds, found


def arange_consts(M, sg, C0):
    """Constant op inputs that are the cache positions 0..C0-1 along one dim (every other dim 1), e.g. litert-torch's
    sliding-window mask builder [1, 1, C]: -> {tensor: dim}. Their values are rebuilt for the new length."""
    import numpy as np
    found = {}
    seen = set()
    for op in sg.operators:
        for t in op.inputs:
            if t < 0 or t in seen:
                continue
            seen.add(t)
            s = PB.shp(sg, t)
            if s.count(C0) != 1 or any(v != 1 for v in s if v != C0):
                continue
            v = M.const(sg, t)  # INT32 / INT64 constants only (None for activations and other types)
            if v is not None and v.size == C0 and np.array_equal(v, np.arange(C0)):
                found[t] = s.index(C0)
    return found


def variant(src, dst, C1, dry_run=False):
    model, data_start, size = R.load_tflite(src)
    M = PB.Model(model, src)
    for si, sg in enumerate(model.subgraphs):
        for op in sg.operators:
            if M.opcode(op) == PB.OP.STABLEHLO_COMPOSITE:
                raise PB.CloneError(f"subgraph {si} has STABLEHLO_COMPOSITE ops (decompositions are not handled here)")
    keys = [name_of(s.signatureKey) for s in model.signatureDefs]
    dec = model.signatureDefs[keys.index("decode")]
    C0 = next(PB.shp(model.subgraphs[dec.subgraphIndex], tm.tensorIndex)[2] for tm in dec.inputs
              if name_of(tm.name).startswith("kv_cache_k_"))
    rep = {"src": src, "C0": C0, "C1": C1, "signatures": {}}
    n_sub0 = len(model.subgraphs)
    cl = PB.Cloner(M)
    for sd in model.signatureDefs:
        key, si = name_of(sd.signatureKey), sd.subgraphIndex
        sg = model.subgraphs[si]
        seeds, found = seeds_for(sg, sd, C0)
        ar = arange_consts(M, sg, C0)
        for t, d in ar.items():
            seeds[t] = (d, 1, 1)
        prop = PB.Propagation(M, si, seeds, C0, key, lambda *a: None)
        summ = prop.summary()
        for w in summ["fixed_windows"]:
            if w["size"] > w["o"] * C1 * w["i"]:
                raise PB.CloneError(f"{key}: a fixed SLICE window of {w['size']} does not fit {w['o'] * C1 * w['i']}")
        outs = {}
        for tm in sd.outputs:
            st = prop.st.get(tm.tensorIndex)
            n = name_of(tm.name)
            if n.startswith("kv_cache_") and st is None:
                raise PB.CloneError(f"{key}: output {n} does not carry C")
            if not n.startswith("kv_cache_") and st is not None:
                raise PB.CloneError(f"{key}: output {n} carries C {st}")
            outs[n] = st
        ni, crep = cl.clone_sg(prop, C1, lambda s: s)
        if ni != len(model.subgraphs) - 1:
            raise PB.CloneError("the copy did not land at the end")
        new_sg = model.subgraphs.pop()
        new_sg.name = sg.name
        import numpy as np
        for t, d in ar.items():  # the positions constant: values 0..C1-1 at the new shape
            x = new_sg.tensors[t]
            dt = PB.NP_OF[x.type]
            x.buffer = cl.buffer_for(np.arange(C1, dtype=dt).reshape([int(v) for v in x.shape]))
        model.subgraphs[si] = new_sg
        rep["signatures"][key] = {"subgraph": si, "C_tensors": crep["tensors_T"], "RESHAPE.shape": crep["RESHAPE.shape"],
                                  "SLICE.size": crep["SLICE.size"], "const_rewritten_in_place": crep["const_rewritten_in_place"],
                                  "const_split_new_tensor": crep["const_split_new_tensor"],
                                  "ambiguous_reshapes": len(summ.get("ambiguous", [])) if isinstance(summ.get("ambiguous"), list) else summ.get("ambiguous"),
                                  "kv_outputs": sum(1 for n in outs if n.startswith("kv_cache_")),
                                  "position_constants": {int(t): int(d) for t, d in ar.items()}}
    if len(model.subgraphs) != n_sub0:
        raise PB.CloneError("subgraph count changed")
    rep["new_buffers"] = len(cl.new_buffers)
    if dry_run:
        return rep
    rep["layout"] = R.write_tflite(model, src, data_start, size, dst)
    rep["dst"] = {"path": dst, "bytes": os.path.getsize(dst)}
    rep["buffer_identity"] = R.buffer_identity(src, dst)
    if not rep["buffer_identity"]["ok"]:
        raise PB.CloneError(f"buffer identity failed: {rep['buffer_identity']}")
    out, _, _ = R.load_tflite(dst)
    back = {}
    for sd in out.signatureDefs:
        sg = out.subgraphs[sd.subgraphIndex]
        ks = [PB.shp(sg, tm.tensorIndex) for tm in sd.inputs if name_of(tm.name).startswith("kv_cache_k_")]
        ms = [PB.shp(sg, tm.tensorIndex) for tm in sd.inputs if name_of(tm.name) == "mask"]
        back[name_of(sd.signatureKey)] = {"kv_k0": ks[0] if ks else None, "mask": ms[0] if ms else None}
    rep["readback"] = back
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--cache", type=int, required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    rep = variant(a.src, a.dst, a.cache, a.dry_run)
    with open(a.report, "w") as f:
        json.dump(rep, f, indent=1, default=str)
    print(json.dumps({k: rep[k] for k in rep if k not in ("layout",)}, indent=1, default=str)[:3000])
    print("CACHE_VARIANT_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
