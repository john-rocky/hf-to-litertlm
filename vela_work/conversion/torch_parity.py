"""Step 3: the export graph in PyTorch (fp32) against the reference embeddings, before any conversion.

    P=python3   # a Python 3.12 with litert-torch 0.10.0, litert-converter, ai-edge-litert 2.2.0, ai-edge-quantizer
    $P scripts/torch_parity.py --L 512 [--limit N] [--scan] [--ln-shift attn12-21=6,mlp12-21=6,final=6] [--tag t]

For every fixture row that fits L: the ids graph's embedding vs oracle/embeddings.npz (cos, max |d|), the embeds
(host-lookup) graph bit-equal to the ids graph, finite. --scan records the fp16 headroom per site of THIS graph at the
valid positions (LayerNorm inputs: |x|, (x-mean)^2, sum of squares; residual stream; GeGLU; attention scores) with the
given --ln-shift applied -> results/fp16_headroom_L{L}{tag}.json: the proof that a shift removes the overflow.
Output results/torch_parity_L{L}{tag}.json. Never overwrites.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from graph import VelaGraph, embed_rows, load_body, load_rope, parse_shift, row_inputs

W = Path(__file__).resolve().parents[1]
FP16_MAX = 65504.0


def dump_json(path, obj):
    path = Path(path)
    assert not path.exists(), f"never overwrite {path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--ln-shift", default="")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default="")
    ap.add_argument("--groups", default="")
    ap.add_argument("--table-sim", default="", help="int8 | int8+patch<K> (the K rows with the largest int8 relative error kept exact) | fp16: the table the graph sees")
    ap.add_argument("--keys-below", default="", help="<litert rows json>:<thresh> — only the fixture rows that gate file reports below thresh")
    ap.add_argument("--pool", choices=["where", "arith"], default="where", help="the pooling form of graph.py (arith -> tag _arith)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    shift = parse_shift(a.ln_shift)
    tag = a.tag or (("_shift" if shift else "") + ("_arith" if a.pool == "arith" else "") + ("_scan" if a.scan else "") + (f"_sim_{a.table_sim.replace('+', '_')}" if a.table_sim else ""))
    out = W / "results" / f"{'fp16_headroom' if a.scan else 'torch_parity'}_L{a.L}{tag}.json"
    assert not out.exists(), f"never overwrite {out}"
    fx = json.loads((W / "fixtures" / "texts.json").read_text())
    ref = np.load(W / "oracle" / "embeddings.npz")
    body = load_body(W / "hf")
    g = VelaGraph(body, a.L, load_rope(W / "oracle" / "rope_L512.npz"), ln_shift=shift, pool_mode=a.pool).eval()
    table = body.embeddings.tok_embeddings.weight.detach().float()
    sim_info = None
    if a.table_sim:
        t = table.numpy()
        if a.table_sim == "fp16":
            ts = t.astype(np.float16).astype(np.float32); patched = []
        else:
            scale = np.abs(t).max(1) / 127.0
            ts = (np.clip(np.rint(t / scale[:, None]), -127, 127) * scale[:, None]).astype(np.float32)
            patched = []
            if "+patch" in a.table_sim:
                k = int(a.table_sim.split("+patch")[1])
                err = np.linalg.norm(ts - t, axis=1) / np.maximum(np.linalg.norm(t, axis=1), 1e-12)
                patched = np.argsort(-err)[:k]
                ts[patched] = t[patched]
        sim_info = dict(spec=a.table_sim, patched_rows=int(len(patched)), patched_ids_head=[int(i) for i in list(patched)[:16]],
                        max_rel_err=float((np.linalg.norm(ts - t, axis=1) / np.maximum(np.linalg.norm(t, axis=1), 1e-12)).max()))
        with torch.no_grad():
            body.embeddings.tok_embeddings.weight.copy_(torch.from_numpy(ts))
            g.pad_row = torch.from_numpy(ts[0]).reshape(1, 1, -1).clone()
        table = torch.from_numpy(ts)
    groups = set(a.groups.split(",")) if a.groups else None
    rows = [r for r in fx["rows"] if r["row_len"] <= a.L and (groups is None or r["group"] in groups) and r["key"] in ref.files]
    if a.keys_below:
        path, thr = a.keys_below.rsplit(":", 1)
        gr = json.loads((W / path).read_text()); gr = gr["rows"] if isinstance(gr, dict) else gr
        keep = {r["key"] for r in gr if r["cos"] < float(thr)}
        rows = [r for r in rows if r["key"] in keep]
    if a.limit:
        rows = rows[: a.limit]
    if a.scan:
        g.taps = {}
    recs = []
    t0 = time.time()
    for i, r in enumerate(rows):
        ids, mask = row_inputs(r["ids"], a.L)
        g.valid = r["row_len"]
        with torch.no_grad():
            e = g.forward_ids(ids, mask)
            e2 = g.forward_embeds(embed_rows(table, ids), mask) if not a.scan else None
        v = e[0].numpy().astype(np.float32)
        rv = ref[r["key"]]
        recs.append(dict(key=r["key"], group=r["group"], row_len=r["row_len"], cos=float(v.astype(np.float64) @ rv.astype(np.float64)), max_abs=float(np.abs(v - rv).max()),
                         finite=bool(np.isfinite(v).all()), embeds_bit_equal=(bool(torch.equal(e, e2)) if e2 is not None else None)))
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    cos = np.array([x["cos"] for x in recs])
    summary = dict(L=a.L, ln_shift=shift, table_sim=sim_info, keys_below=a.keys_below or None, rows=len(recs), finite=sum(x["finite"] for x in recs), cos_min=float(cos.min()), cos_mean=float(cos.mean()),
                   cos_p01=float(np.percentile(cos, 1)), max_abs=max(x["max_abs"] for x in recs),
                   embeds_bit_equal=(all(x["embeds_bit_equal"] for x in recs) if not a.scan else None),
                   by_group={grp: dict(rows=sum(1 for x in recs if x["group"] == grp), cos_min=float(min(x["cos"] for x in recs if x["group"] == grp)))
                             for grp in sorted({x["group"] for x in recs})}, seconds=round(time.time() - t0, 1))
    if a.scan:
        taps = g.taps
        over = {k: v for k, v in taps.items() if v > FP16_MAX}
        near = {k: v for k, v in taps.items() if FP16_MAX >= v > FP16_MAX / 4}
        doc = dict(summary=summary, fp16_max=FP16_MAX, over_fp16=dict(sorted(over.items(), key=lambda kv: -kv[1])), within_4x_of_fp16=near,
                   worst40=dict(sorted(taps.items(), key=lambda kv: -kv[1])[:40]), all_sites=dict(sorted(taps.items())))
        dump_json(out, doc)
        print(json.dumps({k: doc[k] for k in ("summary", "over_fp16", "within_4x_of_fp16")}, indent=1))
        return
    dump_json(out, dict(summary=summary, rows=recs))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
