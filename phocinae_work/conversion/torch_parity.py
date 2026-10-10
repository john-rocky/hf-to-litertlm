"""Step 3: the export graph in PyTorch (fp32) against the publisher's reference, before any conversion.

    python torch_parity.py --L 512 [--limit N] [--scan] [--buckets 256,384,512] [--ln-shift emb=2,final=3,...]

For every oracle row that fits L: the graph's token_logits gathered at the markers vs the reference logits
(max |dlogit|), post-temperature probabilities (max / mean |dp|, argmax), pooled_cls vs the reference CLS row, and the
act head graph fed with the graph's own raw marker logits vs the reference act logits. The host-lookup path
(inputs_embeds from the fp32 table) is checked bit for bit against the ids path on every row.
--scan records the fp16 headroom per site (LayerNorm inputs: |x|, (x-mean)^2, sum of squares; residual stream;
GeGLU product; q k^T scores) at the valid positions of the test rows -> results/fp16_headroom_L{L}.json.
--buckets runs each row at the smallest listed bucket it fits and reports the spread against the --L run.
"""
import argparse
import json
import time

import numpy as np
import torch

from common import HF, W, act_features, dump_json, probs_from_logits, temperatures
from graph import ActHeadGraph, PhocinaeGraph, embed_rows, load_decision_model, row_inputs


def parse_shift(s):
    out = {}
    for part in (s or "").split(","):
        if part.strip():
            k, v = part.split("=")
            out[k.strip()] = int(v)
    return out


def run_rows(g, act, rows, L, temps, table=None, taps=None, limit=0):
    recs = []
    t0 = time.time()
    for i, r in enumerate(rows):
        if r["row_len"] > L:
            continue
        ids, mask, onehot = row_inputs(r["ids_full"], L, qtype=r["qtype"])
        if taps is not None:
            g.valid = r["row_len"]
        with torch.no_grad():
            out = g.forward_ids(ids, mask, onehot)
            tl = out["token_logits"][0].numpy().astype(np.float64)
            cls = out["pooled_cls"][0].numpy().astype(np.float32)
            if table is not None:
                out2 = g.forward_embeds(embed_rows(table, ids), mask, onehot)
                emb_equal = torch.equal(out2["token_logits"], out["token_logits"]) and torch.equal(out2["pooled_cls"], out["pooled_cls"])
            else:
                emb_equal = None
            raw = tl[r["markers"]]
            feats = torch.tensor([act_features(raw)], dtype=torch.float32)
            al = act(out["pooled_cls"], feats)["act_logits"][0].numpy().astype(np.float64)
        p = probs_from_logits(raw, r["qtype"], temps)
        ref_p = np.asarray(r["probs"])
        recs.append(dict(id=r["id"], group=r["group"], row_len=r["row_len"], max_abs_dlogit=float(np.abs(raw - np.asarray(r["logits"])).max()),
                         max_abs_dp=float(np.abs(np.asarray(p) - ref_p).max()), mean_abs_dp=float(np.abs(np.asarray(p) - ref_p).mean()),
                         argmax_equal=int(np.argmax(p)) == r["argmax"], near_tie=r["near_tie"],
                         cls_max_abs=float(np.abs(cls - r["cls"]).max()), act_max_abs=float(np.abs(al - np.asarray(r["act_logits"])).max()),
                         finite=bool(np.isfinite(tl).all() and np.isfinite(cls).all() and np.isfinite(al).all()), embeds_bit_equal=emb_equal))
        if limit and len(recs) >= limit:
            break
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    return recs


def summarize(recs):
    nn_ = [r for r in recs if not r["near_tie"]]
    return dict(rows=len(recs), finite=sum(r["finite"] for r in recs), max_abs_dlogit=max(r["max_abs_dlogit"] for r in recs),
                max_abs_dp=max(r["max_abs_dp"] for r in recs), mean_abs_dp=float(np.mean([r["mean_abs_dp"] for r in recs])),
                argmax_equal_non_near_tie=f"{sum(r['argmax_equal'] for r in nn_)}/{len(nn_)}",
                argmax_equal_near_tie=f"{sum(r['argmax_equal'] for r in recs if r['near_tie'])}/{len(recs) - len(nn_)}",
                cls_max_abs=max(r["cls_max_abs"] for r in recs), act_max_abs=max(r["act_max_abs"] for r in recs),
                embeds_bit_equal=all(r["embeds_bit_equal"] for r in recs) if recs and recs[0]["embeds_bit_equal"] is not None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--buckets", default="")
    ap.add_argument("--ln-shift", default="")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    temps = temperatures()
    oracle = json.loads((W / "oracle" / "oracle.json").read_text())
    fx = {r["id"]: r for r in json.loads((W / "fixtures" / "rows_laya.json").read_text())["rows"]}
    cls = np.load(W / "oracle" / "pooled_cls.npz")
    rows = []
    for o in oracle["rows"]:
        o = dict(o)
        o["ids_full"] = fx[o["id"]]["ids"]
        o["cls"] = cls[o["id"].replace("/", "__")]
        rows.append(o)
    dm, eng = load_decision_model(HF)
    shift = parse_shift(a.ln_shift)
    g = PhocinaeGraph(dm, a.L, ln_shift=shift).eval()
    act = ActHeadGraph(dm).eval()
    table = dm.encoder.embeddings.tok_embeddings.weight.detach().float()
    tag = a.tag or (("_shift" if shift else "") + ("_scan" if a.scan else ""))
    if a.scan:
        g.taps = {}
        recs = run_rows(g, act, [r for r in rows if r["group"] == "test"], a.L, temps, table=None, taps=g.taps, limit=a.limit)
        taps = g.taps
        g.taps = None
        worst = sorted(taps.items(), key=lambda kv: -kv[1])[:40]
        over = {k: v for k, v in taps.items() if v > 65504.0}
        near = {k: v for k, v in taps.items() if 65504.0 >= v > 65504.0 / 4}
        doc = dict(L=a.L, rows=len(recs), ln_shift=shift, fp16_max=65504.0, over_fp16=over, within_4x_of_fp16=near, worst40=dict(worst),
                   parity=summarize(recs), all_sites=dict(sorted(taps.items())))
        dump_json(W / "results" / f"fp16_headroom_L{a.L}{tag}.json", doc)
        print(json.dumps({k: doc[k] for k in ("rows", "over_fp16", "within_4x_of_fp16", "parity")}, indent=1))
        print("worst sites:", json.dumps(dict(worst[:12]), indent=1))
        return
    recs = run_rows(g, act, rows, a.L, temps, table=table, limit=a.limit)
    doc = dict(L=a.L, ln_shift=shift, summary=summarize(recs), by_group={grp: summarize([r for r in recs if r["group"] == grp])
                                                                          for grp in sorted({r["group"] for r in recs})}, rows=recs)
    if a.buckets:
        buckets = [int(x) for x in a.buckets.split(",")]
        graphs = {L: PhocinaeGraph(dm, L, ln_shift=shift).eval() for L in buckets}
        spread = []
        base = {r["id"]: r for r in recs}
        for r in rows:
            Lb = next((L for L in sorted(buckets) if r["row_len"] <= L), None)
            if Lb is None or Lb == a.L or r["id"] not in base:
                continue
            ids, mask, onehot = row_inputs(r["ids_full"], Lb, qtype=r["qtype"])
            with torch.no_grad():
                tl = graphs[Lb].forward_ids(ids, mask, onehot)["token_logits"][0].numpy().astype(np.float64)
            raw = tl[r["markers"]]
            p = probs_from_logits(raw, r["qtype"], temps)
            spread.append(dict(id=r["id"], L=Lb, max_abs_dlogit_vs_ref=float(np.abs(raw - np.asarray(r["logits"])).max()),
                               max_abs_dp_vs_ref=float(np.abs(np.asarray(p) - np.asarray(r["probs"])).max())))
        doc["buckets"] = dict(list=buckets, rows=len(spread), max_abs_dlogit_vs_ref=max(s["max_abs_dlogit_vs_ref"] for s in spread),
                              max_abs_dp_vs_ref=max(s["max_abs_dp_vs_ref"] for s in spread), by_L={L: sum(1 for s in spread if s["L"] == L) for L in buckets})
    dump_json(W / "results" / f"torch_parity_L{a.L}{tag}.json", doc)
    print(json.dumps({k: v for k, v in doc.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
