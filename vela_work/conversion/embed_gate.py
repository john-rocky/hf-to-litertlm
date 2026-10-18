"""The parity bar for a set of embeddings against the reference (shared by the Mac, phone and NPU scorers).

score(vec, fx, oracle, ref) -> dict: per-row cos / max|d| vs oracle/embeddings.npz; pair-similarity Spearman vs the
reference's pair cosines; top-1 retrieval agreement per (group, language) block (every side a against every side b of the
block; near tie = the reference's top-2 gap <= NEAR_TIE excluded); the task metrics (PAWS-X AP per language, STS-B v2
Spearman) next to the reference's on the same rows; bar_core = finite + cos_min + top-1 + pair Spearman (the harness adds
mask liveness: the longest row with 5 valid tokens dropped must move, cos < 0.999).
"""
import numpy as np

from metrics import average_precision, spearman

BAR = dict(cos_min=0.999, top1_agreement="100 % on non-near-tie queries (reference top-2 gap > 0.01)", pair_spearman=0.999,
           liveness_cos_below=0.999)
NEAR_TIE = 0.01


def task_metrics(vec, pairs):
    out = {}
    pc = {p["key"]: float(np.asarray(vec[p["a"]], np.float64) @ np.asarray(vec[p["b"]], np.float64))
          for p in pairs if p["a"] in vec and p["b"] in vec}
    for lang in ("en", "es", "ja", "zh"):
        ps = [p for p in pairs if p["group"] == "pair" and p["lang"] == lang and p["key"] in pc]
        if len(ps) >= 20:
            out[f"pawsx_{lang}_ap"] = average_precision([pc[p["key"]] for p in ps], [p["label"] for p in ps])
    sts = [p for p in pairs if p["group"] == "stsb" and p["key"] in pc]
    if len(sts) >= 20:
        out["stsb_v2_spearman"] = spearman([pc[p["key"]] for p in sts], [p["score"] for p in sts])
    return out, pc


def score(vec, fx, oracle, ref, near_tie=NEAR_TIE):
    rows = [r for r in fx["rows"] if r["key"] in vec]
    recs = []
    for r in rows:
        v = np.asarray(vec[r["key"]], dtype=np.float32).reshape(-1)
        rv = ref[r["key"]]
        finite = bool(np.isfinite(v).all())
        recs.append(dict(key=r["key"], group=r["group"], row_len=r["row_len"], finite=finite, nonfinite=int((~np.isfinite(v)).sum()),
                         cos=float(v.astype(np.float64) @ rv.astype(np.float64)) if finite else None,
                         max_abs=float(np.abs(v - rv).max()) if finite else None, norm=float(np.linalg.norm(v)) if finite else None))
    fin = [x for x in recs if x["finite"]]
    nf = [x for x in recs if not x["finite"]]
    cos = np.array([x["cos"] for x in fin]) if fin else None
    pairs = [p for p in fx["pairs"] if p["a"] in vec and p["b"] in vec]
    cand_metrics, cand_pc = task_metrics(vec, pairs)
    ref_pc = oracle["pair_cos"]
    ref_metrics = {}
    for lang in ("en", "es", "ja", "zh"):
        if f"pawsx_{lang}_ap" in cand_metrics:
            ps = [p for p in pairs if p["group"] == "pair" and p["lang"] == lang]
            ref_metrics[f"pawsx_{lang}_ap"] = average_precision([ref_pc[p["key"]] for p in ps], [p["label"] for p in ps])
    if "stsb_v2_spearman" in cand_metrics:
        sts = [p for p in pairs if p["group"] == "stsb"]
        ref_metrics["stsb_v2_spearman"] = spearman([ref_pc[p["key"]] for p in sts], [p["score"] for p in sts])
    pair_keys = [p["key"] for p in pairs if p["key"] in ref_pc and p["key"] in cand_pc]
    pair_spear = spearman([cand_pc[k] for k in pair_keys], [ref_pc[k] for k in pair_keys]) if len(pair_keys) > 2 else None
    pair_max_abs_dcos = float(max(abs(cand_pc[k] - ref_pc[k]) for k in pair_keys)) if pair_keys else None
    top1 = dict(queries=0, agree=0, near_tie=0, blocks=[])
    for grp in sorted({p["group"] for p in pairs}):
        for lang in sorted({p["lang"] for p in pairs if p["group"] == grp}):
            ps = [p for p in pairs if p["group"] == grp and p["lang"] == lang]
            if len(ps) < 2:
                continue
            A_ref = np.stack([ref[p["a"]] for p in ps]); B_ref = np.stack([ref[p["b"]] for p in ps])
            A = np.stack([np.asarray(vec[p["a"]], np.float32).reshape(-1) for p in ps])
            B = np.stack([np.asarray(vec[p["b"]], np.float32).reshape(-1) for p in ps])
            if not (np.isfinite(A).all() and np.isfinite(B).all()):
                top1["blocks"].append(dict(group=grp, lang=lang, n=len(ps), nonfinite=True))
                continue
            S_ref, S = A_ref @ B_ref.T, A @ B.T
            srt = np.sort(S_ref, axis=1)
            nn_ = (srt[:, -1] - srt[:, -2]) > near_tie
            agree = int((S_ref.argmax(1)[nn_] == S.argmax(1)[nn_]).sum())
            top1["blocks"].append(dict(group=grp, lang=lang, n=len(ps), non_near_tie=int(nn_.sum()), agree=agree))
            top1["queries"] += int(nn_.sum()); top1["agree"] += agree; top1["near_tie"] += int((~nn_).sum())
    bar_core = bool(recs and not nf and float(cos.min()) >= BAR["cos_min"] and top1["agree"] == top1["queries"]
                    and (pair_spear is None or pair_spear >= BAR["pair_spearman"]))
    return dict(rows=dict(run=len(recs), by_group={g: sum(1 for x in recs if x["group"] == g) for g in sorted({x["group"] for x in recs})},
                          nonfinite_rows=len(nf), nonfinite_keys=[x["key"] for x in nf][:20]),
                cos=dict(min=float(cos.min()), mean=float(cos.mean()), p01=float(np.percentile(cos, 1)), p50=float(np.percentile(cos, 50)),
                         by_group={g: float(min(x["cos"] for x in fin if x["group"] == g)) for g in sorted({x["group"] for x in fin})}) if fin else None,
                max_abs=max(x["max_abs"] for x in fin) if fin else None,
                norm=dict(min=min(x["norm"] for x in fin), max=max(x["norm"] for x in fin)) if fin else None,
                pairs=dict(n=len(pair_keys), spearman_vs_reference=pair_spear, max_abs_dcos=pair_max_abs_dcos), top1=top1,
                metrics=dict(candidate=cand_metrics, reference_same_rows=ref_metrics,
                             delta={k: cand_metrics[k] - ref_metrics[k] for k in cand_metrics if k in ref_metrics}),
                bar=dict(**BAR, near_tie=near_tie), bar_core=bar_core, per_row=recs)
