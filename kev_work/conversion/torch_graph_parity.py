"""Graph check in PyTorch: the KevPrefill graph (before export) against the unpatched baseline (tf514_baseline.py,
unpadded rows) and against the reference.

Every question whose row fits L is right-padded to L (pad 248044, valid 1/0) and run through KevPrefill (fp32, eval,
no_grad, torch 1 thread per worker; rows are sharded over --workers processes because one L=1024 row takes ~2.4 s at
1 thread on 0.8B and more threads are slower on this Mac). Per question: real-position hidden max |diff| vs the
baseline's hidden of the same row (cache/r2/tf514_hidden_full.npz), and the head readout vs the reference (argmax,
|dp|, h_sel vs oracle/hidden_<model>.npz). Rows longer than L are skipped and listed.
Also: kev_eager vs transformers' eager_attention_forward on the same q/k/v/mask, bit for bit (random tensors, and the
real q/k/v captured at the attention layers on one padded row), and the concat repeat vs repeat_kv.

-> results/torch_graph_parity_L{L}[_4b].json (never overwritten); cache/r2/torch_graph_hsel_L{L}.npz (h_sel + 3 full
rows, real positions; an intermediate).

--model 4b (d = 2560, 16 q / 4 kv heads, GatedDeltaNet 32 v / 16 k heads; names get _4b, files in cache/r4b):
  - the attention checks take their shapes from the model config; the head interleave of the patch is checked bit for
    bit against repeat_interleave (random tensors and the real query / key of one row) and its calls are counted (must
    be 2 per linear-attention layer and per forward);
  - --phase compute --rows-from oracle/oracle_0.8b.json runs the graph without the 4B reference (the 0.8B reference's
    rows are the 4B rows: same fixtures, same tokenizer) and keeps h_sel + every real position of every row
    in cache/r4b/torch_graph_rows_L{L}_4b.npz; --phase precheck compares those hidden states with the baseline's (no
    reference; the export may start on it, recorded as such); --phase compare (after the 4B reference) asserts the rows
    are the reference's rows, then writes the results json. --phase all (default) runs everything in one go."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from r2_common import (HIDDEN, MODEL, ORACLE_NPZ, RESULT_SUFFIX, K, Clock, Head, add_model_arg, cache_dir,
                       compare_question, dump_json, load_oracle, qkey, select, summarize)

STEP1_NPZ = cache_dir("cache/r2") / f"tf514_hidden_full{RESULT_SUFFIX}.npz"
BAR_HIDDEN_VS_UNPATCHED = 1e-4 if MODEL == "0.8b" else 1e-3   # hidden vs stock: 1e-4 on 0.8B, 1e-3 on 4B
REF_HIDDEN_1E4 = 1e-4


def rows_sha(q):
    return hashlib.sha256(json.dumps([q["row_ids"], q["decide_idx"], q["opt_idx"]]).encode()).hexdigest()


def shard_paths(L, k):
    d = cache_dir("cache/r2")
    return d / f"torch_graph_L{L}{RESULT_SUFFIX}_w{k}.npz", d / f"torch_graph_L{L}{RESULT_SUFFIX}_w{k}.json"


def compute_paths(L):
    d = cache_dir("cache/r2")
    return d / f"torch_graph_rows_L{L}{RESULT_SUFFIX}.npz", d / f"torch_graph_compute_L{L}{RESULT_SUFFIX}.json"


def question_list(rows_from):
    return (json.loads(Path(rows_from).read_text()) if rows_from else load_oracle())["questions"]


def worker(L, k, workers, rows_from):
    torch.set_num_threads(1)
    from kev_graph import KevPrefill, count_interleave, load_text_model, row_inputs
    model, info = load_text_model()
    counter = count_interleave()
    graph = KevPrefill(model, L).eval()
    fits = [q for q in question_list(rows_from) if q["row_len"] <= L]
    mine = fits[k::workers]
    store, rows = {}, []
    with torch.no_grad():
        for q in mine:
            ids, valid = row_inputs(q["row_ids"], L)
            t = time.time()
            h = graph(ids, valid)["hidden"][0].numpy()
            dt = time.time() - t
            n = q["row_len"]
            assert h.shape == (L, HIDDEN)
            rows.append({"key": qkey(q), "seconds": round(dt, 3), "finite_all_positions": bool(np.isfinite(h).all())})
            store[qkey(q)] = select(h, q).astype(np.float32)
            store[qkey(q) + "/real"] = h[:n].astype(np.float32)
    npz, js = shard_paths(L, k)
    np.savez(npz, **store)
    js.write_text(json.dumps({"worker": k, "workers": workers, "load_info": info, "rows": rows,
                              "interleave_calls": counter["calls"], "forwards": len(mine)}))


def attention_bit_checks(L, rows_from):
    """kev_eager vs eager_attention_forward, bit for bit; concat repeat vs repeat_kv; on a ratio>1 GatedDeltaNet the
    patch's head interleave vs repeat_interleave, bit for bit, and its call count on one row."""
    torch.set_num_threads(1)
    from kev_graph import ATTN, KevPrefill, count_interleave, kev_eager, load_text_model, row_inputs
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    import kev_qwen35_patch as P
    model, _ = load_text_model()
    attn_mod = next(m for m in model.modules() if isinstance(m, M.Qwen3_5Attention))
    gdn = [m for m in model.modules() if isinstance(m, M.Qwen3_5GatedDeltaNet)]
    nq, nkv, hd = model.config.num_attention_heads, model.config.num_key_value_heads, attn_mod.head_dim
    n_rep = nq // nkv
    out = {"random": [], "captured": [], "shapes": {"q_heads": nq, "kv_heads": nkv, "head_dim": hd, "n_rep": n_rep}}
    g = torch.Generator().manual_seed(0)
    for n_real in (L, L // 2 + 7, 1):
        q = torch.randn(1, nq, L, hd, generator=g)
        k = torch.randn(1, nkv, L, hd, generator=g)
        v = torch.randn(1, nkv, L, hd, generator=g)
        graph = KevPrefill(model, L)
        valid = torch.zeros(1, L)
        valid[0, :n_real] = 1.0
        mask = graph.causal_const + (1.0 - valid)[:, None, None, :] * -1e4
        a, _ = kev_eager(attn_mod, q, k, v, mask, attn_mod.scaling)
        b, _ = M.eager_attention_forward(attn_mod, q, k, v, mask, attn_mod.scaling)
        rep = torch.cat([k.reshape(nkv, 1, L, hd)] * n_rep, dim=1).reshape(1, nq, L, hd)
        out["random"].append({"n_real": n_real, "bit_equal": bool(torch.equal(a, b)), "max_abs": float((a - b).abs().max()),
                              "repeat_bit_equal": bool(torch.equal(rep, M.repeat_kv(k, n_rep)))})
    ratio = gdn[0].num_v_heads // gdn[0].num_k_heads
    inter = {"ratio": ratio, "num_k_heads": gdn[0].num_k_heads, "num_v_heads": gdn[0].num_v_heads,
             "linear_attention_layers": len(gdn)}
    if ratio > 1:
        x = torch.randn(1, L, gdn[0].num_k_heads, gdn[0].head_k_dim, generator=g)
        inter["random_bit_equal"] = bool(torch.equal(P._litert_interleave_heads(x, ratio), x.repeat_interleave(ratio, dim=2)))
    # real tensors: capture the attention calls and the interleave inputs of one padded row
    captured, inter_inputs = [], []

    def capture(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        captured.append((module, query.clone(), key.clone(), value.clone(), attention_mask.clone(), scaling))
        return kev_eager(module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs)

    counter = count_interleave()
    globs = P.PatchedQwen3_5GatedDeltaNet.forward.__globals__
    counted = globs["_litert_interleave_heads"]

    def keep(x, r):
        if len(inter_inputs) < 4:
            inter_inputs.append((x.clone(), r))
        return counted(x, r)

    globs["_litert_interleave_heads"] = keep
    q0 = next(x for x in question_list(rows_from) if x["id"] == "tv4_000")
    ALL_ATTENTION_FUNCTIONS.register(ATTN, capture)
    try:
        with torch.no_grad():
            KevPrefill(model, L).eval()(*row_inputs(q0["row_ids"], L))
    finally:
        ALL_ATTENTION_FUNCTIONS.register(ATTN, kev_eager)
        globs["_litert_interleave_heads"] = counted
    with torch.no_grad():
        for module, q, k, v, mask, scaling in captured:
            a, _ = kev_eager(module, q, k, v, mask, scaling)
            b, _ = M.eager_attention_forward(module, q, k, v, mask, scaling)
            out["captured"].append({"layer": module.layer_idx, "q": list(q.shape), "k": list(k.shape), "mask": list(mask.shape),
                                    "bit_equal": bool(torch.equal(a, b)), "max_abs": float((a - b).abs().max())})
    inter["calls_one_forward"] = counter["calls"]
    inter["expected_calls_one_forward"] = 2 * len(gdn) if ratio > 1 else 0
    inter["captured"] = [{"shape": list(x.shape), "ratio": r,
                          "bit_equal_vs_repeat_interleave": bool(torch.equal(P._litert_interleave_heads(x, r),
                                                                             x.repeat_interleave(r, dim=2)))}
                         for x, r in inter_inputs]
    inter["pass"] = (inter["calls_one_forward"] == inter["expected_calls_one_forward"]
                     and all(c["bit_equal_vs_repeat_interleave"] for c in inter["captured"])
                     and inter.get("random_bit_equal", True))
    out["interleave"] = inter
    out["all_bit_equal"] = all(x["bit_equal"] for x in out["random"] + out["captured"]) and all(
        x["repeat_bit_equal"] for x in out["random"]) and inter["pass"]
    out["captured_row"] = qkey(q0)
    return out


def compute(a, L):
    """Workers + attention checks -> (store {key: h_sel, key/real: hidden}, record)."""
    for k in range(a.workers):
        for p in shard_paths(L, k):
            assert not p.exists(), f"stale shard {p}"
    t0 = time.time()
    cmd = [sys.executable, __file__, "--model", MODEL, "--L", str(L), "--workers", str(a.workers)]
    if a.rows_from:
        cmd += ["--rows-from", a.rows_from]
    procs = [subprocess.Popen(cmd + ["--worker", str(k)]) for k in range(a.workers)]
    t_attn = time.time()
    attn = attention_bit_checks(L, a.rows_from)
    attn_seconds = round(time.time() - t_attn, 1)
    codes = [p.wait() for p in procs]
    assert codes == [0] * a.workers, codes
    store, wrows, load_info, calls, forwards = {}, {}, None, 0, 0
    for k in range(a.workers):
        npz, js = shard_paths(L, k)
        z = np.load(npz)
        store.update({key: z[key] for key in z.files})
        d = json.loads(js.read_text())
        load_info = d["load_info"]
        wrows.update({r["key"]: r for r in d["rows"]})
        calls += d.get("interleave_calls", 0)
        forwards += d.get("forwards", 0)
    for k in range(a.workers):
        for p in shard_paths(L, k):
            p.unlink()
    fits = [q for q in question_list(a.rows_from) if q["row_len"] <= L]
    record = {"rows_from": a.rows_from or "the model's oracle", "L": L, "workers": a.workers,
              "seconds_workers_wall": round(time.time() - t0, 1), "load_info": load_info,
              "kev_eager_vs_eager": attn, "kev_eager_seconds": attn_seconds, "worker_rows": wrows,
              "row_sha256": {qkey(q): rows_sha(q) for q in fits},
              "interleave_calls_all_workers": calls, "forwards_all_workers": forwards}
    return store, record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--worker", type=int, default=-1)
    ap.add_argument("--phase", choices=["all", "compute", "precheck", "compare"], default="all")
    ap.add_argument("--rows-from", default="", help="phase compute: oracle json whose questions give the rows")
    add_model_arg(ap)
    a = ap.parse_args()
    assert a.model == MODEL
    if a.worker >= 0:
        return worker(a.L, a.worker, a.workers, a.rows_from)
    L = a.L
    out = K / f"results/torch_graph_parity_L{L}{RESULT_SUFFIX}.json"
    out_npz = cache_dir("cache/r2") / f"torch_graph_hsel_L{L}{RESULT_SUFFIX}.npz"
    rows_npz, rec_json = compute_paths(L)
    pre_json = cache_dir("cache/r2") / f"torch_graph_precheck_L{L}{RESULT_SUFFIX}.json"
    clock = Clock()
    started_at = clock.stamp()
    if a.phase in ("all", "compute"):
        assert not out.exists() and not out_npz.exists()
        store, record = compute(a, L)
        if a.phase == "compute":
            assert not rows_npz.exists() and not rec_json.exists()
            np.savez(rows_npz, **store)
            record.update(started_at=started_at, npz=str(rows_npz.relative_to(K)))
            dump_json(rec_json, record)
            print(json.dumps({k: record[k] for k in ("seconds_workers_wall", "kev_eager_seconds", "interleave_calls_all_workers",
                                                     "forwards_all_workers")}, indent=1))
            print(json.dumps({k: v for k, v in record["kev_eager_vs_eager"].items() if k != "captured"}, indent=1))
            return
    else:
        record = json.loads(rec_json.read_text())
        z = np.load(rows_npz)
        store = {key: z[key] for key in z.files}
    step1 = np.load(STEP1_NPZ)
    real_diff = {key[: -len("/real")]: float(np.abs(store[key].astype(np.float64) - step1[key[: -len("/real")]]).max())
                 for key in store if key.endswith("/real")}
    if a.phase == "precheck":
        hv = max(real_diff.values())
        doc = {"what": "torch KevPrefill real-position hidden vs stock 5.14.1 hidden of the same rows (no oracle)",
               "L": L, "model": MODEL, "rows": len(real_diff), "max_abs": hv,
               "bar": BAR_HIDDEN_VS_UNPATCHED, "pass": bool(hv <= BAR_HIDDEN_VS_UNPATCHED), "within_1e-4": bool(hv <= REF_HIDDEN_1E4),
               "attention_all_bit_equal": record["kev_eager_vs_eager"]["all_bit_equal"],
               "interleave": record["kev_eager_vs_eager"].get("interleave"),
               "all_positions_finite": all(r["finite_all_positions"] for r in record["worker_rows"].values()),
               "worst": sorted(real_diff.items(), key=lambda kv: -kv[1])[:5], "at": started_at}
        dump_json(pre_json, doc, overwrite=True)
        print(json.dumps(doc, indent=1))
        return
    oracle = load_oracle()
    ref = np.load(ORACLE_NPZ)
    head = Head()
    fits = [q for q in oracle["questions"] if q["row_len"] <= L]
    same_rows = {qkey(q): record["row_sha256"].get(qkey(q)) == rows_sha(q) for q in fits}
    assert all(same_rows.values()), [k for k, v in same_rows.items() if not v][:5]
    skipped = [{"key": qkey(q), "row_len": q["row_len"]} for q in oracle["questions"] if q["row_len"] > L]
    rows, red_probs, full, keep = [], None, {}, {}
    for q in fits:
        h_sel = store[qkey(q)]
        z_pre, z_post, probs = head(h_sel)
        row = compare_question(q, probs, z_post, h_sel, ref[qkey(q)])
        row.update(record["worker_rows"][qkey(q)])
        row["hidden_real_max_abs_vs_unpatched"] = real_diff[qkey(q)]
        rows.append(row)
        keep[qkey(q)] = h_sel
        if qkey(q) + "/full" in ref.files:
            real = store[qkey(q) + "/real"]
            full[qkey(q)] = float(np.abs(real.astype(np.float64) - ref[qkey(q) + "/full"]).max())
            keep[qkey(q) + "/full"] = real
        if q["id"] == "red_arm_000":
            red_probs = probs
    assert not out.exists() and not out_npz.exists()
    np.savez(out_npz, **keep)
    summary = summarize(rows, red_probs, oracle)
    summary["full_rows_max_abs_hidden_vs_oracle"] = full
    hv = max(r["hidden_real_max_abs_vs_unpatched"] for r in rows)
    secs = [r["seconds"] for r in rows]
    attn = record["kev_eager_vs_eager"]
    doc = {
        "step": f"torch graph: KevPrefill in torch at L={L}, right-padded rows (pad 248044)"
                + (f" (model {MODEL})" if RESULT_SUFFIX else ""), "L": L, "model": MODEL,
        "started_at": started_at, "seconds_wall": clock.seconds(), "seconds_workers_wall": record["seconds_workers_wall"],
        "workers": record["workers"], "torch_threads_per_worker": 1, "load_info": record["load_info"],
        "rows_computed_from": record["rows_from"], "rows_identical_to_oracle": len(same_rows),
        "questions_run": len(rows), "questions_skipped_longer_than_L": skipped,
        "hidden_real_max_abs_vs_unpatched": hv, "bar_hidden_vs_unpatched": BAR_HIDDEN_VS_UNPATCHED,
        "hidden_pass": bool(hv <= BAR_HIDDEN_VS_UNPATCHED), "hidden_within_1e-4": bool(hv <= REF_HIDDEN_1E4),
        "all_positions_finite": all(r["finite_all_positions"] for r in rows),
        "kev_eager_vs_eager": attn, "kev_eager_seconds": record["kev_eager_seconds"],
        "interleave_calls_all_workers": record.get("interleave_calls_all_workers"),
        "forwards_all_workers": record.get("forwards_all_workers"),
        "seconds_per_row_torch_1thread": {"median": float(np.median(secs)), "min": float(np.min(secs)), "max": float(np.max(secs))},
        "summary": summary, "pass": bool(summary["bar_pass"] and hv <= BAR_HIDDEN_VS_UNPATCHED and attn["all_bit_equal"]),
        "hsel_npz": str(out_npz.relative_to(K)), "rows": rows,
    }
    dump_json(out, doc)
    print(json.dumps({k: v for k, v in doc.items() if k not in ("rows", "summary", "kev_eager_vs_eager")}, indent=1))
    print(json.dumps({"overall": summary["overall"], "bar_pass": summary["bar_pass"], "red_arm": summary.get("red_arm"),
                      "flips": summary["flips"], "full": full, "attn_all_bit_equal": attn["all_bit_equal"]}, indent=1))


if __name__ == "__main__":
    main()
