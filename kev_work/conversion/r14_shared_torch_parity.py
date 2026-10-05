"""The Kev-0.8B shared-state pair (r14_shared_state.py) with an r13_kernel form vs the row form with the same form,
every oracle question (402), torch fp32, before any export (gate: row form vs pair <= 1e-4 on the read-out rows h_sel,
<= 1e-5 on the probabilities).

    python r14_shared_torch_parity.py --form R64+sp+ec+dd+vs6 --workers 8

Per request (377): the state = the first n tokens of its rows runs once through StatePrefill at the smallest Ls in
(128, 256, 512, 1024, 2048) that holds it; each question's branch runs through QuestionStep at the smallest Lq in
(64, 128, 192); the reference is KevPrefill (same form) on the whole row at the smallest L in (128, 256, 512, 1024,
2048) that holds the row; RowCheck = StatePrefill(with_hidden) on the whole row must equal KevPrefill bit for bit.
Before the form is applied each worker also runs its questions through the loop-kernel graph (KevPrefill, no rewrite)
= "stock": the pair's total distance from the loop kernel's numbers is recorded (informational; the gate is pair vs
row form of the same form).
State checks: pad amount (Ls 128 vs Ls 256), pad content, guard off (control), short states of 1 / 2 / 3 tokens.
Readout = the oracle's head in numpy float32 (r2_common.Head). Workers: --workers; at most 4 while the file named by
the environment variable KEV_GPU_LOCK (another job timing the GPU, when set) holds an owner line with "timing".
-> results/r14_shared_torch_parity_<form with - >.json (never overwritten); cache/r14/torch_two_phase_hsel_<form>.npz."""
import argparse
import json
import subprocess
import sys
import time

import numpy as np
import torch

from r2_common import PAD_ID, K, Clock, Head, dump_json, load_oracle, qkey, select

LS = (128, 256, 512, 1024, 2048)
LQ = (64, 128, 192)
LROW = (128, 256, 512, 1024, 2048)
CHECK_N = 24
CACHE = K / "cache/r14"
GATE = {"h_sel_max_abs_vs_row": 1e-4, "max_abs_dp_vs_row": 1e-5}


def ftag(form):
    return form.replace("+", "-")


def bucket(n, sizes):
    return next(s for s in sizes if n <= s)


def requests_of(oracle):
    """-> [(request id, n_state, [questions])] in oracle order, after checking the shared state prefix."""
    by_req = {}
    for q in oracle["questions"]:
        by_req.setdefault(q["id"], []).append(q)
    reqs = {r["id"]: r for r in oracle["requests"]}
    out = []
    for rid, qs in by_req.items():
        n = reqs[rid]["state_tokens"]
        s = qs[0]["row_ids"][:n]
        for q in qs:
            assert q["row_ids"][:n] == s and q["row_ids"][n] == 248061, (rid, q["qid"])
            assert q["decide_idx"] >= n and min(q["opt_idx"]) >= n
        out.append((rid, n, qs))
    return out


def relative(q, n):
    r = dict(q)
    r["decide_idx"] = q["decide_idx"] - n
    r["opt_idx"] = [o - n for o in q["opt_idx"]]
    return r


def maxdiff(a, b):
    return float(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).max())


def state_diff(sa, sb, n, names):
    """max |diff| per state kind; k / v compared at the real positions 0..n-1 only."""
    d = {"gdn_state": 0.0, "conv_tail": 0.0, "kv_real": 0.0}
    for name in names:
        a, b = sa[name].numpy(), sb[name].numpy()
        if name.startswith(("k_", "v_")):
            d["kv_real"] = max(d["kv_real"], maxdiff(a[:, :, :n], b[:, :, :n]))
        else:
            d[name.rsplit("_", 1)[0]] = max(d[name.rsplit("_", 1)[0]], maxdiff(a, b))
    return d


def worker(form, k, workers):
    torch.set_num_threads(1)
    import r13_kernel as R
    from kev_graph import KevPrefill, load_text_model, row_inputs
    from r14_shared_state import QuestionStep, StatePrefill, state_inputs
    model, info = load_text_model()
    oracle = load_oracle()
    head = Head()
    reqs = requests_of(oracle)
    mine = list(enumerate(reqs))[k::workers]
    cache = {}

    def mod(kind, *shape):
        key = (kind,) + shape
        if key not in cache:
            cache[key] = {"state": lambda: StatePrefill(model, shape[0]), "question": lambda: QuestionStep(model, *shape),
                          "row": lambda: KevPrefill(model, shape[0]),
                          "rowcheck": lambda: StatePrefill(model, shape[0], with_hidden=True)}[kind]().eval()
        return cache[key]

    # stock pass (shipped graph, no form) on the same questions
    stock = {}
    t0 = time.time()
    with torch.no_grad():
        for _, (rid, n, qs) in mine:
            for q in qs:
                Lr = bucket(q["row_len"], LROW)
                stock[qkey(q)] = select(mod("row", Lr)(*row_inputs(q["row_ids"], Lr))["hidden"][0].numpy(), q)
    t_stock = time.time() - t0
    applied = R.apply(model, form)
    rows, checks, hsel_two = [], [], {}
    with torch.no_grad():
        for ridx, (rid, n, qs) in mine:
            Ls = bucket(n, LS)
            s_ids, s_valid = state_inputs(qs[0]["row_ids"][:n], Ls, PAD_ID)
            t = time.time()
            st = mod("state", Ls)(s_ids, s_valid)
            t_state = time.time() - t
            names = [x for x in st]
            assert len(names) == 48
            chk = {"request": rid, "n_state": n, "Ls": Ls,
                   "gdn_state_max_abs": max(float(st[x].abs().max()) for x in names if x.startswith("gdn_state")),
                   "state_finite": all(bool(torch.isfinite(st[x]).all()) for x in names)}
            if n <= 128:
                s_ids2, s_valid2 = state_inputs(qs[0]["row_ids"][:n], 256, PAD_ID)
                chk["pad_amount_128_vs_256"] = state_diff(st, mod("state", 256)(s_ids2, s_valid2), n, names)
            if ridx < CHECK_N:
                alt = s_ids.clone()
                alt[0, n:] = 248060
                chk["pad_content"] = state_diff(st, mod("state", Ls)(alt, s_valid), n, names)
                chk["pad_content_bit_equal"] = all(v == 0.0 for v in chk["pad_content"].values())
                if n < Ls:
                    chk["guard_off"] = state_diff(st, mod("state", Ls)(s_ids, torch.ones_like(s_valid)), n, names)
            checks.append(chk)
            for q in qs:
                nq = q["row_len"] - n
                Lq = bucket(nq, LQ)
                q_ids, q_valid = state_inputs(q["row_ids"][n:], Lq, PAD_ID)
                t = time.time()
                hq = mod("question", Ls, Lq)(ids=q_ids, valid=q_valid, state_valid=s_valid, **st)["hidden"][0].numpy()
                t_q = time.time() - t
                Lr = bucket(q["row_len"], LROW)
                r_ids, r_valid = row_inputs(q["row_ids"], Lr)
                hr = mod("row", Lr)(r_ids, r_valid)["hidden"][0].numpy()
                hc = mod("rowcheck", Lr)(r_ids, r_valid)["hidden"][0].numpy()
                sel_q = select(hq, relative(q, n))
                sel_r = select(hr, q)
                key = qkey(q)
                sel_s = stock[key]
                _, _, p_q = head(sel_q)
                _, _, p_r = head(sel_r)
                _, _, p_s = head(sel_s.astype(np.float32))
                p_o = np.asarray(q["probs"], np.float64)
                hsel_two[key] = sel_q.astype(np.float32)
                rows.append({
                    "key": key, "request": rid, "source": q["source"], "n_state": n, "n_question": nq, "Ls": Ls,
                    "Lq": Lq, "L_row": Lr, "near_tie_oracle": bool(q["near_tie"]),
                    "max_abs_dp_vs_row": maxdiff(p_q, p_r), "h_sel_max_abs_vs_row": maxdiff(sel_q, sel_r),
                    "hidden_question_max_abs_vs_row": maxdiff(hq[:nq], hr[n:q["row_len"]]),
                    "rowcheck_bit_equal": bool((hc[: q["row_len"]] == hr[: q["row_len"]]).all()),
                    "rowcheck_max_abs": maxdiff(hc[: q["row_len"]], hr[: q["row_len"]]),
                    "max_abs_dp_vs_stock_row": maxdiff(p_q, p_s), "h_sel_max_abs_vs_stock_row": maxdiff(sel_q, sel_s),
                    "row_form_vs_stock_max_abs_dp": maxdiff(p_r, p_s),
                    "argmax_equal_stock_row": int(np.argmax(p_q)) == int(np.argmax(p_s)),
                    "max_abs_dp_vs_oracle": maxdiff(p_q, p_o),
                    "argmax_equal_oracle": q["keys"][int(np.argmax(p_q))] == q["argmax_key"],
                    "argmax_equal_row": int(np.argmax(p_q)) == int(np.argmax(p_r)),
                    "row_max_abs_dp_vs_oracle": maxdiff(p_r, p_o),
                    "probs_two_phase": [float(x) for x in p_q], "probs_row": [float(x) for x in p_r],
                    "finite": bool(np.isfinite(hq[:nq]).all()), "seconds_state": round(t_state, 3),
                    "seconds_question": round(t_q, 3)})
            print(json.dumps({"w": k, "req": rid, "n": n, "q": len(qs),
                              "max_dp_vs_row": max(r["max_abs_dp_vs_row"] for r in rows[-len(qs):])}), flush=True)
    if k == 0:
        checks.append({"short_state": short_state(model, oracle, head)})
    R.reset(model)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE / f"torch_two_phase_hsel_{ftag(form)}_w{k}.npz", **hsel_two)
    (CACHE / f"torch_shared_parity_{ftag(form)}_w{k}.json").write_text(json.dumps(
        {"rows": rows, "checks": checks, "load_info": info, "seconds_stock": round(t_stock, 1),
         "applied": {kk: vv for kk, vv in applied.items() if kk != "tokens"}}))


@torch.no_grad()
def short_state(model, oracle, head):
    """tv4_000's question after a state cut to its first 1, 2 and 3 tokens: two-phase vs the row form of that row."""
    from kev_graph import KevPrefill, row_inputs
    from r14_shared_state import QuestionStep, StatePrefill, state_inputs
    q = next(x for x in oracle["questions"] if x["id"] == "tv4_000")
    n0 = next(r for r in oracle["requests"] if r["id"] == "tv4_000")["state_tokens"]
    branch = q["row_ids"][n0:]
    out = []
    for n in (1, 2, 3):
        row = q["row_ids"][:n] + branch
        s_ids, s_valid = state_inputs(row[:n], 128, PAD_ID)
        st = StatePrefill(model, 128).eval()(s_ids, s_valid)
        tails_zero_rows = {name: int((st[name][0].abs().sum(-1) == 0).sum()) for name in st if name.startswith("conv_tail")}
        q_ids, q_valid = state_inputs(branch, 64, PAD_ID)
        hq = QuestionStep(model, 128, 64).eval()(ids=q_ids, valid=q_valid, state_valid=s_valid, **st)["hidden"][0].numpy()
        r_ids, r_valid = row_inputs(row, 128)
        hr = KevPrefill(model, 128).eval()(r_ids, r_valid)["hidden"][0].numpy()
        rel = {"decide_idx": len(branch) - 1, "opt_idx": [o - n0 for o in q["opt_idx"]]}
        full = {"decide_idx": n + len(branch) - 1, "opt_idx": [o - n0 + n for o in q["opt_idx"]]}
        _, _, pq = head(select(hq, rel))
        _, _, pr = head(select(hr, full))
        out.append({"n_state": n, "conv_tail_zero_rows_per_layer": sorted(set(tails_zero_rows.values())),
                    "hidden_question_max_abs_vs_row": maxdiff(hq[: len(branch)], hr[n: n + len(branch)]),
                    "max_abs_dp_vs_row": maxdiff(pq, pr)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--form", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--worker", type=int, default=-1)
    a = ap.parse_args()
    if a.worker >= 0:
        return worker(a.form, a.worker, a.workers)
    out = K / f"results/r14_shared_torch_parity_{ftag(a.form)}.json"
    assert not out.exists(), f"refusing to overwrite {out}"
    # while the file named by KEV_GPU_LOCK (another job timing the GPU, when set) holds an owner line with "timing", the
    # check starts with at most 4 workers (read at the start; the asked number otherwise)
    import os
    from pathlib import Path as _P
    try:
        gl = _P(os.environ["KEV_GPU_LOCK"]).read_text().strip() if os.environ.get("KEV_GPU_LOCK") else ""
    except OSError:
        gl = ""
    workers_asked = a.workers
    if "timing" in gl:
        a.workers = min(a.workers, 4)
    print(json.dumps({"gpu_lock_at_launch": gl, "workers_asked": workers_asked, "workers": a.workers}), flush=True)
    clock = Clock()
    started = clock.stamp()
    t0 = time.time()
    todo = [k for k in range(a.workers) if not (CACHE / f"torch_shared_parity_{ftag(a.form)}_w{k}.json").exists()]
    reused = [k for k in range(a.workers) if k not in todo]
    procs = [subprocess.Popen([sys.executable, __file__, "--form", a.form, "--workers", str(a.workers), "--worker", str(k)],
                              stdout=open(K / f"logs/r14_shared_parity_{ftag(a.form)}_w{k}.log", "w"),
                              stderr=subprocess.STDOUT) for k in todo]
    codes = [p.wait() for p in procs]
    assert codes == [0] * len(todo), codes
    rows, checks, two, applied, t_stock = [], [], {}, None, []
    for k in range(a.workers):
        d = json.loads((CACHE / f"torch_shared_parity_{ftag(a.form)}_w{k}.json").read_text())
        rows += d["rows"]
        checks += d["checks"]
        info, applied = d["load_info"], d["applied"]
        t_stock.append(d["seconds_stock"])
        two.update(dict(np.load(CACHE / f"torch_two_phase_hsel_{ftag(a.form)}_w{k}.npz")))
    order = {qkey(q): i for i, q in enumerate(load_oracle()["questions"])}
    rows.sort(key=lambda r: order[r["key"]])
    np.savez(CACHE / f"torch_two_phase_hsel_{ftag(a.form)}.npz", **two)
    for k in range(a.workers):
        for p in (CACHE / f"torch_shared_parity_{ftag(a.form)}_w{k}.json",
                  CACHE / f"torch_two_phase_hsel_{ftag(a.form)}_w{k}.npz"):
            p.unlink()
    short = next(c["short_state"] for c in checks if "short_state" in c)
    req_checks = [c for c in checks if "short_state" not in c]
    pa = [c["pad_amount_128_vs_256"] for c in req_checks if "pad_amount_128_vs_256" in c]
    pc = [c for c in req_checks if "pad_content" in c]
    go = [c["guard_off"] for c in req_checks if "guard_off" in c]

    def agg(sel):
        sel = list(sel)
        return {"questions": len(sel),
                "max_abs_dp_vs_row": max(r["max_abs_dp_vs_row"] for r in sel),
                "h_sel_max_abs_vs_row": max(r["h_sel_max_abs_vs_row"] for r in sel),
                "hidden_question_max_abs_vs_row": max(r["hidden_question_max_abs_vs_row"] for r in sel),
                "argmax_equal_row": sum(r["argmax_equal_row"] for r in sel),
                "max_abs_dp_vs_stock_row": max(r["max_abs_dp_vs_stock_row"] for r in sel),
                "h_sel_max_abs_vs_stock_row": max(r["h_sel_max_abs_vs_stock_row"] for r in sel),
                "argmax_equal_stock_row": sum(r["argmax_equal_stock_row"] for r in sel),
                "row_form_vs_stock_max_abs_dp": max(r["row_form_vs_stock_max_abs_dp"] for r in sel),
                "max_abs_dp_vs_oracle": max(r["max_abs_dp_vs_oracle"] for r in sel),
                "mean_question_max_abs_dp_vs_oracle": float(np.mean([r["max_abs_dp_vs_oracle"] for r in sel])),
                "argmax_equal_oracle": sum(r["argmax_equal_oracle"] for r in sel)}

    summary = {
        "all": agg(rows),
        "fits_Ls128_Lq64": agg(r for r in rows if r["n_state"] <= 128 and r["n_question"] <= 64),
        "fits_Ls256_Lq64": agg(r for r in rows if r["n_state"] <= 256 and r["n_question"] <= 64),
        "rowcheck": {"rows": len(rows), "bit_equal": sum(r["rowcheck_bit_equal"] for r in rows),
                     "max_abs": max(r["rowcheck_max_abs"] for r in rows)},
        "nonfinite_questions": sum(not r["finite"] for r in rows),
        "states_finite": all(c["state_finite"] for c in req_checks),
        "gdn_state_max_abs": max(c["gdn_state_max_abs"] for c in req_checks),
        "row_form_vs_oracle_max_abs_dp": max(r["row_max_abs_dp_vs_oracle"] for r in rows),
        "state_checks": {
            "pad_amount_128_vs_256": {"requests": len(pa), **{k: max(d[k] for d in pa) for k in pa[0]}},
            "pad_content": {"requests": len(pc), "bit_equal": sum(c["pad_content_bit_equal"] for c in pc),
                            **{k: max(c["pad_content"][k] for c in pc) for k in pc[0]["pad_content"]}},
            "guard_off_control": {"requests": len(go), **{f"min_{k}": min(d[k] for d in go) for k in go[0]}},
            "short_state": short},
        "gate": GATE,
    }
    summary["pass"] = bool(summary["all"]["max_abs_dp_vs_row"] <= GATE["max_abs_dp_vs_row"]
                           and summary["all"]["h_sel_max_abs_vs_row"] <= GATE["h_sel_max_abs_vs_row"]
                           and summary["all"]["argmax_equal_row"] == len(rows)
                           and summary["rowcheck"]["bit_equal"] == len(rows) and summary["nonfinite_questions"] == 0
                           and all(s["max_abs_dp_vs_row"] <= GATE["max_abs_dp_vs_row"] for s in short))
    doc = {"step": "shared-state pair (torch, form applied) vs the row form with the same form, all questions",
           "form": a.form, "applied": applied, "started_at": started, "seconds_wall": clock.seconds(),
           "seconds_workers_wall": round(time.time() - t0, 1), "seconds_stock_pass_per_worker": t_stock,
           "workers": a.workers, "workers_asked": workers_asked, "gpu_lock_at_launch": gl,
           "workers_rerun": todo, "workers_reused_from_earlier_launch": reused,
           "torch_threads_per_worker": 1, "buckets": {"Ls": LS, "Lq": LQ, "L_row": LROW},
           "load_info": info, "summary": summary, "rows": rows, "request_checks": req_checks,
           "hsel_cache": f"cache/r14/torch_two_phase_hsel_{ftag(a.form)}.npz"}
    dump_json(out, doc)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
