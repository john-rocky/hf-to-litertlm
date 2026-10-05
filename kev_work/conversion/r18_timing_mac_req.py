"""Mac timing of the request points' row form: r14_timing_mac_req.py's "row_best" (one row call per question on the
smaller of the L128 / L256 files that holds that row, both loaded; timing_mac_shared.time_rows) on two file families in
the same run: C7 (tag r17C7-bkzq = the published 128- and 256-token files) and C (tag r14B-vs6), 2 passes in
alternating order. The pairs are timed by r14_timing_mac_req.py (their files are form C).

    python r18_timing_mac_req.py --out results/timing_mac_r18_C7_req.json [--max-load 10] [--wait-s 5400]

Request points (real rows of oracle/oracle_0.8b.json): 1 question = tv4_000, 2 = the first two questions of own_ticket_01
("own_ticket_01_q2"), 3 = own_ticket_01, 5 = own_fiveq_09; also own_order_06 (2 questions) and own_email_03 (3). Protocol:
timing_mac_shared.time_rows (5 warm-up requests, then 20 timed requests; ms = write + run + read-back), Metal float32,
the locks of r14_timing_mac.acquire_both. Never overwrites --out."""
import argparse
import json
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r14_timing_mac import PairS, acquire_both, release_both  # noqa: E402
from timing_mac import K, memory, stamp, swap_used, top_lines  # noqa: E402
from timing_mac_shared import Row, padded, time_rows, time_shared  # noqa: E402

V = "v2_fp16fc_i8emb"
TAG = "r14B-vs6"
ROW_TAGS = ("r17C7-bkzq", "r14B-vs6")   # C7 (the published L128 / L256) and C
POINTS = {128: ["tv4_000", "own_ticket_01_q2", "own_ticket_01", "own_fiveq_09"],
          256: ["tv4_000", "own_order_06", "own_email_03", "own_ticket_01", "own_fiveq_09"]}


def load_points(Ls, Lq, ids):
    """timing_mac_shared.load_requests for the given request ids; "<id>_q2" = that request's first two questions."""
    oracle = json.loads((K / "oracle/oracle_0.8b.json").read_text())
    reqs = {r["id"]: r for r in oracle["requests"]}
    out = {}
    for rid in ids:
        base, cut = (rid[:-3], 2) if rid.endswith("_q2") else (rid, None)
        n = reqs[base]["state_tokens"]
        qs = [q for q in oracle["questions"] if q["id"] == base][:cut]
        assert n <= Ls and all(q["row_len"] - n <= Lq for q in qs), (rid, n, [q["row_len"] - n for q in qs])
        s_ids, s_valid = padded(qs[0]["row_ids"][:n], Ls)
        questions, rows = [], []
        for q in qs:
            sel = [q["decide_idx"] - n] + [o - n for o in q["opt_idx"]]
            questions.append((*padded(q["row_ids"][n:], Lq), sel))
            rows.append({"ids": q["row_ids"], "sel": [q["decide_idx"]] + list(q["opt_idx"])})
        out[rid] = {"n_state": n, "branches": [q["row_len"] - n for q in qs], "row_lens": [q["row_len"] for q in qs],
                    "s_ids": s_ids, "s_valid": s_valid, "questions": questions, "rows": rows}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--max-load", type=float, default=10.0)
    ap.add_argument("--wait-s", type=float, default=5400.0)
    a = ap.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    out = K / a.out
    assert not out.exists(), f"refusing to overwrite {a.out}"
    pts = {Ls: load_points(Ls, 64, ids) for Ls, ids in POINTS.items()}
    allreq = {**pts[128], **pts[256]}
    pairf = {Ls: K / f"exports/kev08b_sharedstate_Ls{Ls}_Lq64_{V}_{TAG}.tflite" for Ls in POINTS}
    rowf = {t: {L: K / f"exports/kev08b_rowprefill_L{L}_{V}_{t}.tflite" for L in (128, 256)} for t in ROW_TAGS}
    for p in [*rowf[ROW_TAGS[0]].values(), *rowf[ROW_TAGS[1]].values()]:
        assert p.exists(), p
    doc = {"what": "Mac timing: request points, the row form only (each row in its smallest bucket), files C7 "
                   "and C, Metal float32",
           "points": {Ls: ids for Ls, ids in POINTS.items()},
           "requests": {rid: {k: v for k, v in r.items() if k in ("n_state", "branches", "row_lens")}
                        for rid, r in allreq.items()},
           "files": {f"row_{t}_L{L}": {"file": str(p.relative_to(K)), "bytes": p.stat().st_size}
                     for t in ROW_TAGS for L, p in rowf[t].items()},
           "protocol": "timing_mac_shared.time_shared / time_rows: 5 warm-up + 20 timed requests, ms = write + run + "
                       "read-back; 2 passes in alternating order",
           "top_before_lock": top_lines(), "swap_before_lock": swap_used(), "started_at": stamp(), "results": {}}
    gfh, hfh, lock = acquire_both(a.max_load, a.wait_s)
    if gfh is None:
        doc.update(status="not measured (locks busy)", lock=lock)
        out.write_text(json.dumps(doc, indent=1) + "\n")
        print(json.dumps(lock, indent=1))
        return
    t0 = time.time()
    try:
        doc["top_at_lock"], doc["swap_at_lock"] = top_lines(), swap_used()
        for p in range(a.passes):
            steps = list(ROW_TAGS) if p % 2 == 0 else list(ROW_TAGS[::-1])
            for st in steps:
                if st in ROW_TAGS:
                    graphs = {L: Row(rowf[st][L], "gpu_f32", 8) for L in (128, 256)}
                    per = {"compile_s": {L: round(g.compile_s, 2) for L, g in graphs.items()},
                           "fully_accelerated": {L: g.fully for L, g in graphs.items()}, "memory_after_compile": memory(),
                           "requests": {}}
                    for rid, req in allreq.items():
                        per["requests"][rid] = time_rows(graphs, req, lambda n: 128 if n <= 128 else 256)
                    for g in graphs.values():
                        g.close()
                    doc["results"][f"pass{p}_rows_best_{st}"] = per
                    print(f"pass{p} rows {st}", {rid: round(v["request_ms"]["median"], 1) for rid, v in per["requests"].items()},
                          flush=True)
                else:
                    Ls = int(st[4:])
                    for share in (True, False):
                        pr = PairS(pairf[Ls], Ls, 64, "gpu_f32", 8, share=share)
                        per = {"compile_s": round(pr.compile_s, 2), "fully_accelerated": pr.fully,
                               "memory_after_compile": memory(), "requests": {}}
                        for rid, req in pts[Ls].items():
                            per["requests"][rid] = {m: time_shared(pr, req, m) for m in ("direct", "host")}
                        pr.close()
                        key = f"pass{p}_pair_Ls{Ls}_{'share' if share else 'noshare'}"
                        doc["results"][key] = per
                        print(key, {rid: round(v["direct"]["request_ms"]["median"], 1) for rid, v in per["requests"].items()},
                              flush=True)
        doc["status"] = "measured"
    finally:
        lock.update(release_both(gfh, hfh))
        doc.update(lock=lock, top_after_release=top_lines(), swap_after=swap_used(), finished_at=stamp(),
                   seconds_wall=round(time.time() - t0, 1), memory_end=memory())
        doc.setdefault("status", "failed (see the log)")
        out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: v for k, v in doc.items() if k != "results"}, indent=1)[:2000])


if __name__ == "__main__":
    main()
