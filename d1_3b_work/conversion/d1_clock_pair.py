"""Round 9 acceptance 4: request-level Mac wall time of the shared-state pair against the row form, through the
CompiledModel API (ai-edge-litert), one measurement window per file and accelerator.

    $EXPORT \
        scripts/d1_clock_pair.py --label <label> --tflite <pair or row file> --accel gpu_f32 [--no-share]
    $EXPORT scripts/d1_clock_pair.py --merge results/real_sharedstate_timing_*.json \
        --out results/timing_mac_pair.json

Requests (fixtures/rows.json ids; the state = the rows' first state_len tokens):
  three  card_text_001: state 19 tokens, questions refund / team / urgency (own tokens 20 / 42 / 35; rows 39 / 61 / 54)
         = the card's 3-question request (row form round 6c: 325.1 ms on Metal fp32)
  one    card_text_001/refund alone (state 19 + 20 = row 39) = the card's 1-question column (round 6c: 108.6 ms)
  two    card_text_001/refund + team (own tokens 20 / 42): the 2-question point between one and three, for the
         question count from which the pair is faster
  most   own_fiveq_09: the fixture request with the most questions (5; state 102, own tokens 37..46, rows 139..148)
A pair file (two signatures) runs a request as state_prefill once + question_step per question, in each hand-over of
--handovers (direct: state_prefill's output buffers passed as question_step's inputs; host: read back to numpy and
written into question_step's own inputs); a row file (one signature, L256) runs one row call per question (the row
form's request). Per request x hand-over: --warmup requests, then --reps timed requests (5 + 20); ms = the wall of the
whole request = every write, run and read-back of the hidden output (the pair's host hand-over also its state read-back
+ write); the host's read-out (float64 softmax of a few logits) is not in it. Per call: the state call (write + run
[+ read-back + write]) and each question call (write + run + read-back) as medians. The first request's probabilities
(host/d1_litert.py readout) are kept per question for --merge to compare across files. load1 (os.getloadavg) before and
after every set; top, vm.swapusage and the window lock line before and after; the process's phys_footprint after the
compile and after the sets (d1_clock_mac.phys_footprint).
GPU options: gpu_f32 = GpuOptions(enforce_f32=True), gpu_default = default precision; a pair file adds
constant_tensor_sharing=True unless --no-share (the sharing A/B).
Outputs (never overwritten): results/real_sharedstate_timing_<label>.json; --merge -> --out (one table: request x file
x accel x hand-over, the sources with their windows, pair / row ratios per request and accel within the merge).
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "scripts"))
sys.path.insert(0, str(K / "host"))       # host/ first: `d1_shared_state` here is the HOST module
import d1_litert as H  # noqa: E402
import d1_shared_state as HS  # noqa: E402
from d1_clock_mac import load1, machine_lines, phys_footprint, stamp, stats  # noqa: E402

assert Path(HS.__file__).resolve() == (K / "host/d1_shared_state.py").resolve(), HS.__file__
REQUESTS = {"three": ("card_text_001", ["refund", "team", "urgency"]), "one": ("card_text_001", ["refund"]),
            "two": ("card_text_001", ["refund", "team"]), "most": ("own_fiveq_09", None),
            # round 10: one row per long bucket (the rows of SPEED_LADDER.md rows 25 / 28 / 31 / 34) and the 3.4k state
            "b512": ("tv4_009", ["answer"]), "b1024": ("own_mid_08k_001", ["discarded"]),
            "b2048": ("own_long_log_10", ["first_failure"]), "long": ("own_long_34k_001", ["refunded_twice"]),
            # round 10: one question in the row classes 65-128 (tv4_000: state 42 + 64) and 129-256 (own_fiveq_09/photo:
            # state 102 + 37, the row of a 5-question record as a request of its own)
            "one128": ("tv4_000", ["answer"]), "one256": ("own_fiveq_09", ["photo"])}
TABLE = K / "cache/real/tables/readout_table.safetensors"
EMBED_TABLE = K / "cache/real/tables/embed_table.safetensors"     # round 10: embeds row graphs and embeds pairs
WARMUP, REPS = 5, 20


def request_rows(name: str) -> tuple[int, list[dict]]:
    rid, qids = REQUESTS[name]
    rows = [r for r in json.loads((K / "fixtures/rows.json").read_text())["rows"] if r["id"] == rid]
    if qids is not None:
        rows = [next(r for r in rows if r["qid"] == q) for q in qids]
    n = rows[0]["state_len"]
    assert all(r["state_len"] == n and r["ids"][:n] == rows[0]["ids"][:n] for r in rows), rid
    return n, rows


def run_pair(path: Path, accel: str, share: bool, threads: int, sets, handovers, warmup: int, reps: int, table,
             embed_table=None) -> dict:
    kind = "cpu" if accel == "cpu" else "gpu"
    t0 = time.perf_counter()
    pair = HS.SharedStatePair(path, kind, "fp32" if accel != "gpu_default" else "default", threads, handover="both",
                              share=share, embed_table=embed_table)
    res = {"file": str(path.relative_to(K)), "bytes": path.stat().st_size, "form": "pair", "accel": accel,
           "input": pair.input,
           "share": share if kind == "gpu" else None, "threads": threads if kind == "cpu" else None,
           "Ls": pair.Ls, "Lq": pair.Lq, "compile_s": round(time.perf_counter() - t0, 3),
           "is_fully_accelerated": pair.fully_accelerated, "options": pair.options_desc,
           "memory_after_compile": phys_footprint(), "warmup": warmup, "reps": reps, "sets": []}
    for name in sets:
        n, rows = request_rows(name)
        state = rows[0]["ids"][:n]
        own = [r["ids"][n:] for r in rows]
        assert n <= pair.Ls and all(len(x) <= pair.Lq for x in own), (name, n, [len(x) for x in own])
        for mode in handovers:
            def request():
                t = time.perf_counter()
                pair.run_state(state, read_back=mode == "host")
                ts = (time.perf_counter() - t) * 1000
                hs, tq = [], []
                for x in own:
                    t1 = time.perf_counter()
                    hs.append(pair.run_question(x, handover=mode))
                    tq.append((time.perf_counter() - t1) * 1000)
                return (time.perf_counter() - t) * 1000, ts, tq, hs

            rec = {"set": name, "request": rows[0]["id"], "keys": [f"{r['id']}/{r['qid']}" for r in rows],
                   "state_tokens": n, "question_tokens": [len(x) for x in own], "handover": mode,
                   "load1_before": load1(), "started_at": stamp()}
            _, _, _, hs = request()
            rec["first_request_finite"] = all(bool(np.isfinite(h[: len(x)]).all()) for h, x in zip(hs, own))
            rec["probs"] = {k: H.readout(h[len(x) - 1], table, r["readout_ids"])
                            for k, h, x, r in zip(rec["keys"], hs, own, rows)}
            for _ in range(warmup - 1):
                request()
            tot, st, qc = [], [], []
            for _ in range(reps):
                a, b, c, _ = request()
                tot.append(a)
                st.append(b)
                qc += c
            rec.update(ms_request=stats(tot), ms_state_call=stats(st), ms_question_call=stats(qc),
                       load1_after=load1(), finished_at=stamp())
            res["sets"].append(rec)
            print(json.dumps({"file": path.name, "accel": accel, "share": share, "set": name, "handover": mode,
                              "median_ms": rec["ms_request"]["median"], "state_ms": rec["ms_state_call"]["median"],
                              "question_ms": rec["ms_question_call"]["median"],
                              "load1": [rec["load1_before"], rec["load1_after"]]}), flush=True)
    res["memory_after_sets"] = phys_footprint()
    pair.close()
    return res


def run_row(path: Path, accel: str, threads: int, sets, warmup: int, reps: int, table, embed_table=None) -> dict:
    kind = "cpu" if accel == "cpu" else "gpu"
    embeds = "_embeds_" in path.name     # round 10: the embeds row graph; the host's gather is inside the timed call
    t0 = time.perf_counter()
    g = (H.LiteRTEmbedsGraph if embeds else H.LiteRTRowGraph)(path, kind, "fp32" if accel != "gpu_default" else "default",
                                                              threads)
    res = {"file": str(path.relative_to(K)), "bytes": path.stat().st_size, "form": "row", "accel": accel,
           "input": "embeds" if embeds else "ids",
           "threads": threads if kind == "cpu" else None, "L": g.L, "compile_s": round(time.perf_counter() - t0, 3),
           "is_fully_accelerated": g.fully_accelerated, "memory_after_compile": phys_footprint(), "warmup": warmup,
           "reps": reps, "sets": []}
    for name in sets:
        n, rows = request_rows(name)
        padded = [H.pad_row(r["ids"], g.L) for r in rows]

        def request():
            t = time.perf_counter()
            hs, tc = [], []
            for ids, valid in padded:
                t1 = time.perf_counter()
                x = np.ascontiguousarray(embed_table.rows(ids[0])[None], dtype=np.float32) if embeds else ids
                hs.append(g(x, valid)[0])
                tc.append((time.perf_counter() - t1) * 1000)
            return (time.perf_counter() - t) * 1000, tc, hs

        rec = {"set": name, "request": rows[0]["id"], "keys": [f"{r['id']}/{r['qid']}" for r in rows],
               "state_tokens": n, "row_tokens": [r["row_len"] for r in rows], "handover": None,
               "load1_before": load1(), "started_at": stamp()}
        _, _, hs = request()
        rec["first_request_finite"] = all(bool(np.isfinite(h[: r["row_len"]]).all()) for h, r in zip(hs, rows))
        rec["probs"] = {k: H.readout(h[r["row_len"] - 1], table, r["readout_ids"]) for k, h, r in zip(rec["keys"], hs, rows)}
        for _ in range(warmup - 1):
            request()
        tot, calls = [], []
        for _ in range(reps):
            a, c, _ = request()
            tot.append(a)
            calls += c
        rec.update(ms_request=stats(tot), ms_row_call=stats(calls), load1_after=load1(), finished_at=stamp())
        res["sets"].append(rec)
        print(json.dumps({"file": path.name, "accel": accel, "set": name, "median_ms": rec["ms_request"]["median"],
                          "call_ms": rec["ms_row_call"]["median"], "load1": [rec["load1_before"], rec["load1_after"]]}),
              flush=True)
    res["memory_after_sets"] = phys_footprint()
    g.close()
    return res


def measure(a) -> int:
    out = K / f"results/{a.out_prefix}{a.label}.json"
    assert not out.exists(), f"refusing to overwrite {out}"
    path = Path(a.tflite) if Path(a.tflite).is_absolute() else K / a.tflite
    table = H.ReadoutTable.from_file(TABLE)
    et = H.EmbedTable(EMBED_TABLE) if "_embeds_" in path.name else None
    sets = [s for s in a.sets.split(",") if s]
    assert all(s in REQUESTS for s in sets), sets
    doc = {"what": "round 9: request-level Mac wall time (pair vs row form), CompiledModel (ai-edge-litert "
                   f"{importlib.metadata.version('ai-edge-litert')}); per request x hand-over {a.warmup} warm-up "
                   f"requests, then {a.reps} timed requests; ms = every write, run and read-back of the request",
           "label": a.label, "machine_before": machine_lines(), "load1_before": load1(), "pid": os.getpid()}
    t0 = time.time()
    if "sharedstate" in path.name:
        doc["result"] = run_pair(path, a.accel, not a.no_share, a.threads, sets, [h for h in a.handovers.split(",") if h],
                                 a.warmup, a.reps, table, et)
    else:
        doc["result"] = run_row(path, a.accel, a.threads, sets, a.warmup, a.reps, table, et)
    doc.update(machine_after=machine_lines(), load1_after=load1(), seconds_wall=round(time.time() - t0, 1),
               finished_at=stamp())
    out.write_text(json.dumps(doc, indent=1) + "\n")
    return 0


def merge(a) -> int:
    out = Path(a.out) if Path(a.out).is_absolute() else K / a.out
    assert not out.exists(), f"refusing to overwrite {out}"
    rows, sources = [], []
    for src in a.merge:
        p = Path(src) if Path(src).is_absolute() else K / src
        d = json.loads(p.read_text())
        r = d["result"]
        sources.append({"file": str(p.relative_to(K)), "label": d["label"], "window": d["machine_before"].get("gpu_lock"),
                        "started": d["machine_before"]["at"], "finished": d.get("finished_at"),
                        "load1_before": d.get("load1_before"), "load1_after": d.get("load1_after"),
                        "top_before": d["machine_before"].get("top"), "swap_before": d["machine_before"].get("swap")})
        form = (f"pair Ls{r['Ls']}+Lq{r['Lq']}" + ("" if r.get("share") in (True, None) else " (no sharing)")
                if r["form"] == "pair" else f"row L{r['L']}") + (" embeds" if r.get("input") == "embeds" else "")
        for s in r["sets"]:
            rows.append({"set": s["set"], "request": s["request"], "keys": s["keys"], "form": form, "file": r["file"],
                         "input": r.get("input", "ids"), "accel": r["accel"], "share": r.get("share"),
                         "handover": s["handover"], "state_tokens": s.get("state_tokens"),
                         "question_tokens": s.get("question_tokens"), "row_tokens": s.get("row_tokens"),
                         "median_ms": s["ms_request"]["median"], "min_ms": s["ms_request"]["min"],
                         "max_ms": s["ms_request"]["max"], "n": s["ms_request"]["n"],
                         "state_call_median_ms": (s.get("ms_state_call") or {}).get("median"),
                         "question_call_median_ms": (s.get("ms_question_call") or {}).get("median"),
                         "row_call_median_ms": (s.get("ms_row_call") or {}).get("median"),
                         "load1": [s["load1_before"], s["load1_after"]], "first_request_finite": s["first_request_finite"],
                         "compile_s": r["compile_s"], "is_fully_accelerated": r["is_fully_accelerated"],
                         "phys_footprint_after_compile": r["memory_after_compile"].get("phys_footprint"),
                         "phys_footprint_after_sets": r["memory_after_sets"].get("phys_footprint"),
                         "lifetime_max_phys_footprint": r["memory_after_sets"].get("lifetime_max_phys_footprint"),
                         "label": d["label"], "started_at": s["started_at"], "probs": s["probs"],
                         "source": str(p.relative_to(K))})
    ratios = []
    for r in rows:      # every pair against every row form of the same request and accelerator (round 10: L128, L256)
        if not r["form"].startswith("pair"):
            continue
        for base in (x for x in rows if x["form"].startswith("row") and x["set"] == r["set"] and x["accel"] == r["accel"]):
            dp = max(max(abs(u - v) for u, v in zip(r["probs"][k], base["probs"][k])) for k in r["keys"])
            ratios.append({"set": r["set"], "accel": r["accel"], "pair": r["form"], "row": base["form"],
                           "handover": r["handover"], "pair_ms": r["median_ms"], "row_ms": base["median_ms"],
                           "row_over_pair": round(base["median_ms"] / r["median_ms"], 3),
                           "max_abs_dp_pair_vs_row_first_request": dp})
    doc = {"what": "round 9: Mac request-level timing, the shared-state pair vs the row form (median of 20 timed "
                   "requests after 5 warm-ups; ms = every write, run and read-back of the request)",
           "sources": sources, "ratios": ratios, "rows": rows}
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print("| request | form | accel | hand-over | median ms | min / max | n | state call | question call | row call |"
          " load1 | window |")
    print("|---|---|---|---|---:|---|---:|---:|---:|---:|---|---|")
    for r in sorted(rows, key=lambda r: (r["set"], r["accel"], r["form"], r["handover"] or "")):
        print(f"| {r['set']} | {r['form']} | {r['accel']} | {r['handover'] or '—'} | {r['median_ms']} | "
              f"{r['min_ms']} / {r['max_ms']} | {r['n']} | {r['state_call_median_ms'] or '—'} | "
              f"{r['question_call_median_ms'] or '—'} | {r['row_call_median_ms'] or '—'} | {r['load1'][0]} / "
              f"{r['load1'][1]} | {r['label']} |")
    print(json.dumps(ratios, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="")
    ap.add_argument("--tflite", default="")
    ap.add_argument("--accel", choices=["gpu_f32", "gpu_default", "cpu"], default="gpu_f32")
    ap.add_argument("--no-share", action="store_true", help="pair on the GPU without constant_tensor_sharing (A/B)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--sets", default="three,one,two,most")
    ap.add_argument("--handovers", default="direct,host")
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--merge", nargs="+", default=[])
    ap.add_argument("--out", default="results/timing_mac_pair.json")
    ap.add_argument("--out-prefix", default="real_sharedstate_timing_", help="results/<prefix><label>.json (round 10: "
                                                                             "real_sharedstate_embeds_timing_)")
    a = ap.parse_args()
    if a.merge:
        return merge(a)
    assert a.label and a.tflite, "--label and --tflite are required"
    return measure(a)


if __name__ == "__main__":
    sys.exit(main())
