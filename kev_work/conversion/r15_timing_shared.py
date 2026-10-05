"""Mac timing per request of the Kev-4B shared-state pair (V2, final kernel) next to the final-kernel row graphs, and the
pair's GPU memory with and without constant tensor sharing. timing_mac_shared.py with the Kev-4B files and hidden size;
the row files are timed one at a time (a Kev-4B row graph holds about 20 GB on the GPU).

    python r15_timing_shared.py --pair Ls128_Lq64 --accel gpu_f32 --share off --tag r15R64-sp-ec-dd-vs6-in1-fn5
        --out results/timing_mac_r15_4b_pair_gpu_f32_noshare.json                           (one command)
    python r15_timing_shared.py --pair Ls128_Lq64 --memory-probe noshare --tag ... --out results/memory_mac_r15_4b_pair_noshare.json

Hold the GPU lock while it runs (the script does not take a lock). Requests (real rows of oracle/oracle_4b.json), by
question count: Ls128 + Lq64: q1 = tv4_000 (state 30, branch 64), q2 = own_ticket_01's first 2 questions (state 109,
branches 49 / 37; the fixtures have no 2-question request that fits Ls128 + Lq64: own_order_06 has a 150-token state
and own_review_05's second branch is 83 tokens), q3 = own_ticket_01 (branches 49 / 37 / 21), q5 = own_fiveq_09 (state
99, branches 33 / 43 / 33 / 29 / 29). Ls256 + Lq64 (--pair Ls256_Lq64): q2 = own_order_06 (state 150), q3 =
own_email_03 (state 167). Forms, one request = the calls that answer all its questions:
  shared_host    state_prefill once + the state outputs read to numpy and written into question_step's inputs + one
                 question_step per question (write + run + read hidden)
  shared_direct  state_prefill once + one question_step per question taking state_prefill's output buffers directly
  row            the row form: every question's row on the smallest row bucket that holds it (L128 if <= 128 tokens,
                 else L256, else L512), each row timed alone on that file (5 warm-up + 20 calls); the request's row
                 time = the sum of its rows' medians (computed: the bucket files are loaded one at a time)
--share on|off: GpuOptions(constant_tensor_sharing) for the pair on the GPU (without sharing the GPU holds the weights
once per signature). Protocol: 5 warm-up requests, then 20 timed requests (median / min / max of the request total);
per-call medians of state_prefill, the state hand-over through the host, question_step and the row calls come from the
timed requests. Every question's readout rows must be finite. Compile seconds per file.
--memory-probe share|noshare: one child process compiles the pair on the GPU (float32 activations) with sharing on or
off, runs one request (the largest of the --pair set) and reports phys_footprint after compile / after it / lifetime
max. Never overwrites --out."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from timing_mac import memory, stamp, swap_used, top_lines  # noqa: E402

K = Path(__file__).resolve().parents[1]
PAD_ID = 248044
HIDDEN = 2560
WARMUP, REPS = 5, 20
V = "v2_fp16fc_i8emb"
REQUESTS = {"Ls128_Lq64": [("q1_tv4_000", "tv4_000", None), ("q2_own_ticket_01_first2", "own_ticket_01", 2),
                            ("q3_own_ticket_01", "own_ticket_01", None), ("q5_own_fiveq_09", "own_fiveq_09", None)],
            "Ls256_Lq64": [("q2_own_order_06", "own_order_06", None), ("q3_own_email_03", "own_email_03", None)]}


def padded(ids, L):
    n = len(ids)
    assert n <= L, (n, L)
    a = np.full((1, L), PAD_ID, dtype=np.int32)
    a[0, :n] = np.asarray(ids, dtype=np.int32)
    v = np.zeros((1, L), dtype=np.float32)
    v[0, :n] = 1.0
    return a, v


def stats(xs):
    xs = np.asarray(xs, dtype=np.float64)
    return {"median": round(float(np.median(xs)), 2), "min": round(float(xs.min()), 2), "max": round(float(xs.max()), 2),
            "n": int(xs.size)}


def options(accel, threads, share=False):
    from ai_edge_litert.compiled_model import CpuOptions, GpuOptions, HardwareAccelerator, Options
    if accel == "gpu_f32":
        return Options(hardware_accelerators=HardwareAccelerator.GPU,
                       gpu_options=GpuOptions(enforce_f32=True, constant_tensor_sharing=share))
    return Options(hardware_accelerators=HardwareAccelerator.CPU, cpu_options=CpuOptions(num_threads=threads))


class Pair:
    def __init__(self, path, Ls, Lq, accel, threads, share=True):
        from ai_edge_litert.compiled_model import CompiledModel
        t0 = time.perf_counter()
        self.model = CompiledModel.from_file(str(path), options=options(accel, threads, share=share))
        self.compile_s = time.perf_counter() - t0
        self.fully = bool(self.model.is_fully_accelerated())
        self.Ls, self.Lq = Ls, Lq
        self.ss, self.sq = f"state_prefill_{Ls}", f"question_step_{Ls}_{Lq}"
        sigs = self.model.get_signature_list()
        self.names = list(sigs[self.ss]["outputs"])
        det = self.model.get_output_tensor_details(self.ss)
        self.numel = {n: int(np.prod(list(det[n]["shape"]))) for n in self.names}
        m = self.model
        self.ins_s = {n: m.create_input_buffer_by_name(self.ss, n) for n in ("ids", "valid")}
        self.outs_s = {n: m.create_output_buffer_by_name(self.ss, n) for n in self.names}
        self.ins_q = {n: m.create_input_buffer_by_name(self.sq, n) for n in sigs[self.sq]["inputs"]}
        self.outs_q = {"hidden": m.create_output_buffer_by_name(self.sq, "hidden")}
        self.direct = {**{n: self.ins_q[n] for n in ("ids", "valid", "state_valid")},
                       **{n: self.outs_s[n] for n in self.names}}

    def request(self, req, mode):
        """-> (total ms, {part: ms}, finite)."""
        parts = {"state": 0.0, "roundtrip": 0.0, "questions": []}
        t = time.perf_counter()
        self.ins_s["ids"].write(req["s_ids"])
        self.ins_s["valid"].write(req["s_valid"])
        self.model.run_by_name(self.ss, self.ins_s, self.outs_s)
        t1 = time.perf_counter()
        parts["state"] = (t1 - t) * 1000
        if mode == "host":
            for n in self.names:
                self.ins_q[n].write(np.asarray(self.outs_s[n].read(self.numel[n], np.float32), dtype=np.float32))
        self.ins_q["state_valid"].write(req["s_valid"])
        t2 = time.perf_counter()
        parts["roundtrip"] = (t2 - t1) * 1000
        finite = True
        inputs = self.ins_q if mode == "host" else self.direct
        for q_ids, q_valid, sel in req["questions"]:
            tq = time.perf_counter()
            self.ins_q["ids"].write(q_ids)
            self.ins_q["valid"].write(q_valid)
            self.model.run_by_name(self.sq, inputs, self.outs_q)
            h = np.asarray(self.outs_q["hidden"].read(self.Lq * HIDDEN, np.float32)).reshape(self.Lq, HIDDEN)
            parts["questions"].append((time.perf_counter() - tq) * 1000)
            finite &= bool(np.isfinite(h[sel]).all())
        return (time.perf_counter() - t) * 1000, parts, finite

    def close(self):
        for b in [*self.ins_s.values(), *self.outs_s.values(), *self.ins_q.values(), *self.outs_q.values()]:
            try:
                b.destroy()
            except Exception:
                pass
        self.model.close()


class Row:
    def __init__(self, path, accel, threads):
        from ai_edge_litert.compiled_model import CompiledModel
        t0 = time.perf_counter()
        self.model = CompiledModel.from_file(str(path), options=options(accel, threads))
        self.compile_s = time.perf_counter() - t0
        self.fully = bool(self.model.is_fully_accelerated())
        self.key = next(iter(self.model.get_signature_list()))
        out = self.model.get_output_tensor_details(self.key)["hidden"]
        self.L = int(list(out["shape"])[1])
        self.ins = {n: self.model.create_input_buffer_by_name(self.key, n) for n in ("ids", "valid")}
        self.outs = {"hidden": self.model.create_output_buffer_by_name(self.key, "hidden")}

    def call(self, ids, valid, sel):
        t = time.perf_counter()
        self.ins["ids"].write(ids)
        self.ins["valid"].write(valid)
        self.model.run_by_name(self.key, self.ins, self.outs)
        h = np.asarray(self.outs["hidden"].read(self.L * HIDDEN, np.float32)).reshape(self.L, HIDDEN)
        return (time.perf_counter() - t) * 1000, bool(np.isfinite(h[sel]).all())

    def close(self):
        for b in [*self.ins.values(), *self.outs.values()]:
            try:
                b.destroy()
            except Exception:
                pass
        self.model.close()


def load_requests(Ls, Lq):
    oracle = json.loads((K / "oracle/oracle_4b.json").read_text())
    reqs = {r["id"]: r for r in oracle["requests"]}
    out = {}
    for name, rid, first in REQUESTS[f"Ls{Ls}_Lq{Lq}"]:
        n = reqs[rid]["state_tokens"]
        qs = [q for q in oracle["questions"] if q["id"] == rid][: first or None]
        s_ids, s_valid = padded(qs[0]["row_ids"][:n], Ls)
        questions, rows = [], []
        for q in qs:
            sel = [q["decide_idx"] - n] + [o - n for o in q["opt_idx"]]
            questions.append((*padded(q["row_ids"][n:], Lq), sel))
            rows.append({"key": f"{q['id']}/{q['qid']}", "ids": q["row_ids"],
                         "sel": [q["decide_idx"]] + list(q["opt_idx"])})
        out[name] = {"request": rid, "questions_used": len(qs), "n_state": n, "branches": [q["row_len"] - n for q in qs], "row_lens": [q["row_len"] for q in qs],
                    "input_tokens": reqs[rid]["usage"]["input_tokens"], "s_ids": s_ids, "s_valid": s_valid,
                    "questions": questions, "rows": rows}
    return out


def time_shared(pair, req, mode):
    for _ in range(WARMUP):
        pair.request(req, mode)
    tot, state, rt, qms, finite = [], [], [], [], True
    for _ in range(REPS):
        t, parts, ok = pair.request(req, mode)
        tot.append(t)
        state.append(parts["state"])
        rt.append(parts["roundtrip"])
        qms += parts["questions"]
        finite &= ok
    return {"request_ms": stats(tot), "state_prefill_ms": stats(state), "roundtrip_ms": stats(rt),
            "question_step_ms": stats(qms), "finite": finite}


def time_rows(g, rows):
    def one():
        tot, per, ok = 0.0, [], True
        for r in rows:
            ids, valid = padded(r["ids"], g.L)
            ms, f = g.call(ids, valid, r["sel"])
            tot += ms
            per.append(ms)
            ok &= f
        return tot, per, ok
    for _ in range(WARMUP):
        one()
    tot, per, finite = [], [], True
    for _ in range(REPS):
        t, p, ok = one()
        tot.append(t)
        per += p
        finite &= ok
    return {"request_ms": stats(tot), "row_call_ms": stats(per), "finite": finite, "rows": len(rows), "L": g.L}


def time_row(g, row):
    """One row alone on a row file: 5 warm-up + 20 calls -> stats of write + run + read."""
    ids, valid = padded(row["ids"], g.L)
    for _ in range(WARMUP):
        g.call(ids, valid, row["sel"])
    ms, finite = [], True
    for _ in range(REPS):
        t, ok = g.call(ids, valid, row["sel"])
        ms.append(t)
        finite &= ok
    return {"ms": stats(ms), "finite": finite, "tokens": len(row["ids"]), "L": g.L}


def run_all(tag, Ls, Lq, accel, threads, share):
    reqs = load_requests(Ls, Lq)
    shared = K / f"exports/kev4b_sharedstate_Ls{Ls}_Lq{Lq}_{V}_{tag}.tflite"
    rows = {}
    for req in reqs.values():
        for r in req["rows"]:
            rows[r["key"]] = r
    bucket = {k: next(L for L in (128, 256, 512) if len(r["ids"]) <= L) for k, r in rows.items()}
    rowf = {L: K / f"exports/kev4b_rowprefill_L{L}_{V}_{tag}.tflite" for L in sorted(set(bucket.values()))}
    for p in [shared, *rowf.values()]:
        assert p.exists(), p
    res = {"share": share, "compile_s": {}, "fully_accelerated": {}, "memory_after_compile": {},
           "requests": {r: {} for r in reqs}, "rows": {}}
    p = Pair(shared, Ls, Lq, accel, threads, share=share)
    res["compile_s"]["shared"], res["fully_accelerated"]["shared"] = round(p.compile_s, 2), p.fully
    res["memory_after_compile"]["shared"] = memory()
    res["state_bytes"] = 4 * sum(p.numel.values())
    res["state_tensors"] = len(p.names)
    for name, req in reqs.items():
        for mode in ("host", "direct"):
            res["requests"][name][f"shared_{mode}"] = time_shared(p, req, mode)
        print(accel, share, name, {m: res["requests"][name][f"shared_{m}"]["request_ms"]["median"]
                                   for m in ("host", "direct")}, flush=True)
    res["memory_end_pair"] = memory()
    p.close()
    for L, path in rowf.items():
        g = Row(path, accel, threads)
        res["compile_s"][f"row_L{L}"], res["fully_accelerated"][f"row_L{L}"] = round(g.compile_s, 2), g.fully
        res["memory_after_compile"][f"row_L{L}"] = memory()
        for k, r in rows.items():
            if bucket[k] == L:
                res["rows"][k] = time_row(g, r)
        g.close()
    for name, req in reqs.items():
        parts = [res["rows"][r["key"]]["ms"]["median"] for r in req["rows"]]
        res["requests"][name]["row_smallest_bucket"] = {
            "request_ms": round(sum(parts), 2), "rows_ms": parts,
            "buckets": [bucket[r["key"]] for r in req["rows"]],
            "note": "sum of the per-row medians, each row alone on its smallest bucket file (computed)"}
        print(accel, name, "row", res["requests"][name]["row_smallest_bucket"]["request_ms"], flush=True)
    res["top_after"] = top_lines()
    meta = {name: {k: v for k, v in r.items() if k in ("request", "questions_used", "n_state", "branches", "row_lens",
                                                      "input_tokens")} for name, r in reqs.items()}
    files = {"shared": {"file": str(shared.relative_to(K)), "bytes": shared.stat().st_size},
             **{f"row_L{L}": {"file": str(q.relative_to(K)), "bytes": q.stat().st_size} for L, q in rowf.items()}}
    return meta, files, res


def memory_child(share, tag, Ls, Lq):
    reqs = load_requests(Ls, Lq)
    path = K / f"exports/kev4b_sharedstate_Ls{Ls}_Lq{Lq}_{V}_{tag}.tflite"
    before = memory()
    g = Pair(path, Ls, Lq, "gpu_f32", 8, share=share)
    after_compile = memory()
    t, parts, ok = g.request(list(reqs.values())[-1], "host")
    after_run = memory()
    g.close()
    print("CHILD_JSON " + json.dumps({"share": share, "file": str(path.relative_to(K)), "compile_s": round(g.compile_s, 2),
                                      "fully_accelerated": g.fully, "finite": ok, "request_ms_first": round(t, 1),
                                      "before": before, "after_compile": after_compile, "after_request": after_run}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accel", choices=["gpu_f32", "cpu8"], default="gpu_f32")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pair", default="Ls128_Lq64")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--memory-probe", choices=["share", "noshare"], default="")
    ap.add_argument("--share", choices=["on", "off"], default="on", help="GPU constant tensor sharing for the timing")
    ap.add_argument("--child", default="")
    a = ap.parse_args()
    Ls, Lq = (int(x[2:]) for x in a.pair.split("_"))
    if a.child:
        return memory_child(a.child == "share", a.tag, Ls, Lq)
    out = K / a.out
    assert not out.exists(), f"refusing to overwrite {a.out}"
    doc = {"share": a.share, "what": ("Mac timing per request of the 4B shared-state pair vs the row form (V2, final kernel)" if not
                    a.memory_probe else f"Mac GPU memory of the 4B shared-state pair, constant tensor sharing {a.memory_probe}"),
           "pair": a.pair, "tag": a.tag, "accel": a.accel, "threads_cpu": a.threads,
           "protocol": f"{WARMUP} warm-up requests, then {REPS} timed requests; ms = wall of all calls of the request "
                       "(write + run + read-back of the full hidden output per call)",
           "top_at_start": top_lines(), "swap_at_start": swap_used(), "started_at": stamp()}
    t0 = time.time()
    try:
        if a.memory_probe:
            r = subprocess.run([sys.executable, __file__, "--tag", a.tag, "--pair", a.pair, "--out", a.out,
                                "--child", a.memory_probe], capture_output=True, text=True, cwd=str(K / "scripts"))
            line = next((ln for ln in r.stdout.splitlines() if ln.startswith("CHILD_JSON ")), None)
            doc["probe"] = (json.loads(line[len("CHILD_JSON "):]) if line else
                            {"returncode": r.returncode, "stderr_tail": r.stderr.splitlines()[-15:]})
            print(json.dumps({k: doc["probe"].get(k) for k in ("share", "compile_s", "fully_accelerated", "finite")}
                             | {"fp_after_compile_gb": round(doc["probe"].get("after_compile", {}).get("phys_footprint", 0) / 1e9, 2),
                                "fp_max_gb": round(doc["probe"].get("after_request", {}).get("lifetime_max_phys_footprint", 0) / 1e9, 2)}),
                  flush=True)
        else:
            doc["requests"], doc["files"], doc["results"] = run_all(a.tag, Ls, Lq, a.accel, a.threads, a.share == "on")
        doc["status"] = "measured"
    finally:
        doc.update(top_at_end=top_lines(), swap_at_end=swap_used(), finished_at=stamp(),
                   seconds_wall=round(time.time() - t0, 1))
        doc.setdefault("status", "failed (see the log)")
        out.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(json.dumps({k: v for k, v in doc.items() if k not in ("results", "probe")}, indent=1, default=str))


if __name__ == "__main__":
    main()
